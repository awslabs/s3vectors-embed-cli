"""embed put / embed query command wiring, with AWS services replaced by fakes."""

import io
import json
import unittest
from unittest import mock

from click.testing import CliRunner
from rich.console import Console

from s3vectors.commands import embed_put as put_mod
from s3vectors.commands import embed_query as query_mod


class FakeBedrockService:
    instances = []

    def __init__(self, session, region, debug=False, console=None):
        self.calls = []
        FakeBedrockService.instances.append(self)

    def embed_with_payload(self, model, payload):
        self.calls.append(("sync", payload))
        return [{"embedding": [0.1, 0.2, 0.3], "metadata": {}}]

    def embed_async_with_payload(self, model, payload, output_uri):
        self.calls.append(("async", payload, output_uri))
        return [{"embedding": [0.4, 0.5, 0.6],
                 "metadata": {"S3VECTORS-EMBED-START-SEC": 30.0, "S3VECTORS-EMBED-END-SEC": 36.0,
                              "S3VECTORS-EMBED-TYPE": "visual-text"}}], "job-xyz"


class FakeS3VectorService:
    instances = []

    def __init__(self, session, region, debug=False, console=None):
        self.puts = []
        self.queries = []
        FakeS3VectorService.instances.append(self)

    def get_index(self, bucket, index):
        return {"index": {"dimension": 3}}

    def put_vectors_batch(self, bucket, index, vectors):
        self.puts.append(vectors)
        return [v["key"] for v in vectors]

    def query_vectors(self, bucket_name, index_name, query_embedding, k, filter_expr, return_metadata, return_distance):
        self.queries.append({"embedding": query_embedding, "k": k, "filter": filter_expr})
        return [{"vectorId": "hit-1", "similarity": 0.12, "metadata": {"x": 1}}]


def _patches(module):
    return [
        mock.patch.object(module, "BedrockService", FakeBedrockService),
        mock.patch.object(module, "S3VectorService", FakeS3VectorService),
        mock.patch.object(module, "get_region", lambda session, region=None: "us-east-1"),
        mock.patch.object(module, "get_current_account_id", lambda session=None: "123456789012"),
    ]


class CommandTestCase(unittest.TestCase):
    module = None

    def setUp(self):
        FakeBedrockService.instances.clear()
        FakeS3VectorService.instances.clear()
        self.buffer = io.StringIO()
        self.console = Console(file=self.buffer, width=400, force_terminal=False, color_system=None)
        for p in _patches(self.module):
            p.start()
            self.addCleanup(p.stop)
        self.runner = CliRunner()

    def output_json(self):
        text = self.buffer.getvalue().strip()
        start = text.index("{")
        return json.loads(text[start:])

    @property
    def bedrock(self):
        return FakeBedrockService.instances[-1]

    @property
    def s3vectors(self):
        return FakeS3VectorService.instances[-1]


COMMON = ["--vector-bucket-name", "vb", "--index-name", "idx"]


class EmbedPut(CommandTestCase):
    module = put_mod

    def invoke(self, *args):
        return self.runner.invoke(put_mod.embed_put, [*COMMON, *args],
                                  obj={"console": self.console, "aws_session": object(), "debug": False})

    def test_text_value_put_merges_user_params_and_stores_vector(self):
        result = self.invoke("--model-id", "amazon.titan-embed-text-v2:0", "--text-value", "hello",
                             "--bedrock-inference-params", '{"normalize": false}', "--key", "my-key",
                             "--metadata", '{"team": "a"}')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.bedrock.calls, [("sync", {"inputText": "hello", "dimensions": 3, "normalize": False})])
        stored = self.s3vectors.puts[0][0]
        self.assertEqual(stored["key"], "my-key")
        self.assertEqual(stored["metadata"], {"team": "a", "S3VECTORS-EMBED-SRC-CONTENT": "hello"})
        out = self.output_json()
        self.assertEqual(out["key"], "my-key")
        self.assertEqual(out["embeddingDimensions"], 3)

    def test_nova_nested_override_reaches_bedrock_in_place(self):
        result = self.invoke("--model-id", "amazon.nova-2-multimodal-embeddings-v1:0", "--text-value", "hello",
                             "--bedrock-inference-params",
                             '{"singleEmbeddingParams": {"text": {"truncationMode": "START"}}}')
        self.assertEqual(result.exit_code, 0, result.output)
        payload = self.bedrock.calls[0][1]
        self.assertEqual(payload["singleEmbeddingParams"]["text"], {"truncationMode": "START", "value": "hello"})
        self.assertEqual(payload["singleEmbeddingParams"]["embeddingPurpose"], "GENERIC_INDEX")

    def test_protected_field_is_rejected_before_any_service_is_created(self):
        for model_id, bad in (
            ("amazon.titan-embed-text-v2:0", '{"dimensions": 256}'),
            ("amazon.nova-2-multimodal-embeddings-v1:0", '{"singleEmbeddingParams": {"embeddingDimension": 1}}'),
            ("amazon.nova-2-multimodal-embeddings-v1:0", '{"singleEmbeddingParams": {"text": "x"}}'),
            ("cohere.embed-english-v3", '{"texts": ["x"]}'),
        ):
            with self.subTest(model=model_id, params=bad):
                result = self.invoke("--model-id", model_id, "--text-value", "hello", "--bedrock-inference-params", bad)
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("Cannot override system-controlled parameters", result.output)
        self.assertEqual(FakeBedrockService.instances, [])
        self.assertEqual(FakeS3VectorService.instances, [])

    def test_invalid_json_params(self):
        result = self.invoke("--model-id", "amazon.titan-embed-text-v2:0", "--text-value", "hi",
                             "--bedrock-inference-params", "{not json")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid JSON", result.output)

    def test_async_only_model_requires_output_uri(self):
        result = self.invoke("--model-id", "twelvelabs.marengo-embed-2-7-v1:0", "--text-value", "hi")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--async-output-s3-uri", result.output)

    def test_twelvelabs_async_put_reports_job_id_and_multiclip_shape(self):
        result = self.invoke("--model-id", "twelvelabs.marengo-embed-2-7-v1:0", "--video", "s3://src/v.mp4",
                             "--async-output-s3-uri", "s3://out/")
        self.assertEqual(result.exit_code, 0, result.output)
        kind, payload, output_uri = self.bedrock.calls[0]
        self.assertEqual(kind, "async")
        self.assertEqual(payload["mediaSource"]["s3Location"], {"uri": "s3://src/v.mp4", "bucketOwner": "123456789012"})
        stored = self.s3vectors.puts[0][0]
        self.assertEqual(stored["metadata"]["S3VECTORS-EMBED-START-SEC"], 30.0)
        self.assertEqual(stored["metadata"]["S3VECTORS-EMBED-SRC-LOCATION"], "s3://src/v.mp4")
        out = self.output_json()
        self.assertEqual(out["key"], stored["key"])

    def test_nova_async_rejects_inline_text(self):
        result = self.invoke("--model-id", "amazon.nova-2-multimodal-embeddings-v1:0", "--text-value", "hi",
                             "--async-output-s3-uri", "s3://out/")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("S3 URI", result.output)
        self.assertEqual(self.bedrock.calls, [])

    def test_unsupported_model(self):
        result = self.invoke("--model-id", "nope.model-v9", "--text-value", "hi")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unsupported model", result.output)

    def test_unsupported_modality_is_rejected_before_services(self):
        for model_id, flag in (("amazon.titan-embed-text-v2:0", "--image"),
                               ("cohere.embed-english-v3", "--video"),
                               ("amazon.titan-embed-image-v1", "--audio")):
            with self.subTest(model=model_id, flag=flag):
                result = self.invoke("--model-id", model_id, flag, "/tmp/x.bin")
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("does not support", result.output)
        self.assertEqual(FakeBedrockService.instances, [])

    def test_multimodal_input_still_allowed_for_titan_image(self):
        with self.runner.isolated_filesystem():
            with open("pic.png", "wb") as f:
                f.write(b"\x89PNG")
            result = self.invoke("--model-id", "amazon.titan-embed-image-v1", "--text-value", "a cat", "--image", "pic.png")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.bedrock.calls[0][1]["inputText"], "a cat")
        self.assertIn("inputImage", self.bedrock.calls[0][1])


class EmbedQuery(CommandTestCase):
    module = query_mod

    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(query_mod, "setup_aws_session", lambda profile=None, region=None: object()),
                  mock.patch.object(query_mod, "Console", lambda *a, **k: self.console)):
            p.start()
            self.addCleanup(p.stop)

    def invoke(self, *args):
        return self.runner.invoke(query_mod.embed_query, [*COMMON, *args], obj={"debug": False})

    def test_text_query_with_cohere_input_type_override(self):
        result = self.invoke("--model-id", "cohere.embed-english-v3", "--text-value", "find",
                             "--bedrock-inference-params", '{"input_type": "search_query"}', "--k", "3",
                             "--filter", '{"x": {"$eq": 1}}')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.bedrock.calls, [("sync", {"texts": ["find"], "input_type": "search_query"})])
        query = self.s3vectors.queries[0]
        self.assertEqual(query["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(query["k"], 3)
        self.assertEqual(json.loads(query["filter"]), {"x": {"$eq": 1}})
        out = self.output_json()
        self.assertEqual(out["results"][0]["Key"], "hit-1")
        self.assertEqual(out["summary"]["queryDimensions"], 3)
        self.assertEqual(out["summary"]["queryType"], "text")

    def test_nova_query_purpose_and_nested_override(self):
        result = self.invoke("--model-id", "amazon.nova-2-multimodal-embeddings-v1:0", "--text-value", "find",
                             "--bedrock-inference-params",
                             '{"singleEmbeddingParams": {"text": {"truncationMode": "NONE"}}}')
        self.assertEqual(result.exit_code, 0, result.output)
        params = self.bedrock.calls[0][1]["singleEmbeddingParams"]
        self.assertEqual(params["embeddingPurpose"], "TEXT_RETRIEVAL")
        self.assertEqual(params["text"]["truncationMode"], "NONE")

    def test_protected_field_rejected_before_services(self):
        result = self.invoke("--model-id", "amazon.nova-2-multimodal-embeddings-v1:0", "--text-value", "find",
                             "--bedrock-inference-params", '{"taskType": "SEGMENTED_EMBEDDING"}')
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Cannot override system-controlled parameters", result.output)
        self.assertEqual(FakeBedrockService.instances, [])

    def test_twelvelabs_video_query_reports_clip_window(self):
        result = self.invoke("--model-id", "twelvelabs.marengo-embed-2-7-v1:0", "--video", "s3://src/v.mp4",
                             "--async-output-s3-uri", "s3://out/", "--bedrock-inference-params",
                             '{"startSec": 30.0, "lengthSec": 6.0, "embeddingOption": ["visual-text"]}')
        self.assertEqual(result.exit_code, 0, result.output)
        kind, payload, _ = self.bedrock.calls[0]
        self.assertEqual(kind, "async")
        self.assertEqual(payload["useFixedLengthSec"], 6.0)  # derived from lengthSec
        out = self.output_json()
        self.assertEqual(out["summary"]["queryStartSec"], 30.0)
        self.assertEqual(out["summary"]["queryEndSec"], 36.0)

    def test_twelvelabs_video_query_requires_clip_params(self):
        result = self.invoke("--model-id", "twelvelabs.marengo-embed-2-7-v1:0", "--video", "s3://src/v.mp4",
                             "--async-output-s3-uri", "s3://out/")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("startSec", result.output)

    def test_unsupported_modality(self):
        result = self.invoke("--model-id", "amazon.titan-embed-text-v2:0", "--image", "/tmp/x.png")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not support image", result.output)


if __name__ == "__main__":
    unittest.main()
