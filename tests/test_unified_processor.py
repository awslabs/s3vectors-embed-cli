"""UnifiedProcessor end to end against fake Bedrock/S3 Vectors services."""

import json
import os
import tempfile
import unittest

from s3vectors.core.unified_processor import UnifiedProcessor
from s3vectors.utils.models import ProcessingInput, SupportedModel


class FakeBedrock:
    """Records the request body and returns canned parsed results."""

    def __init__(self, sync_results=None, async_results=None):
        self.sync_results = sync_results or [{"embedding": [0.1, 0.2], "metadata": {}}]
        self.async_results = async_results or [{"embedding": [0.3, 0.4], "metadata": {}}]
        self.calls = []

    def embed_with_payload(self, model, payload):
        self.calls.append(("sync", model.model_id, payload))
        return self.sync_results

    def embed_async_with_payload(self, model, payload, output_uri):
        self.calls.append(("async", model.model_id, payload, output_uri))
        return self.async_results, "job-123"


class FakeS3Vectors:
    def __init__(self):
        self.puts = []

    def put_vectors_batch(self, bucket, index, vectors):
        self.puts.append((bucket, index, vectors))
        return [v["key"] for v in vectors]


def model(model_id):
    return SupportedModel.from_model_id(model_id)


class SyncPath(unittest.TestCase):
    def setUp(self):
        self.bedrock = FakeBedrock()
        self.processor = UnifiedProcessor(self.bedrock, FakeS3Vectors())

    def test_text_value_put_with_titan(self):
        pi = ProcessingInput("text", {"text": "hello"}, "direct_text_input", {"tag": "a"}, custom_key="k1")
        result = self.processor.process(model("amazon.titan-embed-text-v2:0"), pi, {"normalize": False},
                                        precomputed_dimensions=256, operation="put")
        kind, model_id, payload = self.bedrock.calls[0]
        self.assertEqual(kind, "sync")
        self.assertEqual(payload, {"inputText": "hello", "dimensions": 256, "normalize": False})
        self.assertEqual(result.result_type, "single")
        self.assertIsNone(result.job_id)
        vector = result.vectors[0]
        self.assertEqual(vector["key"], "k1")
        self.assertEqual(vector["data"], {"float32": [0.1, 0.2]})
        self.assertEqual(vector["metadata"], {"tag": "a", "S3VECTORS-EMBED-SRC-CONTENT": "hello"})

    def test_nova_query_text_uses_retrieval_purpose_and_nested_override(self):
        pi = ProcessingInput("text", {"text": "find me"}, "direct_text_input", {})
        self.processor.process(model("amazon.nova-2-multimodal-embeddings-v1:0"), pi,
                               {"singleEmbeddingParams": {"text": {"truncationMode": "START"}}},
                               precomputed_dimensions=1024, operation="query")
        payload = self.bedrock.calls[0][2]
        self.assertEqual(payload["taskType"], "SINGLE_EMBEDDING")
        self.assertEqual(payload["singleEmbeddingParams"]["embeddingPurpose"], "TEXT_RETRIEVAL")
        self.assertEqual(payload["singleEmbeddingParams"]["text"], {"truncationMode": "START", "value": "find me"})

    def test_local_text_file_is_read_and_recorded_in_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "doc.txt")
            with open(path, "w") as f:
                f.write("file body")
            pi = ProcessingInput("text", {"file_path": path}, path, {}, filename_as_key=True, key_prefix="p/")
            result = self.processor.process(model("amazon.titan-embed-text-v1"), pi, {},
                                            precomputed_dimensions=1536, operation="put")
        self.assertEqual(self.bedrock.calls[0][2], {"inputText": "file body"})
        self.assertEqual(result.vectors[0]["key"], "p/doc.txt")
        self.assertEqual(result.vectors[0]["metadata"],
                         {"S3VECTORS-EMBED-SRC-LOCATION": path, "S3VECTORS-EMBED-SRC-CONTENT": "file body"})

    def test_local_image_feeds_titan_cohere_and_nova_in_their_formats(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pic.png")
            with open(path, "wb") as f:
                f.write(b"\x89PNG")
            for model_id, check in (
                ("amazon.titan-embed-image-v1", lambda p: p["inputImage"] == "iVBORw=="),
                ("cohere.embed-english-v3", lambda p: p["images"] == ["data:image/png;base64,iVBORw=="]),
                ("amazon.nova-2-multimodal-embeddings-v1:0",
                 lambda p: p["singleEmbeddingParams"]["image"] == {"format": "png", "source": {"bytes": "iVBORw=="}}),
            ):
                with self.subTest(model=model_id):
                    self.bedrock.calls.clear()
                    pi = ProcessingInput("image", {"file_path": path}, path, {})
                    result = self.processor.process(model(model_id), pi, {}, precomputed_dimensions=1024, operation="put")
                    self.assertTrue(check(self.bedrock.calls[0][2]), self.bedrock.calls[0][2])
                    self.assertEqual(result.vectors[0]["metadata"], {"S3VECTORS-EMBED-SRC-LOCATION": path})

    def test_multimodal_titan(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pic.jpg")
            with open(path, "wb") as f:
                f.write(b"JPG")
            pi = ProcessingInput("multimodal", {"multimodal": {"text": "a cat", "image_path": path}}, path, {})
            result = self.processor.process(model("amazon.titan-embed-image-v1"), pi, {},
                                            precomputed_dimensions=384, operation="put")
        self.assertEqual(self.bedrock.calls[0][2],
                         {"inputText": "a cat", "inputImage": "SlBH", "embeddingConfig": {"outputEmbeddingLength": 384}})
        self.assertEqual(result.vectors[0]["metadata"],
                         {"S3VECTORS-EMBED-SRC-CONTENT": "a cat", "S3VECTORS-EMBED-SRC-LOCATION": path})

    def test_nova_sync_multi_embedding_becomes_multiclip_with_model_metadata(self):
        self.bedrock.sync_results = [
            {"embedding": [1.0], "metadata": {"S3VECTORS-EMBED-TYPE": "VIDEO"}},
            {"embedding": [2.0], "metadata": {"S3VECTORS-EMBED-TYPE": "AUDIO"}},
        ]
        pi = ProcessingInput("video", {"file_path": "s3://b/clip.mp4"}, "s3://b/clip.mp4", {}, custom_key="ignored")
        result = self.processor.process(model("amazon.nova-2-multimodal-embeddings-v1:0"), pi, {},
                                        precomputed_dimensions=1024, operation="put")
        self.assertEqual(result.result_type, "multiclip")
        self.assertEqual([v["metadata"]["S3VECTORS-EMBED-TYPE"] for v in result.vectors], ["VIDEO", "AUDIO"])
        # custom key only applies to single-vector results
        self.assertNotEqual(result.vectors[0]["key"], "ignored")
        self.assertNotEqual(result.vectors[0]["key"], result.vectors[1]["key"])

    def test_bad_user_params_fail_before_any_bedrock_call(self):
        pi = ProcessingInput("text", {"text": "x"}, "direct_text_input", {})
        with self.assertRaises(ValueError):
            self.processor.process(model("amazon.titan-embed-text-v2:0"), pi, {"inputText": "y"},
                                   precomputed_dimensions=256)
        self.assertEqual(self.bedrock.calls, [])

    def test_missing_dimensions_is_an_error(self):
        pi = ProcessingInput("text", {"text": "x"}, "direct_text_input", {})
        with self.assertRaises(ValueError):
            self.processor.process(model("amazon.titan-embed-text-v2:0"), pi, {})


class AsyncPath(unittest.TestCase):
    def setUp(self):
        self.bedrock = FakeBedrock(async_results=[
            {"embedding": [0.1], "metadata": {"S3VECTORS-EMBED-START-SEC": 0.0, "S3VECTORS-EMBED-END-SEC": 5.0}},
            {"embedding": [0.2], "metadata": {"S3VECTORS-EMBED-START-SEC": 5.0, "S3VECTORS-EMBED-END-SEC": 10.0}},
        ])
        self.processor = UnifiedProcessor(self.bedrock, FakeS3Vectors())

    def test_twelvelabs_video_routes_async_and_merges_clip_metadata(self):
        pi = ProcessingInput("video", {"file_path": "s3://src/v.mp4"}, "s3://src/v.mp4", {"base": 1})
        result = self.processor.process(model("twelvelabs.marengo-embed-2-7-v1:0"), pi, {"startSec": 0},
                                        async_output_s3_uri="s3://out/", src_bucket_owner="111122223333",
                                        precomputed_dimensions=1024, operation="put")
        kind, _, payload, output_uri = self.bedrock.calls[0]
        self.assertEqual(kind, "async")
        self.assertEqual(output_uri, "s3://out/")
        self.assertEqual(payload, {"inputType": "video", "startSec": 0,
                                   "mediaSource": {"s3Location": {"uri": "s3://src/v.mp4", "bucketOwner": "111122223333"}}})
        self.assertEqual(result.job_id, "job-123")
        self.assertEqual(result.result_type, "multiclip")
        self.assertEqual(result.vectors[1]["metadata"],
                         {"base": 1, "S3VECTORS-EMBED-SRC-LOCATION": "s3://src/v.mp4",
                          "S3VECTORS-EMBED-START-SEC": 5.0, "S3VECTORS-EMBED-END-SEC": 10.0})

    def test_twelvelabs_without_output_uri_is_rejected(self):
        pi = ProcessingInput("text", {"text": "x"}, "direct_text_input", {})
        with self.assertRaises(ValueError) as cm:
            self.processor.process(model("twelvelabs.marengo-embed-2-7-v1:0"), pi, {}, precomputed_dimensions=1024)
        self.assertIn("--async-output-s3-uri", str(cm.exception))

    def test_nova_async_with_s3_source_builds_segmented_request(self):
        pi = ProcessingInput("audio", {"file_path": "s3://src/a.wav"}, "s3://src/a.wav", {})
        self.processor.process(model("amazon.nova-2-multimodal-embeddings-v1:0"), pi, {},
                               async_output_s3_uri="s3://out/", src_bucket_owner="111122223333",
                               precomputed_dimensions=1024, operation="put")
        payload = self.bedrock.calls[0][2]
        self.assertEqual(payload["taskType"], "SEGMENTED_EMBEDDING")
        self.assertEqual(payload["segmentedEmbeddingParams"]["audio"], {
            "format": "wav", "source": {"s3Location": {"uri": "s3://src/a.wav", "bucketOwner": "111122223333"}},
            "segmentationConfig": {"durationSeconds": 5}})

    def test_nova_async_rejects_local_and_inline_input_without_calling_bedrock(self):
        nova = model("amazon.nova-2-multimodal-embeddings-v1:0")
        for pi in (ProcessingInput("text", {"text": "inline"}, "direct_text_input", {}),
                   ProcessingInput("video", {"file_path": "/tmp/local.mp4"}, "/tmp/local.mp4", {})):
            with self.subTest(content_type=pi.content_type), self.assertRaises(ValueError) as cm:
                self.processor.process(nova, pi, {}, async_output_s3_uri="s3://out/", precomputed_dimensions=1024)
            self.assertIn("S3 URI", str(cm.exception))
        self.assertEqual(self.bedrock.calls, [])

    def test_nova_without_output_uri_stays_sync(self):
        pi = ProcessingInput("text", {"text": "x"}, "direct_text_input", {})
        self.processor.process(model("amazon.nova-2-multimodal-embeddings-v1:0"), pi, {}, precomputed_dimensions=1024)
        self.assertEqual(self.bedrock.calls[0][0], "sync")


class Storage(unittest.TestCase):
    def test_store_vectors_delegates_and_returns_keys(self):
        s3v = FakeS3Vectors()
        processor = UnifiedProcessor(FakeBedrock(), s3v)
        vectors = [{"key": "a", "data": {"float32": [1.0]}, "metadata": {}}]
        self.assertEqual(processor.store_vectors(vectors, "bkt", "idx"), ["a"])
        self.assertEqual(s3v.puts, [("bkt", "idx", vectors)])
        self.assertEqual(processor.store_vectors([], "bkt", "idx"), [])


if __name__ == "__main__":
    unittest.main()
