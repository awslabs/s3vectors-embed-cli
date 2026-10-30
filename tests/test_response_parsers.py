"""Response parsers: sync bodies and async S3 output, normalized to [{embedding, metadata}]."""

import json
import unittest

from s3vectors.utils.models import AsyncOutput, SupportedModel


class FakeS3:
    """Minimal stand-in for boto3 S3 client get_object."""

    def __init__(self, objects):
        self.objects = objects  # {(bucket, key): str}
        self.reads = []

    def get_object(self, Bucket, Key):
        self.reads.append((Bucket, Key))
        if (Bucket, Key) not in self.objects:
            raise KeyError(f"s3://{Bucket}/{Key}")
        body = self.objects[(Bucket, Key)]

        class _Body:
            def read(self_inner):
                return body.encode("utf-8")

        return {"Body": _Body()}


def _vec(n, seed=0.1):
    return [round(seed * (i + 1), 3) for i in range(n)]


class AsyncOutputPaths(unittest.TestCase):
    def test_relative_names_resolve_under_job_prefix(self):
        s3 = FakeS3({("out", "jobs/abc/output.json"): "{}"})
        AsyncOutput(s3, "s3://out/jobs/abc").read_json("output.json")
        self.assertEqual(s3.reads, [("out", "jobs/abc/output.json")])

    def test_prefix_with_trailing_slash_is_not_doubled(self):
        s3 = FakeS3({("out", "jobs/abc/output.json"): "{}"})
        AsyncOutput(s3, "s3://out/jobs/abc/").read_json("output.json")
        self.assertEqual(s3.reads, [("out", "jobs/abc/output.json")])

    def test_bucket_root_output(self):
        s3 = FakeS3({("out", "output.json"): "{}"})
        AsyncOutput(s3, "s3://out").read_json("output.json")
        self.assertEqual(s3.reads, [("out", "output.json")])

    def test_full_uri_bypasses_prefix(self):
        s3 = FakeS3({("other", "x/y.jsonl"): "line"})
        self.assertEqual(AsyncOutput(s3, "s3://out/jobs/abc/").read_text("s3://other/x/y.jsonl"), "line")


class SyncParsers(unittest.TestCase):
    def parse(self, model_id, body):
        return SupportedModel.from_model_id(model_id).parse_response(body)

    def test_titan_v1_and_image(self):
        for model_id in ("amazon.titan-embed-text-v1", "amazon.titan-embed-image-v1"):
            self.assertEqual(self.parse(model_id, {"embedding": [1.0, 2.0], "inputTextTokenCount": 3}),
                             [{"embedding": [1.0, 2.0], "metadata": {}}])

    def test_titan_v2_plain_and_by_type(self):
        self.assertEqual(self.parse("amazon.titan-embed-text-v2:0", {"embedding": [1.0]}),
                         [{"embedding": [1.0], "metadata": {}}])
        self.assertEqual(self.parse("amazon.titan-embed-text-v2:0", {"embeddingsByType": {"binary": [1, 0]}}),
                         [{"embedding": [1, 0], "metadata": {}}])

    def test_cohere(self):
        for model_id in ("cohere.embed-english-v3", "cohere.embed-multilingual-v3"):
            self.assertEqual(self.parse(model_id, {"embeddings": [[0.5, 0.6]], "id": "x", "texts": ["t"]}),
                             [{"embedding": [0.5, 0.6], "metadata": {}}])

    def test_nova_single_embedding(self):
        body = {"embeddings": [{"embeddingType": "TEXT", "embedding": _vec(4), "truncatedCharLength": 8000}]}
        self.assertEqual(self.parse("amazon.nova-2-multimodal-embeddings-v1:0", body), [{
            "embedding": _vec(4),
            "metadata": {"S3VECTORS-EMBED-TYPE": "TEXT", "S3VECTORS-EMBED-TRUNCATED-CHAR-LEN": 8000},
        }])

    def test_nova_audio_video_separate_returns_two(self):
        body = {"embeddings": [{"embeddingType": "VIDEO", "embedding": _vec(3)},
                               {"embeddingType": "AUDIO", "embedding": _vec(3, 0.2)}]}
        results = self.parse("amazon.nova-2-multimodal-embeddings-v1:0", body)
        self.assertEqual([r["metadata"]["S3VECTORS-EMBED-TYPE"] for r in results], ["VIDEO", "AUDIO"])
        self.assertEqual(results[1]["embedding"], _vec(3, 0.2))

    def test_malformed_body_raises_value_error_naming_the_model(self):
        with self.assertRaises(ValueError) as cm:
            self.parse("amazon.titan-embed-text-v2:0", {"unexpected": 1})
        self.assertIn("amazon.titan-embed-text-v2:0", str(cm.exception))
        with self.assertRaises(ValueError):
            self.parse("cohere.embed-english-v3", {"embeddings": []})


class TwelveLabsAsyncParser(unittest.TestCase):
    model = SupportedModel.from_model_id("twelvelabs.marengo-embed-2-7-v1:0")

    def test_clips_with_timing_and_option(self):
        output = {"data": [
            {"embedding": _vec(2), "startSec": 0.0, "endSec": 6.0, "embeddingOption": "visual-text"},
            {"embedding": _vec(2, 0.3), "startSec": 6.0, "endSec": 12.0, "embeddingOption": "visual-text"},
        ]}
        s3 = FakeS3({("out", "job1/output.json"): json.dumps(output)})
        results = self.model.parse_response(AsyncOutput(s3, "s3://out/job1/"))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["metadata"], {"S3VECTORS-EMBED-START-SEC": 0.0, "S3VECTORS-EMBED-END-SEC": 6.0,
                                                  "S3VECTORS-EMBED-TYPE": "visual-text"})
        self.assertEqual(results[1]["embedding"], _vec(2, 0.3))

    def test_text_result_without_timing(self):
        s3 = FakeS3({("out", "j/output.json"): json.dumps({"data": [{"embedding": _vec(2), "embeddingOption": "text"}]})})
        results = self.model.parse_response(AsyncOutput(s3, "s3://out/j/"))
        self.assertEqual(results[0]["metadata"], {"S3VECTORS-EMBED-TYPE": "text"})


class NovaAsyncParser(unittest.TestCase):
    model = SupportedModel.from_model_id("amazon.nova-2-multimodal-embeddings-v1:0")
    PREFIX = "s3://out/job9/"

    def output(self, manifest_entries, jsonl_files=None):
        objects = {("out", "job9/segmented-embedding-result.json"):
                   json.dumps({"embeddingResults": manifest_entries})}
        for key, lines in (jsonl_files or {}).items():
            objects[("out", f"job9/{key}")] = "\n".join(json.dumps(l) for l in lines) + "\n"
        return AsyncOutput(FakeS3(objects), self.PREFIX)

    def test_video_segments(self):
        out = self.output(
            [{"embeddingType": "VIDEO", "status": "SUCCESS", "outputFileUri": f"{self.PREFIX}embedding-video.jsonl"}],
            {"embedding-video.jsonl": [
                {"embedding": _vec(2), "segmentMetadata": {"segmentIndex": 0, "segmentStartSeconds": 0.0, "segmentEndSeconds": 5.0}, "status": "SUCCESS"},
                {"embedding": _vec(2, 0.5), "segmentMetadata": {"segmentIndex": 1, "segmentStartSeconds": 5.0, "segmentEndSeconds": 10.0}, "status": "SUCCESS"},
            ]})
        results = self.model.parse_response(out)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1], {"embedding": _vec(2, 0.5), "metadata": {
            "S3VECTORS-EMBED-TYPE": "VIDEO", "S3VECTORS-EMBED-START-SEC": 5.0,
            "S3VECTORS-EMBED-END-SEC": 10.0, "S3VECTORS-EMBED-SEGMENT-IDX": 1}})

    def test_text_segments_with_char_positions_and_truncation(self):
        out = self.output(
            [{"embeddingType": "TEXT", "status": "SUCCESS", "outputFileUri": f"{self.PREFIX}embedding-text.jsonl"}],
            {"embedding-text.jsonl": [
                {"embedding": _vec(2), "segmentMetadata": {"segmentIndex": 0, "segmentStartCharPosition": 0,
                                                           "segmentEndCharPosition": 999, "truncatedCharLength": 999}},
            ]})
        results = self.model.parse_response(out)
        self.assertEqual(results[0]["metadata"], {
            "S3VECTORS-EMBED-TYPE": "TEXT", "S3VECTORS-EMBED-TRUNCATED-CHAR-LEN": 999,
            "S3VECTORS-EMBED-START-CHAR-POS": 0, "S3VECTORS-EMBED-END-CHAR-POS": 999,
            "S3VECTORS-EMBED-SEGMENT-IDX": 0})

    def test_two_modality_files_are_concatenated(self):
        out = self.output(
            [{"embeddingType": "VIDEO", "status": "SUCCESS", "outputFileUri": f"{self.PREFIX}embedding-video.jsonl"},
             {"embeddingType": "AUDIO", "status": "SUCCESS", "outputFileUri": f"{self.PREFIX}embedding-audio.jsonl"}],
            {"embedding-video.jsonl": [{"embedding": _vec(2), "segmentMetadata": {"segmentIndex": 0}}],
             "embedding-audio.jsonl": [{"embedding": _vec(2), "segmentMetadata": {"segmentIndex": 0}}]})
        types = [r["metadata"]["S3VECTORS-EMBED-TYPE"] for r in self.model.parse_response(out)]
        self.assertEqual(types, ["VIDEO", "AUDIO"])

    def test_blank_lines_are_skipped(self):
        objects = {("out", "job9/segmented-embedding-result.json"): json.dumps({"embeddingResults": [
                       {"embeddingType": "AUDIO", "status": "SUCCESS", "outputFileUri": f"{self.PREFIX}a.jsonl"}]}),
                   ("out", "job9/a.jsonl"): "\n" + json.dumps({"embedding": _vec(2)}) + "\n\n"}
        self.assertEqual(len(self.model.parse_response(AsyncOutput(FakeS3(objects), self.PREFIX))), 1)

    def test_empty_manifest_raises(self):
        with self.assertRaises(Exception) as cm:
            self.model.parse_response(self.output([]))
        self.assertIn("no embeddingResults", str(cm.exception))

    def test_failed_manifest_entry_raises_with_reason_and_fetches_nothing(self):
        out = self.output([{"embeddingType": "VIDEO", "status": "FAILURE", "failureReason": "INVALID_CONTENT",
                            "message": "Unsupported codec", "outputFileUri": f"{self.PREFIX}embedding-video.jsonl"}])
        with self.assertRaises(Exception) as cm:
            self.model.parse_response(out)
        msg = str(cm.exception)
        self.assertIn("VIDEO", msg)
        self.assertIn("INVALID_CONTENT", msg)
        self.assertIn("Unsupported codec", msg)
        self.assertEqual([k for _, k in out._s3.reads], ["job9/segmented-embedding-result.json"])

    def test_success_entry_without_output_uri_raises(self):
        with self.assertRaises(Exception) as cm:
            self.model.parse_response(self.output([{"embeddingType": "TEXT", "status": "SUCCESS"}]))
        self.assertIn("missing outputFileUri", str(cm.exception))

    def test_failed_segment_raises_and_nothing_is_returned(self):
        out = self.output(
            [{"embeddingType": "AUDIO", "status": "SUCCESS", "outputFileUri": f"{self.PREFIX}embedding-audio.jsonl"}],
            {"embedding-audio.jsonl": [
                {"embedding": _vec(2), "segmentMetadata": {"segmentIndex": 0}, "status": "SUCCESS"},
                {"segmentMetadata": {"segmentIndex": 1}, "status": "FAILURE", "failureReason": "RAI_VIOLATION_AUDIO"},
            ]})
        with self.assertRaises(Exception) as cm:
            self.model.parse_response(out)
        self.assertIn("segment 1", str(cm.exception))
        self.assertIn("RAI_VIOLATION_AUDIO", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
