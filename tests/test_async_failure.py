"""BedrockService: reporting a Failed async job with the best available reason."""

import json
import unittest
from unittest import mock

from s3vectors.core.services import BedrockService
from s3vectors.utils.models import SupportedModel


class FakeBody:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode("utf-8")


class FakeS3:
    def __init__(self, objects):
        self.objects = objects

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise Exception("An error occurred (NoSuchKey) when calling the GetObject operation: Not Found")
        return {"Body": FakeBody(self.objects[(Bucket, Key)])}


class FakeRuntime:
    """start_async_invoke succeeds; get_async_invoke reports Failed with the given message."""

    def __init__(self, failure_message):
        self.failure_message = failure_message

    def start_async_invoke(self, **kwargs):
        return {"invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-invoke/job1"}

    def get_async_invoke(self, invocationArn):
        job = {"status": "Failed"}
        if self.failure_message is not None:
            job["failureMessage"] = self.failure_message
        return job


def service(runtime, s3):
    svc = BedrockService.__new__(BedrockService)
    svc.bedrock_runtime = runtime
    svc.s3_client = s3
    svc.debug = False
    svc.console = None
    return svc


NOVA = SupportedModel.from_model_id("amazon.nova-2-multimodal-embeddings-v1:0")
TL = SupportedModel.from_model_id("twelvelabs.marengo-embed-2-7-v1:0")
OUT = "s3://out/jobs/"
MANIFEST = json.dumps({"embeddingResults": [{
    "status": "FAILURE", "embeddingType": "AUDIO_VIDEO_COMBINED", "failureReason": "INVALID_CONTENT",
    "message": "Media bytes are malformed. Please provide a valid media file and try again."}]})


class FailedAsyncJob(unittest.TestCase):
    def run_job(self, model, runtime, s3):
        with self.assertRaises(Exception) as cm:
            service(runtime, s3).embed_async_with_payload(model, {"x": 1}, OUT)
        return str(cm.exception)

    def test_empty_job_message_uses_the_model_manifest(self):
        s3 = FakeS3({("out", "jobs/job1/segmented-embedding-result.json"): MANIFEST})
        msg = self.run_job(NOVA, FakeRuntime(""), s3)
        self.assertIn("INVALID_CONTENT", msg)
        self.assertIn("Media bytes are malformed", msg)
        self.assertIn("AUDIO_VIDEO_COMBINED", msg)

    def test_exception_name_only_is_replaced_by_manifest_detail(self):
        s3 = FakeS3({("out", "jobs/job1/segmented-embedding-result.json"): MANIFEST})
        msg = self.run_job(NOVA, FakeRuntime("ValidationException"), s3)
        self.assertIn("INVALID_CONTENT", msg)

    def test_no_manifest_falls_back_to_job_message(self):
        msg = self.run_job(TL, FakeRuntime("Given file is not a valid image"), FakeS3({}))
        self.assertEqual(msg, "Async embedding failed: Given file is not a valid image")

    def test_no_manifest_and_empty_message_is_still_descriptive(self):
        msg = self.run_job(NOVA, FakeRuntime(""), FakeS3({}))
        self.assertEqual(msg, "Async embedding failed: Bedrock reported the job as Failed without a failure message")

    def test_missing_failure_message_key(self):
        msg = self.run_job(NOVA, FakeRuntime(None), FakeS3({}))
        self.assertTrue(msg.startswith("Async embedding failed: "))
        self.assertNotEqual(msg, "Async embedding failed: ")


if __name__ == "__main__":
    unittest.main()
