"""Request builders: parity with the previous schema-based payloads, plus user-parameter merging."""

import json
import unittest
from pathlib import Path

from s3vectors.utils.models import RequestContext, SupportedModel

FIXTURE = Path(__file__).parent / "fixtures" / "request_payloads.json"


def _ctx(case):
    return RequestContext(
        operation=case["operation"], mode=case["mode"], modality=case["modality"], **case["ctx"]
    )


class RequestPayloadParity(unittest.TestCase):
    """Every builder must produce exactly the request body the schema engine did.

    The fixture was captured from the PR #17 head (schema-based build_payload)
    with no user parameters, across all models, modalities, sync/async and
    put/query.
    """

    def test_fixture_covers_every_model(self):
        cases = json.loads(FIXTURE.read_text())
        covered = {c["model_id"] for c in cases}
        self.assertEqual(covered, {m.model_id for m in SupportedModel})

    def test_builders_match_captured_payloads(self):
        cases = json.loads(FIXTURE.read_text())
        self.assertGreater(len(cases), 40)
        for case in cases:
            with self.subTest(model=case["model_id"], modality=case["modality"],
                              mode=case["mode"], operation=case["operation"],
                              file_path=case["ctx"].get("file_path")):
                model = SupportedModel.from_model_id(case["model_id"])
                self.assertEqual(model.build_request(_ctx(case), {}), case["expected"])


class UserParameterMerge(unittest.TestCase):
    nova = SupportedModel.from_model_id("amazon.nova-2-multimodal-embeddings-v1:0")
    titan_v2 = SupportedModel.from_model_id("amazon.titan-embed-text-v2:0")
    cohere = SupportedModel.from_model_id("cohere.embed-english-v3")
    twelvelabs = SupportedModel.from_model_id("twelvelabs.marengo-embed-2-7-v1:0")

    def nova_text(self, **overrides):
        ctx = RequestContext(operation="put", mode="sync", modality="text", dimensions=1024, text="hi")
        return self.nova.build_request(ctx, overrides)

    def test_flat_model_params_merge_at_root(self):
        ctx = RequestContext(operation="put", mode="sync", modality="text", dimensions=512, text="hi")
        body = self.titan_v2.build_request(ctx, {"normalize": False, "embeddingTypes": ["binary"]})
        self.assertEqual(body, {"inputText": "hi", "dimensions": 512, "normalize": False, "embeddingTypes": ["binary"]})

    def test_cohere_input_type_is_user_overridable(self):
        ctx = RequestContext(operation="query", mode="sync", modality="text", dimensions=1024, text="q")
        body = self.cohere.build_request(ctx, {"input_type": "search_query"})
        self.assertEqual(body["input_type"], "search_query")
        self.assertEqual(body["texts"], ["q"])

    def test_twelvelabs_clip_params_merge_at_root(self):
        ctx = RequestContext(operation="query", mode="async", modality="video", dimensions=1024,
                             file_path="s3://b/v.mp4", src_bucket_owner="123456789012")
        body = self.twelvelabs.build_request(
            ctx, {"startSec": 30.0, "lengthSec": 6.0, "useFixedLengthSec": 6.0, "embeddingOption": ["visual-text"]})
        self.assertEqual(body["inputType"], "video")
        self.assertEqual(body["startSec"], 30.0)
        self.assertEqual(body["embeddingOption"], ["visual-text"])
        self.assertEqual(body["mediaSource"]["s3Location"]["bucketOwner"], "123456789012")

    def test_nested_override_lands_in_place(self):
        body = self.nova_text(singleEmbeddingParams={"text": {"truncationMode": "START"}})
        self.assertEqual(body["singleEmbeddingParams"]["text"], {"truncationMode": "START", "value": "hi"})

    def test_nested_override_keeps_siblings(self):
        body = self.nova_text(singleEmbeddingParams={"embeddingPurpose": "TEXT_RETRIEVAL"})
        params = body["singleEmbeddingParams"]
        self.assertEqual(params["embeddingPurpose"], "TEXT_RETRIEVAL")
        self.assertEqual(params["embeddingDimension"], 1024)
        self.assertEqual(params["text"]["value"], "hi")

    def test_unknown_nested_field_passes_through(self):
        body = self.nova_text(singleEmbeddingParams={"text": {"futureKnob": 3}})
        self.assertEqual(body["singleEmbeddingParams"]["text"]["futureKnob"], 3)

    def test_nova_async_segmentation_override_replaces_default(self):
        ctx = RequestContext(operation="put", mode="async", modality="video", dimensions=1024,
                             file_path="s3://b/v.mp4", src_bucket_owner="123456789012")
        body = self.nova.build_request(
            ctx, {"segmentedEmbeddingParams": {"video": {"segmentationConfig": {"durationSeconds": 12},
                                                         "embeddingMode": "AUDIO_VIDEO_SEPARATE"}}})
        video = body["segmentedEmbeddingParams"]["video"]
        self.assertEqual(video["segmentationConfig"], {"durationSeconds": 12})
        self.assertEqual(video["embeddingMode"], "AUDIO_VIDEO_SEPARATE")
        self.assertEqual(video["format"], "mp4")

    def test_flat_nova_key_is_not_relocated(self):
        # Nested is the only accepted form; a flat key stays at the root.
        body = self.nova_text(truncationMode="START")
        self.assertEqual(body["truncationMode"], "START")
        self.assertEqual(body["singleEmbeddingParams"]["text"]["truncationMode"], "END")

    def test_builder_output_is_not_mutated_by_merge(self):
        ctx = RequestContext(operation="put", mode="sync", modality="text", dimensions=1024, text="hi")
        first = self.nova.build_request(ctx, {"singleEmbeddingParams": {"text": {"truncationMode": "START"}}})
        second = self.nova.build_request(ctx, {})
        self.assertEqual(second["singleEmbeddingParams"]["text"]["truncationMode"], "END")
        self.assertNotEqual(first, second)


class NovaBuilderDetails(unittest.TestCase):
    nova = SupportedModel.from_model_id("amazon.nova-2-multimodal-embeddings-v1:0")

    def test_query_purpose_is_modality_specific(self):
        for modality, expected in (("text", "TEXT_RETRIEVAL"), ("image", "IMAGE_RETRIEVAL"),
                                   ("video", "VIDEO_RETRIEVAL"), ("audio", "AUDIO_RETRIEVAL")):
            ctx = RequestContext(operation="query", mode="sync", modality=modality, dimensions=256,
                                 text="q", file_path=f"s3://b/f.{ {'image': 'png', 'video': 'mp4', 'audio': 'wav'}.get(modality, 'txt') }")
            body = self.nova.build_request(ctx, {})
            self.assertEqual(body["singleEmbeddingParams"]["embeddingPurpose"], expected)

    def test_local_image_uses_inline_bytes_and_normalizes_jpg(self):
        ctx = RequestContext(operation="put", mode="sync", modality="image", dimensions=1024,
                             file_path="/tmp/photo.JPG", image_base64="QUJD")
        image = self.nova.build_request(ctx, {})["singleEmbeddingParams"]["image"]
        self.assertEqual(image, {"format": "jpeg", "source": {"bytes": "QUJD"}})

    def test_sync_s3_image_carries_bucket_owner(self):
        ctx = RequestContext(operation="put", mode="sync", modality="image", dimensions=1024,
                             file_path="s3://b/photo.png", image_base64="QUJD", src_bucket_owner="111122223333")
        image = self.nova.build_request(ctx, {})["singleEmbeddingParams"]["image"]
        self.assertEqual(image["source"], {"s3Location": {"uri": "s3://b/photo.png", "bucketOwner": "111122223333"}})

    def test_async_text_uses_source_not_value(self):
        ctx = RequestContext(operation="put", mode="async", modality="text", dimensions=1024,
                             text="file content", file_path="s3://b/doc.txt")
        text = self.nova.build_request(ctx, {})["segmentedEmbeddingParams"]["text"]
        self.assertNotIn("value", text)
        self.assertEqual(text["source"], {"s3Location": {"uri": "s3://b/doc.txt"}})
        self.assertEqual(text["segmentationConfig"], {})

    def test_async_image_has_no_segmentation_config(self):
        ctx = RequestContext(operation="put", mode="async", modality="image", dimensions=1024,
                             file_path="s3://b/photo.png")
        image = self.nova.build_request(ctx, {})["segmentedEmbeddingParams"]["image"]
        self.assertNotIn("segmentationConfig", image)


if __name__ == "__main__":
    unittest.main()
