"""Non-overridable field enforcement for --bedrock-inference-params."""

import unittest

from s3vectors.utils.models import SupportedModel, deep_merge, validate_overrides


class ValidateOverrides(unittest.TestCase):
    protected = ("schemaVersion", "singleEmbeddingParams.embeddingDimension",
                 "singleEmbeddingParams.text.value", "embeddingConfig.outputEmbeddingLength")

    def assert_blocked(self, params, *expected_paths):
        with self.assertRaises(ValueError) as cm:
            validate_overrides(params, self.protected)
        for path in expected_paths:
            self.assertIn(path, str(cm.exception))

    def test_unrelated_params_pass(self):
        validate_overrides({"normalize": False, "singleEmbeddingParams": {"text": {"truncationMode": "START"}}},
                           self.protected)

    def test_empty_params_pass(self):
        validate_overrides({}, self.protected)

    def test_exact_top_level_path_is_blocked(self):
        self.assert_blocked({"schemaVersion": "x"}, "schemaVersion")

    def test_exact_nested_path_is_blocked(self):
        self.assert_blocked({"singleEmbeddingParams": {"embeddingDimension": 99}},
                            "singleEmbeddingParams.embeddingDimension")

    def test_prefix_of_protected_path_is_blocked(self):
        # Replacing the whole "text" object would drop text.value
        self.assert_blocked({"singleEmbeddingParams": {"text": "oops"}}, "singleEmbeddingParams.text")

    def test_top_level_prefix_is_blocked(self):
        self.assert_blocked({"singleEmbeddingParams": None}, "singleEmbeddingParams")
        self.assert_blocked({"embeddingConfig": {}}, "embeddingConfig")

    def test_path_below_protected_leaf_is_blocked(self):
        self.assert_blocked({"singleEmbeddingParams": {"text": {"value": {"nested": 1}}}},
                            "singleEmbeddingParams.text.value")

    def test_all_conflicts_are_reported(self):
        self.assert_blocked({"schemaVersion": "x", "singleEmbeddingParams": {"embeddingDimension": 1}},
                            "schemaVersion", "singleEmbeddingParams.embeddingDimension")

    def test_similar_prefix_without_dot_is_not_a_conflict(self):
        validate_overrides({"schemaVersionExtra": 1, "singleEmbeddingParamsX": {"embeddingDimension": 1}},
                           self.protected)


class PerModelProtection(unittest.TestCase):
    """What each model refuses, and what it lets through."""

    def model(self, model_id):
        return SupportedModel.from_model_id(model_id)

    def test_titan_v2(self):
        m = self.model("amazon.titan-embed-text-v2:0")
        m.validate_user_params({"normalize": True, "embeddingTypes": ["float"]})
        for bad in ({"inputText": "x"}, {"dimensions": 256}):
            with self.assertRaises(ValueError):
                m.validate_user_params(bad)

    def test_titan_image_protects_only_the_length_inside_embedding_config(self):
        m = self.model("amazon.titan-embed-image-v1")
        with self.assertRaises(ValueError):
            m.validate_user_params({"embeddingConfig": {"outputEmbeddingLength": 256}})
        with self.assertRaises(ValueError):
            m.validate_user_params({"inputImage": "x"})

    def test_cohere_allows_input_type(self):
        m = self.model("cohere.embed-english-v3")
        m.validate_user_params({"input_type": "search_query", "truncate": "END"})
        for bad in ({"texts": ["x"]}, {"images": ["x"]}):
            with self.assertRaises(ValueError):
                m.validate_user_params(bad)

    def test_twelvelabs(self):
        m = self.model("twelvelabs.marengo-embed-2-7-v1:0")
        m.validate_user_params({"startSec": 1, "lengthSec": 5, "embeddingOption": ["audio"]})
        for bad in ({"inputType": "x"}, {"inputText": "x"}, {"mediaSource": {"s3Location": {"uri": "s3://x"}}}):
            with self.assertRaises(ValueError):
                m.validate_user_params(bad)

    def test_nova_allows_documented_user_params_at_their_nested_path(self):
        m = self.model("amazon.nova-2-multimodal-embeddings-v1:0")
        m.validate_user_params({"singleEmbeddingParams": {
            "embeddingPurpose": "TEXT_RETRIEVAL",
            "text": {"truncationMode": "START"},
            "image": {"detailLevel": "DOCUMENT_IMAGE"},
            "video": {"embeddingMode": "AUDIO_VIDEO_SEPARATE"},
        }})
        m.validate_user_params({"segmentedEmbeddingParams": {
            "audio": {"segmentationConfig": {"durationSeconds": 10}},
            "text": {"segmentationConfig": {"maxLengthChars": 1000}},
        }})

    def test_nova_blocks_cli_owned_fields_in_both_task_shapes(self):
        m = self.model("amazon.nova-2-multimodal-embeddings-v1:0")
        bad_cases = [
            {"schemaVersion": "v2"},
            {"taskType": "SEGMENTED_EMBEDDING"},
            {"singleEmbeddingParams": {"embeddingDimension": 256}},
            {"segmentedEmbeddingParams": {"embeddingDimension": 256}},
            {"singleEmbeddingParams": {"text": {"value": "injected"}}},
            {"segmentedEmbeddingParams": {"text": {"source": {"s3Location": {"uri": "s3://evil"}}}}},
            {"singleEmbeddingParams": {"image": {"format": "gif"}}},
            {"segmentedEmbeddingParams": {"video": {"source": {"bytes": "x"}}}},
            {"singleEmbeddingParams": {"audio": "replace-everything"}},
            {"singleEmbeddingParams": []},
        ]
        for bad in bad_cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.validate_user_params(bad)

    def test_non_object_params_are_rejected(self):
        m = self.model("amazon.titan-embed-text-v2:0")
        for bad in (["normalize"], "normalize", 3):
            with self.assertRaises(ValueError):
                m.validate_user_params(bad)


class DeepMerge(unittest.TestCase):
    def test_merges_nested_and_overwrites_leaves(self):
        base = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        merged = deep_merge(base, {"a": {"c": {"x": 9}, "b": 5}, "f": 7})
        self.assertEqual(merged, {"a": {"b": 5, "c": {"d": 2, "x": 9}}, "e": 3, "f": 7})

    def test_non_dict_override_replaces_dict(self):
        self.assertEqual(deep_merge({"a": {"b": 1}}, {"a": [1]}), {"a": [1]})

    def test_inputs_are_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        deep_merge(base, override)
        self.assertEqual(base, {"a": {"b": 1}})
        self.assertEqual(override, {"a": {"c": 2}})


if __name__ == "__main__":
    unittest.main()
