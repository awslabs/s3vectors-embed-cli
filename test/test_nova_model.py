"""Unit tests for Amazon Nova Multimodal Embeddings model integration."""

import pytest
from s3vectors.utils.models import (
    get_model_info,
    validate_user_parameters,
    validate_model_modality,
    determine_content_type,
    prepare_processing_input,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def nova_model():
    """Get Nova model instance."""
    return get_model_info("amazon.nova-2-multimodal-embeddings-v1:0")


@pytest.fixture
def text_content():
    """Standard text content for testing."""
    return {"text": "Hello world", "index": {"dimensions": 1024}}


@pytest.fixture
def image_content():
    """Standard image content for testing."""
    return {"image_base64": "BASE64_IMAGE_DATA", "image_format": "jpeg", "index": {"dimensions": 1024}}


@pytest.fixture
def video_content():
    """Standard video content for testing."""
    return {"media_base64": "BASE64_VIDEO_DATA", "media_format": "mp4", "index": {"dimensions": 1024}}


@pytest.fixture
def audio_content():
    """Standard audio content for testing."""
    return {"media_base64": "BASE64_AUDIO_DATA", "media_format": "wav", "index": {"dimensions": 1024}}


# ============================================================
# Model Registration
# ============================================================

class TestModelRegistration:
    """Tests for Nova model registration and properties."""

    def test_model_found_by_id(self, nova_model):
        assert nova_model is not None

    def test_model_id(self, nova_model):
        assert nova_model.model_id == "amazon.nova-2-multimodal-embeddings-v1:0"

    def test_sync_processing(self, nova_model):
        assert nova_model.is_async() == False

    def test_supported_modalities(self, nova_model):
        assert set(nova_model.capabilities.supported_modalities) == {"text", "image", "video", "audio"}

    def test_no_multimodal_input(self, nova_model):
        """Nova takes one modality per request, not simultaneous multimodal."""
        assert nova_model.supports_multimodal_input() == False

    def test_max_file_size(self, nova_model):
        assert nova_model.capabilities.max_local_file_size == 18 * 1024 * 1024

    def test_description(self, nova_model):
        assert "Nova" in nova_model.capabilities.description


# ============================================================
# Modality Validation
# ============================================================

class TestModalityValidation:
    """Tests for modality validation."""

    @pytest.mark.parametrize("modality", ["text", "image", "video", "audio"])
    def test_valid_modalities_accepted(self, modality):
        validate_model_modality("amazon.nova-2-multimodal-embeddings-v1:0", modality)

    def test_multimodal_rejected(self):
        with pytest.raises(ValueError, match="does not support multimodal"):
            validate_model_modality("amazon.nova-2-multimodal-embeddings-v1:0", "multimodal")


# ============================================================
# Text Payload
# ============================================================

class TestTextPayload:
    """Tests for text embedding payload construction."""

    def test_basic_text_payload(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content)
        assert payload == {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 1024,
                "text": {
                    "truncationMode": "END",
                    "value": "Hello world"
                }
            }
        }

    def test_text_value_preserved(self, nova_model):
        content = {"text": "Specific test content", "index": {"dimensions": 384}}
        payload = nova_model.build_payload("text", content)
        assert payload["singleEmbeddingParams"]["text"]["value"] == "Specific test content"

    def test_default_truncation_mode(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content)
        assert payload["singleEmbeddingParams"]["text"]["truncationMode"] == "END"

    def test_default_embedding_purpose(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content)
        assert payload["singleEmbeddingParams"]["embeddingPurpose"] == "GENERIC_INDEX"

    def test_task_type(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content)
        assert payload["taskType"] == "SINGLE_EMBEDDING"


# ============================================================
# Image Payload
# ============================================================

class TestImagePayload:
    """Tests for image embedding payload construction."""

    def test_basic_image_payload(self, nova_model, image_content):
        payload = nova_model.build_payload("image", image_content)
        assert payload["taskType"] == "SINGLE_EMBEDDING"
        assert payload["singleEmbeddingParams"]["image"]["format"] == "jpeg"
        assert payload["singleEmbeddingParams"]["image"]["source"]["bytes"] == "BASE64_IMAGE_DATA"

    def test_image_dimension(self, nova_model, image_content):
        payload = nova_model.build_payload("image", image_content)
        assert payload["singleEmbeddingParams"]["embeddingDimension"] == 1024

    @pytest.mark.parametrize("fmt", ["jpeg", "png", "gif", "webp"])
    def test_all_image_formats(self, nova_model, fmt):
        content = {"image_base64": "DATA", "image_format": fmt, "index": {"dimensions": 1024}}
        payload = nova_model.build_payload("image", content)
        assert payload["singleEmbeddingParams"]["image"]["format"] == fmt


# ============================================================
# Video Payload
# ============================================================

class TestVideoPayload:
    """Tests for video embedding payload construction."""

    def test_basic_video_payload(self, nova_model, video_content):
        payload = nova_model.build_payload("video", video_content)
        assert payload["taskType"] == "SINGLE_EMBEDDING"
        assert payload["singleEmbeddingParams"]["video"]["format"] == "mp4"
        assert payload["singleEmbeddingParams"]["video"]["source"]["bytes"] == "BASE64_VIDEO_DATA"

    def test_default_embedding_mode(self, nova_model, video_content):
        payload = nova_model.build_payload("video", video_content)
        assert payload["singleEmbeddingParams"]["video"]["embeddingMode"] == "AUDIO_VIDEO_COMBINED"

    @pytest.mark.parametrize("fmt", ["mp4", "mov", "mkv", "webm", "flv", "mpeg", "mpg", "wmv", "3gp"])
    def test_all_video_formats(self, nova_model, fmt):
        content = {"media_base64": "DATA", "media_format": fmt, "index": {"dimensions": 1024}}
        payload = nova_model.build_payload("video", content)
        assert payload["singleEmbeddingParams"]["video"]["format"] == fmt


# ============================================================
# Audio Payload
# ============================================================

class TestAudioPayload:
    """Tests for audio embedding payload construction."""

    def test_basic_audio_payload(self, nova_model, audio_content):
        payload = nova_model.build_payload("audio", audio_content)
        assert payload["taskType"] == "SINGLE_EMBEDDING"
        assert payload["singleEmbeddingParams"]["audio"]["format"] == "wav"
        assert payload["singleEmbeddingParams"]["audio"]["source"]["bytes"] == "BASE64_AUDIO_DATA"

    def test_audio_dimension(self, nova_model, audio_content):
        payload = nova_model.build_payload("audio", audio_content)
        assert payload["singleEmbeddingParams"]["embeddingDimension"] == 1024

    @pytest.mark.parametrize("fmt", ["mp3", "wav", "ogg"])
    def test_all_audio_formats(self, nova_model, fmt):
        content = {"media_base64": "DATA", "media_format": fmt, "index": {"dimensions": 1024}}
        payload = nova_model.build_payload("audio", content)
        assert payload["singleEmbeddingParams"]["audio"]["format"] == fmt


# ============================================================
# Dimensions
# ============================================================

class TestDimensions:
    """Tests for configurable embedding dimensions."""

    @pytest.mark.parametrize("dim", [256, 384, 1024, 3072])
    def test_all_supported_dimensions(self, nova_model, dim):
        content = {"text": "test", "index": {"dimensions": dim}}
        payload = nova_model.build_payload("text", content)
        assert payload["singleEmbeddingParams"]["embeddingDimension"] == dim


# ============================================================
# User Parameter Remapping
# ============================================================

class TestUserParamRemapping:
    """Tests for --bedrock-inference-params remapping into nested structure."""

    def test_embedding_purpose_override(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content, {"embeddingPurpose": "TEXT_RETRIEVAL"})
        assert payload["singleEmbeddingParams"]["embeddingPurpose"] == "TEXT_RETRIEVAL"

    def test_embedding_purpose_preserves_other_fields(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content, {"embeddingPurpose": "CLUSTERING"})
        assert payload["singleEmbeddingParams"]["text"]["value"] == "Hello world"
        assert payload["singleEmbeddingParams"]["embeddingDimension"] == 1024

    def test_truncation_mode_override(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content, {"truncationMode": "START"})
        assert payload["singleEmbeddingParams"]["text"]["truncationMode"] == "START"

    def test_truncation_mode_none(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content, {"truncationMode": "NONE"})
        assert payload["singleEmbeddingParams"]["text"]["truncationMode"] == "NONE"

    def test_detail_level_override(self, nova_model, image_content):
        payload = nova_model.build_payload("image", image_content, {"detailLevel": "DOCUMENT_IMAGE"})
        assert payload["singleEmbeddingParams"]["image"]["detailLevel"] == "DOCUMENT_IMAGE"

    def test_detail_level_preserves_format(self, nova_model, image_content):
        payload = nova_model.build_payload("image", image_content, {"detailLevel": "DOCUMENT_IMAGE"})
        assert payload["singleEmbeddingParams"]["image"]["format"] == "jpeg"
        assert payload["singleEmbeddingParams"]["image"]["source"]["bytes"] == "BASE64_IMAGE_DATA"

    def test_embedding_mode_override(self, nova_model, video_content):
        payload = nova_model.build_payload("video", video_content, {"embeddingMode": "AUDIO_VIDEO_SEPARATE"})
        assert payload["singleEmbeddingParams"]["video"]["embeddingMode"] == "AUDIO_VIDEO_SEPARATE"

    def test_embedding_mode_preserves_video_fields(self, nova_model, video_content):
        payload = nova_model.build_payload("video", video_content, {"embeddingMode": "AUDIO_VIDEO_SEPARATE"})
        assert payload["singleEmbeddingParams"]["video"]["format"] == "mp4"
        assert payload["singleEmbeddingParams"]["video"]["source"]["bytes"] == "BASE64_VIDEO_DATA"

    def test_multiple_overrides(self, nova_model, video_content):
        payload = nova_model.build_payload("video", video_content, {
            "embeddingPurpose": "VIDEO_RETRIEVAL",
            "embeddingMode": "AUDIO_VIDEO_SEPARATE"
        })
        assert payload["singleEmbeddingParams"]["embeddingPurpose"] == "VIDEO_RETRIEVAL"
        assert payload["singleEmbeddingParams"]["video"]["embeddingMode"] == "AUDIO_VIDEO_SEPARATE"

    def test_wrong_modality_param_passes_through(self, nova_model, image_content):
        """truncationMode is a text param - when used with image, it passes to top level."""
        payload = nova_model.build_payload("image", image_content, {"truncationMode": "NONE"})
        assert "truncationMode" not in payload.get("singleEmbeddingParams", {}).get("image", {})
        assert payload.get("truncationMode") == "NONE"

    def test_unknown_param_passes_through(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content, {"unknownParam": "value"})
        assert payload.get("unknownParam") == "value"
        assert "unknownParam" not in payload["singleEmbeddingParams"]

    def test_empty_user_params(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content, {})
        assert payload["singleEmbeddingParams"]["embeddingPurpose"] == "GENERIC_INDEX"

    def test_none_user_params(self, nova_model, text_content):
        payload = nova_model.build_payload("text", text_content, None)
        assert payload["singleEmbeddingParams"]["embeddingPurpose"] == "GENERIC_INDEX"

    @pytest.mark.parametrize("purpose", [
        "GENERIC_INDEX", "GENERIC_RETRIEVAL", "TEXT_RETRIEVAL",
        "IMAGE_RETRIEVAL", "VIDEO_RETRIEVAL", "DOCUMENT_RETRIEVAL",
        "AUDIO_RETRIEVAL", "CLASSIFICATION", "CLUSTERING"
    ])
    def test_all_embedding_purposes(self, nova_model, text_content, purpose):
        payload = nova_model.build_payload("text", text_content, {"embeddingPurpose": purpose})
        assert payload["singleEmbeddingParams"]["embeddingPurpose"] == purpose


# ============================================================
# Response Extraction
# ============================================================

class TestResponseExtraction:
    """Tests for extracting embeddings from Nova API response."""

    def test_text_response(self, nova_model):
        response = {"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2, 0.3]}]}
        assert nova_model.extract_embedding(response) == [0.1, 0.2, 0.3]

    def test_image_response(self, nova_model):
        response = {"embeddings": [{"embeddingType": "IMAGE", "embedding": [0.4, 0.5]}]}
        assert nova_model.extract_embedding(response) == [0.4, 0.5]

    def test_video_combined_response(self, nova_model):
        response = {"embeddings": [{"embeddingType": "AUDIO_VIDEO_COMBINED", "embedding": [0.6, 0.7, 0.8]}]}
        assert nova_model.extract_embedding(response) == [0.6, 0.7, 0.8]

    def test_audio_response(self, nova_model):
        response = {"embeddings": [{"embeddingType": "AUDIO", "embedding": [0.9, 1.0]}]}
        assert nova_model.extract_embedding(response) == [0.9, 1.0]

    def test_video_separate_extracts_first(self, nova_model):
        """AUDIO_VIDEO_SEPARATE returns two embeddings; we extract the first."""
        response = {"embeddings": [
            {"embeddingType": "VIDEO", "embedding": [1.0, 2.0]},
            {"embeddingType": "AUDIO", "embedding": [3.0, 4.0]}
        ]}
        assert nova_model.extract_embedding(response) == [1.0, 2.0]

    def test_response_with_truncation_info(self, nova_model):
        """Response may include truncatedCharLength - extraction still works."""
        response = {"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1], "truncatedCharLength": 5000}]}
        assert nova_model.extract_embedding(response) == [0.1]

    def test_invalid_response_raises(self, nova_model):
        with pytest.raises(ValueError, match="Failed to extract embedding"):
            nova_model.extract_embedding({"unexpected_key": "data"})

    def test_empty_embeddings_array_raises(self, nova_model):
        with pytest.raises((ValueError, IndexError)):
            nova_model.extract_embedding({"embeddings": []})


# ============================================================
# System Key Validation
# ============================================================

class TestSystemKeyValidation:
    """Tests for parameter validation (blocking system key overrides)."""

    @pytest.mark.parametrize("content_type", ["text", "image", "video", "audio"])
    def test_system_keys_are_consistent(self, nova_model, content_type):
        system_keys = nova_model.get_system_keys(content_type)
        assert system_keys == ["taskType", "singleEmbeddingParams"]

    def test_task_type_override_blocked(self):
        system_payload = {"taskType": None, "singleEmbeddingParams": None}
        with pytest.raises(ValueError, match="Cannot override system-controlled parameters"):
            validate_user_parameters(system_payload, {"taskType": "SEGMENTED_EMBEDDING"})

    def test_single_embedding_params_override_blocked(self):
        system_payload = {"taskType": None, "singleEmbeddingParams": None}
        with pytest.raises(ValueError, match="Cannot override system-controlled parameters"):
            validate_user_parameters(system_payload, {"singleEmbeddingParams": {}})

    def test_nova_user_params_allowed(self):
        """Nova-specific params are not top-level system keys, so they pass validation."""
        system_payload = {"taskType": None, "singleEmbeddingParams": None}
        # These should NOT raise
        validate_user_parameters(system_payload, {"embeddingPurpose": "TEXT_RETRIEVAL"})
        validate_user_parameters(system_payload, {"truncationMode": "START"})
        validate_user_parameters(system_payload, {"detailLevel": "DOCUMENT_IMAGE"})
        validate_user_parameters(system_payload, {"embeddingMode": "AUDIO_VIDEO_SEPARATE"})


# ============================================================
# Processing Input Preparation
# ============================================================

class TestProcessingInput:
    """Tests for CLI input → ProcessingInput conversion."""

    def test_text_value_input(self):
        pi = prepare_processing_input("hello", None, None, None, None, False)
        assert pi.content_type == "text"
        assert pi.data == {"text": "hello"}
        assert pi.source_location == "direct_text_input"

    def test_text_file_input(self):
        pi = prepare_processing_input(None, "./doc.txt", None, None, None, False)
        assert pi.content_type == "text"
        assert pi.data == {"file_path": "./doc.txt"}
        assert pi.source_location == "./doc.txt"

    def test_image_input(self):
        pi = prepare_processing_input(None, None, "./photo.jpg", None, None, False)
        assert pi.content_type == "image"
        assert pi.data == {"file_path": "./photo.jpg"}
        assert pi.source_location == "./photo.jpg"

    def test_video_input(self):
        pi = prepare_processing_input(None, None, None, "./clip.mp4", None, False)
        assert pi.content_type == "video"
        assert pi.data == {"file_path": "./clip.mp4"}
        assert pi.source_location == "./clip.mp4"

    def test_audio_input(self):
        pi = prepare_processing_input(None, None, None, None, "./audio.wav", False)
        assert pi.content_type == "audio"
        assert pi.data == {"file_path": "./audio.wav"}
        assert pi.source_location == "./audio.wav"

    def test_s3_image_input(self):
        pi = prepare_processing_input(None, None, "s3://bucket/img.png", None, None, False)
        assert pi.content_type == "image"
        assert pi.data == {"file_path": "s3://bucket/img.png"}

    def test_metadata_passed_through(self):
        pi = prepare_processing_input("hi", None, None, None, None, False, {"key": "val"})
        assert pi.metadata == {"key": "val"}

    def test_custom_key(self):
        pi = prepare_processing_input("hi", None, None, None, None, False, None, "my-key")
        assert pi.custom_key == "my-key"

    def test_filename_as_key(self):
        pi = prepare_processing_input(None, None, "./img.jpg", None, None, False, None, None, True)
        assert pi.filename_as_key == True

    def test_key_prefix(self):
        pi = prepare_processing_input("hi", None, None, None, None, False, None, None, False, "nova/")
        assert pi.key_prefix == "nova/"


# ============================================================
# Content Type Detection
# ============================================================

class TestContentTypeDetection:
    """Tests for determine_content_type function."""

    def test_text_value(self):
        assert determine_content_type("hello", None, None, None, None) == "text"

    def test_text_file(self):
        assert determine_content_type(None, "./f.txt", None, None, None) == "text"

    def test_image(self):
        assert determine_content_type(None, None, "./img.jpg", None, None) == "image"

    def test_video(self):
        assert determine_content_type(None, None, None, "./v.mp4", None) == "video"

    def test_audio(self):
        assert determine_content_type(None, None, None, None, "./a.wav") == "audio"

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="No input type specified"):
            determine_content_type(None, None, None, None, None)
