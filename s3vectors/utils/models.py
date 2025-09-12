"""Model definitions and capabilities for S3 Vectors CLI."""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import click
from s3vectors.utils.multimodal_helpers import build_media_source


@dataclass
class ProcessingInput:
    """Unified input structure for processing."""
    content_type: str  # "text", "image", "video", "audio", "multimodal"
    data: Dict[str, Any]  # Content data
    source_location: str  # Original source location
    metadata: Dict[str, Any]  # Base metadata


def determine_content_type(text_value, text, image, video, audio, is_multimodal=False) -> str:
    """Determine content type from CLI parameters."""
    if is_multimodal:
        return "multimodal"
    if video:
        return "video"
    if audio:
        return "audio"
    if image:
        return "image"
    if text or text_value:
        return "text"
    raise ValueError("No input type specified")


def prepare_processing_input(text_value, text, image, video, audio, is_multimodal, metadata_dict=None) -> ProcessingInput:
    """Prepare unified processing input for both PUT and QUERY operations."""
    metadata = metadata_dict or {}
    
    if is_multimodal:
        return ProcessingInput(
            content_type="multimodal",
            data={"multimodal": {"text": text_value, "image_path": image}},
            source_location=image,  # Use image path as primary source location
            metadata=metadata
        )
    elif text_value:
        return ProcessingInput(
            content_type="text",
            data={"text": text_value},
            source_location="direct_text_input",
            metadata=metadata
        )
    elif text:
        return ProcessingInput(
            content_type="text",
            data={"file_path": text},
            source_location=text,
            metadata=metadata
        )
    elif image:
        return ProcessingInput(
            content_type="image",
            data={"file_path": image},
            source_location=image,
            metadata=metadata
        )
    elif video:
        return ProcessingInput(
            content_type="video",
            data={"file_path": video},
            source_location=video,
            metadata=metadata
        )
    elif audio:
        return ProcessingInput(
            content_type="audio",
            data={"file_path": audio},
            source_location=audio,
            metadata=metadata
        )
    else:
        raise click.ClickException("No valid input provided")


@dataclass
class ModelCapabilities:
    """Capabilities and properties of an embedding model."""
    is_async: bool
    supported_modalities: List[str]  # text, image, video, audio
    description: str
    supports_multimodal_input: bool = False  # Can accept multiple modalities simultaneously
    max_local_file_size: int = None  # Maximum local file size in bytes for async models (None = no limit)
    
    # Schema-based payload and response definitions
    payload_schema: Dict[str, Any] = None
    response_embedding_path: str = None  # Path to extract embedding from response


class SupportedModel(Enum):
    """Enumeration of supported embedding models with their capabilities."""
    
    # Amazon Titan Models
    TITAN_TEXT_V1 = ("amazon.titan-embed-text-v1", ModelCapabilities(
        is_async=False,
        supported_modalities=["text"],
        description="Amazon Titan Text Embeddings v1",
        payload_schema={"inputText": "{content.text}"},
        response_embedding_path="embedding"
    ))
    
    TITAN_TEXT_V2 = ("amazon.titan-embed-text-v2:0", ModelCapabilities(
        is_async=False,
        supported_modalities=["text"],
        description="Amazon Titan Text Embeddings v2",
        payload_schema={
            "inputText": "{content.text}",
            "dimensions": "{index.dimensions}"
            # normalize, embeddingTypes = user parameters (not in schema)
        },
        response_embedding_path="embeddingsByType.*|embedding"  # Handle embeddingsByType with fallback
    ))
    
    TITAN_IMAGE_V1 = ("amazon.titan-embed-image-v1", ModelCapabilities(
        is_async=False,
        supported_modalities=["text", "image"],
        description="Amazon Titan Multimodal Embeddings v1",
        supports_multimodal_input=True,
        payload_schema={
            "text": {
                "inputText": "{content.text}",
                "embeddingConfig": {"outputEmbeddingLength": "{index.dimensions}"}
            },
            "image": {
                "inputImage": "{content.image_base64}",
                "embeddingConfig": {"outputEmbeddingLength": "{index.dimensions}"}
            },
            "multimodal": {
                "inputText": "{content.text}",
                "inputImage": "{content.image_base64}",
                "embeddingConfig": {"outputEmbeddingLength": "{index.dimensions}"}
            }
            # No user parameters in schema = all user params allowed via merge
        },
        response_embedding_path="embedding"
    ))
    
    # Cohere Models
    COHERE_ENGLISH_V3 = ("cohere.embed-english-v3", ModelCapabilities(
        is_async=False,
        supported_modalities=["text", "image"],
        description="Cohere Embed English v3",
        payload_schema={
            "text": {
                "texts": ["{content.text}"],
                "input_type": "search_document"
            },
            "image": {
                "images": ["{content.image}"],
                "input_type": "image"
            }
        },
        response_embedding_path="embeddings[0]"
    ))
    
    COHERE_MULTILINGUAL_V3 = ("cohere.embed-multilingual-v3", ModelCapabilities(
        is_async=False,
        supported_modalities=["text", "image"],
        description="Cohere Embed Multilingual v3",
        payload_schema={
            "text": {
                "texts": ["{content.text}"],
                "input_type": "search_document"
            },
            "image": {
                "images": ["{content.image}"],
                "input_type": "image"
            }
        },
        response_embedding_path="embeddings[0]"
    ))
    
    # TwelveLabs Models
    TWELVELABS_MARENGO_V2_7 = ("twelvelabs.marengo-embed-2-7-v1:0", ModelCapabilities(
        is_async=True,
        supported_modalities=["text", "image", "video", "audio"],
        description="TwelveLabs Marengo Embed 2.7 v1",
        max_local_file_size=36 * 1024 * 1024,  # 36MB limit for local files
        payload_schema={
            "text": {
                "inputType": "text",
                "inputText": "{content.text}"
            },
            "video": {
                "inputType": "video",
                "mediaSource": "{media_source}"
            },
            "audio": {
                "inputType": "audio", 
                "mediaSource": "{media_source}"
            },
            "image": {
                "inputType": "image",
                "mediaSource": "{media_source}"
            }
            # All TwelveLabs user parameters (startSec, lengthSec, etc.) now allowed via merge
        },
        response_embedding_path="embedding"
    ))
    
    def __init__(self, model_id: str, capabilities: ModelCapabilities):
        self.model_id = model_id
        self.capabilities = capabilities
    
    @classmethod
    def from_model_id(cls, model_id: str) -> Optional['SupportedModel']:
        """Get SupportedModel enum from model ID string."""
        for model in cls:
            if model.model_id == model_id:
                return model
        return None
    
    def is_async(self) -> bool:
        """Check if model requires async processing."""
        return self.capabilities.is_async
    
    def get_system_keys(self, content_type: str) -> List[str]:
        """Extract top-level keys from payload schema without building payload."""
        schema = self.capabilities.payload_schema
        if isinstance(schema, dict) and content_type in schema:
            schema = schema[content_type]
        return list(schema.keys()) if isinstance(schema, dict) else []
    
    def supports_modality(self, modality: str) -> bool:
        """Check if model supports a specific modality."""
        return modality in self.capabilities.supported_modalities
    
    def supports_multimodal_input(self) -> bool:
        """Check if model supports multiple modalities simultaneously."""
        return self.capabilities.supports_multimodal_input
    
    def build_payload(self, content_type: str, content: dict, user_params: dict = None, 
                     async_config: dict = None) -> dict:
        """Build model-specific payload using schema."""
        user_params = user_params or {}
        
        # Create context for schema substitution
        context = {
            "model_id": self.model_id,
            "content_type": content_type,
            "content": content,
            "index": content.get("index", {}),  # Flatten index to root level
            "user": user_params,
            "async_config": async_config or {}
        }
        
        # Handle dynamic mediaSource for async multimodal models (video/audio/image)
        if (self.capabilities.is_async and 
            content_type in ["video", "audio", "image"] and 
            content_type in self.capabilities.supported_modalities):
            file_path = content.get("file_path", "")
            src_bucket_owner = async_config.get("src_bucket_owner") if async_config else None
            max_file_size = self.capabilities.max_local_file_size
            context["media_source"] = build_media_source(file_path, src_bucket_owner, max_file_size)
        
        # Handle conditional schemas (like Cohere)
        schema = self.capabilities.payload_schema
        if isinstance(schema, dict) and content_type in schema:
            # Use content_type-specific schema
            schema = schema[content_type]
        
        # Apply schema to get system payload
        system_payload = self._apply_schema(schema, context)
        
        # Deep merge user parameters into system payload
        return self._deep_merge(system_payload, user_params)
    
    def extract_embedding(self, response: dict) -> list:
        """Extract embedding from model response using schema."""
        return self._extract_by_path(response, self.capabilities.response_embedding_path)
    
    def _apply_schema(self, schema: Any, context: dict) -> Any:
        """Recursively apply context to schema template."""

        if isinstance(schema, dict):
            result = {}
            for key, value in schema.items():
                applied_value = self._apply_schema(value, context)
                if applied_value is not None:  # Skip None values
                    result[key] = applied_value
            return result
        elif isinstance(schema, list):
            return [self._apply_schema(item, context) for item in schema]
        elif isinstance(schema, str) and schema.startswith("{") and schema.endswith("}"):
            # Template substitution
            path = schema[1:-1]  # Remove { }
            return self._get_by_path(context, path)
        else:
            return schema
    
    def _deep_merge(self, system_payload: dict, user_params: dict) -> dict:
        """Deep merge user parameters into system payload."""

        result = system_payload.copy()
        
        for key, value in user_params.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Add new key or overwrite non-dict values
                result[key] = value
        
        return result
    
    def _get_by_path(self, obj: dict, path: str) -> Any:
        """Get value from nested dict by dot notation path."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None  # Skip optional parameters
        return current
    
    def _extract_by_path(self, obj: dict, path: str) -> Any:
        """Extract value from response using path like 'embeddings[0]' or 'embeddingsByType.*|embedding'."""
        try:
            # Handle fallback paths with | separator
            if "|" in path:
                paths = path.split("|")
                for fallback_path in paths:
                    try:
                        return self._extract_single_path(obj, fallback_path.strip())
                    except:
                        continue
                # If all paths fail, raise error with the first path
                return self._extract_single_path(obj, paths[0].strip())
            else:
                return self._extract_single_path(obj, path)
        except Exception as e:
            raise ValueError(f"Failed to extract embedding from response using path '{path}': {e}. Response keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}")
    
    def _extract_single_path(self, obj: dict, path: str) -> Any:
        """Extract value from response using a single path."""
        if path.endswith(".*"):
            # Handle dynamic object access like "embeddingsByType.*"
            key = path[:-2]  # Remove ".*"
            if key in obj and isinstance(obj[key], dict):
                # Get first value from the dictionary
                values = list(obj[key].values())
                return values[0] if values else []
            else:
                raise KeyError(f"Key '{key}' not found or not a dictionary")
        elif "[" in path:
            # Handle array access like "embeddings[0]"
            key, index_part = path.split("[", 1)
            index = int(index_part.rstrip("]"))
            return obj[key][index]
        else:
            # Simple key access
            return obj[path]


def validate_user_parameters(system_payload: Dict[str, Any], user_params: Dict[str, Any]) -> None:
    """Validate user parameters don't conflict with system parameters."""
    
    system_fields = set(system_payload.keys())  # Top-level only
    user_fields = set(user_params.keys())       # Top-level only
    
    conflicts = system_fields.intersection(user_fields)
    
    if conflicts:
        conflict_list = sorted(list(conflicts))
        raise ValueError(
            f"Cannot override system-controlled parameters: {conflict_list}. "
            f"These parameters are automatically set based on your CLI inputs."
        )


def get_model_info(model_id: str) -> Optional[SupportedModel]:
    """Get model information from model ID."""
    return SupportedModel.from_model_id(model_id)


def validate_model_modality(model_id: str, modality: str) -> None:
    """Validate that model supports the requested modality."""
    model = get_model_info(model_id)
    if not model:
        raise ValueError(f"Unsupported model: {model_id}")
    
    if not model.supports_modality(modality):
        supported = ", ".join(model.capabilities.supported_modalities)
        raise ValueError(
            f"Model {model_id} does not support {modality} input. "
            f"Supported modalities: {supported}"
        )
