"""Model definitions and capabilities for S3 Vectors CLI."""

import uuid
import base64
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import click
from s3vectors.utils.multimodal_helpers import build_media_source, is_local_path, encode_file_to_base64


@dataclass
class ProcessingInput:
    """Unified input structure for processing."""
    content_type: str  # "text", "image", "video", "audio", "multimodal"
    data: Dict[str, Any]  # Content data
    source_location: str  # Original source location
    metadata: Dict[str, Any]  # Base metadata
    custom_key: Optional[str] = None  # Custom vector key
    filename_as_key: bool = False  # Use filename as vector key
    key_prefix: Optional[str] = None  # Prefix to prepend to all vector keys


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


def prepare_processing_input(text_value, text, image, video, audio, is_multimodal, metadata_dict=None, custom_key=None, filename_as_key=False, key_prefix=None) -> ProcessingInput:
    """Prepare unified processing input for both PUT and QUERY operations."""
    metadata = metadata_dict or {}
    
    if is_multimodal:
        return ProcessingInput(
            content_type="multimodal",
            data={"multimodal": {"text": text_value, "image_path": image}},
            source_location=image,  # Use image path as primary source location
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif text_value:
        return ProcessingInput(
            content_type="text",
            data={"text": text_value},
            source_location="direct_text_input",
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif text:
        return ProcessingInput(
            content_type="text",
            data={"file_path": text},
            source_location=text,
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif image:
        return ProcessingInput(
            content_type="image",
            data={"file_path": image},
            source_location=image,
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif video:
        return ProcessingInput(
            content_type="video",
            data={"file_path": video},
            source_location=video,
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif audio:
        return ProcessingInput(
            content_type="audio",
            data={"file_path": audio},
            source_location=audio,
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
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

    # --- New declarative fields for data-driven model behavior ---
    supports_sync: bool = True  # Whether the model can use invoke_model (sync)
    supports_async: bool = False  # Whether the model can use start_async_invoke
    content_source_format: str = None  # Controls how file content is placed in the payload (None, "media_source", "nova_source")
    content_source_requires_format: bool = False  # When True, content source builder adds a "format" field from file extension
    async_result_config: Dict[str, Any] = None  # Configuration dict for parsing async results from S3
    query_defaults: Dict[str, Dict[str, Any]] = None  # Default params applied during query operations, keyed by content_type
    put_defaults: Dict[str, Dict[str, Any]] = None  # Default params applied during put operations, keyed by content_type
    metadata_mapping: Dict[str, str] = None  # Maps response field names (dot notation) to vector metadata key names
    supported_dimensions: List[int] = None  # Valid dimension values for validation
    default_dimensions: int = None  # Default dimension if not specified by the index
    content_source_params: List[str] = None  # Resolved param keys to include in content source object (e.g., ["embeddingMode", "detailLevel", "segmentationConfig"])
    async_requires_s3_source: bool = False  # When True, async mode requires S3 source (local text uploaded to S3 before submission)


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
    
    # Amazon Nova Models
    NOVA_MME_V1 = ("amazon.nova-2-multimodal-embeddings-v1:0", ModelCapabilities(
        is_async=False,  # Legacy field; not used for routing
        supports_sync=True,
        supports_async=True,
        supported_modalities=["text", "image", "video", "audio"],
        description="Amazon Nova Multimodal Embeddings v1",
        supports_multimodal_input=False,  # One modality per request
        supported_dimensions=[256, 384, 1024, 3072],
        default_dimensions=3072,
        content_source_format="nova_source",
        content_source_requires_format=True,
        async_requires_s3_source=True,
        content_source_params=["embeddingMode", "detailLevel", "segmentationConfig"],
        payload_schema={
            "sync": {
                "text": {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": "{resolved.embeddingPurpose}",
                        "embeddingDimension": "{index.dimensions}",
                        "text": {
                            "truncationMode": "{resolved.truncationMode}",
                            "value": "{content.text}"
                        }
                    }
                },
                "image": {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": "{resolved.embeddingPurpose}",
                        "embeddingDimension": "{index.dimensions}",
                        "image": "{content_source}"
                    }
                },
                "video": {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": "{resolved.embeddingPurpose}",
                        "embeddingDimension": "{index.dimensions}",
                        "video": "{content_source}"
                    }
                },
                "audio": {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": "{resolved.embeddingPurpose}",
                        "embeddingDimension": "{index.dimensions}",
                        "audio": "{content_source}"
                    }
                }
            },
            "async": {
                "text": {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SEGMENTED_EMBEDDING",
                    "segmentedEmbeddingParams": {
                        "embeddingPurpose": "{resolved.embeddingPurpose}",
                        "embeddingDimension": "{index.dimensions}",
                        "text": {
                            "truncationMode": "{resolved.truncationMode}",
                            "source": "{content_source}",
                            "segmentationConfig": "{resolved.segmentationConfig}"
                        }
                    }
                },
                "image": {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SEGMENTED_EMBEDDING",
                    "segmentedEmbeddingParams": {
                        "embeddingPurpose": "{resolved.embeddingPurpose}",
                        "embeddingDimension": "{index.dimensions}",
                        "image": "{content_source}"
                    }
                },
                "video": {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SEGMENTED_EMBEDDING",
                    "segmentedEmbeddingParams": {
                        "embeddingPurpose": "{resolved.embeddingPurpose}",
                        "embeddingDimension": "{index.dimensions}",
                        "video": "{content_source}"
                    }
                },
                "audio": {
                    "schemaVersion": "nova-multimodal-embed-v1",
                    "taskType": "SEGMENTED_EMBEDDING",
                    "segmentedEmbeddingParams": {
                        "embeddingPurpose": "{resolved.embeddingPurpose}",
                        "embeddingDimension": "{index.dimensions}",
                        "audio": "{content_source}"
                    }
                }
            }
        },
        response_embedding_path="embeddings",  # List of embedding objects
        async_result_config={
            "type": "manifest_jsonl",
            "result_file": "segmented-embedding-result.json",
            "embedding_key": "embedding",
            "manifest_results_path": "embeddingResults",
            "manifest_output_uri_key": "outputFileUri",
            "manifest_type_key": "embeddingType",
            "manifest_status_key": "status",
        },
        query_defaults={
            "text": {"embeddingPurpose": "TEXT_RETRIEVAL", "truncationMode": "END", "segmentationConfig": {}},
            "image": {"embeddingPurpose": "IMAGE_RETRIEVAL"},
            "video": {"embeddingPurpose": "VIDEO_RETRIEVAL", "embeddingMode": "AUDIO_VIDEO_COMBINED"},
            "audio": {"embeddingPurpose": "AUDIO_RETRIEVAL"},
        },
        put_defaults={
            "text": {"embeddingPurpose": "GENERIC_INDEX", "truncationMode": "END", "segmentationConfig": {}},
            "image": {"embeddingPurpose": "GENERIC_INDEX"},
            "video": {"embeddingPurpose": "GENERIC_INDEX", "embeddingMode": "AUDIO_VIDEO_COMBINED"},
            "audio": {"embeddingPurpose": "GENERIC_INDEX"},
        },
        metadata_mapping={
            "embeddingType": "S3VECTORS-EMBED-TYPE",
            "segmentMetadata.segmentStartSeconds": "S3VECTORS-EMBED-START-SEC",
            "segmentMetadata.segmentEndSeconds": "S3VECTORS-EMBED-END-SEC",
            "segmentMetadata.segmentStartCharPosition": "S3VECTORS-EMBED-START-CHAR-POS",
            "segmentMetadata.segmentEndCharPosition": "S3VECTORS-EMBED-END-CHAR-POS",
            "segmentMetadata.segmentIndex": "S3VECTORS-EMBED-SEGMENT-IDX",
            "truncatedCharLength": "S3VECTORS-EMBED-TRUNCATED-CHAR-LEN",
        },
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
        response_embedding_path="embedding",
        supports_sync=False,
        supports_async=True,
        content_source_format="media_source",
        async_result_config={
            "type": "single_file",
            "result_file": "output.json",
            "data_path": "data",
            "embedding_key": "embedding",
        },
        metadata_mapping={
            "startSec": "S3VECTORS-EMBED-START-SEC",
            "endSec": "S3VECTORS-EMBED-END-SEC",
            "embeddingOption": "S3VECTORS-EMBED-TYPE",
        },
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
        """Return system-controlled parameter keys for a content type.

        These are keys the CLI populates itself (schema literals like
        schemaVersion/taskType, index-derived values like embeddingDimension,
        and content/source fields) which a user must NOT override via
        --bedrock-inference-params. User-overridable keys are excluded: operation
        defaults referenced via {resolved.X} (e.g. truncationMode,
        embeddingPurpose) and content_source_params (e.g. embeddingMode,
        detailLevel, segmentationConfig).

        The schema is walked recursively and across any sync/async layer so
        nested system keys (e.g. embeddingDimension) are detected for models like
        Nova whose schema is nested. Flat-schema models are unaffected.
        """
        schema = self.capabilities.payload_schema
        if not isinstance(schema, dict):
            return []

        # Collect the schema branch(es) for this content type, descending through
        # any sync/async layer.
        branches = []
        if "sync" in schema and "async" in schema:
            for mode in ("sync", "async"):
                mode_schema = schema.get(mode, {})
                if isinstance(mode_schema, dict) and content_type in mode_schema:
                    branches.append(mode_schema[content_type])
        elif content_type in schema:
            branches.append(schema[content_type])
        else:
            branches.append(schema)

        keys: set = set()
        overridable: set = set()
        for branch in branches:
            self._collect_schema_keys(branch, keys, overridable)

        # content_source_params are user-overridable model parameters.
        if self.capabilities.content_source_params:
            overridable.update(self.capabilities.content_source_params)

        return sorted(keys - overridable)

    def _collect_schema_keys(self, schema: Any, keys: set, overridable: set) -> None:
        """Recursively collect schema keys, marking {resolved.X} keys overridable."""
        if isinstance(schema, dict):
            for key, value in schema.items():
                keys.add(key)
                if (isinstance(value, str) and value.startswith("{resolved.")
                        and value.endswith("}")):
                    overridable.add(key)
                else:
                    self._collect_schema_keys(value, keys, overridable)
        elif isinstance(schema, list):
            for item in schema:
                self._collect_schema_keys(item, keys, overridable)
    
    def supports_modality(self, modality: str) -> bool:
        """Check if model supports a specific modality."""
        return modality in self.capabilities.supported_modalities
    
    def supports_multimodal_input(self) -> bool:
        """Check if model supports multiple modalities simultaneously."""
        return self.capabilities.supports_multimodal_input
    
    def build_payload(self, content_type: str, content: dict, user_params: dict = None, 
                     async_config: dict = None, operation: str = "put") -> dict:
        """Build model-specific payload using schema."""
        user_params = user_params or {}
        
        # Determine processing mode
        is_async_mode = bool(async_config and async_config.get("output_s3_uri"))
        
        # Resolve operation defaults merged with user params
        resolved_params = self._resolve_operation_params(content_type, user_params, operation)
        
        # Create context for schema substitution
        context = {
            "model_id": self.model_id,
            "content_type": content_type,
            "content": content,
            "index": content.get("index", {}),
            "user": user_params,
            "resolved": resolved_params,
            "async_config": async_config or {}
        }
        
        # Build content source if needed
        if self.capabilities.content_source_format:
            file_path = content.get("file_path", "")
            src_bucket_owner = async_config.get("src_bucket_owner") if async_config else None
            if file_path:
                context["content_source"] = self._build_content_source(
                    file_path, src_bucket_owner, content, resolved_params, content_type
                )
                # For backward compatibility with TwelveLabs schemas that use {media_source}
                if self.capabilities.content_source_format == "media_source":
                    context["media_source"] = context["content_source"]
            elif content_type == "text" and is_async_mode and content.get("text"):
                # Some models (e.g., Nova segmented) use a source object for
                # async text instead of an inline value. Only build the
                # content_source if the schema actually references it — models
                # that use {content.text} directly (e.g., TwelveLabs) don't need
                # one and would break if we tried to encode an empty file_path.
                text_schema = self._select_schema(content_type, is_async_mode)
                if self._schema_references_template(text_schema, "content_source"):
                    context["content_source"] = self._build_content_source(
                        "", src_bucket_owner, content, resolved_params, content_type
                    )
        
        # Select schema (sync/async aware)
        schema = self._select_schema(content_type, is_async_mode)
        
        # Apply schema to get system payload
        system_payload = self._apply_schema(schema, context)
        
        # Determine which user params were already consumed by the schema so we
        # don't redundantly re-add them at the payload root. Params consumed via
        # {resolved.X} templates (e.g. truncationMode, embeddingPurpose) and via
        # content_source_params (e.g. embeddingMode, detailLevel, segmentationConfig)
        # are placed into nested locations by the schema; merging them again at the
        # root produces unknown top-level fields that Bedrock rejects (Nova).
        consumed_keys = self._collect_resolved_keys(schema)
        if self.capabilities.content_source_params:
            consumed_keys.update(self.capabilities.content_source_params)
        merge_params = {k: v for k, v in user_params.items() if k not in consumed_keys}
        
        # Deep merge remaining user parameters into system payload
        return self._deep_merge(system_payload, merge_params)
    
    def _schema_references_template(self, schema: Any, template_name: str) -> bool:
        """Check if a schema references a given template variable like {content_source}."""
        if isinstance(schema, dict):
            return any(self._schema_references_template(v, template_name) for v in schema.values())
        elif isinstance(schema, list):
            return any(self._schema_references_template(item, template_name) for item in schema)
        elif isinstance(schema, str):
            return schema == f"{{{template_name}}}"
        return False

    def _collect_resolved_keys(self, schema: Any) -> set:
        """Collect user-param keys consumed by the schema via {resolved.X} templates.
        
        Walks the schema template recursively and returns the set of top-level
        keys referenced through the 'resolved' context (which is the merge of
        operation defaults and user params). These keys are already substituted
        into nested positions and must be excluded from the top-level deep merge.
        Models with flat schemas (Titan/Cohere/TwelveLabs) use no {resolved.X}
        templates, so this returns an empty set and their merge is unchanged.
        """
        keys = set()
        if isinstance(schema, dict):
            for value in schema.values():
                keys |= self._collect_resolved_keys(value)
        elif isinstance(schema, list):
            for item in schema:
                keys |= self._collect_resolved_keys(item)
        elif isinstance(schema, str) and schema.startswith("{") and schema.endswith("}"):
            path = schema[1:-1]
            if path.startswith("resolved."):
                keys.add(path.split(".")[1])
        return keys
    
    def _resolve_operation_params(self, content_type: str, user_params: dict,
                                  operation: str = "put") -> dict:
        """Merge operation defaults with user params. User params win."""
        defaults_map = (self.capabilities.query_defaults if operation == "query"
                        else self.capabilities.put_defaults)
        if not defaults_map:
            return user_params.copy()
        
        defaults = defaults_map.get(content_type, {})
        merged = defaults.copy()
        merged.update(user_params)
        return merged
    
    def _select_schema(self, content_type: str, is_async_mode: bool) -> dict:
        """Select the correct schema branch for content_type and sync/async mode."""
        schema = self.capabilities.payload_schema
        
        if isinstance(schema, dict):
            # Check for sync/async top-level keys
            if "sync" in schema and "async" in schema:
                mode_key = "async" if is_async_mode else "sync"
                schema = schema[mode_key]
            
            # Check for content_type sub-key
            if content_type in schema:
                schema = schema[content_type]
        
        return schema
    
    def _build_content_source(self, file_path: str, src_bucket_owner: str = None,
                              content: dict = None,
                              resolved_params: dict = None,
                              content_type: str = None) -> Any:
        """Build content source object based on model's content_source_format."""
        fmt = self.capabilities.content_source_format
        
        if fmt == "media_source":
            # TwelveLabs style
            return build_media_source(file_path, src_bucket_owner,
                                      self.capabilities.max_local_file_size)
        elif fmt == "nova_source":
            is_text = (content_type == "text")
            
            if is_text:
                # Text follows the text-input flow: inline and local text are
                # already read into content["text"], so encode that string as a
                # bytes SourceObject. Only S3 text is referenced by location (so
                # large objects aren't pulled into the payload). Text
                # SourceObjects carry no "format" field.
                if file_path and not is_local_path(file_path):
                    s3_loc = {"uri": file_path}
                    if src_bucket_owner:
                        s3_loc["bucketOwner"] = src_bucket_owner
                    return {"s3Location": s3_loc}
                text_value = (content or {}).get("text", "")
                return {"bytes": base64.b64encode(text_value.encode("utf-8")).decode("utf-8")}
            
            # Media (image/audio/video): build the whole modality object
            # ({format, source, <media params>}) since the schema maps
            # {content_source} to it directly.
            if is_local_path(file_path):
                base64_data = (content.get("image_base64") if content else None) or encode_file_to_base64(file_path)
                source_inner = {"bytes": base64_data}
            else:
                s3_loc = {"uri": file_path}
                if src_bucket_owner:
                    s3_loc["bucketOwner"] = src_bucket_owner
                source_inner = {"s3Location": s3_loc}
            
            source_obj = {}
            if self.capabilities.content_source_requires_format and file_path:
                ext = Path(file_path).suffix.lstrip(".").lower()
                if ext == "jpg":
                    ext = "jpeg"
                source_obj["format"] = ext
            source_obj["source"] = source_inner
            if resolved_params and self.capabilities.content_source_params:
                for key in self.capabilities.content_source_params:
                    if key in resolved_params:
                        source_obj[key] = resolved_params[key]
            
            return source_obj
        else:
            return None
    
    def extract_embedding(self, response: dict) -> Any:
        """Extract embedding(s) from model response.
        
        Returns either:
        - List[float]: single embedding vector
        - List[Dict]: list of embedding result objects (for multi-embedding responses)
        """
        result = self._extract_by_path(response, self.capabilities.response_embedding_path)
        
        # If result is a list of dicts (multi-embedding), return as-is
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result  # e.g., [{"embedding": [...], "embeddingType": "VIDEO"}, ...]
        
        return result  # Single embedding vector
    
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


def generate_vector_key(custom_key: Optional[str], use_object_key_name: bool, source_location: str, key_prefix: Optional[str] = None) -> str:
    """Generate vector key based on parameters and source location."""
    if custom_key:
        base_key = custom_key
    elif use_object_key_name:
        base_key = extract_key_from_source(source_location)
    else:
        base_key = str(uuid.uuid4())
    
    # Apply key prefix if provided
    if key_prefix:
        return f"{key_prefix}{base_key}"
    else:
        return base_key


def extract_key_from_source(source_location: str) -> str:
    """Extract key from source location (S3 URI or local path)."""
    if source_location.startswith('s3://'):
        # Extract filename from S3 object key
        parts = source_location[5:].split('/', 1)  # Remove 's3://' and split
        object_key = parts[1] if len(parts) > 1 else parts[0]
        return Path(object_key).name  # Get filename from object key
    else:
        # Extract filename from local path
        return Path(source_location).name
