"""Model definitions and capabilities for S3 Vectors CLI.

Every supported model is described by a ModelCapabilities entry made of plain
flags plus three model-specific pieces:

  request_builder(ctx)     -> the Bedrock request body for a RequestContext
  response_parser(output)  -> list of {"embedding": [...], "metadata": {...}},
                              from either a sync invoke_model body (dict) or
                              the S3 output of a completed async job (AsyncOutput)
  non_overridable_fields   -> dotted paths in the request body that the CLI
                              controls and a user may not set through
                              --bedrock-inference-params

User parameters are deep-merged into the built request at any depth, so they
are written in the model's real request shape. For a flat model that is just
{"normalize": false}; for Nova it is e.g.
{"singleEmbeddingParams": {"text": {"truncationMode": "START"}}}.
"""

import json
import uuid
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple
import click
from s3vectors.utils.multimodal_helpers import (
    build_media_source,
    encode_file_to_base64,
    is_local_path,
    parse_s3_uri,
)


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


# ---------------------------------------------------------------------------
# Contract between the generic pipeline and the per-model functions.
# ---------------------------------------------------------------------------


@dataclass
class RequestContext:
    """Everything a request_builder may need. Populated by UnifiedProcessor."""
    operation: str                 # "put" | "query"
    mode: str                      # "sync" (invoke_model) | "async" (start_async_invoke)
    modality: str                  # "text" | "image" | "video" | "audio" | "multimodal"
    dimensions: int                # dimension of the target index
    text: Optional[str] = None     # inline text, or the content of a text file already read
    file_path: Optional[str] = None        # local path or s3:// URI of the source file, if any
    image_base64: Optional[str] = None     # base64 of the image bytes (sync image input)
    image_data_uri: Optional[str] = None   # the same image as data:<mime>;base64,... (Cohere)
    src_bucket_owner: Optional[str] = None # account that owns the source bucket, for cross-account reads


class AsyncOutput:
    """The S3 output of a completed async job, handed to a response_parser.

    Parsers read files by name relative to the job's output prefix
    (e.g. "output.json") or by full s3:// URI, as found in manifests.
    """

    def __init__(self, s3_client, output_s3_uri: str):
        self._s3 = s3_client
        self.bucket, prefix = parse_s3_uri(output_s3_uri)
        self.prefix = prefix if not prefix or prefix.endswith("/") else prefix + "/"

    def read_text(self, location: str) -> str:
        bucket, key = self._resolve(location)
        response = self._s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def read_json(self, location: str) -> Any:
        return json.loads(self.read_text(location))

    def _resolve(self, location: str) -> Tuple[str, str]:
        if location.startswith("s3://"):
            return parse_s3_uri(location)
        return self.bucket, f"{self.prefix}{location}"


@dataclass
class ModelCapabilities:
    """Capabilities and properties of an embedding model."""
    supported_modalities: List[str]  # text, image, video, audio
    description: str
    request_builder: Callable[[RequestContext], Dict[str, Any]]
    response_parser: Callable[[Any], List[Dict[str, Any]]]
    non_overridable_fields: Tuple[str, ...] = ()  # dotted request paths the CLI owns
    supports_multimodal_input: bool = False  # Can accept text + image in one request
    supports_sync: bool = True  # Whether the model can use invoke_model (sync)
    supports_async: bool = False  # Whether the model can use start_async_invoke
    async_requires_s3_source: bool = False  # When True, async input of any modality must already be an S3 URI (inline/local input is rejected, never staged to S3)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return base with override merged in at any depth. Neither input is mutated."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _leaf_paths(params: Dict[str, Any], prefix: str = "") -> Iterator[str]:
    """Dotted path of every leaf value. An empty dict counts as a leaf."""
    for key, value in params.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            yield from _leaf_paths(value, path + ".")
        else:
            yield path


def validate_overrides(user_params: Dict[str, Any], non_overridable: Iterable[str]) -> None:
    """Reject user parameters that would touch a protected request field.

    A user leaf conflicts with a protected path when it is that path, is
    nested inside it, or sits on a prefix of it (replacing the subtree that
    contains the protected field).
    """
    protected = tuple(non_overridable)
    conflicts = []
    for path in _leaf_paths(user_params):
        for field in protected:
            if path == field or path.startswith(field + ".") or field.startswith(path + "."):
                conflicts.append(path)
                break
    if conflicts:
        raise ValueError(
            f"Cannot override system-controlled parameters: {sorted(conflicts)}. "
            f"These parameters are automatically set based on your CLI inputs."
        )


def _single_embedding(embedding: List[float]) -> List[Dict[str, Any]]:
    return [{"embedding": embedding, "metadata": {}}]


# ---------------------------------------------------------------------------
# Amazon Titan
# ---------------------------------------------------------------------------


def build_titan_text_v1_request(ctx: RequestContext) -> Dict[str, Any]:
    return {"inputText": ctx.text}


def build_titan_text_v2_request(ctx: RequestContext) -> Dict[str, Any]:
    return {"inputText": ctx.text, "dimensions": ctx.dimensions}


def build_titan_image_request(ctx: RequestContext) -> Dict[str, Any]:
    """Titan Multimodal: text, image, or both in one request."""
    body: Dict[str, Any] = {}
    if ctx.modality in ("text", "multimodal"):
        body["inputText"] = ctx.text
    if ctx.modality in ("image", "multimodal"):
        body["inputImage"] = ctx.image_base64
    body["embeddingConfig"] = {"outputEmbeddingLength": ctx.dimensions}
    return body


def parse_titan_response(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _single_embedding(body["embedding"])


def parse_titan_text_v2_response(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Titan v2 returns embeddingsByType when embeddingTypes is requested, else embedding."""
    by_type = body.get("embeddingsByType")
    if isinstance(by_type, dict) and by_type:
        return _single_embedding(next(iter(by_type.values())))
    return _single_embedding(body["embedding"])


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------


def build_cohere_request(ctx: RequestContext) -> Dict[str, Any]:
    if ctx.modality == "image":
        return {"images": [ctx.image_data_uri], "input_type": "image"}
    return {"texts": [ctx.text], "input_type": "search_document"}


def parse_cohere_response(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _single_embedding(body["embeddings"][0])


# ---------------------------------------------------------------------------
# Amazon Nova Multimodal Embeddings
# ---------------------------------------------------------------------------

NOVA_SCHEMA_VERSION = "nova-multimodal-embed-v1"
NOVA_MODALITIES = ("text", "image", "video", "audio")

# A static superset: paths that do not occur in a given request are harmless.
NOVA_NON_OVERRIDABLE = ("schemaVersion", "taskType") + tuple(
    f"{root}.{leaf}"
    for root in ("singleEmbeddingParams", "segmentedEmbeddingParams")
    for leaf in ("embeddingDimension",
                 *(f"{m}.{f}" for m in NOVA_MODALITIES for f in ("value", "source", "format")))
)


def build_nova_request(ctx: RequestContext) -> Dict[str, Any]:
    """Nova MME request body.

    Sync is SINGLE_EMBEDDING: one embedding per input, inline text or
    bytes/S3 media. Async is SEGMENTED_EMBEDDING: one embedding per segment,
    S3 sources only (UnifiedProcessor rejects anything else before this runs).
    """
    modality = ctx.modality
    params: Dict[str, Any] = {
        "embeddingPurpose": "GENERIC_INDEX" if ctx.operation == "put" else f"{modality.upper()}_RETRIEVAL",
        "embeddingDimension": ctx.dimensions,
    }

    if modality == "text":
        text: Dict[str, Any] = {"truncationMode": "END"}
        if ctx.mode == "sync":
            text["value"] = ctx.text
        else:
            text["source"] = {"s3Location": _nova_s3_location(ctx)}
            # Required wrapper; left empty so Bedrock applies its own
            # maxLengthChars default.
            text["segmentationConfig"] = {}
        params["text"] = text
    else:
        media: Dict[str, Any] = {"format": _nova_format(ctx.file_path), "source": _nova_source(ctx)}
        if modality == "video":
            media["embeddingMode"] = "AUDIO_VIDEO_COMBINED"
        if ctx.mode == "async" and modality in ("video", "audio"):
            # Bedrock rejects an empty segmentationConfig for video/audio
            # ("required key [durationSeconds] not found"), so the documented
            # default (5 s, valid range 1-30) is sent explicitly.
            media["segmentationConfig"] = {"durationSeconds": 5}
        params[modality] = media

    if ctx.mode == "sync":
        return {"schemaVersion": NOVA_SCHEMA_VERSION, "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": params}
    return {"schemaVersion": NOVA_SCHEMA_VERSION, "taskType": "SEGMENTED_EMBEDDING",
            "segmentedEmbeddingParams": params}


def _nova_s3_location(ctx: RequestContext) -> Dict[str, Any]:
    location = {"uri": ctx.file_path}
    if ctx.src_bucket_owner:
        location["bucketOwner"] = ctx.src_bucket_owner
    return location


def _nova_source(ctx: RequestContext) -> Dict[str, Any]:
    """Nova SourceObject for media: inline bytes for local files, S3 otherwise."""
    if is_local_path(ctx.file_path):
        return {"bytes": ctx.image_base64 or encode_file_to_base64(ctx.file_path)}
    return {"s3Location": _nova_s3_location(ctx)}


def _nova_format(file_path: str) -> str:
    """Nova requires an explicit media format; derive it from the file extension."""
    ext = Path(file_path).suffix.lstrip(".").lower()
    return "jpeg" if ext == "jpg" else ext


_NOVA_SEGMENT_FIELDS = {
    "segmentStartSeconds": "S3VECTORS-EMBED-START-SEC",
    "segmentEndSeconds": "S3VECTORS-EMBED-END-SEC",
    "segmentStartCharPosition": "S3VECTORS-EMBED-START-CHAR-POS",
    "segmentEndCharPosition": "S3VECTORS-EMBED-END-CHAR-POS",
    "segmentIndex": "S3VECTORS-EMBED-SEGMENT-IDX",
}


def parse_nova_response(output) -> List[Dict[str, Any]]:
    """Sync body: {"embeddings": [{embedding, embeddingType?, truncatedCharLength?}]}.
    Async output: manifest referencing one JSONL file per modality."""
    if isinstance(output, AsyncOutput):
        return _parse_nova_async_output(output)
    return [_nova_result(item, item.get("embeddingType")) for item in output["embeddings"]]


def _nova_result(item: Dict[str, Any], embedding_type: Optional[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if embedding_type is not None:
        metadata["S3VECTORS-EMBED-TYPE"] = embedding_type
    segment = item.get("segmentMetadata") or {}
    truncated = item.get("truncatedCharLength", segment.get("truncatedCharLength"))
    if truncated is not None:
        metadata["S3VECTORS-EMBED-TRUNCATED-CHAR-LEN"] = truncated
    for source_key, metadata_key in _NOVA_SEGMENT_FIELDS.items():
        if segment.get(source_key) is not None:
            metadata[metadata_key] = segment[source_key]
    return {"embedding": item.get("embedding", []), "metadata": metadata}


def _parse_nova_async_output(output: AsyncOutput) -> List[Dict[str, Any]]:
    """Fail closed: any failed manifest entry or segment raises with its
    failureReason/message (e.g. RAI_VIOLATION_*, INVALID_CONTENT) rather than
    storing a partial embedding set as if the job were complete."""
    manifest = output.read_json("segmented-embedding-result.json")
    entries = manifest.get("embeddingResults", [])
    if not entries:
        raise Exception("Async embedding job manifest contains no embeddingResults entries")

    failures = [e for e in entries if e.get("status") != "SUCCESS"]
    if failures:
        details = "; ".join(
            f"{entry.get('embeddingType', 'UNKNOWN')}: status={entry.get('status', 'UNKNOWN')}"
            + _failure_detail(entry)
            for entry in failures
        )
        raise Exception(f"Async embedding job reported {len(failures)} failed result(s): {details}")

    missing_uri = [e for e in entries if not e.get("outputFileUri")]
    if missing_uri:
        types = ", ".join(e.get("embeddingType", "UNKNOWN") for e in missing_uri)
        raise Exception(
            f"Async embedding job manifest is malformed: {len(missing_uri)} SUCCESS "
            f"result(s) missing outputFileUri ({types})"
        )

    results = []
    segment_failures = []
    for entry in entries:
        embedding_type = entry.get("embeddingType", "UNKNOWN")
        for line in output.read_text(entry["outputFileUri"]).splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            # Per-segment status: absent means success.
            status = item.get("status")
            if status is not None and status != "SUCCESS":
                index = (item.get("segmentMetadata") or {}).get("segmentIndex", "?")
                segment_failures.append(
                    f"{embedding_type} segment {index}: status={status}" + _failure_detail(item)
                )
                continue
            results.append(_nova_result(item, entry.get("embeddingType")))

    if segment_failures:
        raise Exception(
            f"Async embedding job reported {len(segment_failures)} failed segment(s): "
            + "; ".join(segment_failures)
        )
    return results


def _failure_detail(item: Dict[str, Any]) -> str:
    """Format optional failureReason/message from a manifest entry or JSONL line."""
    detail = ""
    if item.get("failureReason"):
        detail += f", reason={item['failureReason']}"
    if item.get("message"):
        detail += f", message={item['message']}"
    return detail


# ---------------------------------------------------------------------------
# TwelveLabs Marengo
# ---------------------------------------------------------------------------

TWELVELABS_MAX_LOCAL_FILE_SIZE = 36 * 1024 * 1024  # inline base64 limit for local files


def build_twelvelabs_request(ctx: RequestContext) -> Dict[str, Any]:
    if ctx.modality == "text":
        return {"inputType": "text", "inputText": ctx.text}
    return {
        "inputType": ctx.modality,
        "mediaSource": build_media_source(ctx.file_path, ctx.src_bucket_owner, TWELVELABS_MAX_LOCAL_FILE_SIZE),
    }


def parse_twelvelabs_response(output: AsyncOutput) -> List[Dict[str, Any]]:
    """Async output: a single output.json whose "data" array holds one entry per clip."""
    result_data = output.read_json("output.json")
    if isinstance(result_data, dict) and isinstance(result_data.get("data"), list):
        items = result_data["data"]
    else:
        items = result_data if isinstance(result_data, list) else [result_data]
    return [{"embedding": item.get("embedding", []), "metadata": _twelvelabs_metadata(item)} for item in items]


def _twelvelabs_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if item.get("startSec") is not None:
        metadata["S3VECTORS-EMBED-START-SEC"] = item["startSec"]
    if item.get("endSec") is not None:
        metadata["S3VECTORS-EMBED-END-SEC"] = item["endSec"]
    if item.get("embeddingOption") is not None:
        metadata["S3VECTORS-EMBED-TYPE"] = item["embeddingOption"]
    return metadata


class SupportedModel(Enum):
    """Enumeration of supported embedding models with their capabilities."""
    
    # Amazon Titan Models
    TITAN_TEXT_V1 = ("amazon.titan-embed-text-v1", ModelCapabilities(
        supported_modalities=["text"],
        description="Amazon Titan Text Embeddings v1",
        request_builder=build_titan_text_v1_request,
        response_parser=parse_titan_response,
        non_overridable_fields=("inputText",),
    ))
    
    TITAN_TEXT_V2 = ("amazon.titan-embed-text-v2:0", ModelCapabilities(
        supported_modalities=["text"],
        description="Amazon Titan Text Embeddings v2",
        request_builder=build_titan_text_v2_request,
        response_parser=parse_titan_text_v2_response,
        non_overridable_fields=("inputText", "dimensions"),
        # normalize, embeddingTypes: user parameters
    ))
    
    TITAN_IMAGE_V1 = ("amazon.titan-embed-image-v1", ModelCapabilities(
        supported_modalities=["text", "image"],
        description="Amazon Titan Multimodal Embeddings v1",
        supports_multimodal_input=True,
        request_builder=build_titan_image_request,
        response_parser=parse_titan_response,
        non_overridable_fields=("inputText", "inputImage", "embeddingConfig.outputEmbeddingLength"),
    ))
    
    # Cohere Models
    COHERE_ENGLISH_V3 = ("cohere.embed-english-v3", ModelCapabilities(
        supported_modalities=["text", "image"],
        description="Cohere Embed English v3",
        request_builder=build_cohere_request,
        response_parser=parse_cohere_response,
        # input_type is a user parameter (e.g. "search_query" for queries)
        non_overridable_fields=("texts", "images"),
    ))
    
    COHERE_MULTILINGUAL_V3 = ("cohere.embed-multilingual-v3", ModelCapabilities(
        supported_modalities=["text", "image"],
        description="Cohere Embed Multilingual v3",
        request_builder=build_cohere_request,
        response_parser=parse_cohere_response,
        non_overridable_fields=("texts", "images"),
    ))
    
    # Amazon Nova Models
    NOVA_MME_V1 = ("amazon.nova-2-multimodal-embeddings-v1:0", ModelCapabilities(
        supported_modalities=["text", "image", "video", "audio"],
        description="Amazon Nova Multimodal Embeddings v1",
        supports_multimodal_input=False,  # One modality per request
        supports_sync=True,
        supports_async=True,
        async_requires_s3_source=True,
        request_builder=build_nova_request,
        response_parser=parse_nova_response,
        non_overridable_fields=NOVA_NON_OVERRIDABLE,
        # embeddingPurpose, truncationMode, detailLevel, embeddingMode and
        # segmentationConfig are user parameters, set at their nested path:
        # {"singleEmbeddingParams": {"text": {"truncationMode": "START"}}}
    ))

    # TwelveLabs Models
    TWELVELABS_MARENGO_V2_7 = ("twelvelabs.marengo-embed-2-7-v1:0", ModelCapabilities(
        supported_modalities=["text", "image", "video", "audio"],
        description="TwelveLabs Marengo Embed 2.7 v1",
        supports_sync=False,
        supports_async=True,
        request_builder=build_twelvelabs_request,
        response_parser=parse_twelvelabs_response,
        non_overridable_fields=("inputType", "inputText", "mediaSource"),
        # startSec, lengthSec, useFixedLengthSec, embeddingOption: user parameters
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
    
    def supports_modality(self, modality: str) -> bool:
        """Check if model supports a specific modality."""
        return modality in self.capabilities.supported_modalities
    
    def supports_multimodal_input(self) -> bool:
        """Check if model supports multiple modalities simultaneously."""
        return self.capabilities.supports_multimodal_input

    def validate_user_params(self, user_params: Any) -> None:
        """Reject --bedrock-inference-params that touch CLI-controlled fields."""
        if not isinstance(user_params, dict):
            raise ValueError("--bedrock-inference-params must be a JSON object")
        validate_overrides(user_params, self.capabilities.non_overridable_fields)

    def build_request(self, ctx: RequestContext, user_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Model request body with user parameters merged in at any depth."""
        user_params = user_params or {}
        self.validate_user_params(user_params)
        return deep_merge(self.capabilities.request_builder(ctx), user_params)

    def parse_response(self, output) -> List[Dict[str, Any]]:
        """Normalize a sync body or AsyncOutput into [{"embedding", "metadata"}, ...]."""
        try:
            return self.capabilities.response_parser(output)
        except (KeyError, IndexError, TypeError) as e:
            shape = list(output.keys()) if isinstance(output, dict) else type(output).__name__
            raise ValueError(
                f"Unexpected response from {self.model_id}: {e!r}. Response: {shape}"
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
