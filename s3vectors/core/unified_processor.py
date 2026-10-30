"""Unified processing pipeline for sync and async models."""

import base64
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from s3vectors.utils.models import SupportedModel, ProcessingInput, RequestContext, generate_vector_key
from s3vectors.utils.multimodal_helpers import parse_s3_uri


@dataclass
class ProcessingResult:
    """Unified result structure."""
    vectors: List[Dict[str, Any]]  # List of vectors to store
    result_type: str  # "single" or "multiclip"
    job_id: Optional[str] = None


class UnifiedProcessor:
    """Unified processor that handles both sync and async models."""
    
    def __init__(self, bedrock_service, s3vector_service, session=None):
        self.bedrock_service = bedrock_service
        self.s3vector_service = s3vector_service
        self.session = session
    
    def process(self, model: SupportedModel, processing_input: ProcessingInput, 
                user_bedrock_params: Dict[str, Any] = None,
                async_output_s3_uri: str = None, src_bucket_owner: str = None,
                vector_bucket_name: str = None, index_name: str = None,
                precomputed_dimensions: int = None,
                operation: str = "put") -> ProcessingResult:
        """Unified processing method for all input types and models."""
        
        # Step 1: Index dimensions are fetched once by the caller
        if precomputed_dimensions is None:
            raise ValueError("Unexpected error occurred. Index dimensions are not fetched.")
        
        # Step 2: Determine sync/async routing based on model capabilities
        use_async = model.capabilities.supports_async and bool(async_output_s3_uri)
        use_sync = model.capabilities.supports_sync and not use_async

        if not use_async and not use_sync:
            raise ValueError(f"Model {model.model_id} requires --async-output-s3-uri for processing")

        # Step 3: Gather the request context (reads text/image content as needed)
        ctx = self._build_request_context(
            processing_input, precomputed_dimensions, operation,
            mode="async" if use_async else "sync", src_bucket_owner=src_bucket_owner,
        )

        # Models that require an S3 source for async accept only S3 inputs in
        # that mode (Nova docs: "Asynchronous operations can only accept S3
        # inputs"). This applies to every modality. The CLI never stages
        # local input into the async output location.
        if (use_async and model.capabilities.async_requires_s3_source and
                not (ctx.file_path or "").startswith("s3://")):
            raise ValueError(
                f"Model {model.model_id} requires {processing_input.content_type} "
                f"input to be an S3 URI (s3://bucket/key) when using "
                f"--async-output-s3-uri. Inline values and local files are only "
                f"supported in sync mode; upload the file to S3 first."
            )

        # Step 4: Build the request (model builder + user params merged at any depth)
        request_body = model.build_request(ctx, user_bedrock_params or {})
        
        # Step 5: Get embeddings as normalized [{"embedding", "metadata"}, ...]
        if use_async:
            results, job_id = self.bedrock_service.embed_async_with_payload(
                model, request_body, async_output_s3_uri
            )
        else:
            results = self.bedrock_service.embed_with_payload(model, request_body)
            job_id = None
        
        # Step 6: Process results into vectors (unified)
        vectors = self._prepare_vectors(results, processing_input)
        
        # Step 7: Determine result type
        result_type = "multiclip" if len(vectors) > 1 else "single"
        
        return ProcessingResult(vectors=vectors, result_type=result_type, job_id=job_id)
    
    def _build_request_context(self, processing_input: ProcessingInput, index_dimensions: int,
                               operation: str, mode: str, src_bucket_owner: Optional[str]) -> RequestContext:
        """Assemble the RequestContext a model's request_builder works from."""
        ctx = RequestContext(
            operation=operation,
            mode=mode,
            modality=processing_input.content_type,
            dimensions=index_dimensions,
            src_bucket_owner=src_bucket_owner,
        )
        data = processing_input.data
        
        if processing_input.content_type == "text":
            if "file_path" in data and "text" not in data:
                ctx.file_path = data["file_path"]
                # Read the file for models that embed text inline; async
                # (segmented) models reference file_path instead.
                ctx.text = self._read_file_content(ctx.file_path)
                # Expose the text so metadata logic can record it
                data["text"] = ctx.text
            else:
                ctx.text = data.get("text", "")
                
        elif processing_input.content_type == "image":
            if "file_path" in data:
                ctx.file_path = data["file_path"]
                ctx.image_base64, ctx.image_data_uri = self._read_image(ctx.file_path)
            else:
                ctx.image_base64 = data.get("image_base64", "")
                ctx.image_data_uri = data.get("image", "")
                
        elif processing_input.content_type == "multimodal":
            multimodal_data = data.get("multimodal", {})
            ctx.text = multimodal_data.get("text", "")
            image_path = multimodal_data.get("image_path", "")
            if image_path:
                ctx.image_base64, ctx.image_data_uri = self._read_image(image_path)
            
        elif processing_input.content_type in ["video", "audio"]:
            ctx.file_path = data.get("file_path", "")
            
        return ctx
    
    def _read_file_content(self, file_path: str) -> str:
        """Read text file content from local or S3."""
        if file_path.startswith('s3://'):
            bucket, key = parse_s3_uri(file_path)
            s3_client = self.session.client('s3')
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _read_image(self, file_path: str) -> tuple:
        """Read an image (local or S3) as (base64, data URI).

        Titan takes the bare base64 string; Cohere takes the data URI form.
        """
        if file_path.startswith('s3://'):
            bucket, key = parse_s3_uri(file_path)
            s3_client = self.session.client('s3')
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_bytes = response['Body'].read()
        else:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = "image/png" if file_path.lower().endswith('.png') else "image/jpeg"
        return base64_image, f"data:{mime_type};base64,{base64_image}"
    
    def _prepare_vectors(self, results: List[Dict[str, Any]], 
                        processing_input: ProcessingInput) -> List[Dict[str, Any]]:
        """Convert normalized embedding results to vector storage format."""
        vectors = []
        
        for i, result in enumerate(results):
            embedding = result.get('embedding', [])
            if not embedding:
                raise ValueError(f"Missing required embedding in result {i+1}/{len(results)}. Result: {result}")
            
            # Generate vector key based on processing input preferences
            if processing_input.custom_key and len(results) == 1:
                # Use custom key only for single vector results
                vector_key = generate_vector_key(processing_input.custom_key, False, processing_input.source_location, processing_input.key_prefix)
            elif processing_input.filename_as_key and len(results) == 1:
                # Use object key/filename only for single vector results
                vector_key = generate_vector_key(None, True, processing_input.source_location, processing_input.key_prefix)
            else:
                # Generate UUID for multi-vector results or when no key preference specified
                vector_key = generate_vector_key(None, False, processing_input.source_location, processing_input.key_prefix)
            
            # Prepare metadata
            vector_metadata = processing_input.metadata.copy()
            
            # Add standard metadata fields based on input type
            if "file_path" in processing_input.data:
                # File input (--text, --image, --video, --audio) - always add location
                vector_metadata["S3VECTORS-EMBED-SRC-LOCATION"] = processing_input.source_location
                
                # For text files, also add the raw text content
                if processing_input.content_type == "text" and "text" in processing_input.data:
                    vector_metadata["S3VECTORS-EMBED-SRC-CONTENT"] = processing_input.data["text"]
                # For image/video/audio files, S3VECTORS-EMBED-SRC-CONTENT is not added (blank)
            elif processing_input.content_type == "multimodal":
                # Multimodal input (--text-value + --image) - add both content and location
                multimodal_data = processing_input.data.get("multimodal", {})
                vector_metadata["S3VECTORS-EMBED-SRC-CONTENT"] = multimodal_data.get("text", "")
                vector_metadata["S3VECTORS-EMBED-SRC-LOCATION"] = processing_input.source_location
            else:
                # Direct text input (--text-value) - only add content, no location
                if processing_input.content_type == "text" and "text" in processing_input.data:
                    vector_metadata["S3VECTORS-EMBED-SRC-CONTENT"] = processing_input.data["text"]
            
            # Model-specific metadata (clip timing, segment index, embedding type...)
            vector_metadata.update(result.get("metadata", {}))
            
            # Create vector in S3 Vectors API format
            vector = {
                "key": vector_key,
                "data": {
                    "float32": embedding
                },
                "metadata": vector_metadata
            }
            
            vectors.append(vector)
        
        return vectors
    
    def store_vectors(self, vectors: List[Dict[str, Any]], vector_bucket_name: str, 
                     index_name: str) -> List[str]:
        """Store vectors using batch operation."""
        if not vectors:
            return []
        
        self.s3vector_service.put_vectors_batch(vector_bucket_name, index_name, vectors)
        return [v["key"] for v in vectors]
