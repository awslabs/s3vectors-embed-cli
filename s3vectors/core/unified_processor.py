"""Unified processing pipeline for sync and async models."""

import uuid
import base64
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from s3vectors.utils.models import get_model_info, SupportedModel, ProcessingInput
from s3vectors.utils.multimodal_helpers import create_multimodal_metadata


@dataclass
class ProcessingResult:
    """Unified result structure."""
    vectors: List[Dict[str, Any]]  # List of vectors to store
    result_type: str  # "single" or "multiclip"
    job_id: Optional[str] = None
    raw_results: Optional[List[Dict[str, Any]]] = None  # Raw results for timing extraction


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
                precomputed_dimensions: int = None) -> ProcessingResult:
        """Unified processing method for all input types and models."""
        
        # Step 1: Get index dimensions if available
        # Use pre-computed dimensions (required parameter)
        if precomputed_dimensions is None:
            raise ValueError("Unexpected error occurred. Index dimensions are not fetched.")
        
        index_dimensions = precomputed_dimensions
        
        # Step 2: Build content for schema application
        content = self._prepare_content(processing_input, index_dimensions)
        
        # Step 3: Build payload using schema-based system (includes validation and merge)
        user_bedrock_params = user_bedrock_params or {}
        
        # Build final payload with user parameters
        if model.is_async():
            async_config = {
                "output_s3_uri": async_output_s3_uri,
                "src_bucket_owner": src_bucket_owner
            }
            final_payload = model.build_payload(processing_input.content_type, content, user_bedrock_params, async_config)
        else:
            final_payload = model.build_payload(processing_input.content_type, content, user_bedrock_params)
        
        # Step 4: Get embeddings
        if model.is_async():
            raw_results, job_id = self.bedrock_service.embed_async_with_payload(
                model.model_id, final_payload, async_output_s3_uri
            )
        else:
            raw_results = self._embed_sync(model.model_id, final_payload, user_bedrock_params)
            job_id = None
        
        # Step 6: Process results into vectors (unified)
        vectors = self._prepare_vectors(raw_results, processing_input, model)
        
        # Step 7: Determine result type
        result_type = "multiclip" if len(vectors) > 1 else "single"
        
        return ProcessingResult(vectors=vectors, result_type=result_type, job_id=job_id, raw_results=raw_results)
    
    def _prepare_content(self, processing_input: ProcessingInput, index_dimensions: int) -> Dict[str, Any]:
        """Prepare content dictionary for schema application."""
        content = {"index": {"dimensions": index_dimensions}}
        
        if processing_input.content_type == "text":
            if "file_path" in processing_input.data and "text" not in processing_input.data:
                # Read file content for sync models
                file_content = self._read_file_content(processing_input.data["file_path"])
                content["text"] = file_content
                # Update processing_input.data so metadata logic can access the text content
                processing_input.data["text"] = file_content
            else:
                content["text"] = processing_input.data.get("text", "")
                
        elif processing_input.content_type == "image":
            if "file_path" in processing_input.data:
                file_path = processing_input.data["file_path"]
                
                # For async models (TwelveLabs), preserve file_path for media_source
                content["file_path"] = file_path
                
                # For sync models, read and encode image
                base64_image = self._read_image_as_base64(file_path)
                
                # Set both formats to support different models:
                # - Titan expects: {content.image_base64} (just base64 string)
                # - Cohere expects: {content.image} (data URI format)
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    mime_type = "image/jpeg"
                elif file_path.lower().endswith('.png'):
                    mime_type = "image/png"
                else:
                    mime_type = "image/jpeg"  # default
                
                content["image_base64"] = base64_image  # For Titan
                content["image"] = f"data:{mime_type};base64,{base64_image}"  # For Cohere
            else:
                content["image_base64"] = processing_input.data.get("image_base64", "")
                content["image"] = processing_input.data.get("image", "")
                
        elif processing_input.content_type == "multimodal":
            # Handle multimodal input (text + image)
            multimodal_data = processing_input.data.get("multimodal", {})
            content["text"] = multimodal_data.get("text", "")
            
            # Handle image path
            image_path = multimodal_data.get("image_path", "")
            if image_path:
                base64_image = self._read_image_as_base64(image_path)
                
                # Determine MIME type
                if image_path.lower().endswith(('.jpg', '.jpeg')):
                    mime_type = "image/jpeg"
                elif image_path.lower().endswith('.png'):
                    mime_type = "image/png"
                else:
                    mime_type = "image/jpeg"  # default
                
                content["image_base64"] = base64_image  # For Titan
                content["image"] = f"data:{mime_type};base64,{base64_image}"  # For Cohere
            
        elif processing_input.content_type in ["video", "audio"]:
            content["file_path"] = processing_input.data.get("file_path", "")
            
        return content
    
    def _read_file_content(self, file_path: str) -> str:
        """Read text file content from local or S3."""
        if file_path.startswith('s3://'):
            parts = file_path[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            s3_client = self.session.client('s3')
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _read_image_as_base64(self, file_path: str) -> str:
        """Read image file and convert to base64."""
        if file_path.startswith('s3://'):
            parts = file_path[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            s3_client = self.session.client('s3')
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_bytes = response['Body'].read()
        else:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
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
    
    def _embed_sync(self, model_id: str, embedding_input: Dict[str, Any], 
                   user_bedrock_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle sync embedding using model's schema."""
        from s3vectors.utils.models import get_model_info
        
        # Get model object for schema-based embedding extraction
        model = get_model_info(model_id)
        if not model:
            raise ValueError(f"Unsupported model: {model_id}")
        
        # The embedding_input is already the correct payload from _prepare_embedding_input
        # BedrockService.embed_with_payload now uses schema-based embedding extraction
        embedding = self.bedrock_service.embed_with_payload(model, embedding_input)
        
        return [{"embedding": embedding}]
    
    def _prepare_vectors(self, raw_results: List[Dict[str, Any]], 
                        processing_input: ProcessingInput, model: SupportedModel) -> List[Dict[str, Any]]:
        """Convert raw embedding results to vector storage format."""
        vectors = []
        
        for i, result in enumerate(raw_results):
            embedding = result.get('embedding', [])
            if not embedding:
                raise ValueError(f"Missing required embedding in result {i+1}/{len(raw_results)}. Result: {result}")
            
            # Generate unique key
            vector_key = str(uuid.uuid4())
            
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
            
            # Add model-specific metadata for async models
            if model.is_async():
                vector_metadata.update(create_multimodal_metadata(
                    processing_input.content_type, processing_input.source_location, result, i
                ))
            
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
