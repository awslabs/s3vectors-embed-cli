"""Core services for S3 Vectors operations with user agent tracking."""

import json
import base64
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import boto3
from botocore.exceptions import ClientError

from s3vectors.utils.boto_config import get_boto_config, get_user_agent


class BedrockService:
    """Service for Bedrock embedding operations."""
    
    def __init__(self, session: boto3.Session, region: str, debug: bool = False, console=None):
        # Create Bedrock client with user agent tracking
        self.bedrock_runtime = session.client(
            'bedrock-runtime', 
            region_name=region,
            config=get_boto_config()
        )
        self.debug = debug
        self.console = console
        
        if self.debug and self.console:
            self.console.print(f"[dim] BedrockService initialized for region: {region}[/dim]")
            self.console.print(f"[dim] User agent: {get_user_agent()}[/dim]")
    
    def _debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug and self.console:
            self.console.print(f"[dim] {message}[/dim]")
    
    def embed_text(self, model_id: str, text: str, dimensions: Optional[int] = None) -> List[float]:
        """Embed text using Bedrock model with optional dimension specification."""
        start_time = time.time()
        self._debug_log(f"Starting text embedding with model: {model_id}")
        self._debug_log(f"Text length: {len(text)} characters")
        if dimensions:
            self._debug_log(f"Requested dimensions: {dimensions}")
        
        try:
            if model_id.startswith('amazon.titan-embed-text-v2'):
                # Titan Text v2 API specification
                body_dict = {
                    "inputText": text,  # Required
                    "normalize": True,  # Optional, defaults to true
                    "embeddingTypes": ["float"]  # Optional, defaults to float
                }
                if dimensions:
                    # Optional: 1024 (default), 512, 256
                    body_dict["dimensions"] = dimensions
                body = json.dumps(body_dict)
                
            elif model_id.startswith('amazon.titan-embed-text-v1'):
                # Titan Text v1 API specification - only inputText field available
                body = json.dumps({
                    "inputText": text  # Only available field
                })
                # Note: dimensions parameter is ignored for v1 as it's not supported
                
            elif model_id.startswith('amazon.titan-embed-image'):
                # Titan Multimodal Embeddings G1 can handle text-only input
                body_dict = {
                    "inputText": text  # Required for text-only embedding
                }
                if dimensions:
                    # Valid values: 256, 384, 1024 (default)
                    if dimensions not in [256, 384, 1024]:
                        raise ValueError(f"Invalid dimensions for Titan Image v1. Valid values: 256, 384, 1024. Got: {dimensions}")
                    body_dict["embeddingConfig"] = {
                        "outputEmbeddingLength": dimensions
                    }
                body = json.dumps(body_dict)
                
            elif model_id.startswith('cohere.embed'):
                # Cohere models API specification
                body_dict = {
                    "texts": [text],  # Array of strings
                    "input_type": "search_document",  # Default for document embedding
                    "embedding_types": ["float"]  # Default to float embeddings
                }
                if dimensions:
                    # Cohere supports different embedding types but dimensions are model-fixed
                    # Keep float type for compatibility with S3 Vectors
                    body_dict["embedding_types"] = ["float"]
                body = json.dumps(body_dict)
            else:
                raise ValueError(f"Unsupported model: {model_id}")
            
            self._debug_log(f"Making Bedrock API call to model: {model_id}")
            if self.debug and self.console:
                self._debug_log(f"Request body: {body}")
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body,
                contentType='application/json'
            )
            
            elapsed_time = time.time() - start_time
            self._debug_log(f"Bedrock API call completed in {elapsed_time:.2f} seconds")
            
            response_body = json.loads(response['body'].read())
            
            if self.debug and self.console:
                self._debug_log(f"Response body keys: {list(response_body.keys())}")
            
            if model_id.startswith('amazon.titan-embed-text-v2'):
                # Handle embeddingsByType response structure
                if 'embeddingsByType' in response_body:
                    embedding = response_body['embeddingsByType'].get('float', [])
                else:
                    # Fallback to direct embedding field
                    embedding = response_body.get('embedding', [])
                    
            elif model_id.startswith('amazon.titan-embed-text-v1'):
                # v1 returns embedding directly
                embedding = response_body['embedding']
                
            elif model_id.startswith('amazon.titan-embed-image'):
                embedding = response_body['embedding']
                
            elif model_id.startswith('cohere.embed'):
                # Cohere returns embeddings in structured format
                embeddings = response_body.get('embeddings', {})
                if 'float' in embeddings:
                    embedding = embeddings['float'][0]  # First text's float embedding
                else:
                    # Fallback for other response formats
                    embedding = response_body.get('embeddings', [])[0] if response_body.get('embeddings') else []
            
            self._debug_log(f"Generated embedding with {len(embedding)} dimensions")
            total_time = time.time() - start_time
            self._debug_log(f"Total embed_text operation completed in {total_time:.2f} seconds")
            
            return embedding
            
        except ClientError as e:
            self._debug_log(f"Bedrock ClientError: {str(e)}")
            raise Exception(f"Bedrock embedding failed: {e}")
        except Exception as e:
            self._debug_log(f"Unexpected error in embed_text: {str(e)}")
            raise
    
    def embed_image(self, model_id: str, image_data: str, text_input: str = None, dimensions: Optional[int] = None) -> List[float]:
        """Embed image using Bedrock model, with optional text for multimodal and dimension specification."""
        try:
            if model_id.startswith('amazon.titan-embed-image'):
                # Titan Multimodal Embeddings G1 API specification
                body_dict = {}
                
                # At least one of inputText or inputImage is required
                if text_input:
                    body_dict["inputText"] = text_input
                if image_data:
                    body_dict["inputImage"] = image_data
                
                if not text_input and not image_data:
                    raise ValueError("At least one of text_input or image_data is required for Titan Image model")
                
                # Optional embeddingConfig with outputEmbeddingLength
                if dimensions:
                    # Valid values: 256, 384, 1024 (default)
                    if dimensions not in [256, 384, 1024]:
                        raise ValueError(f"Invalid dimensions for Titan Image v1. Valid values: 256, 384, 1024. Got: {dimensions}")
                    body_dict["embeddingConfig"] = {
                        "outputEmbeddingLength": dimensions
                    }
                # If no dimensions specified, model uses default (1024)
                
                body = json.dumps(body_dict)
                
            elif model_id.startswith('cohere.embed'):
                # Cohere image embedding API specification
                # Convert image data to proper data URI format
                import base64
                import mimetypes
                
                # Determine MIME type (assume JPEG if not determinable)
                mime_type = "image/jpeg"  # Default
                
                # Create data URI format required by Cohere
                if not image_data.startswith('data:'):
                    # If it's raw base64, add the data URI prefix
                    data_uri = f"data:{mime_type};base64,{image_data}"
                else:
                    # Already in data URI format
                    data_uri = image_data
                
                body_dict = {
                    "images": [data_uri],  # Array of data URIs
                    "input_type": "image",  # Required for image input
                    "embedding_types": ["float"]  # Default to float embeddings
                }
                if dimensions:
                    # Keep float type for compatibility
                    body_dict["embedding_types"] = ["float"]
                body = json.dumps(body_dict)
            else:
                raise ValueError(f"Unsupported image model: {model_id}")
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body,
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            
            if model_id.startswith('amazon.titan-embed-image'):
                # Titan Image returns embedding directly
                return response_body['embedding']
            elif model_id.startswith('cohere.embed'):
                # Cohere returns embeddings in structured format
                embeddings = response_body.get('embeddings', {})
                if 'float' in embeddings:
                    return embeddings['float'][0]  # First image's float embedding
                else:
                    # Fallback for other response formats
                    return response_body.get('embeddings', [])[0] if response_body.get('embeddings') else []
            
        except ClientError as e:
            raise Exception(f"Bedrock image embedding failed: {e}")


class S3VectorService:
    """Service for S3 Vector operations."""
    
    def __init__(self, session: boto3.Session, region: str, debug: bool = False, console=None):
        # Use S3 Vectors client with new endpoint URL
        endpoint_url = f"https://s3vectors.{region}.api.aws"
        self.s3vectors = session.client(
            's3vectors', 
            region_name=region, 
            endpoint_url=endpoint_url,
            config=get_boto_config()
        )
        self.region = region
        self.debug = debug
        self.console = console
        
        if self.debug and self.console:
            self.console.print(f"[dim] S3VectorService initialized for region: {region}[/dim]")
            self.console.print(f"[dim] Using endpoint: {endpoint_url}[/dim]")
            self.console.print(f"[dim] User agent: {get_user_agent()}[/dim]")
    
    def _debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug and self.console:
            self.console.print(f"[dim] {message}[/dim]")
    
    def put_vector(self, bucket_name: str, index_name: str, vector_id: str, 
                   embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """Put vector into S3 vector index using S3 Vectors API."""
        start_time = time.time()
        self._debug_log(f"Starting put_vector operation")
        self._debug_log(f"Bucket: {bucket_name}, Index: {index_name}, Vector ID: {vector_id}")
        self._debug_log(f"Embedding dimensions: {len(embedding)}")
        if metadata:
            self._debug_log(f"Metadata keys: {list(metadata.keys())}")
        
        try:
            # Prepare vector data according to S3 Vectors API format
            vector_data = {
                "key": vector_id,
                "data": {
                    "float32": embedding  # S3 Vectors expects {"float32": [list of floats]}
                }
            }
            
            # Add metadata if provided
            if metadata:
                vector_data["metadata"] = metadata
            
            # Use S3 Vectors PutVectors API
            params = {
                "vectorBucketName": bucket_name,
                "indexName": index_name,
                "vectors": [vector_data]
            }
            
            self._debug_log(f"Making S3 Vectors put_vectors API call")
            if self.debug and self.console:
                self._debug_log(f"API parameters: {json.dumps({k: v for k, v in params.items() if k != 'vectors'})}")
            
            response = self.s3vectors.put_vectors(**params)
            
            elapsed_time = time.time() - start_time
            self._debug_log(f"S3 Vectors put_vectors completed in {elapsed_time:.2f} seconds")
            self._debug_log(f"Vector stored successfully with ID: {vector_id}")
            
            return vector_id
            
        except ClientError as e:
            self._debug_log(f"S3 Vectors ClientError: {str(e)}")
            raise Exception(f"S3 Vectors put_vectors failed: {e}")
        except Exception as e:
            self._debug_log(f"Unexpected error in put_vector: {str(e)}")
            raise

    def put_vectors_batch(self, bucket_name: str, index_name: str, 
                         vectors: List[Dict[str, Any]]) -> List[str]:
        """Put multiple vectors into S3 vector index using S3 Vectors batch API."""
        start_time = time.time()
        self._debug_log(f"Starting put_vectors_batch operation")
        self._debug_log(f"Bucket: {bucket_name}, Index: {index_name}")
        self._debug_log(f"Batch size: {len(vectors)} vectors")
        
        try:
            # Use S3 Vectors PutVectors API with multiple vectors
            params = {
                "vectorBucketName": bucket_name,
                "indexName": index_name,
                "vectors": vectors
            }
            
            self._debug_log(f"Making S3 Vectors put_vectors batch API call")
            if self.debug and self.console:
                self._debug_log(f"API parameters: {json.dumps({k: v for k, v in params.items() if k != 'vectors'})}")
            
            response = self.s3vectors.put_vectors(**params)
            
            elapsed_time = time.time() - start_time
            self._debug_log(f"S3 Vectors put_vectors batch completed in {elapsed_time:.2f} seconds")
            
            # Extract vector IDs from the batch
            vector_ids = [vector["key"] for vector in vectors]
            self._debug_log(f"Batch stored successfully with {len(vector_ids)} vectors")
            
            return vector_ids
            
        except ClientError as e:
            self._debug_log(f"S3 Vectors ClientError: {str(e)}")
            raise Exception(f"S3 Vectors put_vectors batch failed: {e}")
        except Exception as e:
            self._debug_log(f"Unexpected error in put_vectors_batch: {str(e)}")
            raise
    
    def query_vectors(self, bucket_name: str, index_name: str, 
                     query_embedding: List[float], k: int = 5,
                     filter_expr: Optional[str] = None, 
                     return_metadata: bool = True, 
                     return_distance: bool = True) -> List[Dict[str, Any]]:
        """Query vectors from S3 vector index using S3 Vectors API."""
        try:
            # Use S3 Vectors QueryVectors API
            params = {
                "vectorBucketName": bucket_name,
                "indexName": index_name,
                "queryVector": {
                    "float32": query_embedding  # Query vector also needs float32 format
                },
                "topK": k,  # S3 Vectors uses 'topK' not 'k'
                "returnMetadata": return_metadata,
                "returnDistance": return_distance
            }
            
            # Add filter if provided - parse JSON string to object
            if filter_expr:
                import json
                try:
                    # Parse the JSON string into a Python object
                    filter_obj = json.loads(filter_expr)
                    params["filter"] = filter_obj
                    if self.debug:
                        self.console.print(f"[dim] Filter parsed successfully: {filter_obj}[/dim]")
                except json.JSONDecodeError as e:
                    if self.debug:
                        self.console.print(f"[dim] Filter JSON parse error: {e}[/dim]")
                    # If it's not valid JSON, pass as string (for backward compatibility)
                    params["filter"] = filter_expr
            
            response = self.s3vectors.query_vectors(**params)
            
            # Process response
            results = []
            if 'vectors' in response:
                for vector in response['vectors']:
                    result = {
                        'vectorId': vector.get('key'),
                        'similarity': vector.get('distance', 0.0),
                        'metadata': vector.get('metadata', {})
                    }
                    results.append(result)
            
            return results
            
        except ClientError as e:
            raise Exception(f"S3 Vectors query_vectors failed: {e}")
    
    def get_index(self, bucket_name: str, index_name: str) -> Dict[str, Any]:
        """Get index information including dimensions from S3 Vectors API."""
        try:
            # Use S3 Vectors GetIndex API
            params = {
                "vectorBucketName": bucket_name,
                "indexName": index_name
            }
            
            response = self.s3vectors.get_index(**params)
            return response
            
        except ClientError as e:
            raise Exception(f"S3 Vectors get_index failed: {e}")
