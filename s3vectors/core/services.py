"""Core services for S3 Vectors operations."""

import json
import base64
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import boto3
from botocore.exceptions import ClientError


class BedrockService:
    """Service for Bedrock embedding operations."""
    
    def __init__(self, session: boto3.Session, region: str, debug: bool = False, console=None):
        self.bedrock_runtime = session.client('bedrock-runtime', region_name=region)
        self.debug = debug
        self.console = console
        self.region = region
        
        if self.debug and self.console:
            self.console.print(f"[dim] BedrockService initialized for region: {region}[/dim]")
    
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
            
    def embed_video_async(self, model_id: str, s3_video_uri: str, s3_output_uri: str, 
                         embedding_options: List[str] = None, bucket_owner: str = None,
                         max_wait_time: int = 1800, poll_interval: int = 10, console=None) -> Dict[str, Any]:
        """
        Embed video using Bedrock async API with the specified S3 locations.
        Polls until the job completes and returns the embedding.
        
        Args:
            model_id: The Bedrock model ID (e.g., twelvelabs.marengo-embed-2-7-v1:0)
            s3_video_uri: The S3 URI of the video file (e.g., s3://bucket/path/video.mp4)
            s3_output_uri: The S3 URI where the output should be stored (e.g., s3://bucket/output)
            embedding_options: List of embedding options (e.g., ["visual-text", "audio"])
            bucket_owner: The AWS account ID that owns the bucket (REQUIRED)
            max_wait_time: Maximum time to wait for async job completion in seconds (default: 30 minutes)
            poll_interval: Time between job status checks in seconds (default: 10 seconds)
            console: Rich console for output (optional)
            
        Returns:
            List[float]: The video embedding vector
            
        Raises:
            ValueError: If the video doesn't meet requirements or model is unsupported
            Exception: For API errors or timeout
        """
        if not model_id.startswith('twelvelabs.marengo-embed'):
            raise ValueError(f"Unsupported video embedding model: {model_id}. Use twelvelabs.marengo-embed-2-7-v1:0")
        
        if not s3_video_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI format for video: {s3_video_uri}. Must start with 's3://'")
            
        if not s3_output_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI format for output: {s3_output_uri}. Must start with 's3://'")
            
        if not bucket_owner:
            raise ValueError("The 'bucket_owner' parameter is required for video embedding. Please provide your AWS account ID.")
        
        # Default embedding options if not provided
        if not embedding_options:
            embedding_options = ["visual-text", "audio"]
        
        try:
            # Prepare the model input for async invocation exactly as in your example
            model_input = {
                "inputType": "video",
                "mediaSource": {
                    "s3Location": {
                        "uri": s3_video_uri,
                        "bucketOwner": bucket_owner  # Always include bucketOwner since it's required
                    }
                },
                "embeddingOption": embedding_options
            }
            
            # Prepare output data config
            output_data_config = {
                "s3OutputDataConfig": {
                    "s3Uri": s3_output_uri,
                    "bucketOwner": bucket_owner  # Include bucketOwner in the output config as well
                }
            }
            
            self._debug_log(f"Starting async video embedding with model: {model_id}")
            self._debug_log(f"Video S3 URI: {s3_video_uri}")
            self._debug_log(f"Output S3 URI: {s3_output_uri}")
            
            # Start the async job
            response = self.bedrock_runtime.start_async_invoke(
                modelId=model_id,
                modelInput=model_input,
                outputDataConfig=output_data_config
            )
            
            # Get the invocation ARN from the response
            invocation_arn = response.get('invocationArn')
            if not invocation_arn:
                raise Exception("Failed to get invocationArn from async invoke response")
                
            # Extract the invocation ID from the ARN
            invocation_id = invocation_arn.split('/')[-1]
            self._debug_log(f"Async job started with invocation ID: {invocation_id}")
            
            # Store the invocation ARN for the caller
            self.last_invocation_arn = invocation_arn
            
            # Poll for job completion
            self._debug_log(f"Polling for job completion, will wait up to {max_wait_time} seconds")
            if console:
                console.print(f"[yellow]Async video embedding job started. This may take several minutes...[/yellow]")
                console.print(f"[yellow]Invocation ARN: {invocation_arn}[/yellow]")
            
            start_time = time.time()
            while time.time() - start_time < max_wait_time:
                try:
                    # Check the status of the async job
                    status_response = self.bedrock_runtime.get_async_invoke(
                        invocationArn=invocation_arn
                    )
                    
                    status = status_response.get('status')
                    self._debug_log(f"Job status: {status}")
                    
                    if console and time.time() - start_time > 30:  # Show status updates after 30 seconds
                        elapsed = int(time.time() - start_time)
                        if elapsed % 30 == 0:  # Update every 30 seconds
                            console.print(f"[yellow]Job status: {status} (elapsed: {elapsed} seconds)[/yellow]")
                    
                    if status == 'Completed':
                        if console:
                            console.print(f"[green]Job completed successfully![/green]")
                        
                        # Job completed, now we need to get the results from S3
                        # The output file is named "output.json" in the specified S3 output location
                        output_s3_uri = f"{s3_output_uri.rstrip('/')}/{invocation_id}/output.json"
                        self._debug_log(f"Job completed, retrieving results from {output_s3_uri}")
                        
                        if console:
                            console.print(f"[yellow]Retrieving embedding from {output_s3_uri}[/yellow]")
                        
                        # Parse S3 URI to get bucket and key
                        output_bucket = output_s3_uri.split('/')[2]
                        output_key = '/'.join(output_s3_uri.split('/')[3:])
                        
                        # Create S3 client
                        s3_client = boto3.client('s3', region_name=self.region)
                        
                        # Get the results from S3
                        try:
                            response = s3_client.get_object(Bucket=output_bucket, Key=output_key)
                            result_data = json.loads(response['Body'].read().decode('utf-8'))
                            
                            # Extract embeddings from response based on the expected format
                            # The response should have a "data" array containing objects with "embedding" arrays
                            if 'data' in result_data and isinstance(result_data['data'], list) and len(result_data['data']) > 0:
                                # Group embeddings by type
                                embeddings_by_type = {}
                                for segment in result_data['data']:
                                    if 'embedding' in segment and 'embeddingOption' in segment:
                                        embedding_type = segment['embeddingOption']
                                        if embedding_type not in embeddings_by_type:
                                            embeddings_by_type[embedding_type] = []
                                        embeddings_by_type[embedding_type].append({
                                            'embedding': segment['embedding'],
                                            'startSec': segment.get('startSec', 0),
                                            'endSec': segment.get('endSec', 0)
                                        })
                                
                                self._debug_log(f"Found {len(result_data['data'])} total embeddings")
                                self._debug_log(f"Embedding types: {list(embeddings_by_type.keys())}")
                                
                                # Return all embeddings grouped by type
                                if console:
                                    total_count = sum(len(embs) for embs in embeddings_by_type.values())
                                    console.print(f"[green]Successfully retrieved {total_count} embeddings across {len(embeddings_by_type)} types[/green]")
                                
                                # Return the full embeddings_by_type dictionary
                                return {
                                    'embeddings_by_type': embeddings_by_type,
                                    'total_count': len(result_data['data']),
                                    'types': list(embeddings_by_type.keys())
                                }
                            else:
                                self._debug_log(f"Unexpected response format: {result_data}")
                                raise Exception("No embedding found in the response: Invalid response format")
                            
                        except Exception as e:
                            raise Exception(f"Failed to retrieve results from S3: {str(e)}")
                        
                    elif status == 'Failed':
                        failure_reason = status_response.get('failureMessage', 'Unknown reason')
                        raise Exception(f"Async job failed: {failure_reason}")
                    elif status == 'InProgress':
                        # Job is still in progress, continue polling
                        self._debug_log("Job is still in progress, continuing to poll...")
                        
                    # Wait before polling again
                    time.sleep(poll_interval)
                    
                except ClientError as e:
                    self._debug_log(f"Bedrock ClientError during status check: {str(e)}")
                    raise Exception(f"Failed to check async job status: {e}")
                    
            # If we get here, we've timed out
            raise Exception(f"Async job timed out after {max_wait_time} seconds")
            
        except ClientError as e:
            self._debug_log(f"Bedrock ClientError: {str(e)}")
            raise Exception(f"Bedrock video embedding failed: {e}")
        except Exception as e:
            self._debug_log(f"Unexpected error in embed_video_async: {str(e)}")
            raise



class S3VectorService:
    """Service for S3 Vector operations."""
    
    def __init__(self, session: boto3.Session, region: str, debug: bool = False, console=None):
        # Use S3 Vectors client with new endpoint URL
        endpoint_url = f"https://s3vectors.{region}.api.aws"
        self.s3vectors = session.client('s3vectors', region_name=region, endpoint_url=endpoint_url)
        self.region = region
        self.debug = debug
        self.console = console
        
        if self.debug and self.console:
            self.console.print(f"[dim] S3VectorService initialized for region: {region}[/dim]")
            self.console.print(f"[dim] Using endpoint: {endpoint_url}[/dim]")
    
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
            
            # Call the API and handle the response
            try:
                response = self.s3vectors.put_vectors(**params)
                self._debug_log(f"Response: {response}")
            except Exception as e:
                self._debug_log(f"API call error: {str(e)}")
                raise
            
            elapsed_time = time.time() - start_time
            self._debug_log(f"S3 Vectors put_vectors batch completed in {elapsed_time:.2f} seconds")
            
            # Extract vector IDs from the input batch since the API doesn't return them
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
            seen_vector_ids = set()  # Track seen vector IDs to ensure uniqueness
            seen_segments = set()  # Track seen segments (type_index) to avoid duplicate segments
            
            if 'vectors' in response:
                for vector in response['vectors']:
                    vector_id = vector.get('key')
                    metadata = vector.get('metadata', {})
                    
                    # Skip duplicates
                    if vector_id in seen_vector_ids:
                        if self.debug:
                            self.console.print(f"[dim] Skipping duplicate vector ID: {vector_id}[/dim]")
                        continue
                    
                    # For video embeddings, check if we've already seen this segment type and index
                    embed_type = metadata.get('S3VECTORS-EMBED-TYPE')
                    segment_index = metadata.get('S3VECTORS-EMBED-SEGMENT-INDEX')
                    
                    if embed_type and segment_index is not None:
                        # This is a video segment
                        segment_key = f"{embed_type}_{segment_index}"
                        
                        if segment_key in seen_segments:
                            if self.debug:
                                self.console.print(f"[dim] Skipping duplicate segment: {segment_key} (vector ID: {vector_id})[/dim]")
                            continue
                        
                        seen_segments.add(segment_key)
                    
                    seen_vector_ids.add(vector_id)
                    
                    result = {
                        'vectorId': vector_id,
                        'similarity': vector.get('distance', 0.0),
                        'metadata': metadata
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
