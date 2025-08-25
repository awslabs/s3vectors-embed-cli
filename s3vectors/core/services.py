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
        # Create Bedrock clients with user agent tracking
        self.bedrock_runtime = session.client(
            'bedrock-runtime', 
            region_name=region,
            config=get_boto_config()
        )
        # Create S3 client for TwelveLabs result retrieval
        self.s3_client = session.client(
            's3',
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
    

    

    
    def is_async_model(self, model_id: str) -> bool:
        """Check if model requires async processing."""
        return model_id.startswith('twelvelabs.')
    
    def embed_with_payload(self, model_id: str, payload: Dict[str, Any]) -> List[float]:
        """Embed using direct Bedrock API payload for sync models."""
        start_time = time.time()
        self._debug_log(f"Starting embedding with model: {model_id}")
        self._debug_log(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            body = json.dumps(payload)
            
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
            
            # Extract embedding based on model type - get whatever the user requested
            if model_id.startswith('amazon.titan-embed-text-v2'):
                if 'embeddingsByType' in response_body:
                    # Get the first available embedding type (whatever user requested)
                    embeddings_by_type = response_body['embeddingsByType']
                    embedding = list(embeddings_by_type.values())[0] if embeddings_by_type else []
                else:
                    embedding = response_body.get('embedding', [])
                    
            elif model_id.startswith('amazon.titan-embed-text-v1'):
                embedding = response_body['embedding']
                
            elif model_id.startswith('amazon.titan-embed-image'):
                embedding = response_body['embedding']
                
            elif model_id.startswith('cohere.embed'):
                # Cohere returns embeddings as a direct array
                embeddings = response_body.get('embeddings', [])
                if embeddings and len(embeddings) > 0:
                    embedding = embeddings[0]  # First text's embedding
                else:
                    embedding = []
            else:
                raise ValueError(f"Unsupported model for embed_with_payload: {model_id}")
            
            self._debug_log(f"Generated embedding with {len(embedding)} dimensions")
            total_time = time.time() - start_time
            self._debug_log(f"Total embedding operation completed in {total_time:.2f} seconds")
            
            return embedding
            
        except ClientError as e:
            self._debug_log(f"Bedrock ClientError: {str(e)}")
            raise Exception(f"Bedrock embedding failed: {e}")
        except Exception as e:
            self._debug_log(f"Unexpected error in embed_with_payload: {str(e)}")
            raise
    
    def embed_async_with_payload(self, payload: Dict[str, Any]) -> List[Dict]:
        """Handle async embedding with complete StartAsyncInvoke payload."""
        model_id = payload.get("modelId")
        if not self.is_async_model(model_id):
            raise ValueError(f"Model {model_id} is not an async model")
        
        self._debug_log(f"Starting async embedding: {json.dumps(payload, indent=2)}")
        
        try:
            # Start async job with complete payload
            response = self.bedrock_runtime.start_async_invoke(**payload)
            invocation_arn = response['invocationArn']
            
            self._debug_log(f"Async job started: {invocation_arn}")
            
            # Extract output S3 URI for result retrieval
            output_s3_uri = payload["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
            
            # Wait for completion and retrieve results
            results = self._wait_and_retrieve_twelvelabs_results(invocation_arn, output_s3_uri)
            
            # Add job ID to each result for tracking
            for result in results:
                result['jobId'] = invocation_arn
            
            return results
                
        except ClientError as e:
            self._debug_log(f"Async embedding failed: {str(e)}")
            raise Exception(f"Async embedding failed: {e}")
    

    
    def _wait_and_retrieve_twelvelabs_results(self, invocation_arn: str, output_s3_uri: str) -> List[Dict]:
        """Wait for TwelveLabs job completion and retrieve results."""
        self._debug_log(f"Waiting for TwelveLabs job completion: {invocation_arn}")
        
        # Poll job status
        poll_count = 0
        max_polls = 120  # 20 minutes max (120 * 10 seconds)
        
        while poll_count < max_polls:
            try:
                response = self.bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)
                status = response['status']
                
                self._debug_log(f"Job status: {status} (poll #{poll_count + 1})")
                
                if status == 'Completed':
                    break
                elif status == 'Failed':
                    failure_message = response.get('failureMessage', 'Unknown error')
                    raise Exception(f"TwelveLabs async embedding failed: {failure_message}")
                elif status in ['InProgress', 'Submitted']:
                    time.sleep(10)  # Wait 10 seconds before next poll
                    poll_count += 1
                else:
                    raise Exception(f"Unexpected job status: {status}")
                    
            except ClientError as e:
                self._debug_log(f"Error checking job status: {str(e)}")
                raise Exception(f"Failed to check TwelveLabs job status: {e}")
        
        if poll_count >= max_polls:
            raise Exception("TwelveLabs job timed out after 20 minutes")
        
        # Retrieve results from S3
        return self._get_twelvelabs_results_from_s3(output_s3_uri)
    
    def _get_twelvelabs_results_from_s3(self, output_s3_uri: str) -> List[Dict]:
        """Retrieve TwelveLabs results from S3 output location."""
        # Parse S3 URI
        if not output_s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {output_s3_uri}")
        
        path_part = output_s3_uri[5:]  # Remove 's3://'
        if '/' in path_part:
            bucket, prefix = path_part.split('/', 1)
        else:
            bucket = path_part
            prefix = ""
        
        self._debug_log(f"Retrieving TwelveLabs results from s3://{bucket}/{prefix}")
        
        try:
            # List objects in the output location
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            
            if 'Contents' not in response:
                raise Exception("No results found in S3 output location")
            
            # Debug: List all files found
            all_files = [obj['Key'] for obj in response['Contents']]
            self._debug_log(f"All files found in S3 output location: {all_files}")
            
            # Find the result file (usually ends with .json)
            result_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')]
            
            if not result_files:
                raise Exception("No JSON result files found in S3 output location")
            
            # Try to read all JSON files to find the one with embeddings
            for result_key in result_files:
                self._debug_log(f"Reading TwelveLabs results from s3://{bucket}/{result_key}")
                
                obj_response = self.s3_client.get_object(Bucket=bucket, Key=result_key)
                result_data = json.loads(obj_response['Body'].read().decode('utf-8'))
                
                self._debug_log(f"Content of {result_key}: {json.dumps(result_data, indent=2)}")
                
                # Check if this file contains embeddings
                if self._has_embeddings(result_data):
                    # Handle TwelveLabs format with 'data' array
                    if 'data' in result_data and isinstance(result_data['data'], list):
                        return result_data['data']
                    # Handle both single object and array responses
                    elif isinstance(result_data, list):
                        return result_data
                    else:
                        return [result_data]
            
            # If no file with embeddings found, raise error
            raise Exception("No files with embeddings found in S3 output location")
                
        except ClientError as e:
            self._debug_log(f"Error retrieving results from S3: {str(e)}")
            raise Exception(f"Failed to retrieve TwelveLabs results from S3: {e}")
    
    def _has_embeddings(self, data):
        """Check if the data contains embeddings."""
        if isinstance(data, dict):
            # Check for common embedding keys
            embedding_keys = ['embedding', 'embeddings', 'vector', 'vectors']
            if any(key in data for key in embedding_keys):
                return True
            # Check for TwelveLabs format with 'data' array
            if 'data' in data and isinstance(data['data'], list):
                return any(self._has_embeddings(item) for item in data['data'])
        elif isinstance(data, list):
            # Check if any item in the list has embeddings
            return any(self._has_embeddings(item) for item in data)
        return False


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
