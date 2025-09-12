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
from s3vectors.utils.models import get_model_info


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
        model = get_model_info(model_id)
        return model.is_async() if model else False
    
    def embed_with_payload(self, model, payload: Dict[str, Any]) -> List[float]:
        """Embed using direct Bedrock API payload for sync models."""
        start_time = time.time()
        model_id = model.model_id
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
            
            # Extract embedding using schema-based approach
            embedding = model.extract_embedding(response_body)
            
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
    
    def _extract_job_id_from_arn(self, invocation_arn: str) -> str:
        """Extract Bedrock job ID from invocation ARN."""
        # ARN format: arn:aws:bedrock:region:account:async-invoke/job-id
        return invocation_arn.split('/')[-1]

    def embed_async_with_payload(self, model_id: str, final_payload: Dict[str, Any], 
                               async_output_s3_uri: str) -> tuple[List[Dict], str]:
        """Handle async embedding with model_id, final payload, and output S3 URI."""
        if not self.is_async_model(model_id):
            raise ValueError(f"Model {model_id} is not an async model")
        
        # Construct complete payload for start_async_invoke
        complete_payload = {
            "modelId": model_id,
            "modelInput": final_payload,
            "outputDataConfig": {
                "s3OutputDataConfig": {
                    "s3Uri": async_output_s3_uri
                }
            }
        }
        
        self._debug_log(f"Starting async embedding: {json.dumps(complete_payload, indent=2)}")
        
        try:
            # Start async job with complete payload
            response = self.bedrock_runtime.start_async_invoke(**complete_payload)
            invocation_arn = response['invocationArn']
            
            # Extract the Bedrock job ID
            job_id = self._extract_job_id_from_arn(invocation_arn)
            
            self._debug_log(f"Async job started: {invocation_arn}, Job ID: {job_id}")
            
            # Extract base S3 URI and construct the expected output path
            base_s3_uri = async_output_s3_uri.rstrip('/')
            # Bedrock will create the results at: base_uri/job_id/
            output_s3_uri = f"{base_s3_uri}/{job_id}/"
            
            self._debug_log(f"Looking for results at: {output_s3_uri}")
            
            # Wait for completion and retrieve results
            results = self._wait_and_retrieve_twelvelabs_results(invocation_arn, output_s3_uri)
            
            # Return results with job ID
            return results, job_id
                
        except ClientError as e:
            self._debug_log(f"Async embedding failed: {str(e)}")
            raise Exception(f"Async embedding failed: {e}")

    def _wait_and_retrieve_twelvelabs_results(self, invocation_arn: str, output_s3_uri: str) -> List[Dict]:
        """Wait for TwelveLabs job completion and retrieve results."""
        self._debug_log(f"Waiting for TwelveLabs job completion: {invocation_arn}")

        # Poll job status
        poll_count = 0
        max_polls = 180  # 30 minutes max (180 * 10 seconds)
        
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
            raise Exception("TwelveLabs job timed out after 30 minutes")
        
        # Retrieve results from S3
        return self._get_twelvelabs_results_from_s3(output_s3_uri)
    
    def _get_twelvelabs_results_from_s3(self, output_s3_uri: str) -> List[Dict]:
        """Retrieve TwelveLabs results from S3 output location."""
        # Parse S3 URI
        if not output_s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {output_s3_uri}")
        
        path_part = output_s3_uri[5:]  # Remove 's3://'
        if '/' not in path_part:
            raise ValueError(f"Invalid S3 URI format: {output_s3_uri}")
        
        bucket, prefix = path_part.split('/', 1)
        
        # TwelveLabs always outputs results to output.json
        result_key = f"{prefix}/output.json" if not prefix.endswith('/') else f"{prefix}output.json"
        
        self._debug_log(f"Reading TwelveLabs results from s3://{bucket}/{result_key}")
        
        try:
            obj_response = self.s3_client.get_object(Bucket=bucket, Key=result_key)
            result_data = json.loads(obj_response['Body'].read().decode('utf-8'))
            
            # Handle TwelveLabs format with 'data' array
            if 'data' in result_data and isinstance(result_data['data'], list):
                return result_data['data']
            elif isinstance(result_data, list):
                return result_data
            else:
                return [result_data]
                
        except ClientError as e:
            self._debug_log(f"Error retrieving results from S3: {str(e)}")
            raise Exception(f"Failed to retrieve TwelveLabs results from s3://{bucket}/{result_key}: {e}")
    
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
