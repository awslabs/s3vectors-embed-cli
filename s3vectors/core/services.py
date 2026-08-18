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
    
    def embed_with_payload_raw(self, model, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke model and return full response body."""
        body = json.dumps(payload)
        response = self.bedrock_runtime.invoke_model(
            modelId=model.model_id, body=body, contentType='application/json'
        )
        return json.loads(response['body'].read())

    def _extract_job_id_from_arn(self, invocation_arn: str) -> str:
        """Extract Bedrock job ID from invocation ARN."""
        # ARN format: arn:aws:bedrock:region:account:async-invoke/job-id
        return invocation_arn.split('/')[-1]

    def embed_async_with_payload(self, model, final_payload: Dict[str, Any], 
                               async_output_s3_uri: str) -> tuple[List[Dict], str]:
        """Handle async embedding with model object, final payload, and output S3 URI."""
        if not model.capabilities.supports_async:
            raise ValueError(f"Model {model.model_id} does not support async processing")
        
        model_id = model.model_id
        
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
            results = self._wait_and_retrieve_async_results(invocation_arn, output_s3_uri, model)
            
            # Return results with job ID
            return results, job_id
                
        except ClientError as e:
            self._debug_log(f"Async embedding failed: {str(e)}")
            raise Exception(f"Async embedding failed: {e}")

    def _wait_and_retrieve_async_results(self, invocation_arn: str, output_s3_uri: str, model) -> List[Dict]:
        """Wait for async job completion and retrieve results."""
        self._debug_log(f"Waiting for async job completion: {invocation_arn}")

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
                    raise Exception(f"Async embedding failed: {failure_message}")
                elif status in ['InProgress', 'Submitted']:
                    time.sleep(10)  # Wait 10 seconds before next poll
                    poll_count += 1
                else:
                    raise Exception(f"Unexpected job status: {status}")
                    
            except ClientError as e:
                self._debug_log(f"Error checking job status: {str(e)}")
                raise Exception(f"Failed to check async job status: {e}")
        
        if poll_count >= max_polls:
            raise Exception("Async job timed out after 30 minutes")
        
        # Retrieve results from S3 using model's async_result_config
        return self._get_async_results_from_s3(output_s3_uri, model)
    
    def _get_async_results_from_s3(self, output_s3_uri: str, model) -> list:
        """Get async results from S3 using model's async_result_config."""
        config = model.capabilities.async_result_config
        bucket, prefix = self._parse_s3_uri(output_s3_uri)
        # Ensure prefix ends with /
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        if config["type"] == "single_file":
            return self._parse_single_file_results(bucket, prefix, config)
        elif config["type"] == "manifest_jsonl":
            return self._parse_manifest_jsonl_results(bucket, prefix, config)
        else:
            raise ValueError(f"Unknown async_result_config type: {config['type']}")
    
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

    def _parse_s3_uri(self, s3_uri: str):
        """Parse S3 URI into (bucket, key)."""
        path = s3_uri.replace("s3://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def _read_s3_json(self, bucket: str, key: str) -> dict:
        """Read and parse a JSON file from S3."""
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))

    def _read_s3_text(self, bucket: str, key: str) -> str:
        """Read a text file from S3."""
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')

    def _parse_manifest_jsonl_results(self, bucket: str, prefix: str, config: dict) -> list:
        """Parse Nova-style manifest + JSONL async results from S3."""
        manifest_key = f"{prefix}{config['result_file']}"
        manifest = self._read_s3_json(bucket, manifest_key)

        results = []
        results_path = config.get("manifest_results_path", "embeddingResults")
        output_uri_key = config.get("manifest_output_uri_key", "outputFileUri")

        for modality_result in manifest.get(results_path, []):
            # Skip failed results
            status = modality_result.get(config.get("manifest_status_key", "status"), "")
            if status != "SUCCESS":
                continue

            output_uri = modality_result.get(output_uri_key)
            if output_uri:
                jsonl_bucket, jsonl_key = self._parse_s3_uri(output_uri)
                jsonl_content = self._read_s3_text(jsonl_bucket, jsonl_key)
                embedding_type = modality_result.get(config.get("manifest_type_key", "embeddingType"))

                for line in jsonl_content.strip().split("\n"):
                    if line.strip():
                        entry = json.loads(line)
                        embedding = entry.get(config.get("embedding_key", "embedding"), [])
                        result_dict = {"embedding": embedding}
                        if embedding_type:
                            result_dict["embeddingType"] = embedding_type
                        # Copy all other fields from the JSONL entry
                        for k, v in entry.items():
                            if k != config.get("embedding_key", "embedding"):
                                result_dict[k] = v
                        results.append(result_dict)
        return results

    def _parse_single_file_results(self, bucket: str, prefix: str, config: dict) -> list:
        """Parse single-file async results from S3 (e.g., TwelveLabs output.json)."""
        result_key = f"{prefix}{config['result_file']}"
        result_data = self._read_s3_json(bucket, result_key)

        data_path = config.get("data_path")
        if data_path:
            return result_data.get(data_path, [])
        return result_data if isinstance(result_data, list) else [result_data]


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
