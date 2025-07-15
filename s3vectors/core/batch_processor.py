"""Batch processing functionality for S3 Vectors CLI."""

import os
import json
import uuid
import base64
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path


class ContentType(Enum):
    """Supported content types for batch processing."""
    TEXT = "text"
    IMAGE = "image"


@dataclass
class BatchConfig:
    """Configuration for batch processing operations."""
    max_workers: int = 4
    max_vectors_per_batch: int = 500  # Maximum vectors per batch operation
    
    # File extensions
    text_extensions: List[str] = None
    image_extensions: List[str] = None
    
    def __post_init__(self):
        if self.text_extensions is None:
            self.text_extensions = ['txt', 'md', 'json', 'csv', 'log', 'xml', 'html', 'rtf']
        if self.image_extensions is None:
            self.image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']


class InputProcessor:
    """Processes different types of input (text, files, S3 objects, wildcards)."""
    
    def __init__(self, s3_client):
        self.s3_client = s3_client
    
    def process_input(self, input_value: str, input_type: str = "auto", 
                     bucket_owner: Optional[str] = None, is_image: bool = False) -> Dict[str, Any]:
        """
        Process input and return structured data for embedding.
        
        Args:
            input_value: The input value (text, file path, S3 URI, etc.)
            input_type: Type of input ("text", "file", "s3", "auto")
            bucket_owner: AWS account ID for cross-account S3 access
            is_image: Whether this is for image processing (True) or text processing (False)
            
        Returns:
            Dict with processed input information
        """
        if input_type == "auto":
            input_type = self._detect_input_type(input_value)
        
        # Validate file extension for file inputs
        if input_type in ["local_file", "s3_file"]:
            self._validate_file_extension(input_value, input_type, is_image)
        
        if input_type == "text":
            return self._process_text_value(input_value)
        elif input_type == "local_file":
            return self._process_local_file(input_value)
        elif input_type == "local_wildcard":
            return self._process_local_wildcard(input_value)
        elif input_type == "s3_file":
            return self._process_s3_file(input_value, bucket_owner)
        elif input_type == "s3_wildcard":
            return self._process_s3_wildcard(input_value, bucket_owner)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    
    def _validate_file_extension(self, input_value: str, input_type: str, is_image: bool):
        """
        Validate that the file extension matches the expected parameter type.
        
        Args:
            input_value: The input file path or S3 URI
            input_type: Type of input ("local_file" or "s3_file")
            is_image: True if --image parameter was used, False if --text parameter was used
            
        Raises:
            ValueError: If file extension doesn't match the expected parameter type
        """
        # Extract file extension
        if input_type == "local_file":
            # Handle file:// prefix if present
            actual_path = input_value[7:] if input_value.startswith('file://') else input_value
            extension = Path(actual_path).suffix.lower()[1:]  # Remove the dot
            file_description = f"local file '{actual_path}'"
        elif input_type == "s3_file":
            # Extract extension from S3 URI
            extension = input_value.lower().split('.')[-1] if '.' in input_value else ''
            file_description = f"S3 file '{input_value}'"
        else:
            return  # No validation needed for other types
        
        # Get supported extensions
        config = BatchConfig()
        
        if is_image:
            # --image parameter expects image files
            if extension not in config.image_extensions:
                if extension in config.text_extensions:
                    raise ValueError(
                        f"Text file provided to --image parameter. "
                        f"Found {file_description} with extension '.{extension}'. "
                        f"Use --text parameter for text files."
                    )
                else:
                    raise ValueError(
                        f"Unsupported file type for --image parameter. "
                        f"Found {file_description} with extension '.{extension}'. "
                        f"Supported image extensions: {', '.join(config.image_extensions)}"
                    )
        else:
            # --text parameter expects text files
            if extension not in config.text_extensions:
                if extension in config.image_extensions:
                    raise ValueError(
                        f"Image file provided to --text parameter. "
                        f"Found {file_description} with extension '.{extension}'. "
                        f"Use --image parameter for image files."
                    )
                else:
                    raise ValueError(
                        f"Unsupported file type for --text parameter. "
                        f"Found {file_description} with extension '.{extension}'. "
                        f"Supported text extensions: {', '.join(config.text_extensions)}"
                    )
    
    def _detect_input_type(self, input_value: str) -> str:
        """Detect the type of input based on the value."""
        if input_value.startswith('s3://'):
            if input_value.endswith('*') or input_value.endswith('/*'):
                return "s3_wildcard"
            else:
                return "s3_file"
        elif input_value.startswith('file://'):
            return "local_file"
        elif '*' in input_value or '?' in input_value:
            # Local filesystem wildcard pattern
            return "local_wildcard"
        elif os.path.exists(input_value):
            return "local_file"
        else:
            return "text"
    
    def _process_text_value(self, text: str) -> Dict[str, Any]:
        """Process direct text input."""
        return {
            "type": "text_value",
            "content": text.strip(),
            "metadata": {"S3VECTORS-EMBED-SRC-CONTENT": text.strip()}
        }
    
    def _process_local_file(self, file_path: str) -> Dict[str, Any]:
        """Process local file input."""
        # Remove file:// prefix if present
        if file_path.startswith('file://'):
            actual_path = file_path[7:]
        else:
            actual_path = file_path
        
        if not os.path.exists(actual_path):
            raise ValueError(f"File not found: {actual_path}")
        
        # Determine content type
        path_obj = Path(actual_path)
        extension = path_obj.suffix.lower()[1:]  # Remove the dot
        
        config = BatchConfig()
        if extension in config.text_extensions:
            content_type = ContentType.TEXT
            with open(actual_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif extension in config.image_extensions:
            content_type = ContentType.IMAGE
            with open(actual_path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return {
            "type": "local_file",
            "content_type": content_type,
            "content": content,
            "file_path": actual_path,
            "metadata": {
                k: v for k, v in {
                    "S3VECTORS-EMBED-SRC-CONTENT": content if content_type == ContentType.TEXT else None,
                    "S3VECTORS-EMBED-SRC-LOCATION": f"file://{actual_path}"
                }.items() if v is not None
            }
        }
    
    def _process_local_wildcard(self, pattern: str) -> Dict[str, Any]:
        """Process local filesystem wildcard pattern for batch processing."""
        import glob
        from pathlib import Path
        
        # Expand the wildcard pattern (supports recursive patterns with **)
        matched_files = glob.glob(pattern, recursive=True)
        
        # Filter to only include files (not directories) with supported extensions
        valid_files = []
        config = BatchConfig()
        
        for file_path in matched_files:
            if os.path.isfile(file_path):
                path_obj = Path(file_path)
                extension = path_obj.suffix.lower()[1:]  # Remove the dot
                
                if extension in config.text_extensions or extension in config.image_extensions:
                    valid_files.append(file_path)
        
        if not valid_files:
            raise ValueError(f"No supported files found matching pattern: {pattern}")
        
        return {
            "type": "local_wildcard",
            "pattern": pattern,
            "files": valid_files,
            "total_count": len(valid_files),
            "batch_processing": True
        }
    
    def _process_s3_file(self, s3_uri: str, bucket_owner: Optional[str] = None) -> Dict[str, Any]:
        """Process single S3 file input."""
        bucket, key = self._parse_s3_uri(s3_uri)
        
        # Determine content type from extension
        extension = key.lower().split('.')[-1] if '.' in key else ''
        config = BatchConfig()
        
        if extension in config.text_extensions:
            content_type = ContentType.TEXT
            content = self._read_s3_text_file(bucket, key, bucket_owner)
        elif extension in config.image_extensions:
            content_type = ContentType.IMAGE
            content = self._read_s3_image_file(bucket, key, bucket_owner)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return {
            "type": "s3_file",
            "content_type": content_type,
            "content": content,
            "s3_uri": s3_uri,
            "bucket": bucket,
            "key": key,
            "metadata": {
                k: v for k, v in {
                    "S3VECTORS-EMBED-SRC-CONTENT": content if content_type == ContentType.TEXT else None,
                    "S3VECTORS-EMBED-SRC-LOCATION": s3_uri
                }.items() if v is not None
            }
        }
    
    def _process_s3_wildcard(self, s3_pattern: str, bucket_owner: Optional[str] = None) -> Dict[str, Any]:
        """Process S3 wildcard pattern for batch processing."""
        return {
            "type": "s3_wildcard",
            "pattern": s3_pattern,
            "bucket_owner": bucket_owner,
            "batch_processing": True
        }
    
    def _parse_s3_uri(self, s3_uri: str) -> tuple:
        """Parse S3 URI to extract bucket and key."""
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        
        path_part = s3_uri[5:]  # Remove 's3://'
        if '/' not in path_part:
            raise ValueError(f"Invalid S3 URI format. Must include bucket and key: {s3_uri}")
        
        bucket, key = path_part.split('/', 1)
        return bucket, key
    
    def _read_s3_text_file(self, bucket: str, key: str, bucket_owner: Optional[str] = None) -> str:
        """Read text content from S3."""
        get_params = {'Bucket': bucket, 'Key': key}
        if bucket_owner:
            get_params['ExpectedBucketOwner'] = bucket_owner
        
        try:
            response = self.s3_client.get_object(**get_params)
            return response['Body'].read().decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read S3 file s3://{bucket}/{key}: {str(e)}")
    
    def _read_s3_image_file(self, bucket: str, key: str, bucket_owner: Optional[str] = None) -> str:
        """Read image content from S3 and return as base64."""
        get_params = {'Bucket': bucket, 'Key': key}
        if bucket_owner:
            get_params['ExpectedBucketOwner'] = bucket_owner
        
        try:
            response = self.s3_client.get_object(**get_params)
            image_bytes = response['Body'].read()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read S3 image s3://{bucket}/{key}: {str(e)}")


class BatchProcessor:
    """Handles batch processing operations for S3 Vectors embedding."""
    
    def __init__(self, s3_client, bedrock_service, s3vector_service, config: BatchConfig = None):
        """Initialize BatchProcessor with services and configuration."""
        self.s3_client = s3_client
        self.bedrock_service = bedrock_service
        self.s3vector_service = s3vector_service
        self.config = config or BatchConfig()
        
        # Thread synchronization
        self._batch_lock = threading.Lock()
        
        # Batch state
        self.current_batch = []
        self.current_batch_count = 0  # Track number of vectors instead of size
        self.processed_count = 0
        self.failed_count = 0
        self.processed_keys = []
    
    def process_wildcard_pattern(self, s3_pattern: str, vector_bucket: str, index_name: str,
                               model_id: str, metadata_template: Dict[str, Any],
                               bucket_owner: Optional[str] = None,
                               use_object_key_name: bool = False) -> Dict[str, Any]:
        """Process all files matching a wildcard S3 pattern using streaming approach."""
        try:
            self._reset_batch_state()
            
            # Get index dimensions first
            try:
                index_info = self.s3vector_service.get_index(vector_bucket, index_name)
                # Extract dimensions from index info - it's nested under 'index' key
                index_data = index_info.get('index', {})
                dimensions = index_data.get('dimension')  # Note: singular 'dimension'
                if not dimensions:
                    raise ValueError(f"Failed to get dimensions for index '{index_name}' in bucket '{vector_bucket}'. The index may be corrupted or have invalid metadata.")
            except Exception as e:
                # Check if it's a NotFoundException (index doesn't exist)
                error_str = str(e)
                if "NotFoundException" in error_str or "could not be found" in error_str:
                    raise ValueError(f"Vector index '{index_name}' does not exist in bucket '{vector_bucket}'. Please verify the index name and bucket name are correct, and that the index has been created.")
                elif isinstance(e, ValueError):
                    # Re-raise our own ValueError from above
                    raise e
                else:
                    raise ValueError(f"Failed to access vector index '{index_name}' in bucket '{vector_bucket}': {str(e)}")
            
            # Parse S3 URI
            bucket, prefix = self._parse_s3_wildcard_uri(s3_pattern)
            
            # Process files in streaming chunks
            total_files_found = self._process_files_streaming(
                bucket=bucket,
                prefix=prefix,
                bucket_owner=bucket_owner,
                model_id=model_id,
                vector_bucket=vector_bucket,
                index_name=index_name,
                metadata_template=metadata_template,
                use_object_key_name=use_object_key_name,
                wildcard_pattern=s3_pattern,
                dimensions=dimensions
            )
            
            # Flush any remaining items in batch
            if self.current_batch:
                self._flush_batch(vector_bucket, index_name)
            
            if total_files_found == 0:
                return {
                    'Keys': [],
                    'status': 'success',
                    'pattern': s3_pattern,
                    'processed_count': 0,
                    'failed_count': 0,
                    'message': f'No files found under {s3_pattern}'
                }
            
            calculated_failed = total_files_found - self.processed_count
            print(f"\nBatch processing completed!", flush=True)
            print(f"   Total files found: {total_files_found}", flush=True)
            print(f"   Successfully processed: {self.processed_count}", flush=True)
            print(f"   Failed: {calculated_failed}", flush=True)
            
            return {
                'Keys': self.processed_keys,
                'status': 'success',
                'pattern': s3_pattern,
                'processed_count': self.processed_count,
                'failed_count': self.failed_count,
                'total_files_found': total_files_found
            }
            
        except Exception as e:
            print(f"Failed to process wildcard pattern {s3_pattern}: {str(e)}", flush=True)
            return {
                'Keys': self.processed_keys,
                'status': 'failed',
                'pattern': s3_pattern,
                'processed_count': self.processed_count,
                'failed_count': self.failed_count,
                'error': str(e)
            }
    
    def process_local_wildcard_pattern(self, local_pattern: str, vector_bucket: str, index_name: str,
                                     model_id: str, metadata_template: Dict[str, Any],
                                     use_object_key_name: bool = False) -> Dict[str, Any]:
        """Process all files matching a local filesystem wildcard pattern using batch approach."""
        try:
            self._reset_batch_state()
            
            # Get index dimensions first
            try:
                index_info = self.s3vector_service.get_index(vector_bucket, index_name)
                index_data = index_info.get('index', {})
                dimensions = index_data.get('dimension')
                if not dimensions:
                    raise ValueError(f"Failed to get dimensions for index '{index_name}' in bucket '{vector_bucket}'. The index may be corrupted or have invalid metadata.")
            except Exception as e:
                error_str = str(e)
                if "NotFoundException" in error_str or "could not be found" in error_str:
                    raise ValueError(f"Vector index '{index_name}' does not exist in bucket '{vector_bucket}'. Please verify the index name and bucket name are correct, and that the index has been created.")
                elif isinstance(e, ValueError):
                    raise e
                else:
                    raise ValueError(f"Failed to access vector index '{index_name}' in bucket '{vector_bucket}': {str(e)}")
            
            # Expand the wildcard pattern to get all matching files
            import glob
            from pathlib import Path
            
            matched_files = glob.glob(local_pattern, recursive=True)
            
            # Filter to only include files with supported extensions
            valid_files = []
            for file_path in matched_files:
                if os.path.isfile(file_path):
                    path_obj = Path(file_path)
                    extension = path_obj.suffix.lower()[1:]  # Remove the dot
                    
                    if extension in self.config.text_extensions or extension in self.config.image_extensions:
                        valid_files.append(file_path)
            
            if not valid_files:
                return {
                    'Keys': [],
                    'status': 'success',
                    'pattern': local_pattern,
                    'processed_count': 0,
                    'failed_count': 0,
                    'message': f'No supported files found matching pattern: {local_pattern}'
                }
            
            print(f"Found {len(valid_files)} files matching pattern: {local_pattern}", flush=True)
            
            # Process files in batches
            total_files_found = len(valid_files)
            
            # Process files using thread pool
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all files for processing
                future_to_file = {}
                for file_path in valid_files:
                    future = executor.submit(
                        self._process_single_local_file,
                        file_path, model_id, vector_bucket, index_name,
                        metadata_template, use_object_key_name, dimensions
                    )
                    future_to_file[future] = file_path
                
                # Process completed futures
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        vector_item = future.result()
                        if vector_item:
                            # Use the same pattern as S3 processing
                            self._add_to_batch(vector_item, vector_bucket, index_name)
                                
                    except Exception as e:
                        with self._batch_lock:
                            self.failed_count += 1
                        print(f"Failed to process {file_path}: {str(e)}", flush=True)
            
            # Flush any remaining items in batch
            if self.current_batch:
                self._flush_batch(vector_bucket, index_name)
            
            calculated_failed = total_files_found - self.processed_count
            print(f"\nLocal batch processing completed!", flush=True)
            print(f"   Total files found: {total_files_found}", flush=True)
            print(f"   Successfully processed: {self.processed_count}", flush=True)
            print(f"   Failed: {calculated_failed}", flush=True)
            
            return {
                'Keys': self.processed_keys,
                'status': 'success',
                'pattern': local_pattern,
                'processed_count': self.processed_count,
                'failed_count': self.failed_count,
                'total_files_found': total_files_found
            }
            
        except Exception as e:
            print(f"Failed to process local wildcard pattern {local_pattern}: {str(e)}", flush=True)
            return {
                'Keys': self.processed_keys,
                'status': 'failed',
                'pattern': local_pattern,
                'processed_count': self.processed_count,
                'failed_count': self.failed_count,
                'error': str(e)
            }
    
    def _process_files_streaming(self, bucket: str, prefix: str, bucket_owner: Optional[str],
                               model_id: str, vector_bucket: str, index_name: str,
                               metadata_template: Dict[str, Any], use_object_key_name: bool,
                               wildcard_pattern: str, dimensions: Optional[int]) -> int:
        """Process files in streaming chunks of 1000 with pagination."""
        list_params = {
            'Bucket': bucket,
            'Prefix': prefix,
            'MaxKeys': 1000  # Process in chunks of 1000
        }
        if bucket_owner:
            list_params['ExpectedBucketOwner'] = bucket_owner
        
        continuation_token = None
        total_files_found = 0
        chunk_number = 1
        
        while True:
            # Add continuation token if we have one
            if continuation_token:
                list_params['ContinuationToken'] = continuation_token
            
            try:
                print(f"Processing chunk {chunk_number}...", flush=True)
                
                # Get next batch of up to 1000 files
                response = self.s3_client.list_objects_v2(**list_params)
                objects = response.get('Contents', [])
                
                if not objects:
                    break
                
                # Filter objects by supported extensions for this chunk
                filtered_objects = self._filter_objects_by_extension(objects, bucket, bucket_owner)
                
                if filtered_objects:
                    total_files_found += len(filtered_objects)
                    print(f"Found {len(filtered_objects)} supported files in chunk {chunk_number}", flush=True)
                    
                    # Process this chunk in parallel using existing worker logic
                    self._process_objects_parallel(
                        s3_objects=filtered_objects,
                        model_id=model_id,
                        vector_bucket=vector_bucket,
                        index_name=index_name,
                        metadata_template=metadata_template,
                        use_object_key_name=use_object_key_name,
                        wildcard_pattern=wildcard_pattern,
                        dimensions=dimensions
                    )
                
                # Check if there are more objects to retrieve
                if response.get('IsTruncated', False):
                    continuation_token = response.get('NextContinuationToken')
                    chunk_number += 1
                else:
                    break
                    
            except Exception as e:
                print(f"Error processing chunk {chunk_number}: {str(e)}", flush=True)
                break
        
        return total_files_found
    
    def _reset_batch_state(self):
        """Reset batch processing state."""
        self.current_batch = []
        self.current_batch_count = 0  # Reset vector count instead of size
        self.processed_count = 0
        self.failed_count = 0
        self.processed_keys = []
    
    def _parse_s3_wildcard_uri(self, s3_uri: str) -> tuple:
        """Parse S3 URI with wildcard to get bucket and prefix."""
        if s3_uri.endswith('/*'):
            s3_prefix = s3_uri[:-1]  # Remove the '*' but keep the '/'
        elif s3_uri.endswith('*'):
            s3_prefix = s3_uri[:-1]  # Remove the '*'
        else:
            s3_prefix = s3_uri
        
        if not s3_prefix.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        
        path_part = s3_prefix[5:]  # Remove 's3://'
        if '/' not in path_part:
            return path_part, ''
        
        bucket, prefix = path_part.split('/', 1)
        return bucket, prefix
    
    def _list_s3_objects(self, bucket: str, prefix: str, bucket_owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """List S3 objects matching supported file types with pagination support."""
        list_params = {
            'Bucket': bucket,
            'Prefix': prefix,
            'MaxKeys': 1000  # Explicitly set to maximum for efficiency
        }
        if bucket_owner:
            list_params['ExpectedBucketOwner'] = bucket_owner
        
        all_filtered_objects = []
        continuation_token = None
        
        while True:
            # Add continuation token if we have one
            if continuation_token:
                list_params['ContinuationToken'] = continuation_token
            
            try:
                response = self.s3_client.list_objects_v2(**list_params)
                objects = response.get('Contents', [])
                
                # Filter objects by supported extensions for this batch
                filtered_batch = self._filter_objects_by_extension(objects, bucket, bucket_owner)
                all_filtered_objects.extend(filtered_batch)
                
                # Check if there are more objects to retrieve
                if response.get('IsTruncated', False):
                    continuation_token = response.get('NextContinuationToken')
                    print(f"Retrieved {len(all_filtered_objects)} files so far, continuing...", flush=True)
                else:
                    break
                    
            except Exception as e:
                print(f"Error listing S3 objects: {str(e)}", flush=True)
                break
        
        return all_filtered_objects
    
    def _filter_objects_by_extension(self, objects: List[Dict[str, Any]], bucket: str, bucket_owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter objects by supported file extensions."""
        all_extensions = self.config.text_extensions + self.config.image_extensions
        
        filtered_objects = []
        for obj in objects:
            key = obj['Key']
            
            # Skip directories
            if key.endswith('/'):
                continue
            
            # Check file extension
            extension = key.lower().split('.')[-1] if '.' in key else ''
            if extension in all_extensions:
                # Determine content type
                content_type = (ContentType.TEXT if extension in self.config.text_extensions 
                              else ContentType.IMAGE)
                
                filtered_objects.append({
                    'key': key,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'bucket': bucket,
                    'bucket_owner': bucket_owner,
                    'content_type': content_type,
                    's3_uri': f"s3://{bucket}/{key}"
                })
        
        return filtered_objects
    
    def _process_objects_parallel(self, s3_objects: List[Dict[str, Any]], model_id: str,
                                vector_bucket: str, index_name: str, metadata_template: Dict[str, Any],
                                use_object_key_name: bool, wildcard_pattern: str, dimensions: Optional[int] = None):
        """Process S3 objects in parallel."""
        total_files = len(s3_objects)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_obj = {
                executor.submit(
                    self._process_single_object,
                    s3_obj,
                    model_id,
                    metadata_template,
                    use_object_key_name,
                    wildcard_pattern,
                    dimensions
                ): s3_obj for s3_obj in s3_objects
            }
            
            completed = 0
            for future in as_completed(future_to_obj):
                try:
                    vector_item = future.result()
                    if vector_item:
                        self._add_to_batch(vector_item, vector_bucket, index_name)
                    
                    # Update completed count
                    completed += 1
                    
                except Exception as e:
                    print(f"Task failed: {str(e)}", flush=True)
                    self.failed_count += 1
                    completed += 1
    
    def _process_single_object(self, s3_obj: Dict[str, Any], model_id: str,
                             metadata_template: Dict[str, Any], use_object_key_name: bool,
                             wildcard_pattern: str, dimensions: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Process a single S3 object."""
        try:
            s3_uri = s3_obj['s3_uri']
            content_type = s3_obj['content_type']
            
            # Read file content
            if content_type == ContentType.TEXT:
                bucket, key = s3_uri[5:].split('/', 1)
                content = self._read_s3_text_file(bucket, key, s3_obj.get('bucket_owner'))
                embedding = self.bedrock_service.embed_text(model_id, content, dimensions=dimensions)
                raw_text = content
            else:  # IMAGE
                bucket, key = s3_uri[5:].split('/', 1)
                image_data = self._read_s3_image_file(bucket, key, s3_obj.get('bucket_owner'))
                embedding = self.bedrock_service.embed_image(model_id, image_data, dimensions=dimensions)
                raw_text = f"Image file: {s3_uri}"
            
            # Build metadata based on content type
            if content_type == ContentType.TEXT:
                # Text files: content + file location
                metadata = {
                    "S3VECTORS-EMBED-SRC-CONTENT": raw_text,
                    "S3VECTORS-EMBED-SRC-LOCATION": s3_uri
                }
            else:
                # Image files: only file location (no content since images can't be stored as text)
                metadata = {
                    "S3VECTORS-EMBED-SRC-LOCATION": s3_uri
                }
            
            # Add custom metadata
            if metadata_template:
                metadata.update(metadata_template)
            
            # Generate vector ID
            if use_object_key_name:
                vector_id = s3_obj['key']
            else:
                vector_id = str(uuid.uuid4())
            
            return {
                'key': vector_id,
                'data': {'float32': [float(val) for val in embedding]},
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Failed to process {s3_obj['key']}: {str(e)}", flush=True)
            return None
    
    def _process_single_local_file(self, file_path: str, model_id: str, vector_bucket: str, 
                                 index_name: str, metadata_template: Dict[str, Any],
                                 use_object_key_name: bool, dimensions: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Process a single local file and return vector data."""
        try:
            from pathlib import Path
            
            # Determine content type from extension
            path_obj = Path(file_path)
            extension = path_obj.suffix.lower()[1:]  # Remove the dot
            
            if extension in self.config.text_extensions:
                content_type = ContentType.TEXT
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                embedding = self.bedrock_service.embed_text(model_id, content, dimensions=dimensions)
                raw_text = content
            elif extension in self.config.image_extensions:
                content_type = ContentType.IMAGE
                with open(file_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                embedding = self.bedrock_service.embed_image(model_id, image_data, dimensions=dimensions)
                raw_text = f"Image file: {file_path}"
            else:
                print(f"Unsupported file type: {extension} for file {file_path}", flush=True)
                return None
            
            # Build metadata based on content type
            if content_type == ContentType.TEXT:
                # Text files: content + file location
                metadata = {
                    "S3VECTORS-EMBED-SRC-CONTENT": raw_text,
                    "S3VECTORS-EMBED-SRC-LOCATION": f"file://{file_path}"
                }
            else:
                # Image files: only file location (no content since images can't be stored as text)
                metadata = {
                    "S3VECTORS-EMBED-SRC-LOCATION": f"file://{file_path}"
                }
            
            # Add custom metadata
            if metadata_template:
                metadata.update(metadata_template)
            
            # Generate vector ID
            if use_object_key_name:
                vector_id = path_obj.name  # Use filename as vector ID
            else:
                vector_id = str(uuid.uuid4())
            
            # Return vector data (don't add to batch here)
            return {
                'key': vector_id,
                'data': {'float32': [float(val) for val in embedding]},
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Failed to process local file {file_path}: {str(e)}", flush=True)
            return None
    
    def _read_s3_text_file(self, bucket: str, key: str, bucket_owner: Optional[str] = None) -> str:
        """Read text content from S3 file."""
        get_params = {'Bucket': bucket, 'Key': key}
        if bucket_owner:
            get_params['ExpectedBucketOwner'] = bucket_owner
        
        response = self.s3_client.get_object(**get_params)
        return response['Body'].read().decode('utf-8')
    
    def _read_s3_image_file(self, bucket: str, key: str, bucket_owner: Optional[str] = None) -> str:
        """Read image content from S3 file and return as base64."""
        get_params = {'Bucket': bucket, 'Key': key}
        if bucket_owner:
            get_params['ExpectedBucketOwner'] = bucket_owner
        
        response = self.s3_client.get_object(**get_params)
        image_bytes = response['Body'].read()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _add_to_batch(self, vector_item: Dict[str, Any], vector_bucket: str, index_name: str):
        """Thread-safe method to add items to batch."""
        with self._batch_lock:
            # Check if we need to flush before adding
            if self.current_batch_count >= self.config.max_vectors_per_batch:
                self._flush_batch(vector_bucket, index_name)
            
            # Atomically add item and increment count
            self.current_batch.append(vector_item)
            self.current_batch_count += 1
    
    def _flush_batch(self, vector_bucket: str, index_name: str) -> bool:
        """Flush current batch - called within lock context."""
        if not self.current_batch:
            return True
        
        try:
            # Prepare vectors for batch API call
            vectors = []
            
            for item in self.current_batch:
                vectors.append({
                    'key': item['key'],
                    'data': item['data'],
                    'metadata': item['metadata']
                })
            
            # Use S3VectorService batch method - ONE API CALL for all vectors
            result_keys = self.s3vector_service.put_vectors_batch(
                bucket_name=vector_bucket,
                index_name=index_name,
                vectors=vectors
            )
            
            # Track successful keys
            self.processed_keys.extend(result_keys)
            
            # Update counters
            self.processed_count += len(self.current_batch)
            
            print(f"Batch stored successfully. Total processed: {self.processed_count}", flush=True)
            return True
            
        except Exception as e:
            print(f"Failed to store batch: {str(e)}", flush=True)
            self.failed_count += len(self.current_batch)
            return False
        finally:
            # Always reset batch state
            self.current_batch = []
            self.current_batch_count = 0
