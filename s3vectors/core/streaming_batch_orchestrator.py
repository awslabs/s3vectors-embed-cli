"""Streaming batch orchestrator for efficient processing of large file sets."""

import os
import glob
import threading
from typing import List, Dict, Any, Generator, Tuple
from pathlib import Path
from dataclasses import dataclass

from s3vectors.core.unified_processor import UnifiedProcessor, ProcessingInput
from s3vectors.utils.models import SupportedModel


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    processed_count: int
    failed_count: int
    processed_keys: List[str]
    errors: List[str] = None


class StreamingBatchOrchestrator:
    """Efficient streaming batch processor that handles files in chunks without loading all paths into memory."""
    
    def __init__(self, unified_processor: UnifiedProcessor, max_workers: int = 4, batch_size: int = 500):
        self.unified_processor = unified_processor
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.batch_lock = threading.Lock()
        
    def process_streaming_batch(self, file_pattern: str, content_type: str, 
                               vector_bucket_name: str, index_name: str, model: SupportedModel,
                               metadata: Dict[str, Any], async_output_s3_uri: str = None,
                               src_bucket_owner: str = None, user_bedrock_params: Dict[str, Any] = None,
                               index_dimensions: int = None) -> BatchResult:
        """Process files using streaming approach - no memory loading of all file paths."""
        
        # Detect processing strategy
        strategy = self._detect_processing_strategy(file_pattern)
        
        # Use pre-fetched index dimensions (no API call needed)
        if index_dimensions is None:
            raise ValueError("Failed to fetch index dimensions from S3 Vector index. Cannot proceed without actual index dimensions.")
        
        batch_context = {
            'model': model,
            'index_dimensions': index_dimensions,
            'user_bedrock_params': user_bedrock_params or {},
            'async_output_s3_uri': async_output_s3_uri,
            'src_bucket_owner': src_bucket_owner,
            'batch_size': self.batch_size
        }
        
        # Process using appropriate streaming method
        if strategy == "s3_streaming":
            return self._process_s3_streaming(file_pattern, content_type, vector_bucket_name, 
                                            index_name, metadata, batch_context)
        else:  # local_streaming
            return self._process_local_streaming(file_pattern, content_type, vector_bucket_name, 
                                               index_name, metadata, batch_context)
    
    def _detect_processing_strategy(self, file_pattern: str) -> str:
        """Check whether it is for S3 or Local files."""
        if file_pattern.startswith('s3://') and '*' in file_pattern:
            return "s3_streaming"
        elif '*' in file_pattern or '?' in file_pattern:
            return "local_streaming"
        else:
            raise ValueError(f"Invalid pattern for batch processing: {file_pattern}. Pattern must contain wildcards (* or ?) for batch processing.")
    
    def _process_s3_streaming(self, s3_pattern: str, content_type: str,
                             vector_bucket_name: str, index_name: str, 
                             metadata: Dict[str, Any], batch_context: Dict[str, Any]) -> BatchResult:
        """Stream S3 objects directly without pre-loading all paths."""
        
        # Parse S3 pattern
        s3_path = s3_pattern[5:]  # Remove 's3://'
        if s3_path.endswith('/*'):
            s3_path = s3_path[:-1]  # Remove only '*', keep '/'
        elif s3_path.endswith('*'):
            s3_path = s3_path[:-1]  # Remove '*'
            # Add trailing slash if not present to ensure directory-level filtering
            if not s3_path.endswith('/'):
                s3_path += '/'
        
        bucket, prefix = s3_path.split('/', 1) if '/' in s3_path else (s3_path, '')
        
        # Stream S3 objects in chunks
        s3_client = self.unified_processor.session.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
        
        total_processed = 0
        total_failed = 0
        all_processed_keys = []
        all_errors = []
        
        chunk_number = 0
        for chunk_files in self._stream_s3_chunks(paginator, bucket, prefix):
            if not chunk_files:
                continue
                
            chunk_number += 1
            print(f"Processing chunk {chunk_number}...")
            
            # Process chunk
            processed, failed, keys, errors = self._process_chunk(
                chunk_files, content_type, vector_bucket_name, index_name, metadata, batch_context
            )
            
            total_processed += processed
            total_failed += failed
            all_processed_keys.extend(keys)
            if errors:
                all_errors.extend(errors)
            
            print(f"Completed chunk {chunk_number}: {processed} processed, {failed} failed")
        
        return BatchResult(
            processed_count=total_processed,
            failed_count=total_failed,
            processed_keys=all_processed_keys,
            errors=all_errors if all_errors else None
        )
    
    def _stream_s3_chunks(self, paginator, bucket: str, prefix: str) -> Generator[List[str], None, None]:
        """Generator that yields chunks of S3 file paths."""
        chunk = []
        text_extensions = {'txt', 'md', 'json', 'csv', 'log'}
        image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
        video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
        audio_extensions = {'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg', 'wma'}
        all_extensions = text_extensions | image_extensions | video_extensions | audio_extensions
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                # Skip directories
                if key.endswith('/'):
                    continue
                    
                # Check file extension
                extension = key.lower().split('.')[-1] if '.' in key else ''
                if extension in all_extensions:
                    chunk.append(f"s3://{bucket}/{key}")
                    
                    # Yield chunk when it reaches target size
                    if len(chunk) >= self.batch_size:
                        yield chunk
                        chunk = []
        
        # Yield remaining files
        if chunk:
            yield chunk
    
    def _process_local_streaming(self, local_pattern: str, content_type: str,
                                vector_bucket_name: str, index_name: str, 
                                metadata: Dict[str, Any], batch_context: Dict[str, Any]) -> BatchResult:
        """Stream local files directly without pre-loading all paths."""
        
        total_processed = 0
        total_failed = 0
        all_processed_keys = []
        all_errors = []
        
        print(f"Starting streaming processing of {local_pattern}")
        
        chunk_number = 0
        for chunk_files in self._stream_local_chunks(local_pattern):
            if not chunk_files:
                continue
                
            chunk_number += 1
            print(f"Processing chunk {chunk_number}...")
            
            # Process chunk
            processed, failed, keys, errors = self._process_chunk(
                chunk_files, content_type, vector_bucket_name, index_name, metadata, batch_context
            )
            
            total_processed += processed
            total_failed += failed
            all_processed_keys.extend(keys)
            if errors:
                all_errors.extend(errors)
            
            print(f"Completed chunk {chunk_number}: {processed} processed, {failed} failed")
        
        return BatchResult(
            processed_count=total_processed,
            failed_count=total_failed,
            processed_keys=all_processed_keys,
            errors=all_errors if all_errors else None
        )
    
    def _stream_local_chunks(self, pattern: str) -> Generator[List[str], None, None]:
        """Generator that yields chunks of local file paths using iterator."""
        chunk = []
        text_extensions = {'txt', 'md', 'json', 'csv', 'log'}
        image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
        video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
        audio_extensions = {'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg', 'wma'}
        all_extensions = text_extensions | image_extensions | video_extensions | audio_extensions
        
        # Use iglob for memory-efficient iteration
        for file_path in glob.iglob(pattern, recursive=True):
            if os.path.isfile(file_path):
                path_obj = Path(file_path)
                extension = path_obj.suffix.lower()[1:]  # Remove the dot
                
                if extension in all_extensions:
                    chunk.append(file_path)
                    
                    # Yield chunk when it reaches target size
                    if len(chunk) >= self.batch_size:
                        yield chunk
                        chunk = []
        
        # Yield remaining files
        if chunk:
            yield chunk
    
    def _process_chunk(self, chunk_files: List[str], content_type: str,
                      vector_bucket_name: str, index_name: str, 
                      metadata: Dict[str, Any], batch_context: Dict[str, Any]) -> Tuple[int, int, List[str], List[str]]:
        """Process a chunk of files using UnifiedProcessor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        processed_count = 0
        failed_count = 0
        processed_keys = []
        errors = []
        
        # For video/audio, process sequentially due to async nature
        # For text/image, use parallel processing
        if content_type in ["video", "audio"]:
            return self._process_chunk_async(chunk_files, content_type, vector_bucket_name, 
                                           index_name, metadata, batch_context)
        else:
            return self._process_chunk_sync(chunk_files, content_type, vector_bucket_name,
                                          index_name, metadata, batch_context)
    
    def _process_chunk_sync(self, chunk_files: List[str], content_type: str,
                           vector_bucket_name: str, index_name: str, 
                           metadata: Dict[str, Any], batch_context: Dict[str, Any]) -> Tuple[int, int, List[str], List[str]]:
        """Process text/image files in parallel (existing logic)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        processed_count = 0
        failed_count = 0
        processed_keys = []
        errors = []
        vectors_to_store = []
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {}
            for file_path in chunk_files:
                # Create appropriate ProcessingInput - use file_path key for both S3 and local files
                processing_input = ProcessingInput(content_type, {"file_path": file_path}, file_path, metadata)
                
                future = executor.submit(
                    self.unified_processor.process,
                    batch_context['model'],
                    processing_input,
                    batch_context['user_bedrock_params'],
                    batch_context['async_output_s3_uri'],
                    batch_context['src_bucket_owner'],
                    vector_bucket_name,
                    index_name,
                    batch_context['index_dimensions']  # Pass precomputed dimensions
                )
                future_to_file[future] = file_path
            
            # Collect results with interim status updates
            completed_files = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result and result.vectors:
                        vectors_to_store.extend(result.vectors)
                        processed_keys.extend([v.get("key", "") for v in result.vectors])
                        processed_count += len(result.vectors)
                    else:
                        failed_count += 1
                        errors.append(f"No vectors generated for {file_path}")
                except Exception as e:
                    failed_count += 1
                    errors.append(f"Error processing {file_path}: {str(e)}")
                
                completed_files += 1
                # Print interim status every 100 files
                if completed_files % 100 == 0:
                    print(f"Progress: {completed_files}/{len(chunk_files)} files processed ({processed_count} successful, {failed_count} failed)")
        
        # Store vectors in batch
        if vectors_to_store:
            try:
                self.unified_processor.s3vector_service.put_vectors_batch(
                    vector_bucket_name, index_name, vectors_to_store
                )
                print(f"STORED BATCH: {len(vectors_to_store)} vectors")
            except Exception as e:
                errors.append(f"Batch storage failed: {str(e)}")
                # Mark all as failed if storage fails
                failed_count += processed_count
                processed_count = 0
                processed_keys = []
        
        return processed_count, failed_count, processed_keys, errors
    
    def _process_chunk_async(self, chunk_files: List[str], content_type: str,
                            vector_bucket_name: str, index_name: str, 
                            metadata: Dict[str, Any], batch_context: Dict[str, Any]) -> Tuple[int, int, List[str], List[str]]:
        """Process video/audio files with parallel async processing."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        processed_files_count = 0  # Count files, not vectors
        failed_files_count = 0     # Count files, not vectors
        processed_keys = []
        errors = []
        
        total_files = len(chunk_files)
        print(f"Processing {total_files} {content_type} files with {self.max_workers} concurrent workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {}
            
            # Submit all files for async processing
            for file_path in chunk_files:
                processing_input = ProcessingInput(content_type, {"file_path": file_path}, file_path, metadata)
                
                future = executor.submit(
                    self.unified_processor.process,
                    batch_context['model'],
                    processing_input,
                    batch_context['user_bedrock_params'],
                    batch_context['async_output_s3_uri'],
                    batch_context['src_bucket_owner'],
                    vector_bucket_name,
                    index_name,
                    batch_context['index_dimensions']
                )
                future_to_file[future] = file_path
            
            # Collect results as they complete
            completed_files = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed_files += 1
                
                try:
                    result = future.result()
                    
                    if result and result.vectors:
                        # Store vectors from this file, chunking if needed
                        try:
                            file_vector_count = len(result.vectors)
                            batch_size = batch_context['batch_size']
                            
                            if file_vector_count <= batch_size:
                                # Small file - single API call
                                self.unified_processor.s3vector_service.put_vectors_batch(
                                    vector_bucket_name, index_name, result.vectors
                                )
                            else:
                                # Large file - chunk into batch_size pieces
                                for i in range(0, file_vector_count, batch_size):
                                    chunk = result.vectors[i:i + batch_size]
                                    self.unified_processor.s3vector_service.put_vectors_batch(
                                        vector_bucket_name, index_name, chunk
                                    )
                            
                            processed_files_count += 1  # Count files, not vectors
                            processed_keys.extend([v.get("key", "") for v in result.vectors])
                            
                            print(f"[{completed_files}/{total_files}] Stored {file_vector_count} vectors from {file_path}")
                            
                        except Exception as e:
                            failed_files_count += 1
                            errors.append(f"Storage failed for {file_path}: {str(e)}")
                            
                    else:
                        failed_files_count += 1
                        errors.append(f"No vectors generated for {file_path}")
                        
                except Exception as e:
                    failed_files_count += 1
                    errors.append(f"Error processing {file_path}: {str(e)}")
        
        return processed_files_count, failed_files_count, processed_keys, errors
