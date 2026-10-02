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
                               index_dimensions: int = None, filename_as_key: bool = False, key_prefix: str = None) -> BatchResult:
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
            'batch_size': self.batch_size,
            'filename_as_key': filename_as_key,
            'key_prefix': key_prefix
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
    
    # Hard S3 Vectors service limit: at most 500 vectors per PutVectors call.
    MAX_VECTORS_PER_PUT = 500

    # Modalities whose files each produce a self-contained set of vectors
    # (one per clip/segment) that are stored in their own put_vectors call.
    PER_FILE_STORAGE_MODALITIES = {"video", "audio"}

    def _process_chunk(self, chunk_files: List[str], content_type: str,
                      vector_bucket_name: str, index_name: str, 
                      metadata: Dict[str, Any], batch_context: Dict[str, Any]) -> Tuple[int, int, List[str], List[str]]:
        """Process a chunk of files in parallel using UnifiedProcessor.

        Embedding is one path for all models and modalities: UnifiedProcessor
        owns the sync/async split. Storage strategy depends on the modality:

        - text/image: each file yields one vector, so vectors from all files
          in the chunk are pooled and written in put_vectors calls of up to
          min(batch_size, 500) vectors. A file is never split across calls.
        - video/audio: each file yields one vector per clip/segment, all of
          which are written together in that file's own put_vectors call as
          soon as the file completes. Only a file with more clips than
          min(batch_size, 500) is sliced, to respect the service cap.

        Storage failures are recorded against the files involved and
        processing continues; there is no retry.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        processed_files_count = 0  # Count files, not vectors
        failed_files_count = 0     # Count files, not vectors
        processed_keys = []
        errors = []
        
        total_files = len(chunk_files)
        store_size = min(batch_context['batch_size'], self.MAX_VECTORS_PER_PUT)
        per_file_storage = content_type in self.PER_FILE_STORAGE_MODALITIES
        
        # Pooled buffer for text/image: vectors and the files they came from.
        pooled_vectors: List[Dict[str, Any]] = []
        pooled_files: List[str] = []
        
        def flush_pooled():
            """Write the pooled buffer in one put_vectors call and reset it."""
            nonlocal processed_files_count, failed_files_count
            if not pooled_vectors:
                return
            try:
                self.unified_processor.s3vector_service.put_vectors_batch(
                    vector_bucket_name, index_name, list(pooled_vectors)
                )
                processed_files_count += len(pooled_files)
                processed_keys.extend([v.get("key", "") for v in pooled_vectors])
                print(f"STORED BATCH: {len(pooled_vectors)} vectors from {len(pooled_files)} files")
            except Exception as e:
                failed_files_count += len(pooled_files)
                errors.append(
                    f"Storage failed for {len(pooled_files)} files "
                    f"({', '.join(pooled_files)}): {str(e)}"
                )
            pooled_vectors.clear()
            pooled_files.clear()
        
        print(f"Processing {total_files} {content_type} files with {self.max_workers} concurrent workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {}
            
            # Submit all files for processing
            for file_path in chunk_files:
                processing_input = ProcessingInput(
                    content_type, {"file_path": file_path}, file_path, metadata,
                    filename_as_key=batch_context['filename_as_key'],
                    key_prefix=batch_context['key_prefix']
                )
                
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
                except Exception as e:
                    failed_files_count += 1
                    errors.append(f"Error processing {file_path}: {str(e)}")
                    continue
                
                if not (result and result.vectors):
                    failed_files_count += 1
                    errors.append(f"No vectors generated for {file_path}")
                    continue
                
                file_vector_count = len(result.vectors)
                
                if per_file_storage:
                    # Video/audio: this file's clips go in their own call(s).
                    try:
                        for i in range(0, file_vector_count, store_size):
                            vector_slice = result.vectors[i:i + store_size]
                            self.unified_processor.s3vector_service.put_vectors_batch(
                                vector_bucket_name, index_name, vector_slice
                            )
                        processed_files_count += 1
                        processed_keys.extend([v.get("key", "") for v in result.vectors])
                        print(f"[{completed_files}/{total_files}] Stored {file_vector_count} vectors from {file_path}")
                    except Exception as e:
                        failed_files_count += 1
                        errors.append(f"Storage failed for {file_path}: {str(e)}")
                else:
                    # Text/image: pool across files. Flush first if this file's
                    # vectors wouldn't fit, so a file is never split across calls.
                    if pooled_vectors and len(pooled_vectors) + file_vector_count > store_size:
                        flush_pooled()
                    pooled_vectors.extend(result.vectors)
                    pooled_files.append(file_path)
                    if len(pooled_vectors) >= store_size:
                        flush_pooled()
                    
                    if completed_files % 100 == 0:
                        print(f"Progress: {completed_files}/{total_files} files processed "
                              f"({processed_files_count} stored, {failed_files_count} failed, "
                              f"{len(pooled_files)} pending store)")
        
        # Write whatever is left in the pool after the last file completes.
        flush_pooled()
        
        return processed_files_count, failed_files_count, processed_keys, errors
