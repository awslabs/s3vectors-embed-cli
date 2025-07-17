"""Functions for processing video input for embedding."""

import os
import json
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path

import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from s3vectors.core.services import BedrockService


def process_video_input(video_s3_uri: str, vector_bucket_name: str, index_name: str, model_id: str,
                      bedrock_service: BedrockService, s3vector_service, metadata_dict: Dict[str, Any],
                      s3_output_uri: str, bucket_owner: Optional[str], vector_id: Optional[str], 
                      embedding_options: List[str], console) -> Dict[str, Any]:
    """
    Process video input from S3, generate embedding, and store in S3 Vectors.
    This function will wait for the async job to complete before returning.
    
    Args:
        video_s3_uri: S3 URI of the video file
        vector_bucket_name: S3 vector bucket name
        index_name: Vector index name
        model_id: Bedrock model ID (must be twelvelabs.marengo-embed-2-7-v1:0)
        bedrock_service: Initialized BedrockService
        s3vector_service: Initialized S3VectorService
        metadata_dict: Metadata to attach to the vector
        s3_output_uri: S3 URI for async job output
        bucket_owner: Source bucket owner AWS account ID
        vector_id: Custom vector ID (optional)
        embedding_options: List of embedding options
        console: Rich console for output
        
    Returns:
        Dict with result information
    """
    # Validate model ID
    if not model_id.startswith('twelvelabs.marengo-embed'):
        raise click.ClickException(
            f"Unsupported model for video embedding: {model_id}. "
            f"Use twelvelabs.marengo-embed-2-7-v1:0 for video embedding."
        )
    
    # Create progress context
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Add task for video processing
        task = progress.add_task("Processing video...", total=None)
        
        try:
            # Generate a vector ID if not provided
            vector_key = vector_id if vector_id else str(uuid.uuid4())
            
            # Generate embedding using the async video embedding function
            # This will start the job, poll until completion, and return the embedding
            progress.update(task, description="Generating video embedding (this may take several minutes)...")
            
            # Call the embed_video_async method which will handle polling until completion
            # This now returns a dictionary with all embeddings grouped by type
            embedding_result = bedrock_service.embed_video_async(
                model_id=model_id,
                s3_video_uri=video_s3_uri,
                s3_output_uri=s3_output_uri,
                embedding_options=embedding_options,
                bucket_owner=bucket_owner,
                console=console  # Pass console for status updates
            )
            
            progress.update(task, description="Video embedding generated ✓")
            
            # Add metadata
            base_metadata = metadata_dict.copy()
            base_metadata.update({
                "S3VECTORS-EMBED-SRC-LOCATION": video_s3_uri
            })
            
            # Store vectors in S3 Vectors
            store_task = progress.add_task("Storing vectors in S3 Vectors...", total=None)
            
            # Get embeddings by type from the result
            embeddings_by_type = embedding_result.get('embeddings_by_type', {})
            
            # Check if we have any embeddings
            if not embeddings_by_type:
                raise click.ClickException("No embeddings found in the response")
            
            # Prepare to store all embeddings individually
            if console:
                console.print(f"[yellow]Storing all {embedding_result.get('total_count', 0)} embeddings as separate vectors...[/yellow]")
            
            # Track successful and failed vectors
            successful_vectors = []
            failed_vectors = []
            
            # Process each embedding type
            for embedding_type, embeddings in embeddings_by_type.items():
                # Process each embedding of this type
                for i, embedding_data in enumerate(embeddings):
                    try:
                        # Create a unique ID for this segment
                        segment_id = f"{vector_key}_{embedding_type}_{i}"
                        
                        # Get the embedding vector
                        embedding_vector = embedding_data['embedding']
                        
                        # Create metadata for this segment
                        segment_metadata = base_metadata.copy()
                        segment_metadata.update({
                            "S3VECTORS-EMBED-TYPE": embedding_type,
                            "S3VECTORS-EMBED-SEGMENT-INDEX": i,
                            "S3VECTORS-EMBED-START-SEC": embedding_data.get('startSec', 0),
                            "S3VECTORS-EMBED-END-SEC": embedding_data.get('endSec', 0),
                            "S3VECTORS-EMBED-PARENT-ID": vector_key,
                            "S3VECTORS-EMBED-TOTAL-SEGMENTS": len(embeddings),
                            "S3VECTORS-EMBED-SOURCE": "video"
                        })
                        
                        # Store this vector individually
                        if console and i == 0:  # Only log the first one of each type to avoid spam
                            console.print(f"[yellow]Storing {embedding_type} vectors...[/yellow]")
                        
                        # Use the individual put_vector method
                        s3vector_service.put_vector(
                            bucket_name=vector_bucket_name,
                            index_name=index_name,
                            vector_id=segment_id,
                            embedding=embedding_vector,
                            metadata=segment_metadata
                        )
                        
                        # Track successful vector
                        successful_vectors.append(segment_id)
                        
                    except Exception as e:
                        # Log the error and continue with the next vector
                        if console:
                            console.print(f"[red]Error storing vector {embedding_type}_{i}: {str(e)}[/red]")
                        failed_vectors.append(f"{embedding_type}_{i}")
            
            # Report results
            if console:
                if successful_vectors:
                    console.print(f"[green]Successfully stored {len(successful_vectors)} vectors[/green]")
                if failed_vectors:
                    console.print(f"[red]Failed to store {len(failed_vectors)} vectors[/red]")
                    
            # If all vectors failed, raise an exception
            if not successful_vectors and failed_vectors:
                raise click.ClickException(f"Failed to store any vectors. First error: {failed_vectors[0]}")
                
            # If some vectors failed but some succeeded, continue with what we have
            if failed_vectors and successful_vectors:
                if console:
                    console.print(f"[yellow]Warning: Some vectors failed to store, but continuing with the {len(successful_vectors)} successful ones.[/yellow]")
            
            progress.update(store_task, description="Vector stored in S3 Vectors ✓")
            
            # Return result
            return {
                'type': 'batch',
                'parentKey': vector_key,
                'vectorCount': len(successful_vectors),
                'successfulVectors': len(successful_vectors),
                'failedVectors': len(failed_vectors),
                'bucket': vector_bucket_name,
                'index': index_name,
                'model': model_id,
                'contentType': 'video',
                'embeddingTypes': list(embeddings_by_type.keys()),
                'baseMetadata': base_metadata,
                'keys': successful_vectors,  # Add the keys field for compatibility with _display_results
                'status': 'success' if successful_vectors else 'failed',
                'processed_count': len(successful_vectors) + len(failed_vectors),
                'pattern': video_s3_uri  # Add the pattern field for compatibility with _display_results
            }
            
        except Exception as e:
            raise click.ClickException(f"Failed to process video: {str(e)}")