"""Functions for processing video query input for embedding."""

import os
import json
import uuid
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path

import click
from rich.progress import Progress, SpinnerColumn, TextColumn


def process_video_query(query_input: str, bedrock_service, s3_client, model_id: str,
                      dimensions: int, bucket_owner: Optional[str], s3_output_bucket: str, console) -> List[float]:
    """
    Process video query input, generate embedding, and return the embedding vector.
    
    Args:
        query_input: Text query or file path (local file or S3 URI)
        bedrock_service: Initialized BedrockService
        s3_client: Initialized S3 client
        model_id: Bedrock model ID (must be twelvelabs.marengo-embed-2-7-v1:0)
        dimensions: Embedding dimensions
        bucket_owner: Source bucket owner AWS account ID
        s3_output_bucket: S3 bucket for async output
        console: Rich console for output
        
    Returns:
        List[float]: The query embedding vector
    """
    # Validate model ID
    if not model_id.startswith('twelvelabs.marengo-embed'):
        raise click.ClickException(
            f"Unsupported model for video query: {model_id}. "
            f"Use twelvelabs.marengo-embed-2-7-v1:0 for video queries."
        )
    
    # Create progress context
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Add task for query processing
        task = progress.add_task("Processing query...", total=None)
        
        try:
            # Check if the query is a file path or direct text
            if os.path.exists(query_input) or query_input.startswith('s3://'):
                # It's a file path, but we don't support video file queries yet
                # For now, we'll treat it as a text query
                console.print("[yellow]Note: Video file queries are not yet supported. Treating as text query.[/yellow]")
                query_text = query_input
            else:
                # It's direct text
                query_text = query_input
            
            progress.update(task, description="Generating text embedding for video query...")
            
            # Set up S3 output location for async job
            s3_output_uri = f"s3://{s3_output_bucket}/text-embeddings"
            
            # Generate a unique ID for this query
            query_id = str(uuid.uuid4())
            
            # Prepare the model input for async invocation
            model_input = {
                "inputType": "text",
                "inputText": query_text
            }
            
            # Prepare output data config
            output_data_config = {
                "s3OutputDataConfig": {
                    "s3Uri": s3_output_uri,
                    "bucketOwner": bucket_owner  # Include bucketOwner in the output config
                }
            }
            
            console.print(f"[yellow]Starting async text embedding job for video query...[/yellow]")
            
            # Start the async job
            response = bedrock_service.bedrock_runtime.start_async_invoke(
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
            console.print(f"[yellow]Invocation ARN: {invocation_arn}[/yellow]")
            
            # Poll for job completion
            max_wait_time = 300  # 5 minutes
            poll_interval = 5  # 5 seconds
            
            import time
            start_time = time.time()
            while time.time() - start_time < max_wait_time:
                try:
                    # Check the status of the async job
                    status_response = bedrock_service.bedrock_runtime.get_async_invoke(
                        invocationArn=invocation_arn
                    )
                    
                    status = status_response.get('status')
                    
                    if time.time() - start_time > 10:  # Show status updates after 10 seconds
                        elapsed = int(time.time() - start_time)
                        if elapsed % 10 == 0:  # Update every 10 seconds
                            console.print(f"[yellow]Job status: {status} (elapsed: {elapsed} seconds)[/yellow]")
                    
                    if status == 'Completed':
                        console.print(f"[green]Job completed successfully![/green]")
                        
                        # Job completed, now we need to get the results from S3
                        output_s3_uri = f"{s3_output_uri.rstrip('/')}/{invocation_id}/output.json"
                        console.print(f"[yellow]Retrieving embedding from {output_s3_uri}[/yellow]")
                        
                        # Parse S3 URI to get bucket and key
                        output_bucket = output_s3_uri.split('/')[2]
                        output_key = '/'.join(output_s3_uri.split('/')[3:])
                        
                        # Get the results from S3
                        try:
                            response = s3_client.get_object(Bucket=output_bucket, Key=output_key)
                            result_data = json.loads(response['Body'].read().decode('utf-8'))
                            
                            # Extract embedding from response
                            # Check if the response has the expected format
                            if 'embedding' in result_data:
                                # Direct embedding field
                                embedding = result_data['embedding']
                            elif 'data' in result_data and isinstance(result_data['data'], list) and len(result_data['data']) > 0:
                                # Format with data array (similar to video embeddings)
                                first_segment = result_data['data'][0]
                                embedding = first_segment.get('embedding', [])
                            else:
                                # Dump the response for debugging
                                console.print(f"[yellow]Response format: {json.dumps(result_data, indent=2)}[/yellow]")
                                raise Exception(f"Unexpected response format: {result_data}")
                            
                            if not embedding:
                                raise Exception("No embedding found in the response")
                                
                            console.print(f"[green]Successfully retrieved embedding with {len(embedding)} dimensions[/green]")
                            
                            return embedding
                            
                        except Exception as e:
                            raise Exception(f"Failed to retrieve results from S3: {str(e)}")
                        
                    elif status == 'Failed':
                        failure_reason = status_response.get('failureMessage', 'Unknown reason')
                        raise Exception(f"Async job failed: {failure_reason}")
                        
                    # Wait before polling again
                    time.sleep(poll_interval)
                    
                except Exception as e:
                    raise Exception(f"Failed to check async job status: {str(e)}")
                    
            # If we get here, we've timed out
            raise Exception(f"Async job timed out after {max_wait_time} seconds")
            
        except Exception as e:
            raise click.ClickException(f"Failed to process video query: {str(e)}")