import os
import json
import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from s3vectors.core.services import BedrockService, S3VectorService
from s3vectors.core.unified_processor import UnifiedProcessor
from s3vectors.utils.config import get_region, get_current_account_id
from s3vectors.utils.models import get_model_info, validate_user_parameters, prepare_processing_input, determine_content_type
from s3vectors.core.streaming_batch_orchestrator import StreamingBatchOrchestrator


def _create_progress_context(console):
    """Create progress context for operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    )


def _validate_inputs(text_value, text, image, video, audio, model):
    """Validate input parameters."""
    inputs_provided = sum(bool(x) for x in [text_value, text, image, video, audio])
    
    if inputs_provided == 0:
        raise click.ClickException("At least one input must be provided: --text-value, --text, --image, --video, or --audio")
    
    # Special case: Allow multimodal input for supported models
    is_multimodal_input = (model.supports_multimodal_input() and 
                          text_value and image and not text and not video and not audio)
    
    if inputs_provided > 1 and not is_multimodal_input:
        raise click.ClickException("Only one input type can be specified at a time, except for multimodal input with supported models (--text-value + --image)")
    
    return is_multimodal_input


@click.command()
@click.option('--vector-bucket-name', required=True, help='S3 vector bucket name')
@click.option('--index-name', required=True, help='Vector index name')
@click.option('--model-id', required=True, help='Bedrock embedding model ID')
@click.option('--text-value', help='Direct text input to embed')
@click.option('--text', help='Text file path (local or S3 URI)')
@click.option('--image', help='Image file path (local or S3 URI)')
@click.option('--video', help='Video file path (local or S3 URI)')
@click.option('--audio', help='Audio file path (local or S3 URI)')
@click.option('--async-output-s3-uri', help='S3 URI for async processing results')
@click.option('--bedrock-inference-params', help='Bedrock inference parameters (JSON)')
@click.option('--metadata', help='Additional metadata (JSON)')
@click.option('--src-bucket-owner', help='AWS account ID for cross-account S3 access')
@click.option('--max-workers', type=int, default=4, help='Maximum parallel workers for batch processing')
@click.option('--batch-size', type=click.IntRange(1, 500), default=500, help='Number of vectors per S3 Vector put_vectors call (1-500, default: 500)')
@click.option('--output', type=click.Choice(['json', 'table']), default='json', help='Output format')
@click.option('--region', help='AWS region')
@click.pass_context
def embed_put(ctx, vector_bucket_name, index_name, model_id, text_value, text, image,
                     video, audio, async_output_s3_uri, bedrock_inference_params,
                     metadata, src_bucket_owner, max_workers, batch_size, output, region):
    """Unified embed and store vectors command."""
    
    console = ctx.obj['console']
    session = ctx.obj['aws_session']
    debug = ctx.obj.get('debug', False)
    region = get_region(session, region)
    
    # Auto-assign account ID if not provided
    if not src_bucket_owner:
        src_bucket_owner = get_current_account_id(session)
    
    # Load model properties once at start
    model = get_model_info(model_id)
    if not model:
        raise click.ClickException(f"Unsupported model: {model_id}")
    
    # Parse parameters
    user_bedrock_params = {}
    if bedrock_inference_params:
        try:
            user_bedrock_params = json.loads(bedrock_inference_params)
        except json.JSONDecodeError:
            raise click.ClickException("Invalid JSON in --bedrock-inference-params parameter")
    
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise click.ClickException("Invalid JSON in --metadata parameter")
    
    # Early validation of user parameters before any processing
    if user_bedrock_params:
        try:
            content_type = determine_content_type(text_value, text, image, video, audio)
            system_keys = model.get_system_keys(content_type)
            system_payload = {key: None for key in system_keys}  # Dummy values for validation
            
            # Validate using utility function
            validate_user_parameters(system_payload, user_bedrock_params)
        except ValueError as e:
            raise click.ClickException(str(e))
    
    # Check async model requirements
    if model.is_async() and not async_output_s3_uri:
        raise click.ClickException(
            f"Async models like {model.model_id} require --async-output-s3-uri parameter."
        )
    
    # Validate inputs
    is_multimodal = _validate_inputs(text_value, text, image, video, audio, model)
    
    try:
        # Initialize services
        bedrock_service = BedrockService(session, region, debug=debug, console=console)
        s3vector_service = S3VectorService(session, region, debug=debug, console=console)
        
        # Create unified processor
        processor = UnifiedProcessor(bedrock_service, s3vector_service, session)
        
        # Fetch index dimensions once at the top level
        try:
            index_info = s3vector_service.get_index(vector_bucket_name, index_name)
            index_dimensions = index_info.get("index", {}).get("dimension")
            if not index_dimensions:
                raise click.ClickException(f"Could not determine dimensions for index {index_name}")
        except Exception as e:
            raise click.ClickException(f"Failed to get index information: {str(e)}")
        
        # Prepare processing input
        processing_input = prepare_processing_input(
            text_value, text, image, video, audio, is_multimodal, metadata_dict
        )
        
        # Check for wildcard patterns (streaming batch processing)
        if processing_input.content_type in ["text", "image", "video", "audio"] and "file_path" in processing_input.data:
            file_path = processing_input.data["file_path"]
            if '*' in file_path or '?' in file_path:
                return _process_streaming_batch(
                    file_path, processing_input.content_type, vector_bucket_name, index_name,
                    model, metadata_dict, user_bedrock_params, async_output_s3_uri, 
                    src_bucket_owner, processor, console, output, max_workers, batch_size, index_dimensions
                )
        
        with _create_progress_context(console) as progress:
            # Process with unified pipeline
            progress.add_task("Processing input...", total=None)
            result = processor.process(
                model, processing_input, user_bedrock_params, 
                async_output_s3_uri, src_bucket_owner, vector_bucket_name, index_name, index_dimensions
            )
            
            # Store vectors with batch_size handling
            progress.add_task(f"Storing {len(result.vectors)} vector(s)...", total=None)
            
            # Handle batch_size for single file processing too
            vector_count = len(result.vectors)
            
            if vector_count <= batch_size:
                stored_keys = processor.store_vectors(result.vectors, vector_bucket_name, index_name)
            else:
                stored_keys = []
                for i in range(0, vector_count, batch_size):
                    chunk = result.vectors[i:i + batch_size]
                    chunk_keys = processor.store_vectors(chunk, vector_bucket_name, index_name)
                    stored_keys.extend(chunk_keys)
        
        # Prepare output
        if result.result_type == "multiclip":
            output_result = {
                'type': 'multiclip',
                'bucket': vector_bucket_name,
                'index': index_name,
                'model': model.model_id,
                'contentType': processing_input.content_type,
                'totalVectors': len(stored_keys),
                'keys': stored_keys
            }
            if result.job_id:
                output_result['jobId'] = result.job_id
        else:
            output_result = {
                'key': stored_keys[0],
                'bucket': vector_bucket_name,
                'index': index_name,
                'model': model.model_id,
                'contentType': processing_input.content_type,
                'embeddingDimensions': index_dimensions,
                'metadata': result.vectors[0]['metadata']
            }
        
        console.print_json(data=output_result)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    embed_put()


def _process_streaming_batch(file_path, content_type, vector_bucket_name, index_name,
                           model, metadata_dict, user_bedrock_params, async_output_s3_uri,
                           src_bucket_owner, processor, console, output, max_workers, batch_size, index_dimensions):
    """Process wildcard pattern using efficient streaming batch orchestrator."""

    
    try:
        # Create streaming batch orchestrator
        streaming_orchestrator = StreamingBatchOrchestrator(processor, max_workers, batch_size)
        
        console.print(f"Starting streaming batch processing: {file_path}")
        
        # Process using streaming approach (no pre-loading of file paths)
        batch_result = streaming_orchestrator.process_streaming_batch(
            file_path, content_type, vector_bucket_name, index_name, model,
            metadata_dict, async_output_s3_uri, src_bucket_owner, user_bedrock_params, index_dimensions
        )
        
        # Display results
        if output == "table":
            _display_batch_table(batch_result, console)
        else:
            result_dict = {
                "type": "streaming_batch",
                "bucket": vector_bucket_name,
                "index": index_name,
                "model": model.model_id,
                "contentType": content_type,
                "totalFiles": batch_result.processed_count + batch_result.failed_count,
                "processedFiles": batch_result.processed_count,
                "failedFiles": batch_result.failed_count,
                "totalVectors": len(batch_result.processed_keys),  # Count actual vectors
                "vectorKeys": batch_result.processed_keys[:10] if batch_result.processed_keys else []  # Show first 10
            }
            
            if batch_result.errors:
                result_dict["errors"] = batch_result.errors[:5]  # Show first 5 errors
            
            console.print_json(data=result_dict)
            
            # Print display limit messages after JSON output
            if len(batch_result.processed_keys) > 10:
                console.print(f"[dim]Note: Showing first 10 of {len(batch_result.processed_keys)} vector keys[/dim]")
            
            if batch_result.errors and len(batch_result.errors) > 5:
                console.print(f"[dim]Note: Showing first 5 of {len(batch_result.errors)} errors[/dim]")
        
        return result_dict if output == "json" else None
        
    except Exception as e:
        console.print(f"[red]Streaming batch processing failed: {str(e)}[/red]")
        raise click.ClickException(f"Streaming batch processing failed: {str(e)}")
        
        # Prepare output in unified format
        result = {
            "type": "batch",
            "bucket": vector_bucket_name,
            "index": index_name,
            "model": model.model_id,
            "contentType": content_type,
            "totalFiles": batch_result.total_files,
            "processedFiles": batch_result.processed_count,  # Note: processed_count not processed_files
            "failedFiles": batch_result.failed_count,
            "totalVectors": batch_result.processed_count,    # Each file = 1 vector
            "vectorKeys": batch_result.processed_keys
        }
        
        if output == "table":
            _display_batch_table(result, console)
        else:
            console.print_json(data=result)
        
    except Exception as e:
        raise click.ClickException(f"Failed to process batch {file_path}: {str(e)}")


def _display_batch_table(result, console):
    """Display batch results in table format."""
    from rich.table import Table
    
    table = Table(title="Batch Processing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Files", str(result["totalFiles"]))
    table.add_row("Processed Files", str(result["processedFiles"]))
    table.add_row("Failed Files", str(result["failedFiles"]))
    table.add_row("Total Vectors", str(result["totalVectors"]))
    table.add_row("Model", result["model"])
    table.add_row("Content Type", result["contentType"])
    
    console.print(table)
