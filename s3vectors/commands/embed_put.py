"""Embed and put vectors command with enhanced batch processing."""

import os
import json
import uuid
from typing import Optional

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from s3vectors.core.services import BedrockService, S3VectorService
from s3vectors.core.batch_processor import InputProcessor, BatchProcessor, BatchConfig
from s3vectors.utils.config import get_region
from s3vectors.utils.twelvelabs_helpers import (
    prepare_media_source, read_text_file_content, create_twelvelabs_metadata
)
from s3vectors.utils.bedrock_params import (
    validate_bedrock_params, build_system_payload, 
    build_twelvelabs_system_payload, merge_bedrock_params
)



def _handle_async_multi_results(results, vector_bucket_name, index_name, model_id,
                               s3vector_service, metadata_dict, content_type, source_location, progress):
    """Handle multiple results from async processing (e.g., video clips)."""
    processed_keys = []
    
    for i, result in enumerate(results):
        embedding = result.get('embedding', [])
        if not embedding:
            continue
            
        # Generate unique key for each clip
        clip_key = str(uuid.uuid4())
        
        # Create metadata for this clip
        clip_metadata = metadata_dict.copy()
        clip_metadata.update(create_twelvelabs_metadata(content_type, source_location, result, i))
        
        # Store vector
        _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                                  clip_key, embedding, clip_metadata, f"Storing clip {i+1}/{len(results)}...")
        processed_keys.append(clip_key)
    
    return {
        'type': 'twelvelabs_multiclip',
        'bucket': vector_bucket_name,
        'index': index_name,
        'model': model_id,
        'contentType': content_type,
        'totalVectors': len(processed_keys),
        'keys': processed_keys
    }


def _process_video_input(video_path, vector_bucket_name, index_name, model_id,
                        bedrock_service, s3vector_service, metadata_dict, key, console,
                        user_bedrock_params, async_output_s3_uri, src_bucket_owner, session):
    """Process video input."""
    if not bedrock_service.is_async_model(model_id):
        raise click.ClickException(f"Model {model_id} does not support video input. Use a multimodal model like twelvelabs.marengo-embed-2-7-v1:0")
    
    if not async_output_s3_uri:
        raise click.ClickException(f"Video processing requires --async-output-s3-uri for async model {model_id}")
    
    with _create_progress_context(console) as progress:
        # Build system payload
        system_payload = build_twelvelabs_system_payload(
            model_id, "video", video_path, async_output_s3_uri, src_bucket_owner, session
        )
        
        # Validate and merge user parameters
        validate_bedrock_params(system_payload, user_bedrock_params, model_id)
        final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
        
        # Process async embedding
        progress.add_task("Processing video (this may take several minutes)...", total=None)
        results = bedrock_service.embed_async_with_payload(final_payload)
        
        # Handle results (likely multiple clips)
        if len(results) == 1:
            # Single result
            embedding = results[0].get('embedding', [])
            vector_key = _generate_vector_id_if_needed(key)
            
            # Create metadata
            metadata_dict.update(create_twelvelabs_metadata("video", video_path, results[0]))
            
            # Store vector
            _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                                      vector_key, embedding, metadata_dict)
            
            return _create_result_dict(vector_key, vector_bucket_name, index_name, model_id, 
                                      'video', embedding, metadata_dict)
        else:
            # Multiple results (clips)
            return _handle_async_multi_results(results, vector_bucket_name, index_name, model_id,
                                             s3vector_service, metadata_dict, "video", video_path, progress)


def _process_audio_input(audio_path, vector_bucket_name, index_name, model_id,
                        bedrock_service, s3vector_service, metadata_dict, key, console,
                        user_bedrock_params, async_output_s3_uri, src_bucket_owner, session):
    """Process audio input."""
    if not bedrock_service.is_async_model(model_id):
        raise click.ClickException(f"Model {model_id} does not support audio input. Use a multimodal model like twelvelabs.marengo-embed-2-7-v1:0")
    
    if not async_output_s3_uri:
        raise click.ClickException(f"Audio processing requires --async-output-s3-uri for async model {model_id}")
    
    with _create_progress_context(console) as progress:
        # Build system payload
        system_payload = build_twelvelabs_system_payload(
            model_id, "audio", audio_path, async_output_s3_uri, src_bucket_owner, session
        )
        
        # Validate and merge user parameters
        validate_bedrock_params(system_payload, user_bedrock_params, model_id)
        final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
        
        # Process async embedding
        progress.add_task("Processing audio (this may take a few minutes)...", total=None)
        results = bedrock_service.embed_async_with_payload(final_payload)
        
        # Handle results
        if len(results) == 1:
            # Single result
            embedding = results[0].get('embedding', [])
            vector_key = _generate_vector_id_if_needed(key)
            
            # Create metadata
            metadata_dict.update(create_twelvelabs_metadata("audio", audio_path, results[0]))
            
            # Store vector
            _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                                      vector_key, embedding, metadata_dict)
            
            return _create_result_dict(vector_key, vector_bucket_name, index_name, model_id, 
                                      'audio', embedding, metadata_dict)
        else:
            # Multiple results (clips)
            return _handle_async_multi_results(results, vector_bucket_name, index_name, model_id,
                                             s3vector_service, metadata_dict, "audio", audio_path, progress)


def _create_progress_context(console):
    """Create a standardized progress context."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )


def _generate_vector_id_if_needed(key):
    """Generate vector key if not provided."""
    return key if key else str(uuid.uuid4())


def _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                               vector_key, embedding, metadata_dict, task_description="Storing vector..."):
    """Store vector with progress tracking."""
    result_vector_id = s3vector_service.put_vector(
        bucket_name=vector_bucket_name,
        index_name=index_name,
        vector_id=vector_key,
        embedding=embedding,
        metadata=metadata_dict
    )
    return result_vector_id


def _create_result_dict(vector_key, vector_bucket_name, index_name, model_id, 
                       content_type, embedding, metadata_dict, result_type='single'):
    """Create standardized result dictionary."""
    return {
        'type': result_type,
        'key': vector_key,
        'bucket': vector_bucket_name,
        'index': index_name,
        'model': model_id,
        'contentType': content_type,
        'embeddingDimensions': len(embedding),
        'metadata': metadata_dict
    }


def _display_results(result, output_format, console):
    """Display results in the specified format."""
    if output_format == 'json':
        console.print(json.dumps(result, indent=2))
    elif output_format == 'table':
        if result.get('type') == 'batch':
            # Display batch results
            table = Table(title=f"Batch Processing Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Pattern", result.get('pattern', 'N/A'))
            table.add_row("Total Processed", str(result.get('processed_count', 0)))
            table.add_row("Failed", str(result.get('failed_count', 0)))
            table.add_row("Status", result.get('status', 'unknown'))
            
            console.print(table)
        elif result.get('type') == 'twelvelabs_multiclip':
            # Display multi-clip results
            table = Table(title=f"Multi-Clip Processing Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Content Type", result.get('contentType', 'N/A'))
            table.add_row("Total Vectors", str(result.get('totalVectors', 0)))
            table.add_row("Model", result.get('model', 'N/A'))
            table.add_row("Bucket", result.get('bucket', 'N/A'))
            table.add_row("Index", result.get('index', 'N/A'))
            
            console.print(table)
        else:
            # Display single result
            table = Table(title="Embedding Result")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Vector Key", result.get('key', 'N/A'))
            table.add_row("Content Type", result.get('contentType', 'N/A'))
            table.add_row("Embedding Dimensions", str(result.get('embeddingDimensions', 0)))
            table.add_row("Model", result.get('model', 'N/A'))
            table.add_row("Bucket", result.get('bucket', 'N/A'))
            table.add_row("Index", result.get('index', 'N/A'))
            
            console.print(table)
    else:
        console.print(f"[red]Unknown output format: {output_format}[/red]")





def _get_index_dimensions(s3vector_service, vector_bucket_name, index_name, console, debug=False):
    """Get index dimensions from S3 Vectors API."""
    if debug:
        console.print(f"[dim] Retrieving index dimensions for {index_name}[/dim]")
    
    try:
        index_info = s3vector_service.get_index(vector_bucket_name, index_name)
        
        # Extract dimensions from index info - it's nested under 'index' key
        index_data = index_info.get('index', {})
        dimensions = index_data.get('dimension')  # Note: singular 'dimension'
        if dimensions:
            if debug:
                console.print(f"[dim] Index dimensions: {dimensions}[/dim]")
            return dimensions
        else:
            console.print("[red]Error: Could not retrieve index dimensions from the index metadata[/red]")
            raise click.ClickException(f"Failed to get dimensions for index '{index_name}' in bucket '{vector_bucket_name}'. The index may be corrupted or have invalid metadata.")
            
    except Exception as e:
        # Check if it's a NotFoundException (index doesn't exist)
        error_str = str(e)
        if "NotFoundException" in error_str or "could not be found" in error_str:
            console.print(f"[red]Error: Vector index '{index_name}' not found in bucket '{vector_bucket_name}'[/red]")
            raise click.ClickException(f"Vector index '{index_name}' does not exist in bucket '{vector_bucket_name}'. Please verify the index name and bucket name are correct, and that the index has been created.")
        else:
            console.print(f"[red]Error: Failed to access vector index ({str(e)})[/red]")
            raise click.ClickException(f"Failed to access vector index '{index_name}' in bucket '{vector_bucket_name}': {str(e)}")


@click.command()
@click.option('--vector-bucket-name', required=True, help='S3 bucket name for vector storage')
@click.option('--index-name', required=True, help='Vector index name')
@click.option('--model-id', required=True, help='Bedrock embedding model ID (e.g., amazon.titan-embed-text-v2:0, amazon.titan-embed-image-v1, cohere.embed-english-v3, twelvelabs.marengo-embed-2-7-v1:0)')
@click.option('--text-value', help='Direct text input to embed')
@click.option('--text', help='Text file path (local file, S3 URI, or S3 wildcard pattern like s3://bucket/folder/*.txt)')
@click.option('--image', help='Image file path (local file, S3 URI, or S3 wildcard pattern like s3://bucket/images/*.jpg)')
@click.option('--video', help='Video file path (local file or S3 URI) for multimodal models')
@click.option('--audio', help='Audio file path (local file or S3 URI) for multimodal models')
@click.option('--async-output-s3-uri', help='S3 URI for async processing results (required for async models like TwelveLabs, e.g., s3://my-bucket/path)')
@click.option('--bedrock-inference-params', help='JSON string with model-specific parameters matching Bedrock API format (e.g., \'{"normalize": false}\' for Titan or \'{"modelInput": {"startSec": 30.0}}\' for TwelveLabs)')
@click.option('--key', help='Custom vector key (auto-generated UUID if not provided)')
@click.option('--metadata', help='JSON metadata to attach to the vector (e.g., \'{"category": "docs", "version": "1.0"}\')')
@click.option('--src-bucket-owner', help='Source bucket owner AWS account ID for cross-account S3 access')
@click.option('--use-object-key-name', is_flag=True, help='Use S3 object key as vector key for batch processing')
@click.option('--max-workers', default=4, type=int, help='Maximum parallel workers for batch processing (default: 4)')
@click.option('--output', type=click.Choice(['table', 'json']), default='json', help='Output format (default: json)')
@click.option('--region', help='AWS region (overrides session/config defaults)')
@click.pass_context
def embed_put(ctx, vector_bucket_name, index_name, model_id, text_value, text, image,
              video, audio, async_output_s3_uri, bedrock_inference_params,
              key, metadata, src_bucket_owner, use_object_key_name,
              max_workers, output, region):
    """Embed text, image, video, or audio content and store as vectors in S3.
    
    \b
    SUPPORTED INPUT TYPES:
    • Direct text: --text-value "your text here"
    • Local files: --text /path/to/file.txt or --image /path/to/image.jpg
    • S3 files: --text s3://bucket/file.txt or --image s3://bucket/image.jpg
    • S3 wildcards: --text "s3://bucket/folder/*.txt" or --image "s3://bucket/images/*.jpg"
    • Multimodal: --text-value "description" --image /path/to/image.jpg (Titan Image v1 only)
    • Video/Audio: --video /path/to/video.mp4 or --audio /path/to/audio.wav
    
    \b
    SUPPORTED MODELS:
    • amazon.titan-embed-text-v2:0 (1024, 512, 256 dimensions)
    • amazon.titan-embed-text-v1 (1536 dimensions, fixed)
    • amazon.titan-embed-image-v1 (1024, 384, 256 dimensions, supports text+image)
    • cohere.embed-english-v3 (1024 dimensions, fixed)
    • cohere.embed-multilingual-v3 (1024 dimensions, fixed)
    • twelvelabs.marengo-embed-2-7-v1:0 (1024 dimensions, async processing)
     
    \b
    BEDROCK INFERENCE PARAMETERS:
    Use --bedrock-inference-params to pass model-specific parameters in Bedrock API format:
    • Sync models: Direct invoke_model body parameters
    • Async models: Complete StartAsyncInvoke structure
    
    \b
    BATCH PROCESSING:
    • Supports up to 500 vectors per batch
    • Use wildcards for multiple files: s3://bucket/docs/*.txt
    • Automatic batching for large datasets
    • Parallel processing with --max-workers
    
    \b
    EXAMPLES:
    # Direct text embedding
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --text-value "Hello world"
    
    # Titan Text v2 with custom parameters
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --text-value "Hello world" \\
      --bedrock-inference-params '{"normalize": false, "embeddingTypes": ["binary"]}'
    
    # Cohere with custom parameters
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id cohere.embed-english-v3 --text-value "Search query" \\
      --bedrock-inference-params '{"input_type": "search_query", "truncate": "END"}'
    
    # TwelveLabs text embedding (async)
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id twelvelabs.marengo-embed-2-7-v1:0 --text-value "Spiderman flies through a street" \\
      --async-output-s3-uri s3://my-async-bucket
    
    # TwelveLabs video with custom parameters
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id twelvelabs.marengo-embed-2-7-v1:0 --video ./sample.mp4 \\
      --async-output-s3-uri s3://my-async-bucket \\
      --bedrock-inference-params '{"modelInput": {"startSec": 30.0, "useFixedLengthSec": 5, "embeddingOption": ["visual-text"]}}'
    
    # S3 batch processing
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --text "s3://source-bucket/docs/*.txt" \\
      --metadata '{"category": "documentation"}' --max-workers 4
    
    # Multimodal embedding (Titan Image v1)
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-image-v1 --text-value "A red car" --image car.jpg
    """
    
    console = ctx.obj['console']
    session = ctx.obj['aws_session']
    debug = ctx.obj.get('debug', False)
    region = get_region(session, region)
    
    # Parse bedrock inference parameters
    user_bedrock_params = {}
    if bedrock_inference_params:
        try:
            user_bedrock_params = json.loads(bedrock_inference_params)
        except json.JSONDecodeError:
            raise click.ClickException("Invalid JSON in --bedrock-inference-params parameter")
    
    # Check if this is an async model (TwelveLabs)
    is_async_model = model_id.startswith('twelvelabs.')
    
    if is_async_model and not async_output_s3_uri:
        raise click.ClickException(
            f"Async models like {model_id} require --async-output-s3-uri parameter. "
            "Please provide an S3 URI for async processing results (e.g., s3://my-bucket/path)."
        )
    
    # Input validation
    inputs_provided = sum(bool(x) for x in [text_value, text, image, video, audio])
    if inputs_provided == 0:
        raise click.ClickException("At least one input must be provided: --text-value, --text, --image, --video, or --audio")
    
    # Special case: Allow multimodal input (text-value + image) for Titan Image v1 model
    is_multimodal_titan = (model_id.startswith('amazon.titan-embed-image') and 
                          text_value and image and not text and not video and not audio)
    
    if inputs_provided > 1 and not is_multimodal_titan:
        raise click.ClickException("Only one input type can be specified at a time, except for multimodal input with Titan Image v1 (--text-value + --image)")
    
    if is_multimodal_titan:
        console.print("[dim] Multimodal input detected: Using both text and image for Titan Image v1[/dim]")
    
    try:
        # Initialize services
        bedrock_service = BedrockService(session, region, debug=debug, console=console)
        s3vector_service = S3VectorService(session, region, debug=debug, console=console)
        s3_client = session.client('s3')
        
        # Parse metadata
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise click.ClickException("Invalid JSON in --metadata parameter")
        
        # Determine input type and process accordingly
        if is_multimodal_titan:
            # Handle multimodal input (text + image) for Titan Image v1
            result = _process_multimodal_input(
                text_value, image, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, s3_client, metadata_dict, key, console,
                user_bedrock_params
            )
        elif text_value:
            result = _process_text_value(
                text_value, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, metadata_dict, key, console,
                user_bedrock_params, async_output_s3_uri, src_bucket_owner, session
            )
        elif text:
            result = _process_text_input(
                text, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, s3_client, metadata_dict,
                src_bucket_owner, use_object_key_name, max_workers,
                key, console, region, user_bedrock_params, async_output_s3_uri, session
            )
        elif image:
            result = _process_image_input(
                image, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, s3_client, metadata_dict,
                src_bucket_owner, use_object_key_name, max_workers,
                key, console, region, user_bedrock_params, async_output_s3_uri, session
            )
        elif video:
            result = _process_video_input(
                video, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, metadata_dict, key, console,
                user_bedrock_params, async_output_s3_uri, src_bucket_owner, session
            )
        elif audio:
            result = _process_audio_input(
                audio, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, metadata_dict, key, console,
                user_bedrock_params, async_output_s3_uri, src_bucket_owner, session
            )
        
        # Display results
        _display_results(result, output, console)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise click.ClickException(str(e))


def _process_multimodal_input(text_value, image_path, vector_bucket_name, index_name, model_id,
                             bedrock_service, s3vector_service, s3_client, metadata_dict, key, console,
                             user_bedrock_params):
    """Process multimodal input (text + image) for Titan Image v1."""
    
    # Get index dimensions first
    dimensions = _get_index_dimensions(s3vector_service, vector_bucket_name, index_name, console, debug=bedrock_service.debug)
    
    with _create_progress_context(console) as progress:
        
        # Read and encode image
        image_task = progress.add_task("Processing image...", total=None)
        try:
            import base64
            
            # Check if it's an S3 URI or local file
            if image_path.startswith('s3://'):
                # Parse S3 URI
                path_part = image_path[5:]  # Remove 's3://'
                if '/' not in path_part:
                    raise ValueError(f"Invalid S3 URI format: {image_path}")
                
                bucket, key_name = path_part.split('/', 1)
                
                # Read from S3
                response = s3_client.get_object(Bucket=bucket, Key=key_name)
                image_data = base64.b64encode(response['Body'].read()).decode('utf-8')
            else:
                # Read local file
                with open(image_path, 'rb') as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    
            progress.update(image_task, description="Image processed ✓")
        except Exception as e:
            raise click.ClickException(f"Failed to read image file: {str(e)}")
        
        # Build system payload for multimodal
        system_payload = build_system_payload(model_id, None, "multimodal", dimensions, text_value, image_data, is_query=False)
        
        # Validate and merge user parameters
        validate_bedrock_params(system_payload, user_bedrock_params, model_id)
        final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
        
        # Generate embedding
        embed_task = progress.add_task("Generating multimodal embedding...", total=None)
        embedding = bedrock_service.embed_with_payload(model_id, final_payload)
        progress.update(embed_task, description="Multimodal embedding generated ✓")
        
        # Prepare metadata - add both text and image info
        metadata_dict.update({
            'S3VECTORS-EMBED-SRC-CONTENT': text_value,
            'S3VECTORS-EMBED-SRC-LOCATION': image_path
        })
        
        # Generate vector ID if not provided
        vector_key = _generate_vector_id_if_needed(key)
        
        # Store vector
        _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                                  vector_key, embedding, metadata_dict)
    
    return _create_result_dict(vector_key, vector_bucket_name, index_name, model_id, 
                              'multimodal', embedding, metadata_dict)


def _process_text_value(text_value, vector_bucket_name, index_name, model_id,
                       bedrock_service, s3vector_service, metadata_dict, key, console,
                       user_bedrock_params, async_output_s3_uri=None, src_bucket_owner=None, session=None):
    """Process direct text value input with new unified approach."""
    
    with _create_progress_context(console) as progress:
        
        if bedrock_service.is_async_model(model_id):
            # Handle async models (TwelveLabs)
            if not async_output_s3_uri:
                raise click.ClickException(f"Async model {model_id} requires --async-output-s3-uri")
            
            # Build system payload for async model
            system_payload = build_twelvelabs_system_payload(
                model_id, "text", text_value, async_output_s3_uri, src_bucket_owner, session
            )
            
            # Validate and merge user parameters
            validate_bedrock_params(system_payload, user_bedrock_params, model_id)
            final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
            
            # Process async embedding
            progress.add_task("Processing text with async model (this may take a few minutes)...", total=None)
            results = bedrock_service.embed_async_with_payload(final_payload)
            
            # Handle multiple results for async models
            if len(results) == 1:
                # Single result
                embedding = results[0].get('embedding', [])
                vector_key = _generate_vector_id_if_needed(key)
                
                # Create metadata
                metadata_dict.update(create_twelvelabs_metadata("text", "direct_text_input", results[0]))
                
                # Store vector
                _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                                          vector_key, embedding, metadata_dict)
                
                return _create_result_dict(vector_key, vector_bucket_name, index_name, model_id, 
                                          'text', embedding, metadata_dict)
            else:
                # Multiple results - handle as batch
                return _handle_async_multi_results(results, vector_bucket_name, index_name, model_id,
                                                 s3vector_service, metadata_dict, "text", "direct_text_input",
                                                 progress)
        else:
            # Handle sync models
            # Get index dimensions first
            dimensions = _get_index_dimensions(s3vector_service, vector_bucket_name, index_name, console, debug=bedrock_service.debug)
            
            # Build system payload
            if model_id.startswith('amazon.titan-embed-image'):
                # For Titan Image v1, text goes in text_input parameter
                system_payload = build_system_payload(model_id, None, "text", dimensions, text_input=text_value, is_query=False)
            else:
                # For other models, text goes in input_content
                system_payload = build_system_payload(model_id, text_value, "text", dimensions, is_query=False)
            
            # Validate and merge user parameters
            validate_bedrock_params(system_payload, user_bedrock_params, model_id)
            final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
            
            # Generate embedding
            progress.add_task("Generating embedding...", total=None)
            embedding = bedrock_service.embed_with_payload(model_id, final_payload)
            
            # Prepare metadata
            metadata_dict.update({'S3VECTORS-EMBED-SRC-CONTENT': text_value})
            
            # Generate vector ID and store
            vector_key = _generate_vector_id_if_needed(key)
            _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                                      vector_key, embedding, metadata_dict)
            
            return _create_result_dict(vector_key, vector_bucket_name, index_name, model_id, 
                                      'text', embedding, metadata_dict)


def _process_file_input(file_input, vector_bucket_name, index_name, model_id,
                       bedrock_service, s3vector_service, s3_client, metadata_dict,
                       src_bucket_owner, use_object_key_name, max_workers,
                       vector_id, console, region, is_image=False, user_bedrock_params=None,
                       async_output_s3_uri=None, session=None):
    """Process file input (text or image) - handles both single files and wildcards."""
    
    # Initialize input processor
    input_processor = InputProcessor(s3_client)
    
    try:
        # Process input - since this comes from --text parameter, we know it should be a file
        if file_input.startswith('s3://'):
            if file_input.endswith('*') or file_input.endswith('/*'):
                input_type = "s3_wildcard"
            else:
                input_type = "s3_file"
        elif '*' in file_input or '?' in file_input:
            input_type = "local_wildcard"
        else:
            input_type = "local_file"
            
        processed_input = input_processor.process_input(
            file_input, 
            input_type=input_type, 
            bucket_owner=src_bucket_owner,
            is_image=is_image
        )
        
        if processed_input.get('batch_processing'):
            # Batch processing for wildcards
            return _process_batch(
                processed_input, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, s3_client, metadata_dict,
                use_object_key_name, max_workers, console, user_bedrock_params,
                async_output_s3_uri, session
            )
        else:
            # Single file processing
            return _process_single_file(
                processed_input, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, metadata_dict, vector_id, console, is_image,
                user_bedrock_params, async_output_s3_uri, session
            )
            
    except Exception as e:
        input_type = "image" if is_image else "text"
        raise click.ClickException(f"Failed to process {input_type} input: {str(e)}")


def _process_text_input(text_input, vector_bucket_name, index_name, model_id,
                       bedrock_service, s3vector_service, s3_client, metadata_dict,
                       src_bucket_owner, use_object_key_name, max_workers,
                       vector_id, console, region, user_bedrock_params, async_output_s3_uri, session):
    """Process text input (file or S3 URI or wildcard)."""
    return _process_file_input(text_input, vector_bucket_name, index_name, model_id,
                              bedrock_service, s3vector_service, s3_client, metadata_dict,
                              src_bucket_owner, use_object_key_name, max_workers,
                              vector_id, console, region, is_image=False, 
                              user_bedrock_params=user_bedrock_params, 
                              async_output_s3_uri=async_output_s3_uri, session=session)


def _process_image_input(image_input, vector_bucket_name, index_name, model_id,
                        bedrock_service, s3vector_service, s3_client, metadata_dict,
                        src_bucket_owner, use_object_key_name, max_workers,
                        vector_id, console, region, user_bedrock_params, async_output_s3_uri, session):
    """Process image input (file or S3 URI or wildcard)."""
    return _process_file_input(image_input, vector_bucket_name, index_name, model_id,
                              bedrock_service, s3vector_service, s3_client, metadata_dict,
                              src_bucket_owner, use_object_key_name, max_workers,
                              vector_id, console, region, is_image=True,
                              user_bedrock_params=user_bedrock_params,
                              async_output_s3_uri=async_output_s3_uri, session=session)


def _process_single_file(processed_input, vector_bucket_name, index_name, model_id,
                        bedrock_service, s3vector_service, metadata_dict, key, console, is_image=False,
                        user_bedrock_params=None, async_output_s3_uri=None, session=None):
    """Process single file (text or image) with new unified approach."""
    
    with _create_progress_context(console) as progress:
        
        if bedrock_service.is_async_model(model_id):
            # Handle async models
            if not async_output_s3_uri:
                raise click.ClickException(f"Async model {model_id} requires --async-output-s3-uri")
            
            input_type = "image" if is_image else "text"
            
            # For file inputs with async models, we need to handle the content appropriately
            if input_type == "text":
                # Use the text content directly
                system_payload = build_twelvelabs_system_payload(
                    model_id, "text", processed_input['content'], async_output_s3_uri, None, session
                )
            else:
                # For images, we need to handle the file path, not the base64 content
                file_path = processed_input.get('file_path') or processed_input.get('s3_uri', 'unknown')
                system_payload = build_twelvelabs_system_payload(
                    model_id, "image", file_path, async_output_s3_uri, None, session
                )
            
            # Validate and merge user parameters
            validate_bedrock_params(system_payload, user_bedrock_params, model_id)
            final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
            
            # Process async embedding
            progress.add_task(f"Processing {input_type} with async model...", total=None)
            results = bedrock_service.embed_async_with_payload(final_payload)
            
            # Handle single result (files typically produce single embeddings)
            if results:
                embedding = results[0].get('embedding', [])
                vector_key = _generate_vector_id_if_needed(key)
                
                # Prepare metadata
                final_metadata = processed_input['metadata'].copy()
                final_metadata.update(metadata_dict)
                final_metadata.update(create_twelvelabs_metadata(input_type, 
                                                               processed_input.get('file_path') or processed_input.get('s3_uri', 'unknown'), 
                                                               results[0]))
                
                # Store vector
                _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                                          vector_key, embedding, final_metadata)
                
                return _create_result_dict(vector_key, vector_bucket_name, index_name, model_id, 
                                          input_type, embedding, final_metadata)
        else:
            # Handle sync models
            # Get index dimensions first
            dimensions = _get_index_dimensions(s3vector_service, vector_bucket_name, index_name, console, debug=bedrock_service.debug)
            
            # Build system payload
            input_type = "image" if is_image else "text"
            if model_id.startswith('amazon.titan-embed-image'):
                if is_image:
                    # For Titan Image v1 with image input
                    system_payload = build_system_payload(model_id, None, input_type, dimensions, image_data=processed_input['content'], is_query=False)
                else:
                    # For Titan Image v1 with text input
                    system_payload = build_system_payload(model_id, None, input_type, dimensions, text_input=processed_input['content'], is_query=False)
            else:
                # For other models
                system_payload = build_system_payload(model_id, processed_input['content'], input_type, dimensions, is_query=False)
            
            # Validate and merge user parameters
            validate_bedrock_params(system_payload, user_bedrock_params, model_id)
            final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
            
            # Generate embedding
            progress.add_task("Generating embedding...", total=None)
            embedding = bedrock_service.embed_with_payload(model_id, final_payload)
            
            # Prepare metadata - only use the metadata from processed input and custom metadata
            final_metadata = processed_input['metadata'].copy()
            final_metadata.update(metadata_dict)
            
            # Generate vector ID if not provided
            vector_key = _generate_vector_id_if_needed(key)
            
            # Store vector
            _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                                      vector_key, embedding, final_metadata, "Storing vector in S3...")
            
            content_type = 'image' if is_image else 'text'
            return _create_result_dict(vector_key, vector_bucket_name, index_name, model_id, 
                                      content_type, embedding, final_metadata)


def _process_batch(processed_input, vector_bucket_name, index_name, model_id,
                  bedrock_service, s3vector_service, s3_client, metadata_dict,
                  use_object_key_name, max_workers, console, user_bedrock_params,
                  async_output_s3_uri, session):
    """Process batch wildcard input (S3 or local filesystem)."""
    
    # For now, batch processing with the new parameter system is complex
    # We'll need to update the BatchProcessor to handle the new approach
    # For the initial implementation, let's show a helpful error message
    
    if user_bedrock_params:
        raise click.ClickException(
            "Batch processing with --bedrock-inference-params is not yet supported. "
            "Please process files individually or use the default model parameters for batch operations."
        )
    
    if bedrock_service.is_async_model(model_id):
        raise click.ClickException(
            f"Batch processing with async models like {model_id} is not yet supported. "
            "Please process files individually for async models."
        )
    
    # Initialize batch processor
    config = BatchConfig(
        max_workers=max_workers,
        max_vectors_per_batch=500  # Maximum 500 vectors per batch
    )
    
    batch_processor = BatchProcessor(
        s3_client=s3_client,
        bedrock_service=bedrock_service,
        s3vector_service=s3vector_service,
        config=config
    )
    
    # Process wildcard pattern based on type
    try:
        if processed_input['type'] == 's3_wildcard':
            # S3 wildcard processing
            result = batch_processor.process_wildcard_pattern(
                s3_pattern=processed_input['pattern'],
                vector_bucket=vector_bucket_name,
                index_name=index_name,
                model_id=model_id,
                metadata_template=metadata_dict,
                bucket_owner=processed_input.get('bucket_owner'),
                use_object_key_name=use_object_key_name
            )
        elif processed_input['type'] == 'local_wildcard':
            # Local filesystem wildcard processing
            result = batch_processor.process_local_wildcard_pattern(
                local_pattern=processed_input['pattern'],
                vector_bucket=vector_bucket_name,
                index_name=index_name,
                model_id=model_id,
                metadata_template=metadata_dict,
                use_object_key_name=use_object_key_name
            )
        else:
            raise ValueError(f"Unsupported batch processing type: {processed_input['type']}")
            
    except ValueError as e:
        # Convert ValueError from batch processor to ClickException
        raise click.ClickException(str(e))
    
    return {
        'type': 'batch',
        'pattern': processed_input['pattern'],
        'bucket': vector_bucket_name,
        'index': index_name,
        'model': model_id,
        'processed_count': result['processed_count'],
        'failed_count': result['failed_count'],
        'keys': result['Keys'],
        'status': result['status']
    }






