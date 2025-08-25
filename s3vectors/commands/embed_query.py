"""Embed and query vectors command."""

import os
import json
import base64
from pathlib import Path
from typing import Optional

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from s3vectors.core.services import BedrockService, S3VectorService
from s3vectors.utils.config import get_region
from s3vectors.utils.bedrock_params import (
    validate_bedrock_params, build_system_payload, merge_bedrock_params
)


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


def _validate_query_inputs(query_input, text_value, text, image, video, audio):
    """Validate query input parameters and return the selected input type and value."""
    
    # Count explicit input parameters (excluding deprecated query_input)
    explicit_inputs = [text_value, text, image, video, audio]
    explicit_provided = sum(bool(x) for x in explicit_inputs)
    
    # Check for conflicting inputs
    if explicit_provided > 1:
        provided_types = []
        if text_value: provided_types.append("--text-value")
        if text: provided_types.append("--text")
        if image: provided_types.append("--image")
        if video: provided_types.append("--video")
        if audio: provided_types.append("--audio")
        
        raise click.ClickException(
            f"Only one input type allowed. You provided: {', '.join(provided_types)}. "
            f"Please use only one input parameter."
        )
    
    # Handle backward compatibility with query_input
    if explicit_provided == 0:
        if query_input:
            # Show deprecation warning
            click.echo("⚠️  WARNING: --query-input is deprecated. Use --text-value, --text, --image, --video, or --audio instead.", err=True)
            return "legacy", query_input
        else:
            raise click.ClickException(
                "No query input provided. Please specify one of: "
                "--text-value, --text, --image, --video, --audio"
            )
    
    # Return the provided input type and value
    if text_value:
        return "text_value", text_value
    elif text:
        return "text", text
    elif image:
        return "image", image
    elif video:
        return "video", video
    elif audio:
        return "audio", audio


def _validate_twelvelabs_query_parameters(input_type, user_bedrock_params, async_output_s3_uri, src_bucket_owner):
    """Validate TwelveLabs-specific query parameters."""
    
    # Validate required parameters for TwelveLabs
    if not async_output_s3_uri:
        raise click.ClickException(
            "TwelveLabs queries require an S3 output URI for async processing. "
            "Please provide --async-output-s3-uri parameter."
        )
    
    # Extract parameters from bedrock_inference_params
    model_input = user_bedrock_params.get("modelInput", {}) if user_bedrock_params else {}
    embedding_options = model_input.get("embeddingOption", [])
    start_sec = model_input.get("startSec", 0.0)
    use_fixed_length_sec = model_input.get("useFixedLengthSec", 5.0)
    
    # Convert single embedding option to list if needed
    if isinstance(embedding_options, str):
        embedding_options = [embedding_options]
    
    # Validate embedding options
    if input_type == "video":
        if not embedding_options:
            raise click.ClickException(
                "Video queries require embedding options in --bedrock-inference-params. "
                'Use: --bedrock-inference-params \'{"modelInput": {"embeddingOption": ["visual-text"]}}\''
            )
        valid_options = ["visual-text", "visual-image", "audio"]
        for option in embedding_options:
            if option not in valid_options:
                raise click.ClickException(
                    f"Invalid embedding option '{option}'. Choose from: {', '.join(valid_options)}"
                )
    elif input_type == "audio":
        if embedding_options and embedding_options != ["audio"]:
            raise click.ClickException(
                f"Audio queries only support 'audio' embedding option, got: {embedding_options}"
            )
        # Auto-set embedding_options for audio
        embedding_options = ["audio"]
    
    # Validate time parameters
    if start_sec < 0:
        raise click.ClickException("startSec must be 0 or greater")
    
    if not (2 <= use_fixed_length_sec <= 10):
        raise click.ClickException("useFixedLengthSec must be between 2 and 10 seconds")
    
    return embedding_options, start_sec, use_fixed_length_sec


def _process_text_value_input(text_value, bedrock_service, model_id, dimensions, user_bedrock_params=None):
    """Process direct text value input."""
    content_type = "text"
    
    # Build system payload
    system_payload = build_system_payload(model_id, text_value, "text", dimensions, is_query=True)
    
    # Validate and merge user parameters
    validate_bedrock_params(system_payload, user_bedrock_params, model_id, is_query=True)
    final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
    
    # Generate embedding
    query_embedding = bedrock_service.embed_with_payload(model_id, final_payload)
    return query_embedding, content_type


def _process_text_file_input(text_path, bedrock_service, model_id, dimensions, src_bucket_owner, session, region, debug, console, user_bedrock_params=None):
    """Process text file input (local or S3)."""
    content_type = "text"
    
    if text_path.startswith('s3://'):
        # S3 text file
        _, content = _process_s3_query_input(text_path, src_bucket_owner, session, region, debug, console)
    else:
        # Local text file
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            raise click.ClickException(f"Text file not found: {text_path}")
        except Exception as e:
            raise click.ClickException(f"Error reading text file {text_path}: {str(e)}")
    
    # Build system payload
    system_payload = build_system_payload(model_id, content, "text", dimensions, is_query=True)
    
    # Validate and merge user parameters
    validate_bedrock_params(system_payload, user_bedrock_params, model_id, is_query=True)
    final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
    
    # Generate embedding
    query_embedding = bedrock_service.embed_with_payload(model_id, final_payload)
    return query_embedding, content_type


def _process_image_file_input(image_path, bedrock_service, model_id, dimensions, src_bucket_owner, session, region, debug, console, user_bedrock_params=None):
    """Process image file input (local or S3)."""
    content_type = "image"
    
    if image_path.startswith('s3://'):
        # S3 image file
        _, content = _process_s3_query_input(image_path, src_bucket_owner, session, region, debug, console)
    else:
        # Local image file
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
        except FileNotFoundError:
            raise click.ClickException(f"Image file not found: {image_path}")
        except Exception as e:
            raise click.ClickException(f"Error reading image file {image_path}: {str(e)}")
    
    # Convert bytes to base64 string for bedrock
    if isinstance(content, bytes):
        content = base64.b64encode(content).decode('utf-8')
    
    # Build system payload
    system_payload = build_system_payload(model_id, content, "image", dimensions, is_query=True)
    
    # Validate and merge user parameters
    validate_bedrock_params(system_payload, user_bedrock_params, model_id, is_query=True)
    final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
    
    # Generate embedding
    query_embedding = bedrock_service.embed_with_payload(model_id, final_payload)
    return query_embedding, content_type


def _process_video_file_input(video_path, bedrock_service, model_id, async_output_s3_uri, 
                            src_bucket_owner, session, console, debug, user_bedrock_params=None):
    """Process video file input for TwelveLabs models using unified system."""
    return _process_media_file_input(
        video_path, bedrock_service, model_id, "video", async_output_s3_uri,
        src_bucket_owner, session, console, debug, user_bedrock_params
    )


def _process_media_file_input(media_path, bedrock_service, model_id, media_type, async_output_s3_uri, 
                            src_bucket_owner, session, console, debug, user_bedrock_params=None):
    """Process media file input (image, video, audio) for TwelveLabs models using unified system."""
    from s3vectors.utils.bedrock_params import build_twelvelabs_system_payload, validate_bedrock_params, merge_bedrock_params
    
    # Extract parameters from user_bedrock_params
    model_input_params = user_bedrock_params.get("modelInput", {}) if user_bedrock_params else {}
    embedding_options = model_input_params.get("embeddingOption", [])
    start_sec = model_input_params.get("startSec", 0.0)
    use_fixed_length_sec = model_input_params.get("useFixedLengthSec", 5.0)
    
    # Convert single embedding option to list if needed
    if isinstance(embedding_options, str):
        embedding_options = [embedding_options]
    
    # Validate embedding options for video
    if media_type == "video" and not embedding_options:
        raise click.ClickException(
            "Video queries require embedding options in --bedrock-inference-params. "
            'Use: --bedrock-inference-params \'{"modelInput": {"embeddingOption": ["visual-text"]}}\''
        )
    
    # Auto-set for audio
    if media_type == "audio" and not embedding_options:
        embedding_options = ["audio"]
    
    if debug:
        console.print(f"[dim]Processing {media_type} query: {media_path}[/dim]")
        if media_type in ["video", "audio"]:
            console.print(f"[dim]Time range: {start_sec}s - {start_sec + use_fixed_length_sec}s[/dim]")
        if media_type == "video" and embedding_options:
            console.print(f"[dim]Embedding option: {embedding_options}[/dim]")
    
    # Process media query using unified TwelveLabs system
    try:
        status_msg = f"Processing {media_type} query"
        if media_type in ["video", "audio"]:
            status_msg += f" (clip: {start_sec}s-{start_sec + use_fixed_length_sec}s"
            if media_type == "video" and embedding_options:
                status_msg += f", type: {embedding_options[0] if embedding_options else 'default'}"
            status_msg += ")"
        status_msg += "..."
        
        with console.status(status_msg):
            # Build system payload
            system_payload = build_twelvelabs_system_payload(
                model_id, media_type, media_path, async_output_s3_uri, src_bucket_owner, session
            )
            
            # Validate and merge user parameters (this will add the user's parameters)
            validate_bedrock_params(system_payload, user_bedrock_params, model_id, is_query=True)
            final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
            
            # Process async embedding
            results = bedrock_service.embed_async_with_payload(final_payload)
            
            if not results or len(results) == 0:
                raise click.ClickException(f"No embeddings generated from {media_type} query")
            
            # Use the first (and should be only) embedding
            query_embedding = results[0]['embedding']
            job_id = results[0].get('jobId')
            
            if debug:
                console.print(f"[dim]Generated embedding with {len(query_embedding)} dimensions[/dim]")
                if job_id:
                    console.print(f"[dim]Job ID: {job_id}[/dim]")
            
            return query_embedding, media_type, job_id
            
    except Exception as e:
        raise click.ClickException(f"Error processing {media_type} query: {str(e)}")


def _process_audio_file_input(audio_path, bedrock_service, model_id, async_output_s3_uri, 
                            src_bucket_owner, session, console, debug, user_bedrock_params=None):
    """Process audio file input for TwelveLabs models using unified system."""
    return _process_media_file_input(
        audio_path, bedrock_service, model_id, "audio", async_output_s3_uri,
        src_bucket_owner, session, console, debug, user_bedrock_params
    )


def _process_twelvelabs_text_query(query_input, bedrock_service, async_output_s3_uri, src_bucket_owner, console, debug, model_id, session, user_bedrock_params=None):
    """Process text query for TwelveLabs models using unified system."""
    
    # Validate required parameters
    if not async_output_s3_uri:
        raise click.ClickException(
            "TwelveLabs queries require an S3 output URI for async processing. "
            "Please provide an S3 URI using the --async-output-s3-uri parameter."
        )
    
    # Note: src_bucket_owner is optional - only required for cross-account S3 access
    
    # Validate that query input is text only for legacy parameter
    query_path = Path(query_input)
    if query_path.exists() or query_input.startswith('s3://'):
        raise click.ClickException(
            "Deprecated input --query-input is not supported for TwelveLabs Marengo Embed 2.7 embeddings model. "
            "Use specific input parameters instead:\n"
            "  --text-value for direct text input\n"
            "  --text for text files\n"
            "  --image for image files\n"
            "  --video for video files (requires embedding options in --bedrock-inference-params)\n"
            "  --audio for audio files"
        )
    
    if debug:
        console.print(f"[dim]Processing TwelveLabs text query: {query_input[:50]}...[/dim]")
    
    # Use unified TwelveLabs processing
    try:
        with console.status("Processing text query with TwelveLabs (this may take a few minutes)..."):
            # Build system payload using unified approach
            from s3vectors.utils.bedrock_params import build_twelvelabs_system_payload, validate_bedrock_params, merge_bedrock_params
            
            system_payload = build_twelvelabs_system_payload(
                model_id, "text", query_input, async_output_s3_uri, src_bucket_owner, session
            )
            
            # Validate and merge user parameters
            validate_bedrock_params(system_payload, user_bedrock_params, model_id, is_query=True)
            final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
            
            # Process async embedding using unified method
            results = bedrock_service.embed_async_with_payload(final_payload)
            
            # Extract the first (and only) embedding from results
            if not results or len(results) == 0:
                raise click.ClickException("No embedding generated from TwelveLabs text query")
            
            # For text queries, we expect a single result
            first_result = results[0]
            if 'embedding' not in first_result:
                raise click.ClickException("No embedding found in TwelveLabs response")
            
            query_embedding = first_result['embedding']
            job_id = first_result.get('jobId')
            
            if debug:
                console.print(f"[dim]Generated TwelveLabs text embedding with {len(query_embedding)} dimensions[/dim]")
                if job_id:
                    console.print(f"[dim]Job ID: {job_id}[/dim]")
            
            return query_embedding, job_id
            
    except Exception as e:
        raise click.ClickException(f"Failed to process TwelveLabs text query: {str(e)}")


def _process_standard_query(query_input, bedrock_service, model_id, dimensions, src_bucket_owner, session, region, debug, console, user_bedrock_params=None):
    """Process query for standard (non-TwelveLabs) models."""
    
    # Process query input (text, local file, or S3 file)
    if query_input.startswith('s3://'):
        # S3 file processing
        content_type, content = _process_s3_query_input(query_input, src_bucket_owner, session, region, debug, console)
    else:
        # Local file or direct text processing
        query_path = Path(query_input)
        if query_path.exists() and query_path.is_file():
            # It's a local file
            if query_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                content_type = "image"
                with open(query_path, 'rb') as f:
                    content = f.read()  # Keep as bytes for consistency
            else:
                content_type = "text"
                with open(query_path, 'r', encoding='utf-8') as f:
                    content = f.read()
        else:
            # It's direct text
            content_type = "text"
            content = query_input
    
    # Convert bytes to base64 string for images
    if content_type == "image" and isinstance(content, bytes):
        content = base64.b64encode(content).decode('utf-8')
    
    # Build system payload
    system_payload = build_system_payload(model_id, content, content_type, dimensions, is_query=True)
    
    # Validate and merge user parameters
    validate_bedrock_params(system_payload, user_bedrock_params, model_id, is_query=True)
    final_payload = merge_bedrock_params(system_payload, user_bedrock_params, model_id)
    
    # Generate query embedding
    query_embedding = bedrock_service.embed_with_payload(model_id, final_payload)
    
    return query_embedding, content_type


@click.command()
@click.option('--vector-bucket-name', required=True, help='S3 bucket name for vector storage')
@click.option('--index-name', required=True, help='Vector index name')
@click.option('--model-id', required=True, help='Bedrock embedding model ID (e.g., amazon.titan-embed-text-v2:0, amazon.titan-embed-image-v1, cohere.embed-english-v3, twelvelabs.marengo-embed-2-7-v1:0)')
@click.option('--query-input', help='[DEPRECATED] Query text or file path - use specific input types instead')
@click.option('--text-value', help='Direct text query string')
@click.option('--text', help='Text file path (local file or S3 URI)')
@click.option('--image', help='Image file path (local file or S3 URI)')
@click.option('--video', help='Video file path (local file or S3 URI) - TwelveLabs models only')
@click.option('--audio', help='Audio file path (local file or S3 URI) - TwelveLabs models only')

@click.option('--k', default=5, type=int, help='Number of results to return (default: 5)')
@click.option('--filter', 'filter_expr', help='Filter expression for results (JSON format with operators, e.g., \'{"$and": [{"category": "docs"}, {"version": "1.0"}]}\')')
@click.option('--return-distance', is_flag=True, help='Return similarity distances in results')
@click.option('--return-metadata/--no-return-metadata', default=True, help='Return metadata in results (default: true)')
@click.option('--src-bucket-owner', help='Source bucket owner AWS account ID for cross-account S3 access')
@click.option('--async-output-s3-uri', help='S3 URI for async output (required for TwelveLabs models, e.g., s3://my-async-bucket)')
@click.option('--bedrock-inference-params', help='JSON string with model-specific parameters matching Bedrock API format (e.g., \'{"normalize": false}\' for Titan or \'{"input_type": "search_query"}\' for Cohere)')
@click.option('--output', type=click.Choice(['table', 'json']), default='json', help='Output format (default: json)')
@click.option('--region', help='AWS region (overrides session/config defaults)')
@click.pass_context
def embed_query(ctx, vector_bucket_name, index_name, model_id, query_input, text_value, text, image, video, audio,
                k, filter_expr, return_distance, return_metadata, 
                src_bucket_owner, async_output_s3_uri, bedrock_inference_params, output, region):
    """Embed query input and search for similar vectors in S3.
    
    \b
    SUPPORTED QUERY INPUT TYPES:
    • Direct text: --text-value "search for this text"
    • Local text file: --text /path/to/query.txt
    • Local image file: --image /path/to/image.jpg
    • S3 text file: --text s3://bucket/query.txt
    • S3 image file: --image s3://bucket/image.jpg
    • Video files: --video /path/to/video.mp4 (TwelveLabs models only)
    • Audio files: --audio /path/to/audio.wav (TwelveLabs models only)
    
    \b
    SUPPORTED MODELS:
    • amazon.titan-embed-text-v2:0 (text queries, 1024/512/256 dimensions)
    • amazon.titan-embed-text-v1 (text queries, 1536 dimensions)
    • amazon.titan-embed-image-v1 (text and image queries, 1024/384/256 dimensions)
    • cohere.embed-english-v3 (text queries, 1024 dimensions)
    • cohere.embed-multilingual-v3 (text queries, 1024 dimensions)
    • twelvelabs.marengo-embed-2-7-v1:0 (text, video, audio queries, 1024 dimensions, async processing)
    
    \b
    TWELVELABS VIDEO/AUDIO QUERIES:
    • Video queries: Require embedding options in --bedrock-inference-params
    • Audio queries: Automatically use audio embedding option
    • Time parameters: Configure via --bedrock-inference-params (startSec, useFixedLengthSec)
    • Single embedding approach: Processes one clip for query simplicity
    • Requires --async-output-s3-uri and --src-bucket-owner parameters
    • Processing time: ~60-120 seconds for video/audio queries
    
    \b
    FILTERING:
    • Use JSON format with AWS S3 Vectors API operators
    • Single condition: --filter '{"category": {"$eq": "documentation"}}'
    • Multiple conditions (AND): --filter '{"$and": [{"category": "docs"}, {"version": "1.0"}]}'
    • Multiple conditions (OR): --filter '{"$or": [{"category": "docs"}, {"category": "guides"}]}'
    • Filter by content type: --filter '{"S3VECTORS-EMBED-INPUT-TYPE": "video"}'
    • Filter by embedding option: --filter '{"S3VECTORS-EMBED-OPTION": "visual-image"}'
    
    \b
    EXAMPLES:
    
    # Text query (preferred method)
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --text-value "search text" --k 10
    
    # Text file query
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --text ./query.txt --k 5
    
    # Image query
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-image-v1 --image ./query-image.jpg --k 3
    
    # Video query with defaults (0-5 second clip, visual-text embedding)
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id twelvelabs.marengo-embed-2-7-v1:0 --video ./query-video.mp4 \\
      --bedrock-inference-params '{"modelInput": {"embeddingOption": ["visual-text"]}}' \\
      --async-output-s3-uri s3://my-async-bucket --src-bucket-owner 123456789012
    
    # Video query with custom time range and audio embedding
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id twelvelabs.marengo-embed-2-7-v1:0 --video ./query-video.mp4 \\
      --bedrock-inference-params '{"modelInput": {"embeddingOption": ["audio"], "startSec": 30.0, "useFixedLengthSec": 8}}' \\
      --async-output-s3-uri s3://my-async-bucket --src-bucket-owner 123456789012
    
    # Audio query with custom time range
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id twelvelabs.marengo-embed-2-7-v1:0 --audio ./query-audio.wav \\
      --bedrock-inference-params '{"modelInput": {"startSec": 15.0, "useFixedLengthSec": 6}}' \\
      --async-output-s3-uri s3://my-async-bucket --src-bucket-owner 123456789012
    
    # S3 video query with visual-image embedding
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id twelvelabs.marengo-embed-2-7-v1:0 --video s3://my-bucket/query.mp4 \\
      --bedrock-inference-params '{"modelInput": {"embeddingOption": ["visual-image"]}}' \\
      --async-output-s3-uri s3://my-async-bucket --src-bucket-owner 123456789012
    
    # TwelveLabs cross-modal text search (existing functionality)
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id twelvelabs.marengo-embed-2-7-v1:0 --text-value "red sports car chase" \\
      --async-output-s3-uri s3://my-async-bucket --src-bucket-owner 123456789012 --k 5
    
    # Query with filtering for video content only
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id twelvelabs.marengo-embed-2-7-v1:0 --text-value "police sirens" \\
      --async-output-s3-uri s3://my-async-bucket --src-bucket-owner 123456789012 \\
      --filter '{"S3VECTORS-EMBED-INPUT-TYPE": "video"}' --return-distance
    
    \b
    BACKWARD COMPATIBILITY:
    • --query-input parameter is deprecated but still supported
    • Use specific input types (--text-value, --text, --image, --video, --audio) for better clarity
    • --query-input will be removed in a future version
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
    
    try:
        # Initialize services
        bedrock_service = BedrockService(session, region, debug=debug, console=console)
        s3vector_service = S3VectorService(session, region, debug=debug, console=console)
        
        # Get index dimensions first
        dimensions = _get_index_dimensions(s3vector_service, vector_bucket_name, index_name, console, debug)
        
        # Validate and determine input type
        input_type, input_value = _validate_query_inputs(query_input, text_value, text, image, video, audio)
        
        # Check if this is a TwelveLabs model
        is_twelvelabs_model = bedrock_service.is_async_model(model_id)
        
        # Process input based on type and model
        if is_twelvelabs_model:
            # Early validation for TwelveLabs models
            if not async_output_s3_uri:
                raise click.ClickException(
                    "TwelveLabs models require --async-output-s3-uri parameter. "
                    "Please provide an S3 URI for async processing results."
                )
            
            # Validate TwelveLabs-specific parameters for video/audio queries
            if input_type in ["video", "audio"]:
                embedding_options, start_sec, use_fixed_length_sec = _validate_twelvelabs_query_parameters(
                    input_type, user_bedrock_params, async_output_s3_uri, src_bucket_owner
                )
            
            # TwelveLabs model processing
            if input_type in ["text_value", "legacy"]:
                # Text queries for TwelveLabs (existing implementation)
                query_embedding, job_id = _process_twelvelabs_text_query(
                    input_value, bedrock_service, async_output_s3_uri, src_bucket_owner, console, debug, model_id, session, user_bedrock_params
                )
                content_type = "text"
            elif input_type == "text":
                # Text file queries for TwelveLabs
                if input_value.startswith('s3://'):
                    # S3 text file
                    _, text_content = _process_s3_query_input(input_value, src_bucket_owner, session, region, debug, console)
                else:
                    # Local text file
                    try:
                        with open(input_value, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                    except FileNotFoundError:
                        raise click.ClickException(f"Text file not found: {input_value}")
                    except Exception as e:
                        raise click.ClickException(f"Error reading text file {input_value}: {str(e)}")
                
                query_embedding, job_id = _process_twelvelabs_text_query(
                    text_content, bedrock_service, async_output_s3_uri, src_bucket_owner, console, debug, model_id, session, user_bedrock_params
                )
                content_type = "text"
            elif input_type == "video":
                query_embedding, content_type, job_id = _process_video_file_input(
                    input_value, bedrock_service, model_id, async_output_s3_uri, 
                    src_bucket_owner, session, console, debug, user_bedrock_params
                )
            elif input_type == "audio":
                query_embedding, content_type, job_id = _process_audio_file_input(
                    input_value, bedrock_service, model_id, async_output_s3_uri, 
                    src_bucket_owner, session, console, debug, user_bedrock_params
                )
            elif input_type == "image":
                # Use the same media processing approach as video/audio
                query_embedding, content_type, job_id = _process_media_file_input(
                    input_value, bedrock_service, model_id, "image", async_output_s3_uri, 
                    src_bucket_owner, session, console, debug, user_bedrock_params
                )
            else:
                raise click.ClickException(
                    f"Input type '{input_type}' not supported for TwelveLabs models. "
                    f"Supported types: --text-value, --text, --image, --video, --audio"
                )
        else:
            # Standard model processing (no job ID for sync models)
            job_id = None
            if input_type == "text_value":
                query_embedding, content_type = _process_text_value_input(
                    input_value, bedrock_service, model_id, dimensions, user_bedrock_params
                )
            elif input_type == "text":
                query_embedding, content_type = _process_text_file_input(
                    input_value, bedrock_service, model_id, dimensions, src_bucket_owner, session, region, debug, console, user_bedrock_params
                )
            elif input_type == "image":
                query_embedding, content_type = _process_image_file_input(
                    input_value, bedrock_service, model_id, dimensions, src_bucket_owner, session, region, debug, console, user_bedrock_params
                )
            elif input_type == "legacy":
                # Backward compatibility - use existing logic
                query_embedding, content_type = _process_standard_query(
                    input_value, bedrock_service, model_id, dimensions, src_bucket_owner, session, region, debug, console, user_bedrock_params
                )
            elif input_type in ["video", "audio"]:
                raise click.ClickException(
                    f"Input type '{input_type}' is only supported with TwelveLabs models. "
                    f"Use --model-id twelvelabs.marengo-embed-2-7-v1:0 for video/audio queries."
                )
            else:
                raise click.ClickException(f"Unsupported input type: {input_type}")

        # Search vectors
        results = s3vector_service.query_vectors(
            bucket_name=vector_bucket_name,
            index_name=index_name,
            query_embedding=query_embedding,
            k=k,
            filter_expr=filter_expr,
            return_metadata=return_metadata,  # Pass the CLI parameter to service
            return_distance=return_distance  # Pass the CLI parameter to service
        )
        
        # Display results
        if not results:
            if output == 'json':
                console.print_json(data={"results": [], "summary": {"resultsFound": 0}})
            else:
                console.print("[yellow]No matching vectors found.[/yellow]")
            return
        
        if output == 'json':
            # JSON output
            json_results = []
            for result in results:
                json_result = {
                    "key": result['vectorId'],
                }
                
                if return_distance:
                    json_result["distance"] = result['similarity']
                
                if return_metadata and result.get('metadata'):
                    json_result["metadata"] = result['metadata']
                
                json_results.append(json_result)
            
            # Summary
            summary = {
                "queryType": content_type,
                "model": model_id,
                "index": index_name,
                "resultsFound": len(results),
                "queryDimensions": len(query_embedding)
            }
            
            # Add TwelveLabs-specific information
            if is_twelvelabs_model:
                summary["processingType"] = "async"
                summary["crossModalSearch"] = True
                if job_id:
                    summary["jobId"] = job_id
            
            output_data = {
                "results": json_results,
                "summary": summary
            }
            
            console.print_json(data=output_data)
        else:
            # Table output (default)
            console.print(f"\n[green]Found {len(results)} matching vectors:[/green]\n")
            
            for i, result in enumerate(results, 1):
                table = Table(title=f"Result #{i}")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="white")
                
                table.add_row("Key", result['vectorId'])
                
                if return_distance:
                    table.add_row("Distance", f"{result['similarity']:.4f}")
                
                # Show metadata if available and requested
                metadata = result.get('metadata', {})
                if return_metadata and metadata:
                    for key, value in metadata.items():
                        table.add_row(f"Metadata: {key}", str(value))
                elif return_metadata:
                    table.add_row("Metadata", "[dim]No metadata available[/dim]")
                
                console.print(table)
                console.print()
            
            # Summary
            summary_table = Table(title="Query Summary")
            summary_table.add_column("Property", style="cyan")
            summary_table.add_column("Value", style="green")
            
            summary_table.add_row("Query Type", content_type)
            summary_table.add_row("Model", model_id)
            summary_table.add_row("Index", index_name)
            summary_table.add_row("Results Found", str(len(results)))
            summary_table.add_row("Query Dimensions", str(len(query_embedding)))
            
            # Add TwelveLabs-specific information
            if is_twelvelabs_model:
                summary_table.add_row("Processing Type", "Async (TwelveLabs)")
                summary_table.add_row("Cross-Modal Search", "Enabled")
            
            console.print(summary_table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise click.ClickException(str(e))


def _process_s3_query_input(s3_uri: str, bucket_owner: Optional[str], session, region: str, debug: bool, console) -> tuple:
    """Process S3 query input and return content type and content."""
    try:
        # Parse S3 URI
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        
        s3_path = s3_uri[5:]  # Remove 's3://'
        if '/' not in s3_path:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        
        bucket, key = s3_path.split('/', 1)
        
        if debug:
            console.print(f"[dim]Processing S3 file: bucket={bucket}, key={key}[/dim]")
        
        # Determine content type from extension
        extension = Path(key).suffix.lower()
        
        # Initialize S3 client
        s3_client = session.client('s3', region_name=region)
        
        if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            # Image file
            content = _read_s3_image_file(s3_client, bucket, key, bucket_owner)
            return "image", content
        else:
            # Text file
            content = _read_s3_text_file(s3_client, bucket, key, bucket_owner)
            return "text", content
            
    except Exception as e:
        raise ValueError(f"Failed to process S3 query input {s3_uri}: {str(e)}")


def _read_s3_text_file(s3_client, bucket: str, key: str, bucket_owner: Optional[str] = None) -> str:
    """Read text content from S3 file."""
    get_params = {'Bucket': bucket, 'Key': key}
    if bucket_owner:
        get_params['ExpectedBucketOwner'] = bucket_owner
    
    try:
        response = s3_client.get_object(**get_params)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to read S3 text file s3://{bucket}/{key}: {str(e)}")


def _read_s3_image_file(s3_client, bucket: str, key: str, bucket_owner: Optional[str] = None) -> bytes:
    """Read image content from S3 file and return as bytes."""
    get_params = {'Bucket': bucket, 'Key': key}
    if bucket_owner:
        get_params['ExpectedBucketOwner'] = bucket_owner
    
    try:
        response = s3_client.get_object(**get_params)
        return response['Body'].read()
    except Exception as e:
        raise ValueError(f"Failed to read S3 image file s3://{bucket}/{key}: {str(e)}")
