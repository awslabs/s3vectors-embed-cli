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


def _generate_embedding_with_progress(progress, bedrock_service, model_id, content, 
                                    dimensions, is_image=False, task_description="Generating embedding..."):
    """Generate embedding with progress tracking."""
    if is_image:
        embedding = bedrock_service.embed_image(model_id, content, dimensions=dimensions)
    else:
        embedding = bedrock_service.embed_text(model_id, content, dimensions=dimensions)
    return embedding


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
@click.option('--model-id', required=True, help='Bedrock embedding model ID (e.g., amazon.titan-embed-text-v2:0, amazon.titan-embed-image-v1, cohere.embed-english-v3)')
@click.option('--text-value', help='Direct text input to embed')
@click.option('--text', help='Text file path (local file, S3 URI, or S3 wildcard pattern like s3://bucket/folder/*.txt)')
@click.option('--image', help='Image file path (local file, S3 URI, or S3 wildcard pattern like s3://bucket/images/*.jpg)')
@click.option('--key', help='Custom vector key (auto-generated UUID if not provided)')
@click.option('--metadata', help='JSON metadata to attach to the vector (e.g., \'{"category": "docs", "version": "1.0"}\')')
@click.option('--src-bucket-owner', help='Source bucket owner AWS account ID for cross-account S3 access')
@click.option('--use-object-key-name', is_flag=True, help='Use S3 object key as vector key for batch processing')
@click.option('--max-workers', default=4, type=int, help='Maximum parallel workers for batch processing (default: 4)')
@click.option('--output', type=click.Choice(['table', 'json']), default='json', help='Output format (default: json)')
@click.option('--region', help='AWS region (overrides session/config defaults)')
@click.pass_context
def embed_put(ctx, vector_bucket_name, index_name, model_id, text_value, text, image,
              key, metadata, src_bucket_owner, use_object_key_name,
              max_workers, output, region):
    """Embed text or image content and store as vectors in S3.
    
    \b
    SUPPORTED INPUT TYPES:
    • Direct text: --text-value "your text here"
    • Local files: --text /path/to/file.txt or --image /path/to/image.jpg
    • S3 files: --text s3://bucket/file.txt or --image s3://bucket/image.jpg
    • S3 wildcards: --text "s3://bucket/folder/*.txt" or --image "s3://bucket/images/*.jpg"
    • Multimodal: --text-value "description" --image /path/to/image.jpg (Titan Image v1 only)
    
    \b
    SUPPORTED MODELS:
    • amazon.titan-embed-text-v2:0 (1024, 512, 256 dimensions)
    • amazon.titan-embed-text-v1 (1536 dimensions, fixed)
    • amazon.titan-embed-image-v1 (1024, 384, 256 dimensions, supports text+image)
    • cohere.embed-english-v3 (1024 dimensions, fixed)
    • cohere.embed-multilingual-v3 (1024 dimensions, fixed)
     
    
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
    
    # Local file with custom key
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --text document.txt --key "doc-001"
    
    # S3 batch processing
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --text "s3://source-bucket/docs/*.txt" \\
      --metadata '{"category": "documentation"}' --max-workers 4
    
    # Multimodal embedding (Titan Image v1)
    s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-image-v1 --text-value "A red car" --image car.jpg
    
    # Debug mode for troubleshooting
    s3vectors-embed --debug put --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --text-value "Debug test"
    """
    
    console = ctx.obj['console']
    session = ctx.obj['aws_session']
    debug = ctx.obj.get('debug', False)
    region = get_region(session, region)
    
    # Validate input - at least one input must be provided
    inputs_provided = sum(bool(x) for x in [text_value, text, image])
    if inputs_provided == 0:
        raise click.ClickException("At least one input must be provided: --text-value, --text, or --image")
    
    # Special case: Allow multimodal input (text-value + image) for Titan Image v1 model
    is_multimodal_titan = (model_id.startswith('amazon.titan-embed-image') and 
                          text_value and image and not text)
    
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
                bedrock_service, s3vector_service, s3_client, metadata_dict, key, console
            )
        elif text_value:
            result = _process_text_value(
                text_value, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, metadata_dict, key, console
            )
        elif text:
            result = _process_text_input(
                text, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, s3_client, metadata_dict,
                src_bucket_owner, use_object_key_name, max_workers,
                key, console, region
            )
        elif image:
            result = _process_image_input(
                image, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, s3_client, metadata_dict,
                src_bucket_owner, use_object_key_name, max_workers,
                key, console, region
            )
        
        # Display results
        _display_results(result, output, console)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise click.ClickException(str(e))


def _process_multimodal_input(text_value, image_path, vector_bucket_name, index_name, model_id,
                             bedrock_service, s3vector_service, s3_client, metadata_dict, key, console):
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
                
                bucket, key = path_part.split('/', 1)
                
                # Read from S3
                response = s3_client.get_object(Bucket=bucket, Key=key)
                image_data = base64.b64encode(response['Body'].read()).decode('utf-8')
            else:
                # Read local file
                with open(image_path, 'rb') as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    
            progress.update(image_task, description="Image processed ✓")
        except Exception as e:
            raise click.ClickException(f"Failed to read image file: {str(e)}")
        
        # Generate multimodal embedding
        embedding = bedrock_service.embed_image(model_id, image_data, text_input=text_value, dimensions=dimensions)
        embed_task = progress.add_task("Generating multimodal embedding...", total=None)
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
                       bedrock_service, s3vector_service, metadata_dict, key, console):
    """Process direct text value input."""
    
    # Get index dimensions first
    dimensions = _get_index_dimensions(s3vector_service, vector_bucket_name, index_name, console, debug=bedrock_service.debug)
    
    with _create_progress_context(console) as progress:
        
        # Generate embedding with dimensions
        embedding = _generate_embedding_with_progress(progress, bedrock_service, model_id, 
                                                    text_value, dimensions)
        
        # Prepare metadata - add standard fields for direct text
        metadata_dict.update({
            'S3VECTORS-EMBED-SRC-CONTENT': text_value
        })
        
        # Generate vector ID if not provided
        vector_key = _generate_vector_id_if_needed(key)
        
        # Store vector
        _store_vector_with_progress(progress, s3vector_service, vector_bucket_name, index_name, 
                                  vector_key, embedding, metadata_dict, "Storing vector in S3...")
    
    return _create_result_dict(vector_key, vector_bucket_name, index_name, model_id, 
                              'text', embedding, metadata_dict)


def _process_file_input(file_input, vector_bucket_name, index_name, model_id,
                       bedrock_service, s3vector_service, s3_client, metadata_dict,
                       src_bucket_owner, use_object_key_name, max_workers,
                       vector_id, console, region, is_image=False):
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
                use_object_key_name, max_workers, console
            )
        else:
            # Single file processing
            return _process_single_file(
                processed_input, vector_bucket_name, index_name, model_id,
                bedrock_service, s3vector_service, metadata_dict, vector_id, console, is_image
            )
            
    except Exception as e:
        input_type = "image" if is_image else "text"
        raise click.ClickException(f"Failed to process {input_type} input: {str(e)}")


def _process_text_input(text_input, vector_bucket_name, index_name, model_id,
                       bedrock_service, s3vector_service, s3_client, metadata_dict,
                       src_bucket_owner, use_object_key_name, max_workers,
                       vector_id, console, region):
    """Process text input (file or S3 URI or wildcard)."""
    return _process_file_input(text_input, vector_bucket_name, index_name, model_id,
                              bedrock_service, s3vector_service, s3_client, metadata_dict,
                              src_bucket_owner, use_object_key_name, max_workers,
                              vector_id, console, region, is_image=False)


def _process_image_input(image_input, vector_bucket_name, index_name, model_id,
                        bedrock_service, s3vector_service, s3_client, metadata_dict,
                        src_bucket_owner, use_object_key_name, max_workers,
                        vector_id, console, region):
    """Process image input (file or S3 URI or wildcard)."""
    return _process_file_input(image_input, vector_bucket_name, index_name, model_id,
                              bedrock_service, s3vector_service, s3_client, metadata_dict,
                              src_bucket_owner, use_object_key_name, max_workers,
                              vector_id, console, region, is_image=True)


def _process_single_file(processed_input, vector_bucket_name, index_name, model_id,
                        bedrock_service, s3vector_service, metadata_dict, key, console, is_image=False):
    """Process single file (text or image)."""
    
    # Get index dimensions first
    dimensions = _get_index_dimensions(s3vector_service, vector_bucket_name, index_name, console, debug=bedrock_service.debug)
    
    with _create_progress_context(console) as progress:
        
        # Generate embedding with dimensions
        embedding = _generate_embedding_with_progress(progress, bedrock_service, model_id, 
                                                    processed_input['content'], dimensions, is_image)
        
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
                  use_object_key_name, max_workers, console):
    """Process batch wildcard input (S3 or local filesystem)."""
    
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


def _display_results(result, output_format, console):
    """Display results in the specified format."""
    if output_format == 'json':
        # JSON output
        if result['type'] == 'single':
            json_data = {
                "key": result['key'],
                "bucket": result['bucket'],
                "index": result['index'],
                "model": result['model'],
                "contentType": result['contentType'],
                "embeddingDimensions": result['embeddingDimensions'],
                "metadata": result['metadata']
            }
        else:  # batch
            json_data = {
                "Keys": result['keys'],
                "status": result['status'],
                "pattern": result['pattern'],
                "processed_count": result['processed_count'],
                "bucket": result['bucket'],
                "index": result['index'],
                "model": result['model']
            }
        
        console.print_json(data=json_data)
    else:
        # Table output
        if result['type'] == 'single':
            table = Table(title="Vector Storage Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Key", result['key'])
            table.add_row("Bucket", result['bucket'])
            table.add_row("Index", result['index'])
            table.add_row("Model", result['model'])
            table.add_row("Content Type", result['contentType'])
            table.add_row("Embedding Dimensions", str(result['embeddingDimensions']))
            
            console.print(table)
            console.print(f"\n[green]✓ Successfully stored vector with key: {result['key']}[/green]")
        else:  # batch
            table = Table(title="Batch Processing Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Pattern", result['pattern'])
            table.add_row("Status", result['status'])
            table.add_row("Processed Count", str(result['processed_count']))
            table.add_row("Total Keys", str(len(result['keys'])))
            table.add_row("Bucket", result['bucket'])
            table.add_row("Index", result['index'])
            table.add_row("Model", result['model'])
            
            console.print(table)
            
            if result['status'] == 'success':
                console.print(f"\n[green] Batch processing completed successfully![/green]")
                console.print(f"[green] Processed {result['processed_count']} files[/green]")
            else:
                console.print(f"\n[yellow] Batch processing completed with errors[/yellow]")
                console.print(f"[yellow] Processed: {result['processed_count']} files[/yellow]")
