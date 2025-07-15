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
@click.option('--query-input', required=True, help='Query text or file path (local file or S3 URI)')
@click.option('--k', default=5, type=int, help='Number of results to return (default: 5)')
@click.option('--filter', 'filter_expr', help='Filter expression for results (JSON format with operators, e.g., \'{"$and": [{"category": "docs"}, {"version": "1.0"}]}\')')
@click.option('--return-distance', is_flag=True, help='Return similarity distances in results')
@click.option('--return-metadata/--no-return-metadata', default=True, help='Return metadata in results (default: true)')
@click.option('--src-bucket-owner', help='Source bucket owner AWS account ID for cross-account S3 access')
@click.option('--output', type=click.Choice(['table', 'json']), default='json', help='Output format (default: json)')
@click.option('--region', help='AWS region (overrides session/config defaults)')
@click.pass_context
def embed_query(ctx, vector_bucket_name, index_name, model_id, query_input, 
                k, filter_expr, return_distance, return_metadata, src_bucket_owner, output, region):
    """Embed query input and search for similar vectors in S3.
    
    \b
    SUPPORTED QUERY TYPES:
    • Direct text: --query-input "search for this text"
    • Local text file: --query-input /path/to/query.txt
    • Local image file: --query-input /path/to/image.jpg
    • S3 text file: --query-input s3://bucket/query.txt
    • S3 image file: --query-input s3://bucket/image.jpg
    
    \b
    SUPPORTED MODELS:
    • amazon.titan-embed-text-v2:0 (text queries)
    • amazon.titan-embed-text-v1 (text queries)
    • amazon.titan-embed-image-v1 (text and image queries)
    • cohere.embed-english-v3 (text queries)
    • cohere.embed-multilingual-v3 (text queries)

    
    \b
    FILTERING:
    • Use JSON format with AWS S3 Vectors API operators
    • Single condition: --filter '{"category": {"$eq": "documentation"}}'
    • Multiple conditions (AND): --filter '{"$and": [{"category": "docs"}, {"version": "1.0"}]}'
    • Multiple conditions (OR): --filter '{"$or": [{"category": "docs"}, {"category": "guides"}]}'
    • Comparison operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
    • Logical operators: $and, $or, $not
    • Supports exact matches and basic comparisons
    
    \b
    OUTPUT FORMATS:
    • JSON (default): Machine-readable, complete metadata
    • Table: Human-readable, formatted display
    
    \b
    EXAMPLES:
    # Basic text query
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --query-input "search text" --k 10
    
    # Query with metadata filtering
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --query-input "search text" \\
      --filter '{"category": "docs"}' --return-distance --k 5
    
    # Image query (Titan Image v1)
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-image-v1 --query-input /path/to/query-image.jpg --k 3
    
    # File-based query
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --query-input query-document.txt
    
    # S3 file query with cross-account access
    s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --query-input s3://other-bucket/query.txt \\
      --src-bucket-owner 123456789012
    
    # Debug mode for troubleshooting
    s3vectors-embed --debug query --vector-bucket-name my-bucket --index-name my-index \\
      --model-id amazon.titan-embed-text-v2:0 --query-input "debug query"
    """
    
    console = ctx.obj['console']
    session = ctx.obj['aws_session']
    debug = ctx.obj.get('debug', False)
    region = get_region(session, region)
    
    try:
        # Initialize services
        bedrock_service = BedrockService(session, region, debug=debug, console=console)
        s3vector_service = S3VectorService(session, region, debug=debug, console=console)
        
        # Get index dimensions first
        dimensions = _get_index_dimensions(s3vector_service, vector_bucket_name, index_name, console, debug)
        
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
        
        # Generate query embedding
        if content_type == "text":
            query_embedding = bedrock_service.embed_text(model_id, content, dimensions=dimensions)
        else:
            # For images, convert bytes to base64 string if needed
            if isinstance(content, bytes):
                content = base64.b64encode(content).decode('utf-8')
            query_embedding = bedrock_service.embed_image(model_id, content, dimensions=dimensions)

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
