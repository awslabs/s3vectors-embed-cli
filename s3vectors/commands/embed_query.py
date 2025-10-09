import json
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table

from s3vectors.core.unified_processor import UnifiedProcessor
from s3vectors.core.services import BedrockService, S3VectorService
from s3vectors.utils.config import setup_aws_session, get_region, get_current_account_id
from s3vectors.utils.models import get_model_info, validate_model_modality, prepare_processing_input, determine_content_type


def _validate_query_inputs(query_input, text_value, text, image, video, audio, model):
    """Validate query input parameters."""
    inputs = [query_input, text_value, text, image, video, audio]
    provided_inputs = [inp for inp in inputs if inp is not None]
    
    if len(provided_inputs) == 0:
        raise click.ClickException(
            "No query input provided. Use one of: --text-value, --text, --image, --video, or --audio"
        )
    
    # Handle deprecated --query-input parameter
    if query_input:
        raise click.ClickException(
            "--query-input is deprecated and no longer supported. Use --text-value, --text, --image, --video, or --audio instead."
        )
    
    # Special case: Allow multimodal input for supported models
    is_multimodal_input = (model.supports_multimodal_input() and 
                          text_value and image and not text and not video and not audio)
    
    if len(provided_inputs) > 1 and not is_multimodal_input:
        raise click.ClickException(
            "Multiple query inputs provided. Use only one input type, except for multimodal queries with supported models (--text-value + --image)"
        )
    
    return is_multimodal_input


def _format_query_results(results: Dict[str, Any], output_format: str, console: Console):
    """Format and display query results."""
    if output_format == "table":
        _display_results_table(results, console)
    else:
        console.print_json(data=results)


def _display_results_table(results: Dict[str, Any], console: Console):
    """Display query results in table format."""
    table = Table(title="Query Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Vector Key", style="green")
    table.add_column("Distance", style="yellow")
    table.add_column("Metadata", style="blue")
    
    query_results = results.get("results", [])
    for i, result in enumerate(query_results, 1):
        key = result.get("Key", "N/A")
        distance = f"{result.get('distance', 0):.4f}" if result.get('distance') is not None else "N/A"
        metadata = json.dumps(result.get("metadata", {}), indent=2) if result.get("metadata") else "None"
        
        table.add_row(str(i), key, distance, metadata)
    
    console.print(table)
    
    # Display summary
    summary = results.get("summary", {})
    console.print(f"\nQuery Summary:")
    console.print(f"  Model: {summary.get('model', 'N/A')}")
    console.print(f"  Results Found: {summary.get('resultsFound', 0)}")
    console.print(f"  Query Dimensions: {summary.get('queryDimensions', 'N/A')}")


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
@click.option('--async-output-s3-uri', help='S3 URI for async output (required for TwelveLabs models, e.g., s3://my-bucket/path)')
@click.option('--bedrock-inference-params', help='JSON string with model-specific parameters matching Bedrock API format (e.g., \'{"normalize": false}\' for Titan or \'{"input_type": "search_query"}\' for Cohere)')
@click.option('--output', type=click.Choice(['table', 'json']), default='json', help='Output format (default: json)')
@click.option('--region', help='AWS region (overrides session/config defaults)')
@click.pass_context
def embed_query(ctx, vector_bucket_name, index_name, model_id, query_input, text_value, text, image, video, audio,
                       k, filter_expr, return_distance, return_metadata, 
                       src_bucket_owner, async_output_s3_uri, bedrock_inference_params, output, region):
    """Embed query input and search for similar vectors using UnifiedProcessor.
    
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
    TWELVELABS QUERIES:
    • Single embedding approach: Processes one clip for query simplicity
    • Video queries: Require embedding options in --bedrock-inference-params
    • Audio queries: Automatically use audio embedding option
    • Time parameters: Configure via --bedrock-inference-params (startSec, useFixedLengthSec)
    • Requires --async-output-s3-uri and --src-bucket-owner parameters
    • Processing time: ~60-120 seconds for video/audio queries
    
    \b
    FILTERING:
    • Use JSON format with AWS S3 Vectors API operators
    • Single condition: --filter '{"category": {"$eq": "documentation"}}'
    • Multiple conditions (AND): --filter '{"$and": [{"category": "docs"}, {"version": "1.0"}]}'
    • Multiple conditions (OR): --filter '{"$or": [{"category": "docs"}, {"category": "guides"}]}'
    • Comparison operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
    """
    console = Console()
    
    try:
        # Get model information first for validation
        model = get_model_info(model_id)
        if not model:
            raise click.ClickException(f"Unsupported model: {model_id}")
        
        # Validate inputs
        is_multimodal = _validate_query_inputs(query_input, text_value, text, image, video, audio, model)
        
        # Determine content type for model validation
        content_type = determine_content_type(text_value, text, image, video, audio, is_multimodal)
        
        # Validate model capabilities
        if is_multimodal:
            if not model.supports_multimodal_input():
                raise click.ClickException(f"Model {model_id} does not support multimodal input (text + image)")
        else:
            validate_model_modality(model_id, content_type)
        
        # Validate async model requirements
        if model.is_async() and not async_output_s3_uri:
            raise click.ClickException(
                f"Async models like {model.model_id} require --async-output-s3-uri parameter."
            )
        
        # Parse user parameters
        user_bedrock_params = {}
        if bedrock_inference_params:
            try:
                user_bedrock_params = json.loads(bedrock_inference_params)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"Invalid JSON in --bedrock-inference-params: {e}")
        
        # Parse filter expression
        metadata_filter = None
        if filter_expr:
            try:
                metadata_filter = json.loads(filter_expr)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"Invalid JSON in --filter: {e}")
        
        # Setup AWS session and services
        session = setup_aws_session(profile=None, region=region)
        region = get_region(session, region)
        
        # Auto-assign account ID if not provided
        if not src_bucket_owner:
            src_bucket_owner = get_current_account_id(session)
        
        bedrock_service = BedrockService(session, region, debug=ctx.obj.get('debug', False), console=console)
        s3vector_service = S3VectorService(session, region, debug=ctx.obj.get('debug', False), console=console)
        
        # Create UnifiedProcessor
        processor = UnifiedProcessor(bedrock_service, s3vector_service, session)
        
        # Fetch index dimensions once at the top level (same pattern as PUT)
        try:
            index_info = s3vector_service.get_index(vector_bucket_name, index_name)
            index_dimensions = index_info.get("index", {}).get("dimension")
            if not index_dimensions:
                raise click.ClickException(f"Could not determine dimensions for index {index_name}")
        except Exception as e:
            raise click.ClickException(f"Failed to get index information: {str(e)}")
        
        # Create ProcessingInput
        processing_input = prepare_processing_input(text_value, text, image, video, audio, is_multimodal)
        
        # Process query input to generate embedding (same as PUT)
        with console.status("[bold green]Generating query embedding..."):
            # Validate TwelveLabs query parameters for video/audio
            if model.is_async() and processing_input.content_type in ["video", "audio"]:
                if not user_bedrock_params:
                    user_bedrock_params = {}
                
                # Validate required parameters for video/audio queries
                if "startSec" not in user_bedrock_params or "lengthSec" not in user_bedrock_params:
                    raise click.ClickException('Both start time (startSec) and length (lengthSec) are required in --bedrock-inference-params for video/audio queries. Example: --bedrock-inference-params \'{"startSec": 30.0, "lengthSec": 6.0, "embeddingOption": ["visual-text"]}\'')
                
                # Validate embeddingOption for video queries (audio auto-selects)
                if processing_input.content_type == "video" and "embeddingOption" not in user_bedrock_params:
                    raise click.ClickException('embeddingOption is required for video queries. Specify exactly one: ["visual-text"], ["visual-image"], or ["audio"]. Example: --bedrock-inference-params \'{"startSec": 30.0, "lengthSec": 6.0, "embeddingOption": ["visual-text"]}\'')
                
                # Validate embeddingOption has exactly one value for video queries
                if processing_input.content_type == "video" and "embeddingOption" in user_bedrock_params:
                    embedding_options = user_bedrock_params["embeddingOption"]
                    if not isinstance(embedding_options, list) or len(embedding_options) != 1:
                        raise click.ClickException('embeddingOption must contain exactly one value for video queries. Example: --bedrock-inference-params \'{"embeddingOption": ["visual-text"]}\'')
                
                # Calculate useFixedLengthSec from lengthSec if not explicitly provided
                if "useFixedLengthSec" not in user_bedrock_params:
                    user_bedrock_params["useFixedLengthSec"] = user_bedrock_params["lengthSec"]
                
                # Validate useFixedLengthSec range
                use_fixed_length = user_bedrock_params.get("useFixedLengthSec")
                if use_fixed_length is not None and (use_fixed_length < 2 or use_fixed_length > 10):
                    raise click.ClickException(f"Length of the clip must be between 2-10 seconds, got: {use_fixed_length}")
            
            result = processor.process(
                model=model,
                processing_input=processing_input,
                user_bedrock_params=user_bedrock_params,
                async_output_s3_uri=async_output_s3_uri,
                src_bucket_owner=src_bucket_owner,
                precomputed_dimensions=index_dimensions
            )
        
        # Extract query embedding (TwelveLabs returns multiple, use first for query)
        query_timing = {}  # Store timing info for summary
        
        # For TwelveLabs, capture timing info from raw result before vector processing
        if model.is_async() and processing_input.content_type in ["video", "audio"] and hasattr(result, 'raw_results'):
            if result.raw_results and len(result.raw_results) > 0:
                first_raw = result.raw_results[0]
                if "startSec" in first_raw:
                    query_timing["queryStartSec"] = first_raw["startSec"]
                if "endSec" in first_raw:
                    query_timing["queryEndSec"] = first_raw["endSec"]
        
        if hasattr(result, 'vectors') and result.vectors:
            # Get the embedding from the first vector
            first_vector = result.vectors[0]
            if "embedding" in first_vector:
                query_embedding = first_vector["embedding"]
            elif "data" in first_vector and "float32" in first_vector["data"]:
                query_embedding = first_vector["data"]["float32"]
            else:
                raise click.ClickException(f"No embedding found in result. Available keys: {list(first_vector.keys())}")
        else:
            raise click.ClickException("Failed to generate query embedding - no vectors returned")
        
        # Perform vector similarity search
        with console.status("[bold green]Searching for similar vectors..."):
            search_results = s3vector_service.query_vectors(
                bucket_name=vector_bucket_name,
                index_name=index_name,
                query_embedding=query_embedding,
                k=k,
                filter_expr=json.dumps(metadata_filter) if metadata_filter else None,
                return_metadata=return_metadata,
                return_distance=return_distance
            )
        
        # Format results
        formatted_results = {
            "results": [
                {
                    "Key": result.get("vectorId", ""),
                    "distance": result.get("similarity", 0.0),
                    "metadata": result.get("metadata", {})
                }
                for result in search_results
            ],
            "summary": {
                "queryType": content_type,
                "model": model_id,
                "index": index_name,
                "resultsFound": len(search_results),
                "queryDimensions": len(query_embedding),
                **query_timing  # Add timing info for TwelveLabs queries
            }
        }
        
        # Add distances if requested (already included in results)
        if not return_distance:
            for result in formatted_results["results"]:
                result.pop("distance", None)
        
        # Display results
        _format_query_results(formatted_results, output, console)
        
    except Exception as e:
        raise click.ClickException(str(e))
