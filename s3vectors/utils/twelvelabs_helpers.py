"""TwelveLabs-specific helper utilities for S3 Vectors CLI."""

import os
import json
import base64
import boto3
from pathlib import Path
from typing import Dict, Any, Optional
import click


def _get_current_account_id(session=None) -> str:
    """Get the current AWS account ID using STS."""
    try:
        if session:
            sts = session.client('sts')
        else:
            sts = boto3.client('sts')
        response = sts.get_caller_identity()
        return response['Account']
    except Exception as e:
        raise click.ClickException(f"Failed to get AWS account ID: {str(e)}")


def prepare_media_source(file_path: str, bucket_owner: Optional[str] = None, session=None) -> Dict[str, Any]:
    """
    Prepare mediaSource object for TwelveLabs API.
    
    Args:
        file_path: Local file path or S3 URI
        bucket_owner: AWS account ID for cross-account S3 access
        session: AWS session for account ID detection
        
    Returns:
        Dict containing either base64String or s3Location
    """
    if file_path.startswith('s3://'):
        # S3 location format
        media_source = {
            "s3Location": {
                "uri": file_path
            }
        }
        # TwelveLabs API requires bucketOwner field - auto-detect if not provided
        if bucket_owner:
            media_source["s3Location"]["bucketOwner"] = bucket_owner
        else:
            # Auto-detect current AWS account ID
            media_source["s3Location"]["bucketOwner"] = _get_current_account_id(session)
        return media_source
    else:
        # Local file - convert to base64
        if not os.path.exists(file_path):
            raise click.ClickException(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Check file size limit (36MB for base64)
        file_size_mb = len(file_content) / (1024 * 1024)
        if len(file_content) > 36 * 1024 * 1024:
            raise click.ClickException(f"File too large: {file_size_mb:.1f}MB. Max: 36MB for local files. Consider uploading to S3 first.")
        
        base64_content = base64.b64encode(file_content).decode('utf-8')
        return {"base64String": base64_content}


def validate_twelvelabs_parameters(input_type: str, **kwargs):
    """
    Validate TwelveLabs-specific parameters.
    
    Args:
        input_type: The input type (text, video, audio, image)
        **kwargs: Parameter values to validate
    """
    # Validate useFixedLengthSec
    if "use_fixed_length_sec" in kwargs and kwargs["use_fixed_length_sec"] is not None:
        value = kwargs["use_fixed_length_sec"]
        if not (2 <= value <= 10):
            raise click.ClickException("--use-fixed-length-sec must be between 2 and 10 seconds")
    
    # Validate minClipSec
    if "min_clip_sec" in kwargs and kwargs["min_clip_sec"] is not None:
        value = kwargs["min_clip_sec"]
        if not (1 <= value <= 5):
            raise click.ClickException("--min-clip-sec must be between 1 and 5 seconds")
    
    # Validate embedding options for video
    if input_type == "video" and "embedding_options" in kwargs and kwargs["embedding_options"]:
        valid_options = {"visual-text", "visual-image", "audio"}
        provided_options = set(kwargs["embedding_options"].split(','))
        invalid_options = provided_options - valid_options
        if invalid_options:
            raise click.ClickException(f"Invalid embedding options: {invalid_options}. Valid options: {valid_options}")
    
    # Validate text truncate
    if input_type == "text" and "text_truncate" in kwargs and kwargs["text_truncate"]:
        if kwargs["text_truncate"] not in ["end", "none"]:
            raise click.ClickException("--text-truncate must be 'end' or 'none'")


def add_time_parameters(params: Dict, start_sec: Optional[float], length_sec: Optional[float], 
                       use_fixed_length_sec: Optional[float]):
    """
    Add time-related parameters for video/audio processing.
    
    Args:
        params: Parameter dictionary to update
        start_sec: Start time offset
        length_sec: Duration to process
        use_fixed_length_sec: Fixed duration for each clip
    """
    if start_sec is not None:
        params["startSec"] = start_sec
    if length_sec is not None:
        params["lengthSec"] = length_sec  
    if use_fixed_length_sec is not None:
        params["useFixedLengthSec"] = use_fixed_length_sec


def read_text_file_content(file_path: str) -> str:
    """
    Read text content from local file or S3 URI.
    
    Args:
        file_path: Local file path or S3 URI
        
    Returns:
        Text content as string
    """
    if file_path.startswith('s3://'):
        raise click.ClickException("S3 text files for TwelveLabs should be handled via mediaSource, not inputText")
    
    if not os.path.exists(file_path):
        raise click.ClickException(f"Text file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        raise click.ClickException(f"Unable to read text file (encoding issue): {file_path}")


def create_twelvelabs_metadata(input_type: str, source_location: str, 
                              embedding_data: Dict, clip_index: int = 0) -> Dict[str, Any]:
    """
    Create metadata for TwelveLabs vector storage.
    
    Args:
        input_type: The input type (text, video, audio, image)
        source_location: Original file location
        embedding_data: Embedding response data
        clip_index: Index of the clip (for multi-clip responses)
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        'S3VECTORS-EMBED-SRC-LOCATION': source_location,
        'S3VECTORS-EMBED-MODALITY': input_type
    }
    
    # Add temporal information if available
    if embedding_data.get('startSec') is not None:
        metadata['S3VECTORS-EMBED-START-SEC'] = embedding_data['startSec']
    if embedding_data.get('endSec') is not None:
        metadata['S3VECTORS-EMBED-END-SEC'] = embedding_data['endSec']
    
    # Add embedding type if available
    if embedding_data.get('embeddingOption'):
        metadata['S3VECTORS-EMBED-TYPE'] = embedding_data['embeddingOption']
    
    return metadata
