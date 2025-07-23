"""Boto3 configuration utilities for S3 Vectors CLI."""

from botocore.config import Config
from s3vectors.__version__ import __version__


def get_user_agent():
    """
    Get the CLI user agent string for logging/debugging.
    
    Returns:
        User agent string for display
    """
    return f"s3vectors-embed-cli/{__version__}"


def get_boto_config():
    """
    Get boto3 Config object with CLI user agent tracking.
    
    Returns:
        botocore.config.Config with custom user agent
    """
    return Config(
        user_agent_extra=get_user_agent()
    )
