"""User agent utilities for S3 Vectors CLI tracking."""

from botocore.config import Config
from s3vectors.__version__ import __version__


def get_s3vectors_user_agent_config():
    """
    Get boto3 Config object with S3 Vectors CLI user agent tracking.
    
    Returns:
        botocore.config.Config with custom user agent
    """
    return Config(
        user_agent_extra=f"s3vectors-embed-cli/{__version__}",
        retries={'max_attempts': 3}
    )


def get_s3vectors_user_agent_string():
    """
    Get the S3 Vectors CLI user agent string for logging/debugging.
    
    Returns:
        User agent string for display
    """
    return f"s3vectors-embed-cli/{__version__}"
