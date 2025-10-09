"""AWS configuration utilities."""

import os
import boto3
from botocore.exceptions import NoCredentialsError, ProfileNotFound
import click

def get_current_account_id(session=None) -> str:
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

def setup_aws_session(profile=None, region=None):
    """Set up AWS session with optional profile and region."""
    try:
        if profile:
            session = boto3.Session(profile_name=profile)
        else:
            session = boto3.Session()
        
        # Test credentials
        sts = session.client('sts')
        sts.get_caller_identity()
        
        return session
    except ProfileNotFound:
        raise click.ClickException(f"AWS profile '{profile}' not found")
    except NoCredentialsError:
        raise click.ClickException(
            "AWS credentials not found. Please configure your credentials using:\n"
            "  aws configure\n"
            "or set environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
        )
    except Exception as e:
        raise click.ClickException(f"Failed to setup AWS session: {str(e)}")

def get_region(session, region=None):
    """Get AWS region from session or parameter."""
    if region:
        return region
    
    # Try to get region from session
    session_region = session.region_name
    if session_region:
        return session_region
    
    # Default to us-east-1
    return 'us-east-1'
