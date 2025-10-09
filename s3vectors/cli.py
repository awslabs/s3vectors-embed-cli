#!/usr/bin/env python3
"""Main CLI entry point for S3 Vectors."""

import click
from rich.console import Console
from rich.traceback import install

from s3vectors.commands.embed_put import embed_put
from s3vectors.commands.embed_query import embed_query
from s3vectors.utils.config import setup_aws_session

# Install rich traceback handler
install(show_locals=True)
console = Console()

@click.group()
@click.version_option(version="0.2.0")
@click.option('--profile', help='AWS profile to use')
@click.option('--region', help='AWS region to use')
@click.option('--debug', is_flag=True, help='Enable debug mode with detailed logging')
@click.pass_context
def cli(ctx, profile, region, debug):
    """S3 Vectors Embed CLI - Standalone tool for vector embedding operations with S3 and Bedrock."""
    ctx.ensure_object(dict)
    ctx.obj['aws_session'] = setup_aws_session(profile, region)
    ctx.obj['console'] = console
    ctx.obj['debug'] = debug
    
    if debug:
        console.print("[dim] Debug mode enabled[/dim]")
        console.print(f"[dim] AWS Profile: {profile or 'default'}[/dim]")
        console.print(f"[dim] AWS Region: {region or 'from session/config'}[/dim]")

# Register commands as subcommands
cli.add_command(embed_put, name='put')
cli.add_command(embed_query, name='query')

def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise

if __name__ == '__main__':
    main()
