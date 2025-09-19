import io
import json
import boto3
import pytest
from rich.console import Console
from s3vectors.commands.embed_put import embed_put
from contextlib import redirect_stdout, redirect_stderr

buf_out = io.StringIO()
buf_err = io.StringIO()

@pytest.mark.asyncio
async def test_embed_and_store_text():

    """
    Test for running embedding for a long text supplied via `--text-value` param
    """

    # params for put_vector
    bucket_name = 'my-new-s3-vector-bucket'
    index_name = 'text-vector-index'
    model_id = 'amazon.titan-embed-text-v2:0'
    content_type = 'text'
    dimension_size = 1024  # known at index creation time
    file_path = 'tests/data/long_text.txt'

    # reading a really long text (>> 2048 bytes)
    with open(file_path) as f:
        text_value = f.read()

    # reading only first 10 chars, ensuring that metadata config is < 2048 bytes
    truncated_metadata_content = text_value[:10]
    metadata_config = {"S3VECTORS-EMBED-SRC-CONTENT": truncated_metadata_content}

    # string-ifying for CLI command
    metadata_config_fmt = json.dumps(metadata_config)

    args = [
        "--vector-bucket-name", bucket_name,
        "--index-name", index_name,
        "--model-id", model_id,
        "--text-value", text_value,
        "--metadata", metadata_config_fmt
    ]

    ctx = embed_put.make_context(
        info_name="put",
        args=args,
        obj={
            "console": Console(file=buf_out, force_terminal=False, color_system=None),
            "aws_session": boto3.Session(
                profile_name="default",
                region_name="us-east-1",
            ),
        },
    )

    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        result = embed_put.invoke(ctx)

    console_output = eval(buf_out.getvalue())

    assert console_output["bucket"] == bucket_name
    assert console_output["index"] == index_name
    assert console_output["model"] == model_id
    assert console_output["contentType"] == content_type
    assert console_output["embeddingDimensions"] == dimension_size
    assert console_output["metadata"]["S3VECTORS-EMBED-SRC-CONTENT"] == truncated_metadata_content

@pytest.mark.asyncio
async def test_embed_and_store_text_file():
    """
        Test for running embedding for a long text inside a local text file via `--text` param
        """

    # params for put_vector
    bucket_name = 'my-new-s3-vector-bucket'
    index_name = 'text-vector-index'
    model_id = 'amazon.titan-embed-text-v2:0'
    content_type = 'text'
    dimension_size = 1024  # known at index creation time
    file_path = 'tests/data/long_text.txt'


    # reading a really long text (>> 2048 bytes)
    with open(file_path) as f:
        text_value = f.read()

    # reading only first 10 chars, ensuring that metadata config is < 2048 bytes
    truncated_metadata_content = text_value[:10]

    metadata_config = {"S3VECTORS-EMBED-SRC-CONTENT": truncated_metadata_content}

    # string-ifying for CLI command
    metadata_config_fmt = json.dumps(metadata_config)

    args = [
        "--vector-bucket-name", bucket_name,
        "--index-name", index_name,
        "--model-id", model_id,
        "--text", file_path,
        "--metadata", metadata_config_fmt
    ]

    ctx = embed_put.make_context(
        info_name="put",
        args=args,
        obj={
            "console": Console(file=buf_out, force_terminal=False, color_system=None),
            "aws_session": boto3.Session(
                profile_name="default",
                region_name="us-east-1",
            ),
        },
    )

    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        result = embed_put.invoke(ctx)

    console_output = eval(buf_out.getvalue())

    assert console_output["bucket"] == bucket_name
    assert console_output["index"] == index_name
    assert console_output["model"] == model_id
    assert console_output["contentType"] == content_type
    assert console_output["embeddingDimensions"] == dimension_size
    assert console_output["metadata"]["S3VECTORS-EMBED-SRC-CONTENT"] == truncated_metadata_content


