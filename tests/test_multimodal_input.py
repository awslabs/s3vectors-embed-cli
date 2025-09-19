import io
import json
import boto3
import pytest
from rich.console import Console
from s3vectors.commands.embed_put import embed_put
from contextlib import redirect_stdout, redirect_stderr

buf_out = io.StringIO()
buf_err = io.StringIO()


def format_console_output(buffered_output_string: str):

    # looking at where the dicationary starts
    dictionary_start_idx = buffered_output_string.find('{')

    # strip out dictionary string alone
    buffered_output_dict = buffered_output_string[dictionary_start_idx:]

    # convert back to dictionary and return
    return eval(buffered_output_dict)



@pytest.mark.asyncio
async def test_embed_and_store_image_multimodal():

    # params for put_vector
    bucket_name = 'my-new-s3-vector-bucket'
    index_name = 'text-vector-index'
    model_id = 'amazon.titan-embed-image-v1'
    content_type = 'multimodal'
    dimension_size = 1024  # known at index creation time
    image_path = 'tests/data/sample.png'
    text_value = 'this is a sample image'

    truncated_text_value = "sample image"
    truncated_image_path = "sample.png"

    metadata_config = {
        "S3VECTORS-EMBED-SRC-CONTENT": truncated_text_value,
        "S3VECTORS-EMBED-SRC-LOCATION": truncated_image_path
    }

    # string-ifying for CLI command
    metadata_config_fmt = json.dumps(metadata_config)

    args = [
        "--vector-bucket-name", bucket_name,
        "--index-name", index_name,
        "--model-id", model_id,
        "--text-value", text_value,
        "--image", image_path,
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

    console_output = format_console_output(str(buf_out.getvalue()))

    assert console_output["bucket"] == bucket_name
    assert console_output["index"] == index_name
    assert console_output["model"] == model_id
    assert console_output["contentType"] == content_type
    assert console_output["embeddingDimensions"] == dimension_size
    assert console_output["metadata"]["S3VECTORS-EMBED-SRC-CONTENT"] == truncated_text_value
    assert console_output["metadata"]["S3VECTORS-EMBED-SRC-LOCATION"] == truncated_image_path