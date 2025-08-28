"""Common utilities for Bedrock parameter handling and payload building."""

from typing import Dict, Any, Optional
import click


def extract_all_keys(obj, prefix=""):
    """Recursively extract all keys from a nested dictionary."""
    keys = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key)
            if isinstance(value, dict):
                keys.update(extract_all_keys(value, full_key))
    return keys


def validate_s3_vectors_compatibility(user_params, model_id):
    """Validate that user parameters are compatible with S3 Vectors requirements."""
    
    # S3 Vectors only supports float embeddings
    if model_id.startswith('amazon.titan-embed-text-v2'):
        embedding_types = user_params.get('embeddingTypes', ['float'])
        if embedding_types != ['float'] and 'float' not in embedding_types:
            raise click.ClickException(
                "S3 Vectors only supports float embeddings. "
                f"Requested embeddingTypes: {embedding_types}. "
                "Please use 'float' or remove embeddingTypes parameter (defaults to float)."
            )
        if len(embedding_types) > 1:
            raise click.ClickException(
                "S3 Vectors only supports single embedding type. "
                f"Requested embeddingTypes: {embedding_types}. "
                "Please specify only 'float' or remove embeddingTypes parameter."
            )
    
    elif model_id.startswith('cohere.embed'):
        embedding_types = user_params.get('embedding_types', ['float'])
        if embedding_types != ['float'] and 'float' not in embedding_types:
            raise click.ClickException(
                "S3 Vectors only supports float embeddings. "
                f"Requested embedding_types: {embedding_types}. "
                "Please use 'float' or remove embedding_types parameter (defaults to float)."
            )
        if len(embedding_types) > 1:
            raise click.ClickException(
                "S3 Vectors only supports single embedding type. "
                f"Requested embedding_types: {embedding_types}. "
                "Please specify only 'float' or remove embedding_types parameter."
            )


def validate_bedrock_params(system_payload, user_params, model_id, is_query=False):
    """Validate that user parameters don't conflict with system-controlled parameters."""
    if not user_params:
        return
    
    # Validate S3 Vectors compatibility - only float embeddings are supported
    validate_s3_vectors_compatibility(user_params, model_id)
    
    # For TwelveLabs, check conflicts in modelInput
    if model_id.startswith('twelvelabs.'):
        system_model_input = system_payload.get("modelInput", {})
        user_model_input = user_params.get("modelInput", {})
        
        if user_model_input:
            system_keys = extract_all_keys(system_model_input)
            user_keys = extract_all_keys(user_model_input)
            conflicts = system_keys.intersection(user_keys)
            
            if conflicts:
                raise click.ClickException(
                    f"Cannot override system-controlled parameters in modelInput: {sorted(conflicts)}\n"
                    f"These parameters are automatically set based on your CLI inputs:\n"
                    f"- 'inputType': Set from input type (--video, --audio, --text-value, --image)\n"
                    f"- 'mediaSource': Set from media inputs (--video, --audio, --image) and --src-bucket-owner\n"
                    f"- 'inputText': Set from text inputs (--text-value, --text)\n\n"
                    f"Valid modelInput parameters include: startSec, lengthSec, useFixedLengthSec, embeddingOption, minClipSec, textTruncate"
                )
        
        # Check top-level conflicts
        top_level_system = {k: v for k, v in system_payload.items() if k != "modelInput"}
        top_level_user = {k: v for k, v in user_params.items() if k != "modelInput"}
        
        if top_level_user:
            system_keys = set(top_level_system.keys())
            user_keys = set(top_level_user.keys())
            conflicts = system_keys.intersection(user_keys)
            
            if conflicts:
                raise click.ClickException(
                    f"Cannot override system-controlled top-level parameters: {sorted(conflicts)}\n"
                    f"These parameters are automatically set:\n"
                    f"- 'modelId': Set from --model-id\n"
                    f"- 'outputDataConfig': Set from --async-output-s3-uri"
                )
    else:
        # For sync models, check direct payload conflicts
        system_keys = extract_all_keys(system_payload)
        user_keys = extract_all_keys(user_params)
        conflicts = system_keys.intersection(user_keys)
        
        if conflicts:
            conflict_explanations = []
            for conflict in sorted(conflicts):
                if conflict in ["inputText", "texts"]:
                    conflict_explanations.append(f"- '{conflict}': Set from text inputs (--text-value, --text)")
                elif conflict in ["inputImage", "images"]:
                    conflict_explanations.append(f"- '{conflict}': Set from image inputs (--image)")
                elif conflict in ["dimensions", "embeddingConfig.outputEmbeddingLength"]:
                    conflict_explanations.append(f"- '{conflict}': Set from S3 Vector index dimensions")
                else:
                    conflict_explanations.append(f"- '{conflict}': System-controlled parameter")
            
            raise click.ClickException(
                f"Cannot override system-controlled parameters: {sorted(conflicts)}\n"
                f"These parameters are automatically set based on your CLI inputs:\n" +
                "\n".join(conflict_explanations) + "\n\n" +
                f"Use --bedrock-inference-params for model-specific parameters like normalize, embeddingTypes, input_type, truncate, etc."
            )


def build_system_payload(model_id, input_content, input_type, dimensions=None, text_input=None, image_data=None, is_query=False):
    """Build the system-controlled payload based on model and inputs."""
    if model_id.startswith('amazon.titan-embed-text-v2'):
        payload = {"inputText": input_content}
        if dimensions:
            payload["dimensions"] = dimensions
        return payload
        
    elif model_id.startswith('amazon.titan-embed-text-v1'):
        return {"inputText": input_content}
        
    elif model_id.startswith('amazon.titan-embed-image'):
        payload = {}
        if text_input:
            payload["inputText"] = text_input
        if image_data:
            payload["inputImage"] = image_data
        if input_type == "text" and input_content:
            payload["inputText"] = input_content
        elif input_type == "image" and input_content:
            payload["inputImage"] = input_content
        if dimensions:
            payload["embeddingConfig"] = {"outputEmbeddingLength": dimensions}
        return payload
        
    elif model_id.startswith('cohere.embed'):
        if input_type == "image":
            # Cohere requires data URI format, not raw base64
            if not input_content.startswith('data:'):
                # Convert raw base64 to data URI (assume JPEG if not specified)
                data_uri = f"data:image/jpeg;base64,{input_content}"
            else:
                data_uri = input_content
            
            return {
                "images": [data_uri],
                "input_type": "image"
            }
        else:
            # For Cohere text embeddings, use appropriate input_type based on operation
            cohere_input_type = "search_query" if is_query else "search_document"
            return {
                "texts": [input_content],
                "input_type": cohere_input_type
            }
    
    else:
        raise ValueError(f"Unsupported model for system payload: {model_id}")


def build_twelvelabs_system_payload(model_id, input_type, input_content, async_output_s3_uri, src_bucket_owner=None, session=None):
    """Build system-controlled payload for TwelveLabs models."""
    from s3vectors.utils.twelvelabs_helpers import prepare_media_source
    
    model_input = {"inputType": input_type}
    
    if input_type == "text":
        model_input["inputText"] = input_content
    else:
        # For media types, prepare media source
        model_input["mediaSource"] = prepare_media_source(input_content, src_bucket_owner, session)
    
    # Use the base URI directly - let Bedrock handle the path structure
    # Ensure the S3 URI ends with a slash for proper path construction
    base_uri = async_output_s3_uri.rstrip('/')
    output_s3_uri = f"{base_uri}/"
    
    return {
        "modelId": model_id,
        "modelInput": model_input,
        "outputDataConfig": {
            "s3OutputDataConfig": {
                "s3Uri": output_s3_uri
            }
        }
    }


def merge_bedrock_params(system_payload, user_params, model_id):
    """Merge user parameters into system payload."""
    if not user_params:
        return system_payload
    
    if model_id.startswith('twelvelabs.'):
        # For TwelveLabs, merge modelInput and handle other top-level params
        merged_payload = system_payload.copy()
        
        if "modelInput" in user_params:
            merged_payload["modelInput"].update(user_params["modelInput"])
        
        # Add any other user parameters (excluding system-controlled ones)
        for key, value in user_params.items():
            if key not in ["modelId", "outputDataConfig", "modelInput"]:
                merged_payload[key] = value
        
        return merged_payload
    else:
        # For sync models, simple merge
        merged_payload = system_payload.copy()
        merged_payload.update(user_params)
        return merged_payload