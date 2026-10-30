# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-29

### Added
- Amazon Nova Multimodal Embeddings (`amazon.nova-2-multimodal-embeddings-v1:0`) support for text, image, video and audio inputs
- Sync (InvokeModel) and async (StartAsyncInvoke) processing for Nova; async is used when `--async-output-s3-uri` is provided, and accepts S3 inputs only
- Nova embedding dimensions 256, 384, 1024 and 3072, taken from the target index dimension
- Segmented embeddings for Nova async text, video and audio, stored with segment metadata (start/end seconds, character positions, segment index, embedding type)
- Nova video embedding modes `AUDIO_VIDEO_COMBINED` (default) and `AUDIO_VIDEO_SEPARATE`
- Separate Nova embedding purposes for indexing (`GENERIC_INDEX`) and querying (modality-specific `*_RETRIEVAL`)
- Client-side rejection of `--bedrock-inference-params` that override system-controlled request fields
- Client-side rejection of input modalities a model does not support

### Changed
- `--bedrock-inference-params` is deep-merged into the model's request in its native shape, e.g. `{"singleEmbeddingParams": {"text": {"truncationMode": "START"}}}` for Nova; Titan, Cohere and TwelveLabs requests are flat, so their parameters are unaffected
- Model request building and response parsing moved to per-model functions; requests sent for existing models are unchanged

### Fixed
- Failed async jobs and failed Nova segments report the model's failure reason instead of storing a partial result or printing an empty error

## [0.2.1] - 2025-01-12

### Added
- `--key` parameter for custom vector keys
- `--use-object-key-name` flag to use S3 object key or filename as vector key
- Vector key management documentation in README

### Fixed
- Table output format error for batch processing operations
- Validation to prevent `--use-object-key-name` with `--text-value`

## [0.2.0] - 2025-01-XX

### Added
- TwelveLabs Marengo Embed 2.7 model support for multimodal embeddings
- Video query support with `--video` parameter
- Audio query support with `--audio` parameter  
- Image query support with `--image` parameter for TwelveLabs models
- Cross-modal search capabilities (text-to-video, video-to-text, etc.)

### Deprecated
- `--query-input` parameter - use specific input types (`--text-value`, `--text`, `--image`, `--video`, `--audio`) instead

## [0.1.1] - 2024-07-23

### Added
- User agent tracking for CLI usage analytics

## [0.1.0] - 2024-07-15

### Added
- Initial release with basic S3 Vectors functionality
- Support for Amazon Titan and Cohere embedding models
- Vector storage and query operations
- S3 integration for vector indexes

[Unreleased]: https://github.com/awslabs/s3vectors-embed-cli/compare/v0.1.1...HEAD
[0.2.0]: https://github.com/awslabs/s3vectors-embed-cli/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/awslabs/s3vectors-embed-cli/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/awslabs/s3vectors-embed-cli/releases/tag/v0.1.0
