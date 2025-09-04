# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-01-XX

### Added
- TwelveLabs Marengo Embed 2.7 model support for multimodal embeddings
- Video query support with `--video` parameter and `--embedding-options`
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
