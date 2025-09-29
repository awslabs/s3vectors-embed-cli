# PR: Add parallel async processing for video/audio batch operations

## Title
Add parallel async processing for video/audio batch operations

## Description
Enhances video/audio batch processing with configurable parallel workers for concurrent file processing.

### Changes Made
- **Parallel Async Processing**: Video/audio files now process concurrently using ThreadPoolExecutor
- **User-Controlled Concurrency**: `--max-workers` parameter controls parallel execution
- **File Extension Support**: Added video (`mp4`, `avi`, `mov`, `mkv`, `wmv`, `flv`, `webm`) and audio (`mp3`, `wav`, `flac`, `aac`, `m4a`, `ogg`, `wma`) extensions to batch processing
- **Fixed File Counting**: Corrected `processedFiles` to count actual files instead of vectors for async processing
- **Enhanced Documentation**: Updated README.md with comprehensive batch processing examples and processing strategy details

### Files Modified
- `s3vectors/core/streaming_batch_orchestrator.py`: Added parallel async processing and video/audio file extensions
- `s3vectors/commands/embed_put.py`: Extended wildcard processing to video/audio and fixed vector counting
- `README.md`: Updated batch processing documentation with examples and processing strategy table

### Testing
- Tested with local video batch processing (`/path/videos/*`)
- Tested with S3 video batch processing (`s3://bucket/videos/*`)
- Tested with various worker configurations (1, 2, 4 workers)
- Verified correct file counting vs vector counting
- Tested large-scale image batch processing (602 files)

### Backward Compatibility
- No breaking changes
- Existing text/image batch processing unchanged
- Default behavior preserved (sync for text/image, async for video/audio)

### Related
- Builds on TwelveLabs model support from PR #7
- Enhances existing video/audio batch processing with parallel execution

## Commit Message
```
feat: add parallel async processing for video/audio batch operations

- Add parallel ThreadPoolExecutor for video/audio batch processing
- Support video/audio file extensions in wildcard patterns
- Fix processedFiles counting to count files instead of vectors
- Add user-controlled concurrency via --max-workers parameter
- Update documentation with processing strategy and examples
- Enable concurrent processing of video/audio files in batch operations
```

## Self Sign-off
```
Signed-off-by: Vaibhav Sabharwal <vibhusabharwal@gmail.com>
```
