# Amazon S3 Vectors Embed CLI

Amazon S3 Vectors Embed CLI is a standalone command-line tool that simplifies the process of working with vector embeddings in S3 Vectors. You can create vector embeddings for your data using Amazon Bedrock and store and query them in your S3 vector index using single commands. 

**Amazon S3 Vectors Embed CLI is in preview release and is subject to change.**

## Supported Commands

**s3vectors-embed put**: Embed text, file content, or S3 objects and store them as vectors in an S3 vector index.
You can create and ingest vector embeddings into an S3 vector index using a single put command. You specify the data input you want to create an embedding for, an Amazon Bedrock embeddings model ID, your S3 vector bucket name, and S3 vector index name. The command supports several input formats including text data, a local text or image file, an S3 image or text object or prefix. The command generates embeddings using the dimensions configured in your S3 vector index properties. If you are ingesting embeddings for several objects in an S3 prefix or local file path, it automatically uses batch processes to maximize throughput. 

**Note**: Each file is processed as a single embedding. Document chunking is not currently supported. 

**s3vectors-embed query**: Embed a query input and search for similar vectors in an S3 vector index.
You can perform similarity queries for vector embeddings in your S3 vector index using a single query command. You specify your query input, an Amazon Bedrock embeddings model ID, the vector bucket name, and vector index name. The command accepts several types of query inputs like a text string, an image file, or a single S3 text or image object. The command generates embeddings for your query using the input embeddings model and then performs a similarity search to find the most relevant matches. You can control the number of results returned, apply metadata filters to narrow your search, and choose whether to include similarity distance in the results for comprehensive analysis.

### Supported Input Types

**Note**: 
Starting version 0.2.0, this CLI has introduced a unified `--bedrock-inference-params` paramter for all model-specific parameters.
Additionally, the `--query-input` parameter in the query command has been replaced with the following individual parameters instead: 

- **`--text-value`**: Direct text query string (preferred for text queries)
- **`--text`**: Text file path (local file or S3 URI)
- **`--image`**: Image file path (local file or S3 URI)
- **`--video`**: Video file path (local file or S3 URI) - TwelveLabs models only
- **`--audio`**: Audio file path (local file or S3 URI) - TwelveLabs models only

## Installation and Configuration
### Prerequisites
- Python 3.9 or higher
- To execute the CLI, you will need AWS credentials configured. 
- Update your AWS account with appropriate permissions to use Amazon Bedrock and S3 Vectors
- Access to an Amazon Bedrock embedding model
- Create an Amazon S3 vector bucket and vector index to store your embeddings

### Quick Install (Recommended)
```bash
pip install s3vectors-embed-cli
```

### Development Install
```bash
# Clone the repository
git clone https://github.com/awslabs/s3vectors-embed-cli
cd s3vectors-embed-cli

# Install in development mode
pip install -e .
```

**Note**: All dependencies are automatically installed when you install the package via pip.

### Quick Start

#### **Put Examples**

1. **Embed text and store them as vectors in your S3 vector index:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "Hello, world!"
```

2. **Process local text files:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "./documents/sample.txt"
```

3. **Process image files using a local file path:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-image-v1 \
  --image "./images/photo.jpg"
```

4. **Process files from a local file path using wildcard characters:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "./documents/*.txt"
```

5. **Process files from an S3 general purpose bucket using wildcard characters:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "s3://bucket/path/*"
```

6. **Add metadata alongside your vectors:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "s3://my-bucket/sample.txt"
  --metadata '{"category": "technology", "version": "1.0"}'
```

7. **Use custom model parameters:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "Sample text with custom parameters" \
  --bedrock-inference-params '{"normalize": false}'
```

#### ** Examples for the TwelveLabs Marengo Embedding Model (Async Processing)**

**Note:** For the TwelveLabs model (`twelvelabs.marengo-embed-2-7-v1:0`), Bedrock processes data asynchronously and first stores the embedding output in a general purpose S3 bucket that you specify. 

7. **TwelveLabs embeddings for text data :**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --text-value "Spiderman flies through a street and catches a car with his web" \
  --async-output-s3-uri s3://my-async-bucket
```

8. **TwelveLabs embeddings for a local video file (up to 36MB for TwelveLabs models):**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video ./sample.mp4 \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"useFixedLengthSec": 5, "minClipSec": 2, "embeddingOption": ["visual-text", "audio"]}'
```

9. **TwelveLabs embeddings for an S3 URI video input (up to 2GB, recommended for large files):**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video s3://my-bucket/large-video.mp4 \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"useFixedLengthSec": 5, "embeddingOption": ["visual-text", "audio"]}' \
  --src-bucket-owner 123456789012  # Optional: only needed for cross-account access
```

10. **TwelveLabs embeddings for a local audio file:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --audio ./audio.wav \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"startSec": 10.0, "lengthSec": 30.0}'
```

11. **TwelveLabs embeddings for an S3 URI audio input:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --audio s3://my-bucket/audio.wav \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"startSec": 10.0, "lengthSec": 30.0}' \
  --src-bucket-owner 123456789012  # Optional: only needed for cross-account access
```

12. **TwelveLabs image embeddings:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --image ./photo.jpg \
  --async-output-s3-uri s3://my-async-bucket
```

13. **TwelveLabs embeddings for a video file using additional options:**
```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video s3://my-bucket/video.mp4 \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"embeddingOption": ["visual-text", "visual-image", "audio"], "useFixedLengthSec": 3, "minClipSec": 2, "startSec": 5.0, "lengthSec": 60.0}' \
  --metadata '{"source": "marketing", "campaign": "2024-q1"}' \
  --src-bucket-owner 123456789012  # Optional: only needed for cross-account access
```

#### **Query Examples**

1. **Direct text query:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "query text" \
  --k 10
```

2. **Query using a local text file:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "./query.txt" \
  --k 5 \
  --output table
```

3. **Query using an S3 text file:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "s3://my-bucket/query.txt" \
  --k 3 
```

4. **Image query:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-image-v1 \
  --image "./query-image.jpg" \
  --k 5
```

5. **TwelveLabs: cross-modal text search:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --text-value "red sports car chase" \
  --async-output-s3-uri s3://my-async-bucket \
  --k 5
```

6. **TwelveLabs: Query using a video input with the default time range (0-5 second clip):**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video "./query-video.mp4" \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"embeddingOption": ["visual-text"]}' \
  --k 5
```

7. **TwelveLabs: Query using a video input with a custom time range:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video "./query-video.mp4" \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"embeddingOption": ["audio"], "startSec": 30.0, "useFixedLengthSec": 8}' \
  --k 5
```

8. **TwelveLabs: Query using an audio input :**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --audio "./query-audio.wav" \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"startSec": 15.0, "useFixedLengthSec": 6}' \
  --k 5
```

9. **TwelveLabs: Query using a visual-image embedding from a video input:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video "s3://my-bucket/query-video.mp4" \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"embeddingOption": ["visual-image"]}' \
  --k 5 \
  --src-bucket-owner 123456789012  # Optional: only needed for cross-account access
```

10. **Titan Text: Query with metadata filters:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "query text" \
  --filter '{"category": {"$eq": "technology"}}' \
  --k 10 \
  --return-metadata
```

11. **Titan Text: Query with multiple metadata filters (AND):**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "query text" \
  --filter '{"$and": [{"category": "technology"}, {"version": "1.0"}]}' \
  --k 10 \
  --return-metadata
```

12. **Titan Text: Query with multiple metadata filters (OR):**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "query text" \
  --filter '{"$or": [{"category": "docs"}, {"category": "guides"}]}' \
  --k 5
```

13. **Titan Text: Query with metadata filters (comparison operators):**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "query text" \
  --filter '{"$and": [{"category": "tech"}, {"version": {"$gte": "1.0"}}]}' \
  --k 10
```

14. **Cohere Embed v3: Query with custom model parameters:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id cohere.embed-english-v3 \
  --text-value "search query with custom truncation" \
  --bedrock-inference-params '{"truncate": "END"}' \
  --k 5 \
  --return-distance
```
15. **TwelveLabs Marengo Embed 2.7: S3 video batch processing with high concurrency**
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video "s3://bucket/videos/*" \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"embeddingOption": ["visual-image", "audio"], "startSec": 0, "useFixedLengthSec": 4}' \
  --metadata '{"batch": "multimodal_processing"}' \
  --max-workers 4

### Command Parameters

#### Global Options
- `--debug`: Enable debug mode with detailed logging for troubleshooting
- `--profile`: AWS profile name to use from ~/.aws/credentials
- `--region`: AWS region name (overrides session/config defaults)

#### Put Command Parameters
Required:
- `--vector-bucket-name`: Name of the S3 vector bucket 
- `--index-name`: Name of the vector index in your vector index to store the vector embeddings
- `--model-id`: Bedrock model ID to use for generating embeddings (e.g., amazon.titan-embed-text-v2:0, twelvelabs.marengo-embed-2-7-v1:0)

Input Options (one required):
- `--text-value`: Direct text input to embed
- `--text`: Text input - supports multiple input types:
  - **Local file**: `./document.txt`
  - **Local files with wildcard characters**: `./data/*.txt`, `~/docs/*.md`
  - **S3 object**: `s3://bucket/path/file.txt`
  - **S3 path with wildcard characters**: `s3://bucket/path/*` (prefix-based, not extension-based)
- `--image`: Image input - supports multiple input types:
  - **Local file**: `./document.jpg`
  - **Local wildcard**: `./data/*.jpg`
  - **S3 object**: `s3://bucket/path/file.jpg`
  - **S3 path with wildcard characters**: `s3://bucket/path/*` (prefix-based, not extension-based)
- `--video`: Video input for TwelveLabs models - supports:
  - **Local file**: `./video.mp4` (up to 36MB for TwelveLabs models)
  - **S3 URI**: `s3://bucket/path/video.mp4` 
- `--audio`: Audio input for TwelveLabs models - supports:
  - **Local file**: `./audio.wav` (up to 36MB for TwelveLabs models)
  - **S3 URI**: `s3://bucket/path/audio.wav`

Optional:
- `--key`: Uniquely identifies each vector in the vector index (default: auto-generated UUID)
- `--metadata`: Additional metadata associated with the vector; provided as JSON string
- `--bedrock-inference-params`: Model-specific parameters passed to Bedrock (JSON format, e.g., `'{"normalize": false}'`)
- `--src-bucket-owner`: AWS account ID for cross-account S3 access to input files (optional, only needed when input S3 files are in a different AWS account)
- `--max-workers`: Maximum parallel workers for batch processing (default: 4)
- `--batch-size`: Number of vectors per S3 Vector put_vectors call (1-500, default: 500)
- `--output`: Output format (json or table, default: json)

**TwelveLabs-Specific Parameters:**
- `--async-output-s3-uri`: S3 URI for async processing results (required for TwelveLabs models, e.g., s3://my-bucket/path)

**TwelveLabs Model Parameters (via `--bedrock-inference-params`):**
Use `--bedrock-inference-params '{...}'` for TwelveLabs-specific options:
- `embeddingOption`: Array for video: `["visual-text"]`, `["visual-image"]`, `["audio"]`, or combinations
- `startSec`: Start time in seconds for video/audio processing
- `lengthSec`: Duration to process in seconds
- `useFixedLengthSec`: Fixed duration for each clip in seconds (2-10 for video)
- `minClipSec`: Minimum clip duration for video processing (1-5 seconds)
- `textTruncate`: How to handle text exceeding 77 tokens (`"end"` or `"none"`, default: `"end"`)

**Example:**
```bash
--bedrock-inference-params '{"embeddingOption": ["visual-text", "audio"], "startSec": 30.0, "useFixedLengthSec": 5}'
```

#### Query Command Parameters

**Core Required Parameters:**
- `--vector-bucket-name`: Name of the S3 vector bucket
- `--index-name`: Name of the vector index 
- `--model-id`: Bedrock model ID to use for generating embeddings (e.g., amazon.titan-embed-text-v2:0, twelvelabs.marengo-embed-2-7-v1:0)

**Query Input Parameters (One Required):**
- `--text-value`: Direct text query string
- `--text`: Text file path (local file or S3 URI)
- `--image`: Image file path (local file or S3 URI)
- `--video`: Video file path (local file or S3 URI) - TwelveLabs models only
- `--audio`: Audio file path (local file or S3 URI) - TwelveLabs models only

**TwelveLabs-Specific Parameters:**
- `--async-output-s3-uri`: S3 URI for async output (required for TwelveLabs models, e.g., s3://my-bucket/path)
- `--src-bucket-owner`: AWS account ID for cross-account S3 access to input files (optional, only needed when input S3 files are in a different AWS account)

**TwelveLabs Model Parameters (via `--bedrock-inference-params`):**
Use `--bedrock-inference-params '{...}'` for TwelveLabs query options:
- `embeddingOption`: Array for video queries: `["visual-text"]`, `["visual-image"]`, `["audio"]` (required for video, auto-selected for audio)
- `startSec`: Start time in seconds (default: 0)
- `useFixedLengthSec`: Fixed clip duration in seconds (default: 5, range: 2-10)

**Example:**
```bash
--bedrock-inference-params '{"embeddingOption": ["visual-text"], "startSec": 30.0, "useFixedLengthSec": 8}'
```

**Optional Parameters:**
- `--k`: Number of results to return (default: 5)
- `--filter`: Filter expression for metadata-based filtering (JSON format with AWS S3 Vectors API operators)
- `--bedrock-inference-params`: Model-specific parameters passed to Bedrock (JSON format, e.g., `'{"truncate": "END"}'`)
- `--return-metadata`: Include metadata in results (default: true)
- `--return-distance`: Include similarity distance scores
- `--output`: Output format (table or json, default: json)
- `--region`: AWS region name

**Query Examples:**

```bash
# Direct text query (preferred method)
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --text-value "search text" --k 10

# Text file query
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --text ./query.txt --k 5

# Image query
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-image-v1 --image ./query-image.jpg --k 3

# S3 file query with cross-account access
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --text s3://other-bucket/query.txt \
  --src-bucket-owner 123456789012

# TwelveLabs cross-modal search
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 --text-value "red sports car chase" \
  --async-output-s3-uri s3://my-async-bucket --k 5

```

### Model Compatibility
| Model | Type | Dimensions | Use Case | API Type |
|-------|------|------------|----------|----------|
| `amazon.titan-embed-text-v2:0` | Text | 1024, 512, 256 | Modern text embedding | Sync |
| `amazon.titan-embed-text-v1` | Text | 1536 | Legacy text embedding | Sync |
| `amazon.titan-embed-image-v1` | Multimodal (Text + Image) | 1024, 384, 256 | Text and image embedding | Sync |
| `cohere.embed-english-v3` | Multimodal (Text or Image) | 1024 | Advanced English text or image embedding | Sync |
| `cohere.embed-multilingual-v3` | Multimodal (Text or Image) | 1024 | Multilingual text or image embedding | Sync |
| `twelvelabs.marengo-embed-2-7-v1:0` | Multimodal (Video, Audio, Text, Image) | 1024 | Video and audio understanding | **Async** |

**Note**: TwelveLabs models require async processing (~60 seconds) and an S3 general purpose bucket location to store the interim embedding results. 

## Advanced Model Parameters

### **Bedrock Inference Parameters (`--bedrock-inference-params`)**

The `--bedrock-inference-params` parameter allows you to pass model-specific parameters directly to Amazon Bedrock embedding models. This unified parameter system works across all models and operations (PUT and QUERY), providing fine-grained control over embedding generation.

#### **Model-Specific Parameters**

##### **Amazon Titan Text Models**
```bash
# Titan Text v2 - Control normalization
s3vectors-embed put \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "Sample text" \
  --bedrock-inference-params '{"normalize": false}'

# Titan Text v2 - Set dimensions (if supported by your index)
s3vectors-embed put \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "Sample text" \
  --bedrock-inference-params '{"dimensions": 512}'
```

##### **Cohere Models**
```bash
# Cohere - Control text truncation
s3vectors-embed put \
  --model-id cohere.embed-english-v3 \
  --text-value "Long text content..." \
  --bedrock-inference-params '{"truncate": "END"}'

# Cohere - Multiple parameters
s3vectors-embed query \
  --model-id cohere.embed-english-v3 \
  --text-value "Search query" \
  --bedrock-inference-params '{"truncate": "END"}' \
  --k 5
```

##### **TwelveLabs Models**
```bash
# TwelveLabs - Text truncation
s3vectors-embed put \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --text-value "Long text content..." \
  --async-output-s3-uri s3://my-bucket \
  --bedrock-inference-params '{"textTruncate": "end"}'

# TwelveLabs - Video processing parameters
s3vectors-embed put \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video s3://bucket/video.mp4 \
  --async-output-s3-uri s3://my-bucket \
  --bedrock-inference-params '{"startSec": 30.0, "useFixedLengthSec": 10, "embeddingOption": ["visual-text"]}'
```
#### **Valid Parameters by Model**

| Model | Valid Parameters | Example Values |
|-------|------------------|----------------|
| **Titan Text v2** | `normalize` | `false`|
| **Titan Text v1** | None (basic model) | N/A |
| **Titan Image v1** | `normalize` | `false`|
| **Cohere English/Multilingual** | `truncate` | `"END"`, `"NONE"` |
| **TwelveLabs Marengo** | `textTruncate`, `startSec`, `lengthSec`, `useFixedLengthSec`, `embeddingOption`, `minClipSec` | `"end"`, `30.0`, `60.0`, `5`, `["visual-text"]`, `2` |

#### **Best Practices**

1. **Use Model Documentation**: Refer to Amazon Bedrock model documentation for available parameters
2. **Test Parameters**: Validate parameters with small datasets before batch processing
3. **JSON Validation**: Ensure JSON is properly formatted (use online validators if needed)
4. **Parameter Naming**: Use exact parameter names as specified in Bedrock documentation
5. **Value Types**: Match data types (strings, numbers, booleans, arrays) as required

#### **Common Use Cases**

##### **Performance Optimization**
```bash
# Disable normalization for faster processing (if acceptable for your use case)
--bedrock-inference-params '{"normalize": false}'
```

##### **Text Handling**
```bash
# Control how long text is handled
--bedrock-inference-params '{"truncate": "END"}'  # Cohere
--bedrock-inference-params '{"textTruncate": "end"}'  # TwelveLabs
```

##### **Video Processing Control**
```bash
# Precise video segment processing
--bedrock-inference-params '{"startSec": 45.0, "useFixedLengthSec": 8, "embeddingOption": ["visual-text", "audio"]}'
```

#### **Error Handling**

The CLI provides clear error messages for parameter issues:

```bash
# Invalid parameter name
Error: Cannot override system-controlled parameters: ['inputText']

# Invalid JSON format
Error: Invalid JSON in --bedrock-inference-params: Expecting ',' delimiter

# Model-specific guidance
Valid parameters include: startSec, lengthSec, useFixedLengthSec, embeddingOption, minClipSec, textTruncate
```

## Metadata Filtering

### **Supported Operators**

#### **Comparison Operators**
- `$eq`: Equal to
- `$ne`: Not equal to  
- `$gt`: Greater than
- `$gte`: Greater than or equal to
- `$lt`: Less than
- `$lte`: Less than or equal to
- `$in`: Value in array
- `$nin`: Value not in array

#### **Logical Operators**
- `$and`: Logical AND (all conditions must be true)
- `$or`: Logical OR (at least one condition must be true)
- `$not`: Logical NOT (condition must be false)

### **Filter Examples**

#### **Single Condition Filters**
```bash
# Exact match
--filter '{"category": {"$eq": "documentation"}}'

# Not equal
--filter '{"status": {"$ne": "archived"}}'

# Greater than or equal
--filter '{"version": {"$gte": "2.0"}}'

# Value in list
--filter '{"category": {"$in": ["docs", "guides", "tutorials"]}}'
```

#### **Multiple Condition Filters**
```bash
# AND condition (all must be true)
--filter '{"$and": [{"category": "tech"}, {"version": "1.0"}]}'

# OR condition (at least one must be true)  
--filter '{"$or": [{"category": "docs"}, {"category": "guides"}]}'

# Complex nested conditions
--filter '{"$and": [{"category": "tech"}, {"$or": [{"version": "1.0"}, {"version": "2.0"}]}]}'

# NOT condition
--filter '{"$not": {"category": {"$eq": "archived"}}}'
```

#### **Advanced Filter Examples**
```bash
# Multiple AND conditions with comparison operators
--filter '{"$and": [{"category": "documentation"}, {"version": {"$gte": "1.0"}}, {"status": {"$ne": "draft"}}]}'

# OR with nested AND conditions
--filter '{"$or": [{"$and": [{"category": "docs"}, {"version": "1.0"}]}, {"$and": [{"category": "guides"}, {"version": "2.0"}]}]}'

# Using $in with multiple values
--filter '{"$and": [{"category": {"$in": ["docs", "guides"]}}, {"language": {"$eq": "en"}}]}'
```

### **Important Notes**

1. **JSON Format**: Filters must be valid JSON strings
2. **Quotes**: Use single quotes around the entire filter and double quotes inside JSON
3. **Case Sensitivity**: String comparisons are case-sensitive
3. **Data Types**: Ensure filter values match the data types in your metadata

## Metadata

The Amazon S3 Vectors Embed CLI automatically adds standard metadata fields to help track and manage your vector embeddings. Understanding these fields is important for filtering and troubleshooting your vector data.

### Standard Metadata Fields

The CLI automatically adds the following metadata fields to every vector:

#### `S3VECTORS-EMBED-SRC-CONTENT`
- **Purpose**: Stores the original text content. Configure this field as *nonFilterableMetadataKeys* while creating S3 vector index to store large text.
- **Behavior**:
  - **Direct text input** (`--text-value`): Contains the actual text content
  - **Text files**: Contains the full text content of the file
  - **Image files**: N/A (images don't have textual content to store) 

**Examples**:
```bash
# Direct text - stores the actual text
--text-value "Hello world" 
# Metadata: {"S3VECTORS-EMBED-SRC-CONTENT": "Hello world"}

# Text file - stores file content
--text document.txt
# Metadata: {"S3VECTORS-EMBED-SRC-CONTENT": "Contents of document.txt..."}

# Image file - no SOURCE_CONTENT field added
--image photo.jpg
# Metadata: {}
```

#### `S3VECTORS-EMBED-SRC-LOCATION`
- **Purpose**: Tracks the original file location
- **Behavior**:
  - **Text files**: Contains the file path or S3 URI
  - **Image files**: Contains the file path or S3 URI
  - **Direct text**: Not added (no file involved)

**Examples**:
```bash
# Local text file
--text /path/to/document.txt
# Metadata: {
#   "S3VECTORS-EMBED-SRC-CONTENT": "File contents...",
#   "S3VECTORS-EMBED-SRC-LOCATION": "file:///path/to/document.txt"
# }

# S3 text file
--text s3://my-bucket/docs/file.txt
# Metadata: {
#   "S3VECTORS-EMBED-SRC-CONTENT": "File contents...",
#   "S3VECTORS-EMBED-SRC-LOCATION": "s3://my-bucket/docs/file.txt"
# }

# Image file (local or S3)
--image /path/to/photo.jpg
# Metadata: {
#   "S3VECTORS-EMBED-SRC-LOCATION": "file:///path/to/photo.jpg"
# }

--image s3://my-bucket/images/photo.jpg
# Metadata: {
#   "S3VECTORS-EMBED-SRC-LOCATION": "s3://my-bucket/images/photo.jpg"
# }
```

### Additional Metadata

You can add your own metadata using the `--metadata` parameter with JSON format:

```bash
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "Sample text" \
  --metadata '{"category": "documentation", "version": "1.0", "author": "team-a"}'
```

**Result**: Your metadata is merged with the two standard metadata fields:
```json
{
  "S3VECTORS-EMBED-SRC-CONTENT": "Sample text",
  "category": "documentation",
  "version": "1.0", 
  "author": "team-a"
}
```

## Output Formats

The CLI provides a simple output by default with an optional debug mode for more detailed information like progress information.

### Simple Output (Default)

The CLI provides a simple output without progress indicators:

```bash
# PUT output
s3vectors-embed put --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --text-value "Hello"
```
**Output:**
```
{
  "key": "abc-123-def-456",
  "bucket": "my-bucket",
  "index": "my-index",
  "model": "amazon.titan-embed-text-v2:0",
  "contentType": "text",
  "embeddingDimensions": 1024,
  "metadata": {
    "S3VECTORS-EMBED-SRC-CONTENT": "Hello"
  }
}
```

### Debug option

Use `--debug` for comprehensive operational details:

```bash
# Debug mode provides detailed logging
s3vectors-embed --debug put --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --text-value "Hello"
```

The CLI supports two output formats for query results:

### JSON Format (Default)
- **Machine-readable**: Perfect for programmatic processing
- **Complete data**: Shows full metadata content without truncation
- **Structured**: Easy to parse and integrate with other tools

```bash
# Uses JSON by default
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --text-value "search text"

# Explicit JSON format (same as default)
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --text-value "search text" --output json
```

**JSON Output Example:**
```json
{
  "results": [
    {
      "Key": "abc123-def456-ghi789",
      "distance": 0.2345,
      "metadata": {
        "S3VECTORS-EMBED-SRC-CONTENT": "Complete text content without any truncation...",
        "S3VECTORS-EMBED-SRC-LOCATION": "s3://bucket/path/file.txt",
        "category": "documentation",
        "author": "team-a"
      }
    }
  ],
  "summary": {
    "queryType": "text",
    "model": "amazon.titan-embed-text-v2:0",
    "index": "my-index",
    "resultsFound": 1,
    "queryDimensions": 1024
  }
}
```

### Table Format
- **Human-readable**: Easy to read and analyze visually
- **Complete data**: Shows full metadata content without truncation
- **Formatted**: Clean tabular display with proper alignment

```bash
# Explicit table format
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --text-value "search text" --output table
```

## Wildcard Character Support

The CLI supports powerful wildcard characters in the input path for processing multiple files efficiently:

### **Local Filesystem Patterns (NEW)**

- **Basic wildcards**: `./data/*.txt` - all .txt files in data directory
- **Home directory**: `~/documents/*.md` - all .md files in user's documents
- **Recursive patterns**: `./docs/**/*.txt` - all .txt files recursively
- **Multiple extensions**: `./files/*.{txt,md,json}` - multiple file types
- **Question mark**: `./file?.txt` - single character wildcard

**Examples:**
```bash
# Process all text files in current directory
s3vectors-embed put --vector-bucket-name bucket --index-name idx \
  --model-id amazon.titan-embed-text-v2:0 --text "./*.txt"

# Process all markdown files in home directory
s3vectors-embed put --vector-bucket-name bucket --index-name idx \
  --model-id amazon.titan-embed-text-v2:0 --text "~/notes/*.md"

# Process files with pattern matching
s3vectors-embed put --vector-bucket-name bucket --index-name idx \
  --model-id amazon.titan-embed-text-v2:0 --text "./doc?.txt"
```


**Important**: S3 wildcards work with prefixes, not file extensions. Use `s3://bucket/path/*` not `s3://bucket/path/*.ext`

**Examples:**
```bash
# Process all files under an S3 prefix
s3vectors-embed put --vector-bucket-name bucket --index-name idx \
  --model-id amazon.titan-embed-text-v2:0 --text "s3://bucket/path1/*"

```

### **Important Differences: Local vs S3 Wildcards**

**Local Filesystem Wildcards:**
- ✅ Support file extensions: `./data/*.txt`, `./docs/*.json`
- ✅ Support complex patterns: `./files/*.{txt,md}`, `./doc?.txt`
- ✅ Support recursive patterns: `./docs/**/*.md`

**S3 Wildcards:**
- ✅ Support prefix patterns: `s3://bucket/docs/*`, `s3://bucket/2024/reports/*`
- ❌ **Do NOT support extension filtering**: `s3://bucket/path/*.json` won't work
- ❌ **Do NOT support complex patterns**: Use prefix-based organization instead 

**Best Practices:**
- **For S3**: Organize files by prefix/path structure: `s3://bucket/json-files/*`
- **For Local**: Use full wildcard capabilities: `./data/*.{json,txt}`

### **Pattern Processing Features**

- **Batch Processing**: Large file sets automatically batched 
- **Parallel Processing**: Configurable workers for concurrent processing
- **Error Handling**: Individual file failures don't stop batch processing and do not fail the whole batch.
- **Progress Tracking**: Clear reporting of processed vs failed files
- **File Type Filtering**: CLI automatically filters supported file types after pattern expansion

## Batch Processing

The CLI supports efficient batch processing for multiple files using both local and S3 wildcard characters in the input path. Video and audio batch processing is supported with parallel async processing.

### **Batch Processing Features**

- **Automatic batching**: Large datasets are automatically split into batches of 500 vectors
- **Dual processing strategies**: 
  - **Sync models** (text/image): Parallel processing with batch storage
  - **Async models** (text/image/video/audio): Parallel async processing with per-file storage
- **Error resilience**: Individual file failures don't stop batch processing
- **Performance optimization**: Efficient memory usage and API call batching
- **User-controlled concurrency**: Configure parallel workers with `--max-workers`

### **Processing Strategy by Content Type**

The CLI automatically selects the optimal processing strategy based on content type:

| Content Type | Processing Mode | API Used | Batch Strategy | Output |
|--------------|----------------|----------|----------------|---------|
| **Text**  | Sync  | `InvokeModel` | Parallel batch storage | Single vector per file |
| **Image** | Sync  | `InvokeModel` | Parallel batch storage | Single vector per file |
| **Text**  | Async | `StartAsyncInvoke` | Parallel batch storage | Single vector per file |
| **Image** | Async | `StartAsyncInvoke` | Parallel batch storage | Single vector per file |
| **Video** | Async | `StartAsyncInvoke` | Per-file storage | Multiple vectors per file |
| **Audio** | Async | `StartAsyncInvoke` | Per-file storage | Multiple vectors per file |

### Batch Processing Examples

**Text/Image Batch Processing:**
```bash
# Process all local text files
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "./documents/*.txt" \
  --metadata '{"source": "local_batch", "category": "documents"}' \
  --max-workers 4

# S3 image files batch processing
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-image-v1 \
  --image "s3://bucket/images/*" \
  --metadata '{"category": "images", "source": "batch_upload"}' \
  --max-workers 2

# TwelveLabs large image batch 
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --image "s3://bucket/animals/*" \
  --async-output-s3-uri s3://my-async-bucket \
  --metadata '{"batch": "animal_images"}' \
  --max-workers 4
```

**Video/Audio Batch Processing:**
```bash
# Local video batch processing 
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video "/path/to/videos/*" \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"embeddingOption": ["visual-text"], "startSec": 3, "useFixedLengthSec": 5}' \
  --metadata '{"batch": "video_processing", "date": "2025-09-14"}' \
  --max-workers 2

# S3 video batch processing 
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --video "s3://bucket/videos/*" \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"embeddingOption": ["visual-image", "audio"], "startSec": 0, "useFixedLengthSec": 4}' \
  --metadata '{"batch": "multimodal_processing"}' \
  --max-workers 4

# Audio batch processing
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id twelvelabs.marengo-embed-2-7-v1:0 \
  --audio "s3://bucket/audio/*" \
  --async-output-s3-uri s3://my-async-bucket \
  --bedrock-inference-params '{"startSec": 10, "useFixedLengthSec": 8}' \
  --max-workers 2
```

### **Batch Processing Output**

**Text/Image Batch Output (Sync):**
```bash
Processing chunk 1...
Found 94 supported files in chunk 1
STORED BATCH: 94 vectors
Completed chunk 1: 94 processed, 0 failed

{
  "type": "streaming_batch",
  "contentType": "text",
  "totalFiles": 94,
  "processedFiles": 94,
  "failedFiles": 0,
  "totalVectors": 94
}
```

**Video/Audio Batch Output (Async):**
```bash
Processing 4 video files with 2 concurrent workers...
[1/4] Stored 6 vectors from /path/video1.mp4
[2/4] Stored 14 vectors from /path/video2.mp4
[3/4] Stored 1 vectors from /path/video3.mp4
[4/4] Stored 45 vectors from /path/video4.mp4
Completed chunk 1: 4 processed, 0 failed

{
  "type": "streaming_batch",
  "contentType": "video",
  "totalFiles": 4,
  "processedFiles": 4,
  "failedFiles": 0,
  "totalVectors": 66
}
```

### Troubleshooting

#### Use Debug Mode for Troubleshooting

For troubleshooting, first enable debug mode to get detailed information in the output:

```bash
# Add --debug to any command for detailed logging
s3vectors-embed --debug put --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --text-value "test"
```

Debug mode provides:
- **API request/response details**: See exact payloads sent to Bedrock and S3 Vectors
- **Performance timing**: Identify slow operations
- **Configuration validation**: Verify AWS settings and service initialization
- **Error context**: Detailed error messages with full context

#### Troubleshooting Issues

1. **AWS Credentials Not Found**
```bash
# Error: Unable to locate credentials
# Solution: Configure AWS credentials
aws configure
# Or set environment variables:
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Debug with credentials issue:
s3vectors-embed --debug put ... 
# Will show: "BedrockService initialization failed" with details
```

2. **Vector index Not Found**
```bash
# Error: ResourceNotFoundException: Vector index not found
# Solution: Ensure the vector index exists and you have correct permissions
aws s3 ls s3vectors://your-bucket

# Debug output will show:
# S3 Vectors ClientError: ResourceNotFoundException...
```

3. **Model Access Issues**
```bash
# Error: AccessDeniedException: Unable to access Bedrock model
# Solution: Verify Bedrock model access and permissions
aws bedrock list-foundation-models

# Debug output will show:
# Bedrock ClientError: AccessDeniedException...
# Request body: {...} (shows what was attempted)
```

4. **Service Unavailable Errors**
```bash
# Error: ServiceUnavailableException
# Debug output provides context:
# S3 Vectors ClientError: ServiceUnavailableException when calling PutVectors
# API parameters: {"vectorBucketName": "...", "indexName": "..."}
```

5. **Bedrock Inference Parameters Issues**
```bash
# Error: Cannot override system-controlled parameters
# Solution: Use only model-specific parameters, avoid system-controlled ones
s3vectors-embed put \
  --model-id amazon.titan-embed-text-v2:0 \
  --text-value "test" \
  --bedrock-inference-params '{"normalize": false}'  # ✅ Valid

# Invalid JSON format
# Error: Invalid JSON in --bedrock-inference-params
# Solution: Validate JSON format (use single quotes around, double quotes inside)
--bedrock-inference-params '{"parameter": "value"}'  # ✅ Correct
--bedrock-inference-params "{'parameter': 'value'}"  # ❌ Wrong

# Debug output shows parameter validation:
# Cannot override system-controlled parameters: ['inputText']
# Valid parameters for this model: normalize, dimensions
```

## Repository Structure
```
s3vectors-embed-cli/
├── s3vectors/                           # Main package directory
│   ├── cli.py                          # Main CLI entry point
│   ├── commands/                       # Command implementations
│   │   ├── embed_put.py               # Vector embedding and storage
│   │   └── embed_query.py             # Vector similarity search
│   ├── core/                          # Core functionality
│   │   ├── unified_processor.py       # Unified processing logic
│   │   ├── services.py               # Bedrock and S3Vector services
│   │   └── streaming_batch_orchestrator.py  # Batch processing
│   └── utils/                         # Utility functions
│       ├── config.py                 # AWS configuration management
│       ├── models.py                 # Model definitions and capabilities
│       └── multimodal_helpers.py     # Multimodal processing helpers
├── setup.py                          # Package installation configuration
├── pyproject.toml                    # Modern Python packaging configuration
├── requirements.txt                  # Python dependencies
├── LICENSE                           # Apache 2.0 license
```