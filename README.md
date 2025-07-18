# Amazon S3 Vectors Embed CLI

Amazon S3 Vectors Embed CLI is a standalone command-line tool that simplifies the process of working with vector embeddings in S3 Vectors. You can create vector embeddings for your data using Amazon Bedrock and store and query them in your S3 vector index using single commands. 

**Amazon S3 Vectors Embed CLI is in preview release and is subject to change.**

## Supported Commands

**s3vectors-embed put**: Embed text, file content, or S3 objects and store them as vectors in an S3 vector index.
You can create and ingest vector embeddings into an S3 vector index using a single put command. You specify the data input you want to create an embedding for, an Amazon Bedrock embeddings model ID, your S3 vector bucket name, and S3 vector index name. The command supports several input formats including text data, a local text or image file, an S3 image or text object or prefix. The command generates embeddings using the dimensions configured in your S3 vector index properties. If you are ingesting embeddings for several objects in an S3 prefix or local file path, it automatically uses batch processes to maximize throughput. 

**Note**: Each file is processed as a single embedding. Document chunking is not currently supported. 

**s3vectors-embed query**: Embed a query input and search for similar vectors in an S3 vector index.
You can perform similarity queries for vector embeddings in your S3 vector index using a single query command. You specify your query input, an Amazon Bedrock embeddings model ID, the vector bucket name, and vector index name. The command accepts several types of query inputs like a text string, an image file, or a single S3 text or image object. The command generates embeddings for your query using the input embeddings model and then performs a similarity search to find the most relevant matches. You can control the number of results returned, apply metadata filters to narrow your search, and choose whether to include similarity distance in the results for comprehensive analysis.


## Installation and Configuration
### Prerequisites
- Python 3.8 or higher
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

#### **Query Examples**

1. **Query with no filters:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --query-input "query text" \
  --k 10
```

2. **Query using a local text file as input:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --query-input "./query.txt" \
  --k 5 \
  --output table
```

3. **Query using an S3 text file as input:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --query-input "s3://my-bucket/image.jpeg" \
  --k 3 
```

4. **Query with metadata filters:**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --query-input "query text" \
  --filter '{"category": {"$eq": "technology"}}' \
  --k 10 \
  --return-metadata
```

5. **Query with multiple metadata filters (AND):**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --query-input "query text" \
  --filter '{"$and": [{"category": "technology"}, {"version": "1.0"}]}' \
  --k 10 \
  --return-metadata
```

6. **Query with multiple metadata filters (OR):**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --query-input "query text" \
  --filter '{"$or": [{"category": "docs"}, {"category": "guides"}]}' \
  --k 5
```

7. **Query with metadata filters (comparison operators):**
```bash
s3vectors-embed query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --query-input "query text" \
  --filter '{"$and": [{"category": "tech"}, {"version": {"$gte": "1.0"}}]}' \
  --k 10
```


### Command Parameters

#### Global Options
- `--debug`: Enable debug mode with detailed logging for troubleshooting
- `--profile`: AWS profile name to use from ~/.aws/credentials
- `--region`: AWS region name (overrides session/config defaults)

#### Put Command Parameters
Required:
- `--vector-bucket-name`: Name of the S3 vector bucket 
- `--index-name`: Name of the vector index in your vector index to store the vector embeddings
- `--model-id`: Bedrock model ID to use for generating embeddings (e.g., amazon.titan-embed-text-v2:0)

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

Optional:
- `--key`: Uniquely identifies each vector in the vector index (default: auto-generated UUID)
- `--metadata`: Additional metadata associated with the vector; provided as JSON string
- `--bucket-owner`: AWS account ID for cross-account S3 access
- `--output`: Output format (json or table, default: json)

#### Query Command Parameters
Required:
- `--vector-bucket-name`: Name of the S3 vector bucket
- `--index-name`: Name of the vector index 
- `--model-id`: Bedrock model ID to use for generating embeddings (e.g., amazon.titan-embed-text-v2:0)
- `--query-input`: Query text or file path (local file or S3 URI)

Optional:
- `--k`: Number of results to return (default: 5)
- `--filter`: Filter expression for metadata-based filtering (JSON format with AWS S3 Vectors API operators)
- `--return-metadata`: Include metadata in results (default: true)
- `--return-distance`: Include similarity distance
- `--output`: Output format (table or json, default: json)
- `--region`: AWS region name

Example with all optional parameters:
```bash
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --query-input "search query" \
  --k 10 --filter '{"$and": [{"category": "tech"}, {"version": {"$gte": "1.0"}}]}' --return-metadata \
  --return-distance --output table --region us-west-2
```

### Model Compatibility
| Model | Type | Dimensions | Use Case |
|-------|------|------------|----------|
| `amazon.titan-embed-text-v2:0` | Text | 1024, 512, 256 | Modern text embedding |
| `amazon.titan-embed-text-v1` | Text | 1536 | Legacy text embedding |
| `amazon.titan-embed-image-v1` | Multimodal (Text + Image) | 1024, 384, 256 | Text and image embedding |
| `cohere.embed-english-v3` | Multimodal (Text or Image) | 1024 | Advanced English text or image embedding |
| `cohere.embed-multilingual-v3` | Multimodal (Text or Image) | 1024 | Multilingual text or image embedding |

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
  --model-id amazon.titan-embed-text-v2:0 --query-input "search text"

# Explicit JSON format (same as default)
s3vectors-embed query --vector-bucket-name my-bucket --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 --query-input "search text" --output json
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
  --model-id amazon.titan-embed-text-v2:0 --query-input "search text" --output table
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

The CLI supports efficient batch processing for multiple files using both local and S3 wildcard characters in the input path

### **Batch Processing Features**

- **Automatic batching**: Large datasets are automatically split into batches of 500 vectors
- **Parallel processing**: Configurable worker threads for concurrent file processing
- **Error resilience**: Individual file failures don't stop batch processing
- **Performance optimization**: Efficient memory usage and API call batching

### Batch Processing Examples

**Local files batch processing (NEW):**
```bash
# Process all local text files
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "./documents/*.txt" \
  --metadata '{"source": "local_batch", "category": "documents"}' \
  --max-workers 4

# Process files from multiple directories
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "~/data/**/*.md" \
  --max-workers 2
```

**S3 files batch processing:**
```bash
# Text files batch processing
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-text-v2:0 \
  --text "s3://bucket/text/*" \
  --metadata '{"category": "documents", "batch": "2024-01"}' \
  --max-workers 4

# Image files batch processing
s3vectors-embed put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id amazon.titan-embed-image-v1 \
  --image "s3://bucket/images/*" \
  --metadata '{"category": "images", "source": "batch_upload"}' \
  --max-workers 2
```

### **Batch Processing Output**

```bash
# Example output for local wildcard processing
Processing chunk 1...
Found 94 supported files in chunk 1
Batch stored successfully. Total processed: 94

Batch processing completed!
   Total files found: 94
   Successfully processed: 94
   Failed: 0
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

4. **Performance Issues**
```bash
# Use debug mode to identify bottlenecks:
s3vectors-embed --debug put ...

# Debug output shows timing:
#  Bedrock API call completed in 2.45 seconds (slow)
#  S3 Vectors put_vectors completed in 0.15 seconds (normal)
```

5. **Service Unavailable Errors**
```bash
# Error: ServiceUnavailableException
# Debug output provides context:
# S3 Vectors ClientError: ServiceUnavailableException when calling PutVectors
# API parameters: {"vectorBucketName": "...", "indexName": "..."}
```

## Repository Structure
```
s3vectors-embed-cli/
├── s3vectors/                    # Main package directory
│   ├── cli.py                    # Main CLI entry point
│   ├── commands/                 # Command implementations
│   │   ├── embed_put.py         # Vector embedding and storage
│   │   └── embed_query.py       # Vector similarity search
│   ├── core/                    # Core functionality
│   │   ├── batch_processor.py   # Batch processing implementation
│   │   └── services.py         # Bedrock and S3Vector services
│   └── utils/                   # Utility functions
│       └── config.py           # AWS configuration management
├── setup.py                    # Package installation configuration
├── pyproject.toml              # Modern Python packaging configuration
├── requirements.txt            # Python dependencies
├── LICENSE                     # Apache 2.0 license
```
