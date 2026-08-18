#!/bin/bash
# Nova Embedding Model - Complete Integration Test Script with Output Validation
# Usage: chmod +x run_nova_tests.sh && ./run_nova_tests.sh
# Output: /tmp/nova-test-results.txt
#
# Prerequisites:
#   - AWS credentials configured
#   - s3vectors-embed CLI installed (pip install s3vectors-embed-cli)
#   - Vector bucket "photo-discovery-bucket" with index "photo-discovery-index" created
#   - Sample images in s3://workshop-samples-10d4f9c0/images/images/photos/
#   - For video/audio tests: upload test files to s3://workshop-samples-10d4f9c0/test_data/

VECTOR_BUCKET="photo-discovery-bucket"
INDEX_NAME="photo-discovery-index"
REGION="us-east-1"
MODEL_ID="amazon.nova-2-multimodal-embeddings-v1:0"
SAMPLES_BUCKET="workshop-samples-10d4f9c0"
OUTPUT_FILE="/tmp/nova-test-results.txt"

# Colors for terminal
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Initialize output file
echo "============================================================" | tee "$OUTPUT_FILE"
echo "NOVA EMBEDDING MODEL - COMPLETE INTEGRATION TEST RESULTS" | tee -a "$OUTPUT_FILE"
echo "Date: $(date)" | tee -a "$OUTPUT_FILE"
echo "Model: $MODEL_ID" | tee -a "$OUTPUT_FILE"
echo "Vector Bucket: $VECTOR_BUCKET" | tee -a "$OUTPUT_FILE"
echo "Index: $INDEX_NAME" | tee -a "$OUTPUT_FILE"
echo "Region: $REGION" | tee -a "$OUTPUT_FILE"
echo "Samples Bucket: $SAMPLES_BUCKET" | tee -a "$OUTPUT_FILE"
echo "============================================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

PASS=0
FAIL=0
SKIP=0

# ============================================================
# VALIDATION HELPERS
# ============================================================

# Check if output contains a string
assert_contains() {
    local output="$1"
    local expected="$2"
    local label="$3"
    if echo "$output" | grep -q "$expected"; then
        return 0
    else
        echo "  ⚠️  ASSERTION FAILED: Expected output to contain '$expected' ($label)" | tee -a "$OUTPUT_FILE"
        return 1
    fi
}

# Check if output does NOT contain a string
assert_not_contains() {
    local output="$1"
    local unexpected="$2"
    local label="$3"
    if echo "$output" | grep -q "$unexpected"; then
        echo "  ⚠️  ASSERTION FAILED: Output should NOT contain '$unexpected' ($label)" | tee -a "$OUTPUT_FILE"
        return 1
    else
        return 0
    fi
}

# Check JSON field value using grep (no jq dependency)
assert_json_field() {
    local output="$1"
    local field="$2"
    local expected_value="$3"
    if echo "$output" | grep -q "\"$field\": $expected_value\|\"$field\":$expected_value\|\"$field\": \"$expected_value\""; then
        return 0
    else
        echo "  ⚠️  ASSERTION FAILED: Expected \"$field\" to be $expected_value" | tee -a "$OUTPUT_FILE"
        return 1
    fi
}

# ============================================================
# TEST RUNNER
# ============================================================

run_test() {
    local test_id="$1"
    local description="$2"
    local expect_fail="$3"  # "true" if we expect failure
    # VALIDATIONS is set before calling run_test

    echo "------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
    echo "[$test_id] $description" | tee -a "$OUTPUT_FILE"
    echo "------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
    echo "Command:" | tee -a "$OUTPUT_FILE"
    echo "  $CMD" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    OUTPUT=$(eval "$CMD" 2>&1)
    EXIT_CODE=$?

    echo "Output:" | tee -a "$OUTPUT_FILE"
    echo "$OUTPUT" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    local validation_passed=true

    if [ "$expect_fail" = "true" ]; then
        if [ $EXIT_CODE -ne 0 ]; then
            # Run validations on error output
            if [ -n "$VALIDATIONS" ]; then
                eval "$VALIDATIONS" || validation_passed=false
            fi
            if [ "$validation_passed" = true ]; then
                echo -e "${GREEN}Result: ✅ PASS (expected error, output validated)${NC}" | tee -a "$OUTPUT_FILE"
                PASS=$((PASS + 1))
            else
                echo -e "${RED}Result: ❌ FAIL (error occurred but output validation failed)${NC}" | tee -a "$OUTPUT_FILE"
                FAIL=$((FAIL + 1))
            fi
        else
            echo -e "${RED}Result: ❌ FAIL (expected error but succeeded)${NC}" | tee -a "$OUTPUT_FILE"
            FAIL=$((FAIL + 1))
        fi
    else
        if [ $EXIT_CODE -eq 0 ]; then
            # Run validations on success output
            if [ -n "$VALIDATIONS" ]; then
                eval "$VALIDATIONS" || validation_passed=false
            fi
            if [ "$validation_passed" = true ]; then
                echo -e "${GREEN}Result: ✅ PASS (output validated)${NC}" | tee -a "$OUTPUT_FILE"
                PASS=$((PASS + 1))
            else
                echo -e "${RED}Result: ❌ FAIL (command succeeded but output validation failed)${NC}" | tee -a "$OUTPUT_FILE"
                FAIL=$((FAIL + 1))
            fi
        else
            echo -e "${RED}Result: ❌ FAIL${NC}" | tee -a "$OUTPUT_FILE"
            FAIL=$((FAIL + 1))
        fi
    fi

    # Reset validations
    VALIDATIONS=""
    echo "" | tee -a "$OUTPUT_FILE"
}

skip_test() {
    local test_id="$1"
    local description="$2"
    local reason="$3"

    echo "------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
    echo "[$test_id] $description" | tee -a "$OUTPUT_FILE"
    echo "------------------------------------------------------------" | tee -a "$OUTPUT_FILE"
    echo -e "${YELLOW}Result: ⏭️  SKIPPED ($reason)${NC}" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    SKIP=$((SKIP + 1))
}

# ============================================================
# SETUP: Create test data files
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== SETUP: Creating test data ==========" | tee -a "$OUTPUT_FILE"

# Create a local text file for testing
echo "Amazon Nova Multimodal Embeddings is a state-of-the-art model for semantic search. It supports text, images, video, and audio through a unified embedding space." > /tmp/nova_test_sample.txt
echo "Created /tmp/nova_test_sample.txt" | tee -a "$OUTPUT_FILE"

# Upload text file to S3 for S3 text test
aws s3 cp /tmp/nova_test_sample.txt s3://$SAMPLES_BUCKET/test_data/sample.txt --quiet 2>/dev/null
echo "Uploaded test text file to s3://$SAMPLES_BUCKET/test_data/sample.txt" | tee -a "$OUTPUT_FILE"

# Check if video/audio test files exist in S3
VIDEO_EXISTS=$(aws s3 ls s3://$SAMPLES_BUCKET/test_data/clip.mp4 2>/dev/null | wc -l)
AUDIO_EXISTS=$(aws s3 ls s3://$SAMPLES_BUCKET/test_data/audio.mp3 2>/dev/null | wc -l)
LONG_VIDEO_EXISTS=$(aws s3 ls s3://$SAMPLES_BUCKET/test_data/long_video_60s.mp4 2>/dev/null | wc -l)

echo "Video test file (clip.mp4): $([ $VIDEO_EXISTS -gt 0 ] && echo 'Found' || echo 'Not found - video tests will be skipped')" | tee -a "$OUTPUT_FILE"
echo "Audio test file (audio.mp3): $([ $AUDIO_EXISTS -gt 0 ] && echo 'Found' || echo 'Not found - audio tests will be skipped')" | tee -a "$OUTPUT_FILE"
echo "Long video file (long_video_60s.mp4): $([ $LONG_VIDEO_EXISTS -gt 0 ] && echo 'Found' || echo 'Not found - long video test will be skipped')" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# ============================================================
# N1: TEXT EMBEDDINGS
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N1: TEXT EMBEDDINGS ==========" | tee -a "$OUTPUT_FILE"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'A golden retriever playing fetch in a sunny park' \
  --metadata '{\"test_id\": \"N1.1\", \"category\": \"text_basic\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\"" "has key field" &&
assert_json_field "$OUTPUT" "embeddingDimensions" "1024" &&
assert_json_field "$OUTPUT" "contentType" "text" &&
assert_contains "$OUTPUT" "\"model\": \"amazon.nova-2-multimodal-embeddings-v1:0\"" "correct model" &&
assert_contains "$OUTPUT" "S3VECTORS-EMBED-SRC-CONTENT" "has source content metadata"
'
run_test "N1.1" "Basic text embedding"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'Testing truncation mode START which omits characters from the beginning of text when exceeding token limits' \
  --bedrock-inference-params '{\"truncationMode\": \"START\"}' \
  --metadata '{\"test_id\": \"N1.2\", \"category\": \"text_truncation\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\"" "has key field" &&
assert_json_field "$OUTPUT" "embeddingDimensions" "1024" &&
assert_json_field "$OUTPUT" "contentType" "text"
'
run_test "N1.2" "Text with truncationMode=START override"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text '/tmp/nova_test_sample.txt' \
  --metadata '{\"test_id\": \"N1.3\", \"category\": \"text_local_file\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\"" "has key field" &&
assert_json_field "$OUTPUT" "contentType" "text" &&
assert_contains "$OUTPUT" "S3VECTORS-EMBED-SRC-LOCATION" "has source location metadata" &&
assert_contains "$OUTPUT" "S3VECTORS-EMBED-SRC-CONTENT" "has source content metadata"
'
run_test "N1.3" "Local text file embedding"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text 's3://$SAMPLES_BUCKET/test_data/sample.txt' \
  --metadata '{\"test_id\": \"N1.4\", \"category\": \"text_s3_file\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\"" "has key field" &&
assert_json_field "$OUTPUT" "contentType" "text" &&
assert_contains "$OUTPUT" "s3://" "has S3 source location"
'
run_test "N1.4" "S3 text file embedding"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'golden retriever dog playing' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"TEXT_RETRIEVAL\"}' \
  --k 3 \
  --return-distance \
  --return-metadata"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results array" &&
assert_contains "$OUTPUT" "\"resultsFound\"" "has resultsFound" &&
assert_not_contains "$OUTPUT" "\"resultsFound\": 0" "results not empty" &&
assert_json_field "$OUTPUT" "queryDimensions" "1024" &&
assert_contains "$OUTPUT" "\"distance\"" "has distance scores"
'
run_test "N1.5" "Text query with TEXT_RETRIEVAL purpose"

# ============================================================
# N2: IMAGE EMBEDDINGS
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N2: IMAGE EMBEDDINGS ==========" | tee -a "$OUTPUT_FILE"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --image 's3://$SAMPLES_BUCKET/images/images/photos/cat_on_sofa.jpg' \
  --metadata '{\"test_id\": \"N2.1\", \"category\": \"cat\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\"" "has key field" &&
assert_json_field "$OUTPUT" "contentType" "image" &&
assert_json_field "$OUTPUT" "embeddingDimensions" "1024" &&
assert_contains "$OUTPUT" "S3VECTORS-EMBED-SRC-LOCATION" "has source location" &&
assert_contains "$OUTPUT" "cat_on_sofa.jpg" "location contains filename"
'
run_test "N2.1" "Single S3 image embedding"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --image 's3://$SAMPLES_BUCKET/images/images/photos/beach_sunset.jpg' \
  --bedrock-inference-params '{\"detailLevel\": \"DOCUMENT_IMAGE\"}' \
  --metadata '{\"test_id\": \"N2.2\", \"category\": \"landscape\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\"" "has key field" &&
assert_json_field "$OUTPUT" "contentType" "image" &&
assert_json_field "$OUTPUT" "embeddingDimensions" "1024" &&
assert_contains "$OUTPUT" "beach_sunset.jpg" "location contains filename"
'
run_test "N2.2" "Image with DOCUMENT_IMAGE detail level"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --image 's3://$SAMPLES_BUCKET/images/images/photos/*' \
  --metadata '{\"test_id\": \"N2.3\", \"batch\": \"all_photos\"}' \
  --max-workers 4"
VALIDATIONS='
assert_contains "$OUTPUT" "\"processedFiles\"" "has processedFiles" &&
assert_not_contains "$OUTPUT" "\"processedFiles\": 0" "processed at least 1 file" &&
assert_contains "$OUTPUT" "\"failedFiles\": 0" "zero failures" &&
assert_contains "$OUTPUT" "\"totalVectors\"" "has totalVectors count" &&
assert_json_field "$OUTPUT" "contentType" "image"
'
run_test "N2.3" "Batch embed all photos from S3 prefix"

# ============================================================
# N3: VIDEO EMBEDDINGS
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N3: VIDEO EMBEDDINGS ==========" | tee -a "$OUTPUT_FILE"

if [ $VIDEO_EXISTS -gt 0 ]; then
    CMD="s3vectors-embed put \
      --vector-bucket-name $VECTOR_BUCKET \
      --index-name $INDEX_NAME \
      --region $REGION \
      --model-id $MODEL_ID \
      --video 's3://$SAMPLES_BUCKET/test_data/clip.mp4' \
      --metadata '{\"test_id\": \"N3.1\", \"category\": \"video_combined\"}'"
    VALIDATIONS='
    assert_contains "$OUTPUT" "\"key\"" "has key field" &&
    assert_json_field "$OUTPUT" "contentType" "video" &&
    assert_json_field "$OUTPUT" "embeddingDimensions" "1024"
    '
    run_test "N3.1" "Video embedding (AUDIO_VIDEO_COMBINED default)"

    CMD="s3vectors-embed put \
      --vector-bucket-name $VECTOR_BUCKET \
      --index-name $INDEX_NAME \
      --region $REGION \
      --model-id $MODEL_ID \
      --video 's3://$SAMPLES_BUCKET/test_data/clip.mp4' \
      --bedrock-inference-params '{\"embeddingMode\": \"AUDIO_VIDEO_SEPARATE\"}' \
      --metadata '{\"test_id\": \"N3.2\", \"category\": \"video_separate\"}'"
    VALIDATIONS='
    assert_contains "$OUTPUT" "\"key\"" "has key field" &&
    assert_json_field "$OUTPUT" "contentType" "video" &&
    assert_json_field "$OUTPUT" "embeddingDimensions" "1024"
    '
    run_test "N3.2" "Video with AUDIO_VIDEO_SEPARATE mode"

    CMD="s3vectors-embed query \
      --vector-bucket-name $VECTOR_BUCKET \
      --index-name $INDEX_NAME \
      --region $REGION \
      --model-id $MODEL_ID \
      --text-value 'video content action scene' \
      --bedrock-inference-params '{\"embeddingPurpose\": \"VIDEO_RETRIEVAL\"}' \
      --k 3 \
      --return-distance \
      --return-metadata"
    VALIDATIONS='
    assert_contains "$OUTPUT" "\"results\"" "has results array" &&
    assert_json_field "$OUTPUT" "queryDimensions" "1024"
    '
    run_test "N3.3" "Text-to-video cross-modal query"
else
    skip_test "N3.1" "Video embedding (AUDIO_VIDEO_COMBINED default)" "No clip.mp4 in s3://$SAMPLES_BUCKET/test_data/"
    skip_test "N3.2" "Video with AUDIO_VIDEO_SEPARATE mode" "No clip.mp4 in s3://$SAMPLES_BUCKET/test_data/"
    skip_test "N3.3" "Text-to-video cross-modal query" "No video embeddings to query"
fi

# ============================================================
# N4: AUDIO EMBEDDINGS
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N4: AUDIO EMBEDDINGS ==========" | tee -a "$OUTPUT_FILE"

if [ $AUDIO_EXISTS -gt 0 ]; then
    CMD="s3vectors-embed put \
      --vector-bucket-name $VECTOR_BUCKET \
      --index-name $INDEX_NAME \
      --region $REGION \
      --model-id $MODEL_ID \
      --audio 's3://$SAMPLES_BUCKET/test_data/audio.mp3' \
      --metadata '{\"test_id\": \"N4.1\", \"category\": \"audio\"}'"
    VALIDATIONS='
    assert_contains "$OUTPUT" "\"key\"" "has key field" &&
    assert_json_field "$OUTPUT" "contentType" "audio" &&
    assert_json_field "$OUTPUT" "embeddingDimensions" "1024"
    '
    run_test "N4.1" "Audio embedding (mp3)"

    CMD="s3vectors-embed query \
      --vector-bucket-name $VECTOR_BUCKET \
      --index-name $INDEX_NAME \
      --region $REGION \
      --model-id $MODEL_ID \
      --text-value 'speech conversation dialogue' \
      --bedrock-inference-params '{\"embeddingPurpose\": \"AUDIO_RETRIEVAL\"}' \
      --k 3 \
      --return-distance \
      --return-metadata"
    VALIDATIONS='
    assert_contains "$OUTPUT" "\"results\"" "has results array" &&
    assert_json_field "$OUTPUT" "queryDimensions" "1024"
    '
    run_test "N4.2" "Text-to-audio cross-modal query"
else
    skip_test "N4.1" "Audio embedding (mp3)" "No audio.mp3 in s3://$SAMPLES_BUCKET/test_data/"
    skip_test "N4.2" "Text-to-audio cross-modal query" "No audio embeddings to query"
fi

# ============================================================
# N5: CROSS-MODAL QUERY (Text → Image)
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N5: CROSS-MODAL QUERY (Text → Image) ==========" | tee -a "$OUTPUT_FILE"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'a dog playing in the park' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"IMAGE_RETRIEVAL\"}' \
  --k 5 \
  --return-distance \
  --return-metadata"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results array" &&
assert_not_contains "$OUTPUT" "\"resultsFound\": 0" "results not empty" &&
assert_json_field "$OUTPUT" "queryDimensions" "1024" &&
assert_contains "$OUTPUT" "\"distance\"" "has distance scores"
'
run_test "N5.1" "Text-to-image search: dog"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'a cat sitting on furniture' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"IMAGE_RETRIEVAL\"}' \
  --k 5 \
  --return-distance \
  --return-metadata"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results array" &&
assert_not_contains "$OUTPUT" "\"resultsFound\": 0" "results not empty" &&
assert_contains "$OUTPUT" "cat_on_sofa" "cat image in results"
'
run_test "N5.2" "Text-to-image search: cat (expects cat_on_sofa in results)"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'beautiful sunset over the ocean' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"IMAGE_RETRIEVAL\"}' \
  --k 3 \
  --return-distance \
  --return-metadata"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results array" &&
assert_not_contains "$OUTPUT" "\"resultsFound\": 0" "results not empty" &&
assert_contains "$OUTPUT" "beach_sunset" "sunset image in results"
'
run_test "N5.3" "Text-to-image search: sunset (expects beach_sunset in results)"

# ============================================================
# N6: IMAGE → IMAGE QUERY
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N6: IMAGE-TO-IMAGE QUERY ==========" | tee -a "$OUTPUT_FILE"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --image 's3://$SAMPLES_BUCKET/images/images/photos/golden_retriver_beach.jpg' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"IMAGE_RETRIEVAL\"}' \
  --k 5 \
  --return-distance \
  --return-metadata"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results array" &&
assert_not_contains "$OUTPUT" "\"resultsFound\": 0" "results not empty" &&
assert_contains "$OUTPUT" "golden_retriver" "golden retriever image in results" &&
assert_json_field "$OUTPUT" "queryType" "image"
'
run_test "N6.1" "Image-to-image similarity search (expects same/similar images)"

# ============================================================
# N7: METADATA FILTERS
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N7: METADATA FILTERS ==========" | tee -a "$OUTPUT_FILE"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'animal' \
  --filter '{\"category\": {\"\$eq\": \"cat\"}}' \
  --k 5 \
  --return-distance \
  --return-metadata"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results array" &&
assert_contains "$OUTPUT" "\"category\": \"cat\"" "all results have category=cat"
'
run_test "N7.1" "Query with metadata filter (category=cat)"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'nature scene' \
  --filter '{\"\$or\": [{\"category\": \"cat\"}, {\"category\": \"landscape\"}]}' \
  --k 5 \
  --return-distance \
  --return-metadata"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results array"
'
run_test "N7.2" "Query with OR metadata filter"

# ============================================================
# N8: KEY MANAGEMENT
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N8: KEY MANAGEMENT ==========" | tee -a "$OUTPUT_FILE"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'Custom key test' \
  --key 'nova-test-001' \
  --metadata '{\"test_id\": \"N8.1\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\": \"nova-test-001\"" "key matches expected value"
'
run_test "N8.1" "Custom vector key (expects key=nova-test-001)"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --image 's3://$SAMPLES_BUCKET/images/images/photos/golden_retriver_beach_2.jpg' \
  --filename-as-key \
  --metadata '{\"test_id\": \"N8.2\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\": \"golden_retriver_beach_2.jpg\"" "key is filename"
'
run_test "N8.2" "Filename as vector key (expects key=golden_retriver_beach_2.jpg)"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'Key prefix test' \
  --key 'doc-001' \
  --key-prefix 'nova/' \
  --metadata '{\"test_id\": \"N8.3\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\": \"nova/doc-001\"" "key has prefix applied"
'
run_test "N8.3" "Key with prefix (expects key=nova/doc-001)"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --image 's3://$SAMPLES_BUCKET/images/images/photos/golden_retriver_beach_3.jpg' \
  --filename-as-key \
  --key-prefix 'photos/' \
  --metadata '{\"test_id\": \"N8.4\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\": \"photos/golden_retriver_beach_3.jpg\"" "key has prefix + filename"
'
run_test "N8.4" "Filename as key with prefix (expects key=photos/golden_retriver_beach_3.jpg)"

# ============================================================
# N9: OUTPUT FORMATS & DEBUG
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N9: OUTPUT FORMATS & DEBUG ==========" | tee -a "$OUTPUT_FILE"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'dog' \
  --k 3 \
  --return-distance \
  --output table"
VALIDATIONS='
assert_contains "$OUTPUT" "Query Results" "table header present" &&
assert_contains "$OUTPUT" "Query Summary" "summary section present" &&
assert_contains "$OUTPUT" "Results Found" "results found in summary"
'
run_test "N9.1" "Table output format"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'sunset beach' \
  --k 3 \
  --return-distance \
  --output json"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "JSON has results key" &&
assert_contains "$OUTPUT" "\"summary\"" "JSON has summary key"
'
run_test "N9.2" "JSON output format (explicit)"

CMD="s3vectors-embed --debug put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'Debug mode test for Nova model'"
VALIDATIONS='
assert_contains "$OUTPUT" "Debug mode enabled" "debug banner shown" &&
assert_contains "$OUTPUT" "BedrockService initialized" "service init logged" &&
assert_contains "$OUTPUT" "S3VectorService initialized" "vector service init logged" &&
assert_contains "$OUTPUT" "Payload:" "payload logged" &&
assert_contains "$OUTPUT" "SINGLE_EMBEDDING" "Nova payload visible in debug" &&
assert_contains "$OUTPUT" "Bedrock API call completed" "API timing logged" &&
assert_contains "$OUTPUT" "1024 dimensions" "dimensions logged"
'
run_test "N9.3" "Debug mode shows payload, timing, and dimensions"

# ============================================================
# N10: ERROR HANDLING & VALIDATION
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N10: ERROR HANDLING & VALIDATION ==========" | tee -a "$OUTPUT_FILE"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'This should fail' \
  --bedrock-inference-params '{\"taskType\": \"SEGMENTED_EMBEDDING\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "Cannot override system-controlled parameters" "correct error message" &&
assert_contains "$OUTPUT" "taskType" "mentions blocked parameter"
'
run_test "N10.1" "Blocked system key override (expected error)" "true"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'Nova does not need async bucket'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\"" "succeeds without async bucket" &&
assert_json_field "$OUTPUT" "contentType" "text"
'
run_test "N10.2" "No --async-output-s3-uri needed (sync model)"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'This should fail' \
  --bedrock-inference-params '{\"singleEmbeddingParams\": {\"embeddingPurpose\": \"CLUSTERING\"}}'"
VALIDATIONS='
assert_contains "$OUTPUT" "Cannot override system-controlled parameters" "correct error message" &&
assert_contains "$OUTPUT" "singleEmbeddingParams" "mentions blocked parameter"
'
run_test "N10.3" "Blocked singleEmbeddingParams override (expected error)" "true"

if [ $LONG_VIDEO_EXISTS -gt 0 ]; then
    CMD="s3vectors-embed put \
      --vector-bucket-name $VECTOR_BUCKET \
      --index-name $INDEX_NAME \
      --region $REGION \
      --model-id $MODEL_ID \
      --video 's3://$SAMPLES_BUCKET/test_data/long_video_60s.mp4' \
      --metadata '{\"test_id\": \"N10.4\"}'"
    VALIDATIONS='
    assert_contains "$OUTPUT" "error\|Error\|failed\|Failed" "error message present"
    '
    run_test "N10.4" "Video exceeding 30s (expected Bedrock error)" "true"
else
    skip_test "N10.4" "Video exceeding 30s (expected Bedrock error)" "No long_video_60s.mp4 in s3://$SAMPLES_BUCKET/test_data/"
fi

# ============================================================
# N11: EMBEDDING PURPOSE VARIATIONS
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "========== N11: EMBEDDING PURPOSE VARIATIONS ==========" | tee -a "$OUTPUT_FILE"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'animals in nature' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"GENERIC_RETRIEVAL\"}' \
  --k 5 \
  --return-distance"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results" &&
assert_json_field "$OUTPUT" "queryDimensions" "1024"
'
run_test "N11.1" "Query with GENERIC_RETRIEVAL purpose"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'Classification embedding test' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"CLASSIFICATION\"}' \
  --metadata '{\"test_id\": \"N11.2\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\"" "has key field" &&
assert_json_field "$OUTPUT" "embeddingDimensions" "1024"
'
run_test "N11.2" "Put with CLASSIFICATION purpose"

CMD="s3vectors-embed put \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'Clustering embedding test' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"CLUSTERING\"}' \
  --metadata '{\"test_id\": \"N11.3\"}'"
VALIDATIONS='
assert_contains "$OUTPUT" "\"key\"" "has key field" &&
assert_json_field "$OUTPUT" "embeddingDimensions" "1024"
'
run_test "N11.3" "Put with CLUSTERING purpose"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'document with text and tables' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"DOCUMENT_RETRIEVAL\"}' \
  --k 3 \
  --return-distance"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results" &&
assert_json_field "$OUTPUT" "queryDimensions" "1024"
'
run_test "N11.4" "Query with DOCUMENT_RETRIEVAL purpose"

CMD="s3vectors-embed query \
  --vector-bucket-name $VECTOR_BUCKET \
  --index-name $INDEX_NAME \
  --region $REGION \
  --model-id $MODEL_ID \
  --text-value 'music and speech' \
  --bedrock-inference-params '{\"embeddingPurpose\": \"AUDIO_RETRIEVAL\"}' \
  --k 3 \
  --return-distance"
VALIDATIONS='
assert_contains "$OUTPUT" "\"results\"" "has results" &&
assert_json_field "$OUTPUT" "queryDimensions" "1024"
'
run_test "N11.5" "Query with AUDIO_RETRIEVAL purpose"

# ============================================================
# SUMMARY
# ============================================================
echo "" | tee -a "$OUTPUT_FILE"
echo "============================================================" | tee -a "$OUTPUT_FILE"
echo "TEST SUMMARY" | tee -a "$OUTPUT_FILE"
echo "============================================================" | tee -a "$OUTPUT_FILE"
TOTAL=$((PASS + FAIL))
echo "Total Run: $TOTAL" | tee -a "$OUTPUT_FILE"
echo "Passed:    $PASS" | tee -a "$OUTPUT_FILE"
echo "Failed:    $FAIL" | tee -a "$OUTPUT_FILE"
echo "Skipped:   $SKIP" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED ($PASS/$TOTAL) — Output validated${NC}" | tee -a "$OUTPUT_FILE"
else
    echo -e "${RED}❌ $FAIL TEST(S) FAILED ($PASS/$TOTAL passed)${NC}" | tee -a "$OUTPUT_FILE"
fi

if [ $SKIP -gt 0 ]; then
    echo -e "${YELLOW}⏭️  $SKIP test(s) skipped (upload video/audio files to enable)${NC}" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    echo "To enable video/audio tests, upload files to S3:" | tee -a "$OUTPUT_FILE"
    echo "  aws s3 cp clip.mp4 s3://$SAMPLES_BUCKET/test_data/clip.mp4" | tee -a "$OUTPUT_FILE"
    echo "  aws s3 cp audio.mp3 s3://$SAMPLES_BUCKET/test_data/audio.mp3" | tee -a "$OUTPUT_FILE"
    echo "  aws s3 cp long_video_60s.mp4 s3://$SAMPLES_BUCKET/test_data/long_video_60s.mp4" | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"
echo "Results saved to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
