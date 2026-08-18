# 📊 AMAZON NOVA MULTIMODAL EMBEDDINGS - TEST REPORT

## Test Configuration
- **Vector Bucket**: `<YOUR_VECTOR_BUCKET>`
- **Index**: `<YOUR_INDEX_NAME>` (1024 dimensions recommended)
- **Region**: `us-east-1`
- **Date**: 2026-08-XX
- **CLI Version**: 0.3.0
- **Model**: `amazon.nova-2-multimodal-embeddings-v1:0`

### Prerequisites
- S3 vector bucket created with a vector index (dimensions: 256, 384, 1024, or 3072)
- Nova Multimodal Embeddings model enabled in your AWS account
- Test files prepared:
  - A local text file (e.g., `./test_data/sample.txt`)
  - A local image file (e.g., `./test_data/test_image.jpg`)
  - A local video file ≤30 seconds (e.g., `./test_data/clip.mp4`)
  - A local audio file ≤30 seconds (e.g., `./test_data/audio.mp3`)
  - A text-heavy image/document (e.g., `./test_data/invoice.png`)
  - Multiple image files for batch test (e.g., `./test_data/images/*.jpg`)

---

## 🧪 TEST RESULTS

### Test Series N1: Nova Text Embeddings

#### Test N1.1: Direct Text Input - Basic
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "Amazon Nova generates high-quality multimodal embeddings for search and retrieval applications" \
  --metadata '{"test_scenario": "nova_text_basic", "model": "nova-multimodal", "test_id": "N1.1"}'
```
**Expected**: ✅ SUCCESS - Single vector stored with 1024 dimensions
**Result**: 

**QUERY Command:**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "multimodal embeddings search retrieval" \
  --k 1 \
  --return-metadata \
  --return-distance \
  --filter '{"test_id": "N1.1"}'
```
**Expected**: ✅ SUCCESS - 1 result found
**Result**: 

---

#### Test N1.2: Direct Text Input - With truncationMode Override
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "This tests truncation behavior when text exceeds the maximum token limit of the model. The START mode will omit characters from the beginning of the text." \
  --bedrock-inference-params '{"truncationMode": "START"}' \
  --metadata '{"test_scenario": "nova_text_truncation", "model": "nova-multimodal", "test_id": "N1.2"}'
```
**Expected**: ✅ SUCCESS - Embedding generated with START truncation mode
**Result**: 

---

#### Test N1.3: Local Text File Input
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text "./test_data/sample.txt" \
  --metadata '{"test_scenario": "nova_text_file", "model": "nova-multimodal", "test_id": "N1.3"}'
```
**Expected**: ✅ SUCCESS - File content embedded as single vector
**Result**: 

---

#### Test N1.4: S3 Text File Input
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text "s3://<YOUR_BUCKET>/test_data/sample.txt" \
  --metadata '{"test_scenario": "nova_text_s3", "model": "nova-multimodal", "test_id": "N1.4"}'
```
**Expected**: ✅ SUCCESS - S3 file content embedded
**Result**: 

---

### Test Series N2: Nova Image Embeddings

#### Test N2.1: Local Image File - Standard
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --image "./test_data/test_image.jpg" \
  --metadata '{"test_scenario": "nova_image_basic", "model": "nova-multimodal", "test_id": "N2.1", "image_type": "standard"}'
```
**Expected**: ✅ SUCCESS - Image embedded with STANDARD_IMAGE detail level (default)
**Result**: 

**QUERY Command (cross-modal text→image):**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "description of the test image content" \
  --bedrock-inference-params '{"embeddingPurpose": "IMAGE_RETRIEVAL"}' \
  --k 3 \
  --return-metadata \
  --return-distance
```
**Expected**: ✅ SUCCESS - Cross-modal retrieval finds the image
**Result**: 

---

#### Test N2.2: Document Image - High Resolution (DOCUMENT_IMAGE)
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --image "./test_data/invoice.png" \
  --bedrock-inference-params '{"detailLevel": "DOCUMENT_IMAGE"}' \
  --metadata '{"test_scenario": "nova_image_document", "model": "nova-multimodal", "test_id": "N2.2", "image_type": "document"}'
```
**Expected**: ✅ SUCCESS - Document image embedded at high resolution
**Result**: 

**QUERY Command (document retrieval):**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "invoice total amount due payment" \
  --bedrock-inference-params '{"embeddingPurpose": "DOCUMENT_RETRIEVAL"}' \
  --k 1 \
  --return-metadata \
  --filter '{"test_id": "N2.2"}'
```
**Expected**: ✅ SUCCESS - Document found via text query
**Result**: 

---

#### Test N2.3: S3 Image File
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --image "s3://<YOUR_BUCKET>/test_data/test_image.jpg" \
  --metadata '{"test_scenario": "nova_image_s3", "model": "nova-multimodal", "test_id": "N2.3"}'
```
**Expected**: ✅ SUCCESS - S3 image embedded
**Result**: 

---

### Test Series N3: Nova Video Embeddings (≤30 seconds)

#### Test N3.1: Local Video - Combined Audio+Video (Default)
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --video "./test_data/clip.mp4" \
  --metadata '{"test_scenario": "nova_video_combined", "model": "nova-multimodal", "test_id": "N3.1", "mode": "AUDIO_VIDEO_COMBINED"}'
```
**Expected**: ✅ SUCCESS - Single combined embedding generated
**Result**: 

**QUERY Command (cross-modal text→video):**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "description of video clip content" \
  --bedrock-inference-params '{"embeddingPurpose": "VIDEO_RETRIEVAL"}' \
  --k 1 \
  --return-metadata \
  --return-distance \
  --filter '{"test_id": "N3.1"}'
```
**Expected**: ✅ SUCCESS - Video found via text query
**Result**: 

---

#### Test N3.2: Local Video - Separate Audio and Video Embeddings
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --video "./test_data/clip.mp4" \
  --bedrock-inference-params '{"embeddingMode": "AUDIO_VIDEO_SEPARATE"}' \
  --metadata '{"test_scenario": "nova_video_separate", "model": "nova-multimodal", "test_id": "N3.2", "mode": "AUDIO_VIDEO_SEPARATE"}'
```
**Expected**: ✅ SUCCESS - Note: CLI stores the first embedding (video portion) from the response
**Result**: 

---

### Test Series N4: Nova Audio Embeddings (≤30 seconds)

#### Test N4.1: Local Audio File
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --audio "./test_data/audio.mp3" \
  --metadata '{"test_scenario": "nova_audio_basic", "model": "nova-multimodal", "test_id": "N4.1"}'
```
**Expected**: ✅ SUCCESS - Audio embedded as single vector
**Result**: 

**QUERY Command (cross-modal text→audio):**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "description of audio content speech" \
  --bedrock-inference-params '{"embeddingPurpose": "AUDIO_RETRIEVAL"}' \
  --k 1 \
  --return-metadata \
  --return-distance \
  --filter '{"test_id": "N4.1"}'
```
**Expected**: ✅ SUCCESS - Audio found via text query
**Result**: 

---

#### Test N4.2: Local WAV Audio File
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --audio "./test_data/audio.wav" \
  --metadata '{"test_scenario": "nova_audio_wav", "model": "nova-multimodal", "test_id": "N4.2"}'
```
**Expected**: ✅ SUCCESS - WAV format audio embedded
**Result**: 

---

### Test Series N5: Nova embeddingPurpose Optimization

#### Test N5.1: GENERIC_INDEX (Put) → TEXT_RETRIEVAL (Query)
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "Machine learning models can be trained on large datasets to recognize patterns in unstructured data" \
  --metadata '{"test_scenario": "nova_purpose_index", "model": "nova-multimodal", "test_id": "N5.1"}'
```
**Expected**: ✅ SUCCESS - Default GENERIC_INDEX purpose used
**Result**: 

**QUERY with TEXT_RETRIEVAL:**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "machine learning pattern recognition" \
  --bedrock-inference-params '{"embeddingPurpose": "TEXT_RETRIEVAL"}' \
  --k 3 \
  --return-distance
```
**Expected**: ✅ SUCCESS - Optimized retrieval returns relevant results
**Result**: 

---

#### Test N5.2: GENERIC_INDEX (Put) → GENERIC_RETRIEVAL (Query)
**QUERY with GENERIC_RETRIEVAL:**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "machine learning pattern recognition" \
  --bedrock-inference-params '{"embeddingPurpose": "GENERIC_RETRIEVAL"}' \
  --k 3 \
  --return-distance
```
**Expected**: ✅ SUCCESS - Mixed-modality retrieval works
**Result**: 

---

### Test Series N6: Nova Batch Processing

#### Test N6.1: Local Image Wildcard Batch
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --image "./test_data/images/*.jpg" \
  --metadata '{"test_scenario": "nova_batch_images", "model": "nova-multimodal", "test_id": "N6.1"}' \
  --max-workers 4
```
**Expected**: ✅ SUCCESS - All images processed in parallel batch
**Result**: 

---

#### Test N6.2: Local Text Wildcard Batch
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text "./test_data/*.txt" \
  --metadata '{"test_scenario": "nova_batch_text", "model": "nova-multimodal", "test_id": "N6.2"}' \
  --max-workers 4
```
**Expected**: ✅ SUCCESS - All text files processed in batch
**Result**: 

---

#### Test N6.3: S3 Prefix Batch Processing
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --image "s3://<YOUR_BUCKET>/test_data/images/*" \
  --metadata '{"test_scenario": "nova_batch_s3", "model": "nova-multimodal", "test_id": "N6.3"}' \
  --max-workers 2
```
**Expected**: ✅ SUCCESS - S3 images batch processed
**Result**: 

---

### Test Series N7: Nova Key Management

#### Test N7.1: Custom Key
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "Custom key test for Nova model" \
  --key "nova-custom-key-001" \
  --metadata '{"test_scenario": "nova_custom_key", "test_id": "N7.1"}'
```
**Expected**: ✅ SUCCESS - Vector stored with key `nova-custom-key-001`
**Result**: 

---

#### Test N7.2: Filename as Key
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --image "./test_data/test_image.jpg" \
  --filename-as-key \
  --metadata '{"test_scenario": "nova_filename_key", "test_id": "N7.2"}'
```
**Expected**: ✅ SUCCESS - Vector stored with key `test_image.jpg`
**Result**: 

---

#### Test N7.3: Key Prefix
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "Key prefix test" \
  --key "doc-001" \
  --key-prefix "nova/" \
  --metadata '{"test_scenario": "nova_key_prefix", "test_id": "N7.3"}'
```
**Expected**: ✅ SUCCESS - Vector stored with key `nova/doc-001`
**Result**: 

---

### Test Series N8: Nova Output Formats & Debug

#### Test N8.1: JSON Output (Default)
**QUERY Command:**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "test query" \
  --k 3 \
  --output json
```
**Expected**: ✅ SUCCESS - JSON output with results array and summary
**Result**: 

---

#### Test N8.2: Table Output
**QUERY Command:**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "test query" \
  --k 3 \
  --output table
```
**Expected**: ✅ SUCCESS - Formatted table output
**Result**: 

---

#### Test N8.3: Debug Mode
**PUT Command:**
```bash
s3vectors-embed --debug put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "Debug mode test for Nova"
```
**Expected**: ✅ SUCCESS - Debug output shows payload, timing, and API details
**Result**: 

---

### Test Series N9: Nova Error Handling & Validation

#### Test N9.1: Invalid taskType Override (Should Be Blocked)
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "This should fail" \
  --bedrock-inference-params '{"taskType": "SEGMENTED_EMBEDDING"}'
```
**Expected**: ❌ ERROR - "Cannot override system-controlled parameters: ['taskType']"
**Result**: 

---

#### Test N9.2: Video Exceeding 30 Seconds
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --video "./test_data/long_video_60s.mp4"
```
**Expected**: ❌ ERROR - Bedrock should reject video exceeding 30s duration
**Result**: 

---

#### Test N9.3: No Async Bucket Required (Unlike TwelveLabs)
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --video "./test_data/clip.mp4"
```
**Expected**: ✅ SUCCESS - No --async-output-s3-uri needed (sync model)
**Result**: 

---

### Test Series N10: Nova Metadata & Filters

#### Test N10.1: Rich Metadata with Filters
**PUT Command:**
```bash
s3vectors-embed put \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "Product description for filtering test - electronics category" \
  --metadata '{"category": "electronics", "brand": "acme", "version": "2.0", "test_id": "N10.1"}'
```
**Expected**: ✅ SUCCESS
**Result**: 

**QUERY with complex filter:**
```bash
s3vectors-embed query \
  --vector-bucket-name <YOUR_VECTOR_BUCKET> \
  --index-name <YOUR_INDEX_NAME> \
  --region us-east-1 \
  --model-id amazon.nova-2-multimodal-embeddings-v1:0 \
  --text-value "electronics product" \
  --filter '{"$and": [{"category": {"$eq": "electronics"}}, {"version": {"$gte": "1.0"}}]}' \
  --k 5 \
  --return-metadata
```
**Expected**: ✅ SUCCESS - Results filtered by category AND version
**Result**: 

---

## 📊 TEST SUMMARY

| Series | Category | Tests | Passed | Failed |
|--------|----------|-------|--------|--------|
| N1 | Text Embeddings | 4 | | |
| N2 | Image Embeddings | 3 | | |
| N3 | Video Embeddings | 2 | | |
| N4 | Audio Embeddings | 2 | | |
| N5 | embeddingPurpose Optimization | 2 | | |
| N6 | Batch Processing | 3 | | |
| N7 | Key Management | 3 | | |
| N8 | Output Formats & Debug | 3 | | |
| N9 | Error Handling & Validation | 3 | | |
| N10 | Metadata & Filters | 1 | | |
| **Total** | | **26** | | |

## 🔑 Key Observations

### Nova vs TwelveLabs Comparison
| Feature | Nova | TwelveLabs |
|---------|------|------------|
| API Type | Sync (InvokeModel) | Async (StartAsyncInvoke) |
| Processing Time | ~1-3 seconds | ~60-120 seconds |
| Max Video/Audio | 30 seconds | 2GB / 2 hours |
| Dimensions | 256, 384, 1024, 3072 | 1024 (fixed) |
| Requires async bucket | ❌ No | ✅ Yes |
| Multi-clip segmentation | ❌ No (single embedding) | ✅ Yes |
| embeddingPurpose | ✅ Yes (9 options) | ❌ No |

### Performance Notes
- Nova text embedding latency: ~X ms
- Nova image embedding latency: ~X ms
- Nova video embedding latency: ~X ms
- Nova batch throughput (4 workers): ~X vectors/min

---

## ✅ CONCLUSION

**Nova Integration Status**: 
**Ready for Release**: 
