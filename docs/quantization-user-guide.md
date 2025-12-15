# Model Quantization User Guide

## What is Model Quantization?

Model quantization reduces the size and memory requirements of AI models by converting high-precision weights (16-bit or 32-bit floats) to lower precision formats (2-bit to 8-bit integers). This makes models:

- **Smaller**: 4x-16x reduction in file size
- **Faster**: Reduced memory bandwidth requirements
- **More Accessible**: Run larger models on consumer hardware

## Quick Start

### 1. Start the Quantization Worker

```bash
cd /path/to/llama-nexus
docker-compose --profile quantize up quantization-worker
```

### 2. Open the UI

Navigate to **Quantization** in the sidebar, or go to: `http://localhost:3002/quantization`

### 3. Create a Quantization Job

Click **New Job** and follow the wizard:

**Step 1: Select Model**
- Enter a job name (e.g., "Llama 2 7B Quantization")
- Enter a HuggingFace model ID (e.g., `meta-llama/Llama-2-7b-hf`)

**Step 2: Choose Formats**
- Select **GGUF** (recommended for llama.cpp)
- GPTQ and AWQ coming soon

**Step 3: Configure Quantization Types**
- Choose one or more quantization levels:
  - **Q4_K_M** (default, good balance)
  - **Q5_K_M** (higher quality)
  - **Q8_0** (near-lossless)
  - See "Quantization Types" section for details

**Step 4: Review and Create**
- Check disk space and time estimates
- Click **Create Job**

### 4. Monitor Progress

The job card will show:
- Current status (downloading, preparing, quantizing)
- Progress percentage
- Number of completed outputs

### 5. Access Outputs

Once complete, outputs are stored in the Docker volume and registered in the Model Registry.

## Quantization Type Guide

### Recommended for Most Users

| Type | Size Factor | Quality | Use Case |
|------|-------------|---------|----------|
| **Q4_K_M** | ~31% | Good | Best balance of size and quality |
| **Q5_K_M** | ~38% | Better | When you have more storage |
| **Q8_0** | ~56% | Excellent | Near-lossless quality |

### For Specific Needs

| Type | Size Factor | Quality | Use Case |
|------|-------------|---------|----------|
| Q2_K | ~14% | Lower | Extreme compression, testing |
| Q3_K_M | ~21% | Fair | Tight storage constraints |
| Q4_K_S | ~29% | Good | Slightly smaller than Q4_K_M |
| Q5_K_S | ~36% | Better | Slightly smaller than Q5_K_M |
| Q6_K | ~50% | Very Good | High quality needed |
| F16 | 100% | Perfect | No quantization (baseline) |

### Quick Recommendations

**For 7B Models:**
- **Recommended**: Q4_K_M or Q5_K_M
- **High Quality**: Q6_K or Q8_0
- **Limited Storage**: Q3_K_M or Q4_K_S

**For 13B+ Models:**
- **Recommended**: Q4_K_M
- **If Storage Allows**: Q5_K_M
- **Tight Constraints**: Q3_K_M

**For 70B Models:**
- **Recommended**: Q4_K_M (saves ~100GB!)
- **Extreme**: Q2_K or Q3_K_M

## Understanding Disk Requirements

### Example: Llama 2 7B

Original model (FP16): ~14 GB

Quantized outputs:
- Q4_K_M: ~4 GB (saves 10 GB)
- Q5_K_M: ~5 GB (saves 9 GB)
- Q8_0: ~7 GB (saves 7 GB)
- All three: ~16 GB total

**Tip**: The system will show estimated disk space before you create the job.

## API Usage

### Create Job via API

```bash
curl -X POST http://localhost:8700/api/v1/quantize/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Quantization Job",
    "source_model": "meta-llama/Llama-2-7b-hf",
    "output_formats": ["gguf"],
    "gguf_quant_types": ["Q4_K_M", "Q5_K_M"]
  }'
```

### Get Estimate

```bash
curl -X POST http://localhost:8700/api/v1/quantize/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "source_model": "meta-llama/Llama-2-7b-hf",
    "output_formats": ["gguf"],
    "gguf_quant_types": ["Q4_K_M"]
  }'
```

### Check Job Status

```bash
curl http://localhost:8700/api/v1/quantize/jobs/{job_id}
```

### List All Jobs

```bash
curl http://localhost:8700/api/v1/quantize/jobs
```

## Tips and Best Practices

### 1. Start Small
Test with a small model first (e.g., TinyLlama-1.1B) to verify everything works.

### 2. Multiple Quantizations
Create multiple quantization levels in one job to compare quality vs size trade-offs.

### 3. Check Estimates
Always review the disk space and time estimates before creating large jobs.

### 4. GPU Memory
The worker needs GPU access. Make sure your GPU has enough VRAM for the source model.

### 5. Storage Management
Quantized files can be large. Monitor your disk usage and delete old jobs when done.

### 6. Model Selection
Not all models quantize equally well:
- Larger models (13B+) see bigger benefits from quantization
- Smaller models (< 3B) may show more quality loss at low quantization

## Troubleshooting

### Job Stuck in "Queued"

**Problem**: Worker isn't running or can't connect to Redis

**Solution**:
```bash
# Check worker status
docker-compose ps quantization-worker

# View worker logs
docker-compose logs quantization-worker

# Restart worker
docker-compose restart quantization-worker
```

### "Failed to Download Model"

**Problem**: Model doesn't exist or no HuggingFace token

**Solution**:
1. Verify model ID exists on HuggingFace
2. Check if model requires authentication
3. Add `HUGGINGFACE_TOKEN` to `.env` file

### "Out of Memory"

**Problem**: GPU doesn't have enough VRAM

**Solution**:
- Try quantizing smaller models
- Close other GPU-intensive applications
- Use a GPU with more VRAM

### Job Failed During Quantization

**Problem**: Conversion error or corrupted model

**Solution**:
1. Check worker logs for detailed error
2. Verify source model is compatible with llama.cpp
3. Try a different model or quantization type

## Advanced Configuration

### Custom Worker Settings

Edit `docker-compose.yml` to adjust worker configuration:

```yaml
quantization-worker:
  environment:
    - MODELS_DIR=/custom/path/models
    - QUANTIZATION_OUTPUT_DIR=/custom/path/outputs
    - CUDA_VISIBLE_DEVICES=0,1  # Use multiple GPUs
```

### Calibration Datasets (Coming Soon)

For GPTQ and AWQ quantization, you'll be able to specify custom calibration datasets for better quality.

## FAQ

**Q: Can I quantize my own local models?**  
A: Currently only HuggingFace models are supported. Local model support is planned.

**Q: How long does quantization take?**  
A: Depends on model size and GPU. Typical times:
- 7B model: 5-10 minutes
- 13B model: 10-15 minutes  
- 70B model: 30-60 minutes

**Q: Can I cancel a running job?**  
A: Yes, click the **Cancel** button on the job card or use the API.

**Q: Where are the output files stored?**  
A: In the Docker volume `quantization_output`. Access via:
```bash
docker volume inspect llama-nexus_quantization_output
```

**Q: Can I use quantized models immediately?**  
A: Yes! They're automatically registered in the Model Registry and can be deployed via the Deploy page.

**Q: Is my original model affected?**  
A: No, quantization creates new files. The original model is unchanged.

**Q: What about GPTQ and AWQ?**  
A: Infrastructure is in place, implementation coming soon. These are more complex quantization methods that require calibration.

## Next Steps

After quantizing your models:

1. **Deploy**: Use the Deploy page to run your quantized models
2. **Compare**: Use the Benchmark page to compare quantization quality
3. **Share**: Quantized models can be shared or uploaded to HuggingFace

## Support

For issues or questions:
- Check the worker logs: `docker-compose logs quantization-worker`
- Review the implementation summary: `docs/quantization-implementation-summary.md`
- Open an issue on GitHub
