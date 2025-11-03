# Setup and Installation Guide

## Prerequisites

- Python 3.8 or higher
- Internet connection (required for first-time model download from HuggingFace)
- 4GB+ RAM (8GB recommended for training)
- Optional: CUDA-compatible GPU for faster training

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/optimusprimeg/AI-Agent-Prototype.git
cd AI-Agent-Prototype
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 2.0+ (deep learning framework)
- Transformers 4.30+ (Hugging Face library for pre-trained models)
- PEFT 0.4+ (Parameter-Efficient Fine-Tuning)
- scikit-learn 1.3+ (evaluation metrics)
- NumPy 1.24+ (numerical operations)

**Note:** The first time you run the training script, it will download the DistilBERT model (~250MB) from HuggingFace. This requires an internet connection.

### 4. Verify Installation

```bash
python -c "import torch, transformers, peft, sklearn, numpy; print('✓ All dependencies installed successfully')"
```

## First-Time Setup

### Step 1: Generate Synthetic Data

```bash
python src/data_generator.py
```

**Output:**
- `data/processed/train.json` (840 samples)
- `data/processed/val.json` (180 samples)
- `data/processed/test.json` (180 samples)

**Time:** < 1 second

### Step 2: Train the Model

```bash
python src/models/fine_tuned_model.py
```

**Training Time:**
- GPU (8GB VRAM): 10-15 minutes
- CPU (16GB RAM): 30-45 minutes

**Output:**
- Trained model in `models/expense_classifier/`
- Training logs
- Model achieves >80% accuracy on validation set

### Step 3: Evaluate the Model

```bash
python src/evaluate.py
```

**Output:**
- Accuracy, Precision, Recall, F1-Score
- Per-category performance metrics
- Qualitative evaluation with sample predictions
- Results saved to `logs/evaluation_results.json`

### Step 4: Run the CLI

**Interactive Mode:**
```bash
python src/main.py --mode interactive
```

Enter receipt items and get instant categorization.

**Batch Mode:**
```bash
# Create sample receipt
echo "1. Coffee and pastry
2. Uber ride
3. Movie ticket" > my_receipt.txt

# Process it
python src/main.py --mode batch --input my_receipt.txt --output results.json
```

## Quick Start (All-in-One)

Run the complete demo:

```bash
python demo.py
```

This script will:
1. Generate synthetic data
2. Train the model
3. Evaluate performance
4. Run sample categorization

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Install missing dependency:
```bash
pip install <module_name>
```

### Issue: HuggingFace Connection Error

**Symptom:** "We couldn't connect to 'https://huggingface.co'"

**Solution:** 
- Check internet connection
- HuggingFace models are downloaded on first use
- Models are cached locally after first download

### Issue: Out of Memory During Training

**Solution:**
- Reduce batch size in `src/models/fine_tuned_model.py`:
  ```python
  batch_size=8  # Instead of 16
  ```
- Close other applications
- Use a machine with more RAM

### Issue: Training Takes Too Long

**Solutions:**
- Use a GPU if available
- Reduce number of epochs for quick testing:
  ```python
  epochs=3  # Instead of 10
  ```
- Use smaller dataset for experimentation

## System Requirements

### Minimum Requirements
- CPU: 2 cores
- RAM: 4GB
- Disk: 1GB free space
- OS: Linux, macOS, Windows 10+

### Recommended for Training
- CPU: 4+ cores or GPU with CUDA support
- RAM: 8GB+
- Disk: 2GB free space
- Internet: Required for initial model download

## Environment Variables (Optional)

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Disable HuggingFace telemetry
export HF_HUB_DISABLE_TELEMETRY=1

# Set PyTorch device
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

## Offline Mode

If you have limited internet access:

1. Download models on a machine with internet:
```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('distilbert-base-uncased'); AutoModel.from_pretrained('distilbert-base-uncased')"
```

2. Copy the cache directory (`~/.cache/huggingface/`) to your offline machine

3. Set the cache path:
```bash
export HF_HOME=/path/to/copied/cache
```

## Next Steps

After setup:
1. Review [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
2. Read [DATA_SCIENCE_REPORT.md](docs/DATA_SCIENCE_REPORT.md) for methodology
3. Check [interaction_logs.txt](logs/interaction_logs.txt) for development insights
4. Experiment with the CLI in interactive mode
5. Try customizing categories in `src/data_generator.py`

## Support

For issues or questions:
1. Check this guide first
2. Review the documentation in `docs/`
3. Check GitHub Issues

## License

Educational use only - DS Internship Assignment
