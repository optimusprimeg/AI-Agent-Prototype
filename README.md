# AI Agent Prototype for Expense Categorization

**Student Name:** Optimus Prime  
**University:** Autonomous University  
**Department:** Computer Science  
**Assignment:** DS Internship - AI Agent Development

## Project Overview

This project implements an AI agent system that automates the categorization of personal expenses from receipt text. The system uses a fine-tuned language model with a multi-agent architecture to reason about receipts, plan categorization tasks, and execute classifications into predefined expense categories.

## Features

- **Multi-Agent Architecture**: Separate Planner and Executor agents for reasoning, planning, and execution
- **Fine-Tuned Model**: DistilBERT with LoRA (parameter-efficient fine-tuning) for expense categorization
- **8 Expense Categories**: Food, Transportation, Utilities, Entertainment, Healthcare, Shopping, Housing, Education
- **CLI Interface**: Interactive and batch modes for easy use
- **Comprehensive Evaluation**: Quantitative metrics (accuracy, precision, recall, F1) and qualitative analysis
- **Synthetic Data**: 1200+ generated receipt samples with train/val/test splits

## Architecture

The system consists of:

1. **Data Preprocessor**: Extracts and cleans receipt items
2. **Planner Agent**: Reasons about receipts and creates categorization plans
3. **Executor Agent**: Executes categorization using fine-tuned model
4. **Fine-Tuned Model**: DistilBERT + LoRA for accurate classification
5. **Evaluator**: Computes metrics and generates reports
6. **CLI**: User-friendly command-line interface

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/optimusprimeg/AI-Agent-Prototype.git
cd AI-Agent-Prototype

# Install dependencies
pip install -r requirements.txt
```

**Important Note:** The first time you run the training script, it will download the DistilBERT model (~250MB) from HuggingFace. This requires an internet connection. The model will be cached locally for future use.

### Usage

#### 1. Generate Synthetic Data

```bash
python src/data_generator.py
```

This creates training, validation, and test datasets in `data/processed/`.

#### 2. Train the Model

```bash
python src/models/fine_tuned_model.py
```

Training takes approximately 10-15 minutes on GPU or 30-45 minutes on CPU.  
The trained model is saved to `models/expense_classifier/`.

#### 3. Evaluate the Model

```bash
python src/evaluate.py
```

Displays accuracy, precision, recall, F1-score, and sample predictions.  
Results are saved to `logs/evaluation_results.json`.

#### 4. Run the AI Agent

**Interactive Mode:**

```bash
python src/main.py --mode interactive
```

Enter receipt items line by line, type 'DONE' when finished.

**Batch Mode:**

```bash
# Create a sample receipt file
echo "1. Coffee and pastry
2. Lunch sandwich  
3. Uber ride to office
4. Movie ticket
5. Prescription medication
6. Electric bill payment" > sample_receipt.txt

# Process the receipt
python src/main.py --mode batch --input sample_receipt.txt --output results.json
```

## Example Output

```
================================================================================
AI AGENT EXPENSE CATEGORIZATION SYSTEM
================================================================================

Phase 1: REASONING & PLANNING
--------------------------------------------------------------------------------
  Receipt length: 145 characters
  Structured format: True
  Estimated items: 6

Phase 2: EXECUTION
--------------------------------------------------------------------------------
  Processing 6 tasks...
  [T001] 'Coffee and pastry' -> Food (92.3%)
  [T002] 'Lunch sandwich' -> Food (88.7%)
  [T003] 'Uber ride to office' -> Transportation (95.1%)
  [T004] 'Movie ticket' -> Entertainment (91.2%)
  [T005] 'Prescription medication' -> Healthcare (93.8%)
  [T006] 'Electric bill payment' -> Utilities (89.5%)

Phase 3: AGGREGATION
--------------------------------------------------------------------------------
  Categorized into 5 categories

================================================================================
CATEGORIZATION RESULTS
================================================================================

Food (2 items, avg confidence: 90.5%)
----------------------------------------
  • Coffee and pastry
  • Lunch sandwich

Transportation (1 items, avg confidence: 95.1%)
----------------------------------------
  • Uber ride to office

Entertainment (1 items, avg confidence: 91.2%)
----------------------------------------
  • Movie ticket

Healthcare (1 items, avg confidence: 93.8%)
----------------------------------------
  • Prescription medication

Utilities (1 items, avg confidence: 89.5%)
----------------------------------------
  • Electric bill payment

================================================================================
Total Items Categorized: 6
Categories Used: 5
Overall Confidence: 91.7%
================================================================================
```

## Project Structure

```
AI-Agent-Prototype/
├── src/
│   ├── agents/
│   │   ├── planner.py        # Planner Agent (reasoning & planning)
│   │   └── executor.py       # Executor Agent (execution)
│   ├── models/
│   │   └── fine_tuned_model.py  # DistilBERT + LoRA model
│   ├── utils/
│   │   └── preprocessor.py   # Receipt text preprocessor
│   ├── data_generator.py     # Synthetic data generation
│   ├── evaluate.py           # Model evaluation
│   └── main.py              # CLI application
├── data/
│   └── processed/           # Train/val/test datasets
├── models/
│   └── expense_classifier/  # Trained model artifacts
├── logs/
│   ├── evaluation_results.json
│   └── interaction_logs.txt
├── docs/
│   ├── ARCHITECTURE.md      # System architecture documentation
│   └── DATA_SCIENCE_REPORT.md  # Data science methodology and results
├── requirements.txt
└── README.md
```

## Model Details

**Base Model:** DistilBERT (distilbert-base-uncased)
- 66M parameters
- 40% smaller than BERT
- Fast inference

**Fine-Tuning Method:** LoRA (Low-Rank Adaptation)
- Rank: 16
- Alpha: 32
- Trainable parameters: ~0.5%
- Target modules: q_lin, v_lin (attention layers)

**Training Data:**
- 1,200 synthetic receipt items
- 840 training samples
- 180 validation samples
- 180 test samples

**Expected Performance:**
- Accuracy: >80%
- Precision: >75%
- Recall: >75%
- F1-Score: >75%

## Documentation

- **[Architecture Document](docs/ARCHITECTURE.md)**: Detailed system architecture, component descriptions, and design decisions
- **[Data Science Report](docs/DATA_SCIENCE_REPORT.md)**: Fine-tuning methodology, evaluation metrics, and results
- **[Interaction Logs](logs/interaction_logs.txt)**: Simulated prompts and development process

## Requirements

**Software:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- scikit-learn 1.3+

**Hardware (Minimum):**
- RAM: 4GB for inference, 8GB for training
- Disk: 1GB for data and models
- CPU: Multi-core processor
- GPU: Optional (speeds up training significantly)

## Technical Highlights

1. **Parameter-Efficient Fine-Tuning**: Uses LoRA to train only 0.5% of parameters
2. **Multi-Agent Design**: Clear separation of reasoning, planning, and execution
3. **Comprehensive Evaluation**: Both quantitative metrics and qualitative analysis
4. **User-Friendly CLI**: Interactive and batch processing modes
5. **Extensible Architecture**: Easy to add new categories or integrate with other systems

## Future Enhancements

- Multi-language support
- Amount extraction from receipts
- Vendor/merchant recognition
- RAG (Retrieval-Augmented Generation) for similar past categorizations
- Web-based UI
- Mobile app integration
- Active learning from user corrections

## License

This project is for educational purposes as part of the DS Internship Assignment.

## Contact

**Optimus Prime**  
Autonomous University  
Department of Computer Science

---

*This AI agent demonstrates practical application of modern NLP techniques, multi-agent systems, and parameter-efficient fine-tuning for automating real-world tasks.*