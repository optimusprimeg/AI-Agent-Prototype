# AI Agent Architecture Document

## System Overview

The AI Agent Expense Categorization System is an intelligent automation solution that categorizes personal expenses from receipt text. The system uses a multi-agent architecture with a fine-tuned language model to provide accurate and reliable expense classification.

## Architecture Components

### 1. Data Preprocessor
**Location:** `src/utils/preprocessor.py`

**Purpose:** Handles preprocessing of receipt text and extraction of individual items.

**Key Features:**
- Text cleaning and normalization
- Pattern-based item extraction (supports multiple formats: numbered lists, bullet points, etc.)
- Batch processing capabilities

**Implementation Details:**
- Uses regex patterns to identify and extract items from various receipt formats
- Removes special characters while preserving semantic meaning
- Prepares text for model input with consistent formatting

### 2. Fine-Tuned Model
**Location:** `src/models/fine_tuned_model.py`

**Base Model:** DistilBERT (distilbert-base-uncased)

**Fine-Tuning Method:** LoRA (Low-Rank Adaptation) via PEFT library

**Purpose:** Provides accurate expense category classification using parameter-efficient fine-tuning.

**Model Configuration:**
- Task Type: Sequence Classification
- LoRA Rank (r): 16
- LoRA Alpha: 32
- LoRA Dropout: 0.1
- Target Modules: q_lin, v_lin (attention layers)
- Max Sequence Length: 128 tokens

**Why LoRA?**
- **Parameter Efficiency:** Only ~0.5% of parameters are trainable, reducing computational requirements
- **Memory Efficient:** Significantly lower memory footprint compared to full fine-tuning
- **Fast Training:** Converges quickly with fewer resources
- **Preserves Base Model:** Original model weights remain frozen, preventing catastrophic forgetting
- **Domain Adaptation:** Effectively adapts base model to financial/expense domain

**Training Details:**
- Optimizer: AdamW with weight decay
- Learning Rate Schedule: Linear warmup
- Batch Size: 16
- Epochs: 10 (with early stopping)

### 3. Multi-Agent System

#### 3.1 Planner Agent
**Location:** `src/agents/planner.py`

**Role:** Reasoning and Planning phases of the AI agent

**Responsibilities:**
1. **Reasoning Phase:**
   - Analyzes receipt text structure
   - Assesses preprocessing requirements
   - Estimates number of items and complexity

2. **Planning Phase:**
   - Extracts individual items from receipt
   - Creates structured task list for categorization
   - Maintains task status and progress tracking

**Key Methods:**
- `reason_about_receipt()`: Analyzes receipt characteristics
- `create_plan()`: Generates execution plan with tasks
- `get_pending_tasks()`: Retrieves uncompleted tasks
- `update_task_status()`: Updates task completion status

#### 3.2 Executor Agent
**Location:** `src/agents/executor.py`

**Role:** Execution phase of the AI agent

**Responsibilities:**
1. **Task Execution:**
   - Receives tasks from Planner Agent
   - Uses fine-tuned model for categorization
   - Returns results with confidence scores

2. **Result Aggregation:**
   - Groups items by category
   - Calculates category-level statistics
   - Computes average confidence per category

**Key Methods:**
- `execute_task()`: Categorizes single item
- `execute_batch()`: Processes multiple items efficiently
- `aggregate_results()`: Summarizes results by category

### 4. Evaluator
**Location:** `src/evaluate.py`

**Purpose:** Comprehensive model evaluation with quantitative and qualitative metrics.

**Metrics Computed:**
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Per-class metrics
- Confusion matrix
- Average confidence scores

**Evaluation Types:**
1. **Quantitative:** Statistical metrics on test set
2. **Qualitative:** Manual review of sample predictions

### 5. CLI Interface
**Location:** `src/main.py`

**Purpose:** User-facing command-line interface for system interaction.

**Operating Modes:**
1. **Interactive Mode:** Real-time receipt input and categorization
2. **Batch Mode:** Process receipts from files

**Features:**
- User-friendly input/output
- Detailed result visualization
- Export results to JSON

## Data Flow

```
User Input (Receipt Text)
    ↓
Data Preprocessor
    ↓
Planner Agent (Reasoning & Planning)
    ↓
Task Queue
    ↓
Executor Agent (Execution with Fine-tuned Model)
    ↓
Result Aggregation
    ↓
Output to User
```

## Interaction Flow (Detailed)

1. **User provides receipt text** via CLI (interactive or batch mode)

2. **Planner Agent - Reasoning:**
   - Analyzes receipt structure
   - Determines if preprocessing is needed
   - Estimates complexity

3. **Planner Agent - Planning:**
   - Extracts individual items using preprocessor
   - Creates task for each item (task_id, item, status)
   - Generates execution plan

4. **Executor Agent - Execution:**
   - Receives pending tasks from Planner
   - For each task:
     - Tokenizes item text
     - Runs through fine-tuned model
     - Obtains category prediction + confidence
     - Updates task with results

5. **Executor Agent - Aggregation:**
   - Groups categorized items by category
   - Calculates statistics (count, avg confidence)
   - Prepares summary

6. **CLI displays results:**
   - Organized by category
   - Shows item lists
   - Displays confidence scores
   - Provides summary statistics

## Model Selection Rationale

### Base Model: DistilBERT
**Reasons for Selection:**
- **Lightweight:** 40% smaller than BERT, faster inference
- **Strong Performance:** Retains 97% of BERT's language understanding
- **Well-suited for Classification:** Proven track record on sequence classification tasks
- **Practical:** Lower resource requirements for training and deployment
- **Pre-trained Knowledge:** Already understands general English, reducing training needs

### Fine-Tuning Approach: LoRA
**Reasons for Selection:**
- **Parameter Efficiency:** Only trains ~0.5% of model parameters
- **Resource Constraints:** Enables fine-tuning on consumer hardware
- **Quick Iteration:** Faster training cycles for experimentation
- **Storage Efficient:** Model checkpoint sizes are minimal
- **Reliability:** Proven effective for domain adaptation tasks
- **No Catastrophic Forgetting:** Base model knowledge preserved

## Categories Supported

The system categorizes expenses into 8 primary categories:

1. **Food** - Meals, groceries, dining, beverages
2. **Transportation** - Rides, fuel, parking, public transit
3. **Utilities** - Bills for electric, water, internet, phone
4. **Entertainment** - Movies, concerts, subscriptions, events
5. **Healthcare** - Medical visits, prescriptions, insurance
6. **Shopping** - Clothing, electronics, home goods
7. **Housing** - Rent, mortgage, repairs, maintenance
8. **Education** - Tuition, courses, books, supplies

## System Requirements

**Software Dependencies:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- scikit-learn 1.3+

**Hardware Recommendations:**
- Training: GPU with 8GB+ VRAM (or CPU with 16GB+ RAM)
- Inference: CPU sufficient (4GB+ RAM)

## Performance Characteristics

**Expected Metrics (Target):**
- Accuracy: >80%
- Precision: >75%
- Recall: >75%
- F1-Score: >75%
- Average Confidence: >85%

**Inference Speed:**
- ~50-100ms per item (CPU)
- ~10-20ms per item (GPU)

## Extensibility

The architecture supports easy extension:

1. **Adding Categories:** Update data generator with new category examples
2. **Improving Model:** Swap base model or adjust LoRA parameters
3. **Additional Agents:** Add specialized agents for specific tasks
4. **RAG Integration:** Can add retrieval system for similar past categorizations
5. **API Interface:** CLI can be wrapped with REST API

## Security Considerations

- No sensitive data is logged
- Model runs locally (no external API calls)
- User data remains on local system
- Receipt text not transmitted externally

## Future Enhancements

1. **Multi-language Support:** Extend to non-English receipts
2. **Amount Extraction:** Parse monetary amounts from items
3. **Vendor Recognition:** Identify merchants from receipts
4. **Confidence Thresholds:** Flag low-confidence predictions for review
5. **Active Learning:** Improve model from user corrections
6. **RAG Integration:** Leverage past categorizations
7. **Web Interface:** Browser-based UI for easier access
