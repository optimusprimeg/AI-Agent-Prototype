"""
Fine-Tuned Expense Categorization Model
Uses DistilBERT with LoRA for parameter-efficient fine-tuning.
Optimized for CPU with limited RAM (under 16GB).
"""

import json
import torch
import psutil
import os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import numpy as np
from tqdm import tqdm


class ExpenseDataset(Dataset):
    """Dataset for expense categorization with memory-efficient tokenization."""
    
    def __init__(self, data_path: Path, tokenizer, label2id: dict, precompute_encodings: bool = True):
        """Initialize dataset with optional pre-tokenization for memory efficiency."""
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.precompute_encodings = precompute_encodings
        
        # Pre-tokenize to avoid repeated tokenization during training
        if precompute_encodings:
            print(f"Pre-tokenizing {len(self.data)} samples for memory efficiency...")
            self.encodings = []
            for item in tqdm(self.data, desc="Tokenizing", leave=False):
                encoding = self.tokenizer(
                    item['item'],
                    padding='max_length',
                    truncation=True,
                    max_length=64,  # Reduced from 128 for memory savings
                    return_tensors='pt'
                )
                self.encodings.append({
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(self.label2id[item['category']])
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.precompute_encodings:
            return self.encodings[idx]
        else:
            item = self.data[idx]
            encoding = self.tokenizer(
                item['item'],
                padding='max_length',
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.label2id[item['category']])
            }


class ExpenseCategorizationModel:
    """Fine-tuned model for expense categorization."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """Initialize the model."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label2id = {}
        self.id2label = {}
    
    def get_system_info(self):
        """Get system resource information."""
        cpu_count = os.cpu_count() or 4
        memory_gb = psutil.virtual_memory().total / (1024**3)
        has_cuda = torch.cuda.is_available()
        
        return {
            'cpu_cores': cpu_count,
            'memory_gb': memory_gb,
            'has_cuda': has_cuda
        }
    
    def get_optimal_batch_size(self, memory_gb: float, has_cuda: bool, quick_mode: bool):
        """Calculate optimal batch size based on available resources."""
        if has_cuda:
            # GPU has its own memory
            return 64 if quick_mode else 32
        else:
            # CPU mode - be conservative with batch size
            if memory_gb < 8:
                return 4 if quick_mode else 2
            elif memory_gb < 16:
                return 8 if quick_mode else 4
            else:
                return 16 if quick_mode else 8
    
    def get_dataloader_workers(self, cpu_cores: int, has_cuda: bool):
        """Get optimal number of DataLoader workers."""
        if has_cuda:
            return min(4, cpu_cores)
        else:
            # For CPU training, limit workers to avoid overhead
            return min(2, max(1, cpu_cores // 2))
        
    def prepare_labels(self, train_data_path: Path):
        """Prepare label mappings from training data."""
        with open(train_data_path, 'r') as f:
            data = json.load(f)
        
        categories = sorted(list(set(item['category'] for item in data)))
        self.label2id = {label: idx for idx, label in enumerate(categories)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        print(f"Found {len(categories)} categories: {categories}")
        return categories
    
    def setup_model(self, num_labels: int):
        """Setup base model with LoRA."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"]  # DistilBERT attention layers
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_path: Path, val_path: Path, output_dir: Path, 
              epochs: int = 3, batch_size: int = None, quick_mode: bool = False):
        """Train the model with CPU-optimized settings."""
        
        # Get system information
        sys_info = self.get_system_info()
        print(f"\n{'='*60}")
        print(f"System Resources:")
        print(f"  CPU Cores: {sys_info['cpu_cores']}")
        print(f"  RAM: {sys_info['memory_gb']:.1f} GB")
        print(f"  CUDA Available: {sys_info['has_cuda']}")
        print(f"{'='*60}\n")
        
        # Auto-adjust batch size if not specified
        if batch_size is None:
            batch_size = self.get_optimal_batch_size(
                sys_info['memory_gb'], 
                sys_info['has_cuda'],
                quick_mode
            )
            print(f"Auto-adjusted batch size: {batch_size} (based on {sys_info['memory_gb']:.1f}GB RAM)")
        
        # Prepare labels
        categories = self.prepare_labels(train_path)
        
        # Setup model
        self.setup_model(num_labels=len(categories))
        
        # Prepare datasets with pre-tokenization for efficiency
        print("\nPreparing training data...")
        train_dataset = ExpenseDataset(train_path, self.tokenizer, self.label2id, precompute_encodings=True)
        print("Preparing validation data...")
        val_dataset = ExpenseDataset(val_path, self.tokenizer, self.label2id, precompute_encodings=True)
        
        # Get optimal DataLoader workers
        num_workers = self.get_dataloader_workers(sys_info['cpu_cores'], sys_info['has_cuda'])
        
        # Training arguments - CPU-optimized
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=min(50, len(train_dataset) // batch_size // 2),  # Adaptive warmup
            weight_decay=0.01,
            logging_dir=str(output_dir / 'logs'),
            logging_steps=max(10, len(train_dataset) // batch_size // 5),  # Adaptive logging
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            push_to_hub=False,
            report_to="none",
            fp16=False,  # Disable FP16 - not well supported on CPU
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=False,  # Critical: Disable pin_memory for CPU
            remove_unused_columns=True,
            disable_tqdm=False,  # Enable progress bars
            save_total_limit=2,  # Save only 2 checkpoints to save disk space
            gradient_accumulation_steps=1,  # No accumulation for simplicity
        )
        
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  DataLoader workers: {num_workers}")
        print(f"  Pin memory: False (CPU mode)")
        print(f"  FP16: False (CPU compatibility)")
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train with progress visualization
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}\n")
        train_result = self.trainer.train()
        
        # Display training summary
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Training loss: {train_result.training_loss:.4f}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Training steps: {train_result.global_step}")
        
        # Save model
        self.save_model(output_dir)
        
        # Create visualizations (support package-style and script-style invocation)
        # When this module is executed as a script (python src/models/fine_tuned_model.py)
        # relative imports (from ..utils...) will fail with "attempted relative import"
        # so try multiple import styles and fall back gracefully.
        create_all_visualizations = None
        try:
            # Preferred: package-relative import when running as a package (python -m src.models.fine_tuned_model)
            from ..utils.visualizer import create_all_visualizations
        except Exception:
            try:
                # Try absolute import assuming 'src' is on PYTHONPATH
                from src.utils.visualizer import create_all_visualizations
            except Exception:
                try:
                    # Fallback: top-level import if running with project root on sys.path
                    from utils.visualizer import create_all_visualizations
                except Exception as e:
                    print(f"Note: Could not import visualizer to create visualizations: {e}")

        if create_all_visualizations:
            try:
                create_all_visualizations(output_dir, train_path.parent)
            except Exception as e:
                print(f"Note: Could not create visualizations: {e}")
        
        return train_result
    
    def save_model(self, output_dir: Path):
        """Save the trained model."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Save label mappings
        with open(output_dir / 'label_mappings.json', 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, indent=2)
        
        print(f"\nModel saved to {output_dir}")
    
    def load_model(self, model_dir: Path):
        """Load a trained model."""
        # Load label mappings
        with open(model_dir / 'label_mappings.json', 'r') as f:
            mappings = json.load(f)
            self.label2id = mappings['label2id']
            self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Load model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model = base_model
        
        print(f"Model loaded from {model_dir}")
    
    def predict(self, text: str) -> tuple:
        """Predict category for a single item."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        category = self.id2label[predicted_class]
        return category, confidence


def main():
    """Train the expense categorization model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train expense categorization model (CPU-optimized)')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto-adjusted if not specified)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training mode (1 epoch, auto batch size)')
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        epochs = 1
        batch_size = args.batch_size  # Will be auto-adjusted
        quick_mode = True
        print("\n🚀 Quick training mode enabled")
        print("   Batch size will be auto-adjusted for your system")
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        quick_mode = False
        if batch_size is None:
            print(f"\n⚙️  Training with {epochs} epochs")
            print("   Batch size will be auto-adjusted for your system")
        else:
            print(f"\n⚙️  Training with {epochs} epochs, batch size {batch_size}")
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    data_dir = base_dir / 'data' / 'processed'
    model_dir = base_dir / 'models' / 'expense_classifier'
    
    # Initialize model
    model = ExpenseCategorizationModel()
    
    # Train with CPU-optimized settings
    model.train(
        train_path=data_dir / 'train.json',
        val_path=data_dir / 'val.json',
        output_dir=model_dir,
        epochs=epochs,
        batch_size=batch_size,
        quick_mode=quick_mode
    )
    
    print("\n✅ Training complete!")



if __name__ == "__main__":
    main()
