"""
Fine-Tuned Expense Categorization Model
Uses DistilBERT with LoRA for parameter-efficient fine-tuning.
"""

import json
import torch
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


class ExpenseDataset(Dataset):
    """Dataset for expense categorization."""
    
    def __init__(self, data_path: Path, tokenizer, label2id: dict):
        """Initialize dataset."""
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.label2id = label2id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['item'],
            padding='max_length',
            truncation=True,
            max_length=128,
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
              epochs: int = 10, batch_size: int = 16):
        """Train the model."""
        # Prepare labels
        categories = self.prepare_labels(train_path)
        
        # Setup model
        self.setup_model(num_labels=len(categories))
        
        # Prepare datasets
        train_dataset = ExpenseDataset(train_path, self.tokenizer, self.label2id)
        val_dataset = ExpenseDataset(val_path, self.tokenizer, self.label2id)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(output_dir / 'logs'),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            push_to_hub=False,
            report_to="none"
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        print("\nStarting training...")
        train_result = self.trainer.train()
        
        # Save model
        self.save_model(output_dir)
        
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
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    model_dir = base_dir / 'models' / 'expense_classifier'
    
    # Initialize model
    model = ExpenseCategorizationModel()
    
    # Train
    model.train(
        train_path=data_dir / 'train.json',
        val_path=data_dir / 'val.json',
        output_dir=model_dir,
        epochs=10,
        batch_size=16
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
