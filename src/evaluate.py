"""
Model Evaluation Script
Evaluates the fine-tuned model on test data and generates detailed metrics.
"""

import json
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import numpy as np
try:
    # Package-relative (when running as package)
    from ..models.fine_tuned_model import ExpenseCategorizationModel
except Exception:
    try:
        # Absolute import when 'src' is on PYTHONPATH or running from project root
        from src.models.fine_tuned_model import ExpenseCategorizationModel
    except Exception:
        try:
            # Top-level import when script directory is src
            from models.fine_tuned_model import ExpenseCategorizationModel
        except Exception as e:
            raise ImportError(f"Could not import ExpenseCategorizationModel: {e}")


class ModelEvaluator:
    """Evaluates the expense categorization model."""
    
    def __init__(self, model_path: Path):
        """Initialize evaluator with trained model."""
        self.model = ExpenseCategorizationModel()
        self.model.load_model(model_path)
        self.results = {}
    
    def evaluate_on_test_set(self, test_data_path: Path) -> dict:
        """
        Evaluate model on test dataset.
        
        Args:
            test_data_path: Path to test data JSON file
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        print(f"\nEvaluating on {len(test_data)} test samples...")
        
        # Make predictions
        true_labels = []
        pred_labels = []
        confidences = []
        
        for item in test_data:
            true_label = item['category']
            pred_label, confidence = self.model.predict(item['item'])
            
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            confidences.append(confidence)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                true_labels, pred_labels, average=None, zero_division=0,
                labels=sorted(list(set(true_labels)))
            )
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'avg_confidence': np.mean(confidences),
            'total_samples': len(test_data),
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'confidences': confidences
        }
        
        return self.results
    
    def generate_classification_report(self) -> str:
        """Generate detailed classification report."""
        if not self.results:
            return "No evaluation results available. Run evaluate_on_test_set first."
        
        report = classification_report(
            self.results['true_labels'],
            self.results['pred_labels'],
            zero_division=0
        )
        return report
    
    def generate_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix."""
        if not self.results:
            return None
        
        cm = confusion_matrix(
            self.results['true_labels'],
            self.results['pred_labels']
        )
        return cm
    
    def qualitative_evaluation(self, test_data_path: Path, num_samples: int = 10):
        """
        Perform qualitative evaluation on sample items.
        
        Args:
            test_data_path: Path to test data
            num_samples: Number of samples to evaluate
        """
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Select random samples
        import random
        samples = random.sample(test_data, min(num_samples, len(test_data)))
        
        print("\n" + "="*80)
        print("QUALITATIVE EVALUATION - Sample Predictions")
        print("="*80)
        
        correct = 0
        for i, item in enumerate(samples, 1):
            true_label = item['category']
            pred_label, confidence = self.model.predict(item['item'])
            
            is_correct = true_label == pred_label
            if is_correct:
                correct += 1
            
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            
            print(f"\n{i}. Item: '{item['item']}'")
            print(f"   True Category: {true_label}")
            print(f"   Predicted: {pred_label} (confidence: {confidence:.2%})")
            print(f"   Status: {status}")
        
        print(f"\n{'='*80}")
        print(f"Sample Accuracy: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
        print(f"{'='*80}")
    
    def save_results(self, output_path: Path):
        """Save evaluation results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'accuracy': float(self.results['accuracy']),
            'precision_weighted': float(self.results['precision_weighted']),
            'recall_weighted': float(self.results['recall_weighted']),
            'f1_weighted': float(self.results['f1_weighted']),
            'avg_confidence': float(self.results['avg_confidence']),
            'total_samples': int(self.results['total_samples']),
            'classification_report': self.generate_classification_report()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def print_summary(self):
        """Print evaluation summary."""
        if not self.results:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*80)
        print("QUANTITATIVE EVALUATION RESULTS")
        print("="*80)
        print(f"\nTest Set Size: {self.results['total_samples']} samples")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {self.results['accuracy']:.4f} ({self.results['accuracy']*100:.2f}%)")
        print(f"  Precision: {self.results['precision_weighted']:.4f}")
        print(f"  Recall:    {self.results['recall_weighted']:.4f}")
        print(f"  F1-Score:  {self.results['f1_weighted']:.4f}")
        print(f"  Avg Confidence: {self.results['avg_confidence']:.4f} ({self.results['avg_confidence']*100:.2f}%)")
        print("\nDetailed Classification Report:")
        print(self.generate_classification_report())
        print("="*80)


def main():
    """Run model evaluation."""
    base_dir = Path(__file__).parent.parent  # Go up to project root
    model_dir = base_dir / 'models' / 'expense_classifier'
    data_dir = base_dir / 'data' / 'processed'
    test_data_path = data_dir / 'test.json'
    results_path = base_dir / 'logs' / 'evaluation_results.json'
    
    # Check if model exists
    if not model_dir.exists():
        print("Error: Trained model not found.")
        print("Please train the model first using:")
        print("  python src/models/fine_tuned_model.py")
        return
    
    # Initialize evaluator
    print("Initializing Model Evaluator...")
    evaluator = ModelEvaluator(model_dir)
    
    # Quantitative evaluation
    evaluator.evaluate_on_test_set(test_data_path)
    evaluator.print_summary()
    
    # Qualitative evaluation
    evaluator.qualitative_evaluation(test_data_path, num_samples=10)
    
    # Save results
    evaluator.save_results(results_path)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
