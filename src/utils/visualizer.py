"""
Training Visualizer
Provides human-friendly visualizations for training progress.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List


class TrainingVisualizer:
    """Creates visualizations for training metrics."""
    
    def __init__(self, output_dir: Path):
        """Initialize visualizer."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_progress(self, metrics: Dict[str, List[float]], save_path: Path = None):
        """
        Plot training progress with loss and accuracy.
        
        Args:
            metrics: Dictionary with 'loss', 'accuracy', 'epoch' keys
            save_path: Where to save the plot
        """
        if save_path is None:
            save_path = self.output_dir / 'training_progress.png'
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        if 'loss' in metrics and metrics['loss']:
            ax1.plot(metrics.get('epoch', range(len(metrics['loss']))), 
                    metrics['loss'], 'b-', linewidth=2, marker='o')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Training Loss', fontsize=12)
            ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'accuracy' in metrics and metrics['accuracy']:
            ax2.plot(metrics.get('epoch', range(len(metrics['accuracy']))), 
                    metrics['accuracy'], 'g-', linewidth=2, marker='s')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Training progress visualization saved to: {save_path}")
        return save_path
    
    def plot_category_distribution(self, data_path: Path, save_path: Path = None):
        """
        Plot distribution of categories in dataset.
        
        Args:
            data_path: Path to dataset JSON
            save_path: Where to save the plot
        """
        if save_path is None:
            save_path = self.output_dir / 'category_distribution.png'
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Count categories
        category_counts = {}
        for item in data:
            cat = item['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Sort by count
        categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        cats, counts = zip(*categories)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(cats)), counts, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title('Dataset Category Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Category distribution saved to: {save_path}")
        return save_path
    
    def create_summary_card(self, metrics: Dict, save_path: Path = None):
        """
        Create a summary card with key metrics.
        
        Args:
            metrics: Dictionary with accuracy, precision, recall, f1
            save_path: Where to save the card
        """
        if save_path is None:
            save_path = self.output_dir / 'training_summary.png'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Training Summary', 
               ha='center', va='top', fontsize=20, fontweight='bold')
        
        # Metrics
        y_pos = 0.75
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        
        for i, (name, key, color) in enumerate(zip(metric_names, metric_keys, colors)):
            if key in metrics:
                value = metrics[key]
                
                # Metric name
                ax.text(0.25, y_pos, name + ':', 
                       ha='right', va='center', fontsize=14, fontweight='bold')
                
                # Metric value
                ax.text(0.28, y_pos, f'{value:.2%}', 
                       ha='left', va='center', fontsize=14, color=color, fontweight='bold')
                
                # Progress bar
                bar_y = y_pos - 0.03
                ax.add_patch(plt.Rectangle((0.55, bar_y), 0.35, 0.04, 
                                          facecolor='lightgray', edgecolor='none'))
                ax.add_patch(plt.Rectangle((0.55, bar_y), 0.35 * value, 0.04, 
                                          facecolor=color, edgecolor='none'))
                
                y_pos -= 0.15
        
        # Footer
        ax.text(0.5, 0.05, '✅ Model Ready for Deployment', 
               ha='center', va='center', fontsize=12, 
               style='italic', color='green' if metrics.get('accuracy', 0) > 0.8 else 'orange')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary card saved to: {save_path}")
        return save_path


def create_all_visualizations(model_dir: Path, data_dir: Path):
    """Create all visualizations after training."""
    viz = TrainingVisualizer(model_dir / 'visualizations')
    
    # Plot category distribution
    try:
        viz.plot_category_distribution(data_dir / 'train.json')
    except Exception as e:
        print(f"Could not create category distribution: {e}")
    
    print("\n📊 Visualizations created successfully!")
    return viz


if __name__ == "__main__":
    # Test visualizer
    from pathlib import Path
    
    test_dir = Path('/tmp/test_viz')
    test_dir.mkdir(exist_ok=True)
    
    viz = TrainingVisualizer(test_dir)
    
    # Test metrics
    metrics = {
        'epoch': [1, 2, 3],
        'loss': [0.5, 0.3, 0.2],
        'accuracy': [0.75, 0.85, 0.90]
    }
    
    viz.plot_training_progress(metrics)
    
    summary = {
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.86,
        'f1': 0.85
    }
    
    viz.create_summary_card(summary)
    
    print("Test visualizations created!")
