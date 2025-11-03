"""
Executor Agent
Responsible for executing the categorization tasks using the fine-tuned model.
"""

from typing import Dict, List
from pathlib import Path

# Import ExpenseCategorizationModel with fallbacks so this module works both
# when executed as a script and when imported as a package.
try:
    # Preferred when running as a package
    from ..models.fine_tuned_model import ExpenseCategorizationModel
except Exception:
    try:
        # Absolute import with 'src' on PYTHONPATH or when running from project root
        from src.models.fine_tuned_model import ExpenseCategorizationModel
    except Exception:
        try:
            # Top-level import when script directory is src
            from models.fine_tuned_model import ExpenseCategorizationModel
        except Exception as e:
            raise ImportError(f"Could not import ExpenseCategorizationModel: {e}")


class ExecutorAgent:
    """
    Executor Agent: Executes categorization tasks using the fine-tuned model.
    
    This agent implements the execution phase of the AI agent:
    - Takes tasks from the Planner
    - Uses the fine-tuned model to categorize items
    - Returns categorization results with confidence scores
    """
    
    def __init__(self, model_path: Path = None):
        """
        Initialize the Executor Agent.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model = ExpenseCategorizationModel()
        self.model_loaded = False
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: Path):
        """Load the fine-tuned model."""
        try:
            self.model.load_model(model_path)
            self.model_loaded = True
            print(f"\n[Executor Agent] Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"\n[Executor Agent] Error loading model: {e}")
            self.model_loaded = False
    
    def execute_task(self, task: Dict) -> Dict:
        """
        Execute a single categorization task.
        
        Args:
            task: Task dictionary from Planner
            
        Returns:
            Updated task with category and confidence
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Cannot execute tasks.")
        
        item_text = task['item']
        
        # Use model to predict category
        category, confidence = self.model.predict(item_text)
        
        # Update task
        result = task.copy()
        result['category'] = category
        result['confidence'] = confidence
        result['status'] = 'completed'
        
        print(f"  [{task['task_id']}] '{item_text[:40]}...' -> {category} ({confidence:.2%})")
        
        return result
    
    def execute_batch(self, tasks: List[Dict]) -> List[Dict]:
        """
        Execute a batch of categorization tasks.
        
        Args:
            tasks: List of task dictionaries from Planner
            
        Returns:
            List of updated tasks with categories and confidence scores
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Cannot execute tasks.")
        
        print(f"\n[Executor Agent - Execution]")
        print(f"  Processing {len(tasks)} tasks...")
        
        results = []
        for task in tasks:
            result = self.execute_task(task)
            results.append(result)
        
        print(f"  Completed {len(results)} tasks")
        
        return results
    
    def aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate categorization results by category.
        
        Args:
            results: List of completed tasks
            
        Returns:
            Dictionary with aggregated results by category
        """
        aggregated = {}
        
        for result in results:
            category = result['category']
            if category not in aggregated:
                aggregated[category] = {
                    'items': [],
                    'count': 0,
                    'avg_confidence': 0.0
                }
            
            aggregated[category]['items'].append(result['item'])
            aggregated[category]['count'] += 1
        
        # Calculate average confidence per category
        for category in aggregated:
            confidences = [r['confidence'] for r in results if r['category'] == category]
            aggregated[category]['avg_confidence'] = sum(confidences) / len(confidences)
        
        print(f"\n[Executor Agent - Aggregation]")
        print(f"  Categorized into {len(aggregated)} categories")
        for category, data in aggregated.items():
            print(f"  {category}: {data['count']} items (avg confidence: {data['avg_confidence']:.2%})")
        
        return aggregated


def main():
    """Test the Executor Agent."""
    # This is a placeholder test - actual model needs to be trained first
    print("Executor Agent Test")
    print("=" * 50)
    print("\nNote: To test the Executor Agent, first train the model using:")
    print("  python src/models/fine_tuned_model.py")
    print("\nThen the Executor Agent can be tested via the main CLI:")
    print("  python src/main.py")


if __name__ == "__main__":
    main()
