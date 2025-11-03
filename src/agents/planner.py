"""
Planner Agent
Responsible for reasoning about receipt content and planning the categorization process.
"""

from typing import List, Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.preprocessor import ReceiptPreprocessor
else:
    from ..utils.preprocessor import ReceiptPreprocessor


class PlannerAgent:
    """
    Planner Agent: Reasons about the receipt and breaks it down into categorization tasks.
    
    This agent implements the reasoning and planning phases of the AI agent:
    - Analyzes the receipt text structure
    - Extracts individual items
    - Creates a task plan for categorization
    """
    
    def __init__(self):
        """Initialize the Planner Agent."""
        self.preprocessor = ReceiptPreprocessor()
        self.current_plan = None
    
    def reason_about_receipt(self, receipt_text: str) -> Dict:
        """
        Reasoning Phase: Analyze the receipt and understand its structure.
        
        Args:
            receipt_text: Raw receipt text
            
        Returns:
            Dictionary with reasoning results
        """
        reasoning = {
            "receipt_length": len(receipt_text),
            "has_structure": any(pattern in receipt_text for pattern in ['1.', '2.', '-', '*', '•']),
            "estimated_items": receipt_text.count('\n') + 1,
            "preprocessing_needed": True
        }
        
        print(f"\n[Planner Agent - Reasoning]")
        print(f"  Receipt length: {reasoning['receipt_length']} characters")
        print(f"  Structured format: {reasoning['has_structure']}")
        print(f"  Estimated items: {reasoning['estimated_items']}")
        
        return reasoning
    
    def create_plan(self, receipt_text: str) -> Dict:
        """
        Planning Phase: Create a structured plan for categorizing receipt items.
        
        Args:
            receipt_text: Raw receipt text
            
        Returns:
            Dictionary containing the execution plan
        """
        # First, reason about the receipt
        reasoning = self.reason_about_receipt(receipt_text)
        
        # Extract items from receipt
        items = self.preprocessor.extract_items_from_receipt(receipt_text)
        
        # Create tasks for each item
        tasks = []
        for idx, item in enumerate(items):
            task = {
                "task_id": f"T{idx+1:03d}",
                "item": item,
                "status": "pending",
                "category": None,
                "confidence": None
            }
            tasks.append(task)
        
        # Create the plan
        plan = {
            "reasoning": reasoning,
            "total_tasks": len(tasks),
            "tasks": tasks,
            "status": "ready"
        }
        
        self.current_plan = plan
        
        print(f"\n[Planner Agent - Planning]")
        print(f"  Created {len(tasks)} categorization tasks")
        print(f"  Plan status: {plan['status']}")
        
        return plan
    
    def get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks from the current plan."""
        if self.current_plan is None:
            return []
        
        return [task for task in self.current_plan['tasks'] if task['status'] == 'pending']
    
    def update_task_status(self, task_id: str, category: str, confidence: float):
        """Update the status of a task after execution."""
        if self.current_plan is None:
            return
        
        for task in self.current_plan['tasks']:
            if task['task_id'] == task_id:
                task['status'] = 'completed'
                task['category'] = category
                task['confidence'] = confidence
                break
    
    def get_plan_summary(self) -> Dict:
        """Get a summary of the current plan execution."""
        if self.current_plan is None:
            return {"status": "no_plan"}
        
        completed = sum(1 for task in self.current_plan['tasks'] if task['status'] == 'completed')
        pending = sum(1 for task in self.current_plan['tasks'] if task['status'] == 'pending')
        
        return {
            "total_tasks": self.current_plan['total_tasks'],
            "completed": completed,
            "pending": pending,
            "progress_percentage": (completed / self.current_plan['total_tasks'] * 100) if self.current_plan['total_tasks'] > 0 else 0
        }


def main():
    """Test the Planner Agent."""
    planner = PlannerAgent()
    
    # Sample receipt
    sample_receipt = """
    1. Coffee and pastry
    2. Lunch sandwich
    3. Uber ride to downtown
    4. Movie ticket
    5. Prescription medication
    6. Electric bill payment
    7. Groceries from store
    """
    
    print("Testing Planner Agent")
    print("=" * 50)
    
    # Create plan
    plan = planner.create_plan(sample_receipt)
    
    print("\n[Plan Details]")
    for task in plan['tasks']:
        print(f"  {task['task_id']}: {task['item']}")
    
    # Get summary
    summary = planner.get_plan_summary()
    print(f"\n[Plan Summary]")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Pending: {summary['pending']}")


if __name__ == "__main__":
    main()
