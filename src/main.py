"""
Main CLI Application
Command-line interface for the AI Agent Expense Categorization System.
"""

import argparse
from pathlib import Path
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent


class ExpenseCategorizationCLI:
    """CLI for the expense categorization AI agent."""
    
    def __init__(self, model_path: Path):
        """Initialize the CLI with agents."""
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent(model_path)
        
        if not self.executor.model_loaded:
            raise ValueError("Failed to load model. Please train the model first.")
    
    def categorize_receipt(self, receipt_text: str) -> dict:
        """
        Categorize a receipt using the multi-agent system.
        
        Args:
            receipt_text: Raw receipt text
            
        Returns:
            Dictionary containing categorization results
        """
        print("\n" + "="*80)
        print("AI AGENT EXPENSE CATEGORIZATION SYSTEM")
        print("="*80)
        
        # Phase 1: Reasoning and Planning (Planner Agent)
        print("\nPhase 1: REASONING & PLANNING")
        print("-" * 80)
        plan = self.planner.create_plan(receipt_text)
        
        # Phase 2: Execution (Executor Agent)
        print("\nPhase 2: EXECUTION")
        print("-" * 80)
        pending_tasks = self.planner.get_pending_tasks()
        results = self.executor.execute_batch(pending_tasks)
        
        # Update planner with results
        for result in results:
            self.planner.update_task_status(
                result['task_id'],
                result['category'],
                result['confidence']
            )
        
        # Phase 3: Aggregation
        print("\nPhase 3: AGGREGATION")
        print("-" * 80)
        aggregated = self.executor.aggregate_results(results)
        
        return {
            'plan': plan,
            'results': results,
            'aggregated': aggregated
        }
    
    def display_results(self, categorization_results: dict):
        """Display categorization results in a user-friendly format."""
        print("\n" + "="*80)
        print("CATEGORIZATION RESULTS")
        print("="*80)
        
        aggregated = categorization_results['aggregated']
        results = categorization_results['results']
        
        # Display by category
        for category, data in sorted(aggregated.items()):
            print(f"\n{category} ({data['count']} items, avg confidence: {data['avg_confidence']:.1%})")
            print("-" * 40)
            for item in data['items']:
                print(f"  • {item}")
        
        print("\n" + "="*80)
        print(f"Total Items Categorized: {len(results)}")
        print(f"Categories Used: {len(aggregated)}")
        overall_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"Overall Confidence: {overall_confidence:.1%}")
        print("="*80)


def interactive_mode(cli: ExpenseCategorizationCLI):
    """Run the CLI in interactive mode."""
    print("\n" + "="*80)
    print("INTERACTIVE MODE - AI Agent Expense Categorization")
    print("="*80)
    print("\nEnter receipt items (one per line). Type 'DONE' when finished.")
    print("Type 'QUIT' to exit.\n")
    
    while True:
        print("-" * 80)
        print("Enter receipt text (type 'DONE' on a new line when finished):")
        
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'DONE':
                break
            if line.strip().upper() == 'QUIT':
                print("\nExiting...")
                return
            lines.append(line)
        
        if not lines:
            print("No input provided. Please try again.")
            continue
        
        receipt_text = '\n'.join(lines)
        
        try:
            results = cli.categorize_receipt(receipt_text)
            cli.display_results(results)
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")
        
        print("\n" + "="*80)
        cont = input("Categorize another receipt? (y/n): ")
        if cont.lower() != 'y':
            print("\nExiting...")
            break


def batch_mode(cli: ExpenseCategorizationCLI, input_file: Path, output_file: Path = None):
    """Process receipt from file."""
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    with open(input_file, 'r') as f:
        receipt_text = f.read()
    
    results = cli.categorize_receipt(receipt_text)
    cli.display_results(results)
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            # Convert results to JSON-serializable format
            output = {
                'aggregated': {
                    cat: {
                        'items': data['items'],
                        'count': data['count'],
                        'avg_confidence': float(data['avg_confidence'])
                    }
                    for cat, data in results['aggregated'].items()
                },
                'total_items': len(results['results'])
            }
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='AI Agent Expense Categorization System'
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'batch'],
        default='interactive',
        help='Operation mode (default: interactive)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input file path (for batch mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (for batch mode)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/expense_classifier',
        help='Path to trained model directory'
    )
    
    args = parser.parse_args()
    
    # Get paths
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / args.model
    
    # Check if model exists
    if not model_path.exists():
        print("Error: Trained model not found.")
        print(f"Expected location: {model_path}")
        print("\nPlease train the model first using:")
        print("  python src/models/fine_tuned_model.py")
        return
    
    try:
        # Initialize CLI
        cli = ExpenseCategorizationCLI(model_path)
        
        # Run appropriate mode
        if args.mode == 'interactive':
            interactive_mode(cli)
        else:  # batch mode
            if not args.input:
                print("Error: --input required for batch mode")
                return
            
            input_path = Path(args.input)
            output_path = Path(args.output) if args.output else None
            batch_mode(cli, input_path, output_path)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
