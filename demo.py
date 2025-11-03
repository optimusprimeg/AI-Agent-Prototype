#!/usr/bin/env python3
"""
Demo Script for AI Agent Expense Categorization System

This script demonstrates the complete workflow:
1. Generate synthetic data
2. Train the model
3. Evaluate the model
4. Run a sample categorization
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        return False
    
    print(f"\n✅ Success: {description} completed")
    return True


def main():
    """Run the complete demo workflow."""
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║        AI AGENT EXPENSE CATEGORIZATION SYSTEM - DEMO                       ║
    ║                                                                            ║
    ║        Student: Optimus Prime                                              ║
    ║        University: Autonomous University                                   ║
    ║        Department: Computer Science                                        ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    base_dir = Path(__file__).parent
    
    # Check if already trained
    model_dir = base_dir / 'models' / 'expense_classifier'
    if model_dir.exists() and (model_dir / 'config.json').exists():
        print("\n📌 Model already trained. Skipping training steps.")
        print("   Delete 'models/expense_classifier/' to retrain from scratch.")
        skip_training = True
    else:
        skip_training = False
        
        # Step 1: Generate Data
        if not run_command(
            "python src/data_generator.py",
            "Step 1/4: Generating Synthetic Receipt Data"
        ):
            return 1
        
        # Step 2: Train Model
        print("\n📝 Note: Model training will take approximately:")
        print("   - GPU (8GB VRAM): 10-15 minutes")
        print("   - CPU (16GB RAM): 30-45 minutes")
        
        response = input("\nContinue with training? (y/n): ")
        if response.lower() != 'y':
            print("\n⚠️  Demo cancelled. Run 'python demo.py' again to continue.")
            return 0
        
        if not run_command(
            "python src/models/fine_tuned_model.py",
            "Step 2/4: Training Fine-Tuned Model (This will take some time...)"
        ):
            return 1
        
        # Step 3: Evaluate Model
        if not run_command(
            "python src/evaluate.py",
            "Step 3/4: Evaluating Model on Test Set"
        ):
            return 1
    
    # Step 4: Run Sample Categorization
    print(f"\n{'='*80}")
    print("Step 4/4: Running Sample Expense Categorization")
    print(f"{'='*80}")
    
    # Create a sample receipt
    sample_receipt = base_dir / 'sample_receipt.txt'
    with open(sample_receipt, 'w') as f:
        f.write("""1. Morning coffee and bagel
2. Lunch at restaurant
3. Uber ride to meeting
4. Movie ticket for evening show
5. Prescription medication refill
6. Internet bill payment
7. New running shoes
8. Rent payment for apartment
9. Online course subscription
10. Groceries from supermarket""")
    
    print(f"\nCreated sample receipt: {sample_receipt}")
    print("\nSample receipt contents:")
    print("-" * 40)
    with open(sample_receipt, 'r') as f:
        print(f.read())
    print("-" * 40)
    
    # Run categorization
    print("\n🚀 Running AI Agent Categorization...\n")
    result = subprocess.run(
        f"python src/main.py --mode batch --input sample_receipt.txt",
        shell=True,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print("\n❌ Error: Categorization failed")
        return 1
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    
    print("\n📚 Next Steps:")
    print("   1. Review the architecture: docs/ARCHITECTURE.md")
    print("   2. Read the data science report: docs/DATA_SCIENCE_REPORT.md")
    print("   3. Check interaction logs: logs/interaction_logs.txt")
    print("   4. Try interactive mode: python src/main.py --mode interactive")
    print("   5. View evaluation results: logs/evaluation_results.json")
    
    print("\n💡 Tips:")
    print("   - Use interactive mode for real-time categorization")
    print("   - Check evaluation metrics in logs/evaluation_results.json")
    print("   - Model is saved in models/expense_classifier/")
    print("   - Processed data is in data/processed/")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
