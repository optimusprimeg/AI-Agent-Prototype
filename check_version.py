#!/usr/bin/env python3
"""
Version Check Script
Verifies you have the CPU-optimized training code.
"""

import sys
from pathlib import Path

def check_version():
    """Check if the CPU-optimized version is present."""
    
    print("=" * 60)
    print("CPU-Optimized Training Version Check")
    print("=" * 60)
    
    # Check file exists
    model_file = Path(__file__).parent / 'src' / 'models' / 'fine_tuned_model.py'
    if not model_file.exists():
        print(f"❌ ERROR: {model_file} not found!")
        return False
    
    # Read the file
    with open(model_file, 'r') as f:
        content = f.read()
    
    checks = {
        "Pin memory disabled": "dataloader_pin_memory=False" in content,
        "Auto batch size": "get_optimal_batch_size" in content,
        "Pre-tokenization": "precompute_encodings=True" in content,
        "System info detection": "get_system_info" in content,
        "Progress bars": "from tqdm import tqdm" in content,
        "Resource monitoring": "import psutil" in content,
    }
    
    print("\nFeature Checks:")
    all_passed = True
    for feature, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {feature}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ You have the CPU-optimized version!")
        print("\nYou can now run:")
        print("  python src/models/fine_tuned_model.py --quick")
    else:
        print("❌ You have an OLD version!")
        print("\nPlease run:")
        print("  git pull origin copilot/fix-135869538-1088827371-3115b681-fff6-436b-a386-90ee8910cb52")
        print("\nThen clear Python cache:")
        print("  find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true")
        print("  find . -name '*.pyc' -delete 2>/dev/null || true")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = check_version()
    sys.exit(0 if success else 1)
