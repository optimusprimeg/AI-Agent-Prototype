"""
Data Preprocessor
Handles preprocessing of receipt text and items for the expense categorization model.
"""

import re
from typing import List, Dict, Tuple


class ReceiptPreprocessor:
    """Preprocesses receipt text and extracts items."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.item_patterns = [
            r'^\d+[\.\)]\s*(.+)',  # 1. item or 1) item
            r'^-\s*(.+)',           # - item
            r'^\*\s*(.+)',          # * item
            r'^•\s*(.+)',           # • item
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s\-\(\),\.]', '', text)
        return text.strip()
    
    def extract_items_from_receipt(self, receipt_text: str) -> List[str]:
        """Extract individual items from receipt text."""
        lines = receipt_text.strip().split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match line item patterns
            matched = False
            for pattern in self.item_patterns:
                match = re.match(pattern, line)
                if match:
                    item = match.group(1).strip()
                    items.append(self.clean_text(item))
                    matched = True
                    break
            
            # If no pattern matched, treat whole line as item
            if not matched and len(line) > 3:
                items.append(self.clean_text(line))
        
        return items
    
    def prepare_for_model(self, item: str) -> str:
        """Prepare item text for model input."""
        return self.clean_text(item)
    
    def batch_prepare(self, items: List[str]) -> List[str]:
        """Prepare a batch of items for model input."""
        return [self.prepare_for_model(item) for item in items]


def main():
    """Test the preprocessor."""
    preprocessor = ReceiptPreprocessor()
    
    # Test receipt text
    sample_receipt = """
    1. Coffee and pastry
    2. Lunch sandwich
    3. Uber ride to downtown
    4. Movie ticket
    5. Prescription medication
    """
    
    print("Sample receipt text:")
    print(sample_receipt)
    print("\nExtracted items:")
    
    items = preprocessor.extract_items_from_receipt(sample_receipt)
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")


if __name__ == "__main__":
    main()
