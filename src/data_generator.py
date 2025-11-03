"""
Synthetic Receipt Data Generator
Generates receipt items with categories for training the expense categorization model.
"""

import json
import random
from typing import List, Dict
from pathlib import Path


class ReceiptDataGenerator:
    """Generates synthetic receipt data for training and testing."""
    
    # Predefined categories and sample items
    CATEGORIES = {
        "Food": [
            "Coffee and pastry", "Lunch sandwich", "Dinner at restaurant", "Groceries",
            "Pizza delivery", "Breakfast burrito", "Salad bowl", "Sushi platter",
            "Ice cream cone", "Fast food meal", "Bagel with cream cheese", "Protein shake",
            "Chicken tikka masala", "Caesar salad", "Burger and fries", "Pasta carbonara",
            "Steak dinner", "Fish and chips", "Tacos", "Burrito bowl", "Smoothie",
            "Energy bar", "Trail mix", "Fresh fruit", "Vegetables", "Bread and milk",
            "Cheese and crackers", "Deli sandwich", "Soup and bread", "Fried rice"
        ],
        "Transportation": [
            "Uber ride", "Lyft to airport", "Bus ticket", "Subway fare", "Train ticket",
            "Gas station fuel", "Parking fee", "Taxi fare", "Car wash", "Oil change",
            "Vehicle registration", "Highway toll", "Airport shuttle", "Bike rental",
            "Scooter rental", "Metro card", "Bus pass", "Parking garage", "Car rental",
            "Ride share", "Gasoline", "Diesel fuel", "Tire rotation", "Vehicle inspection"
        ],
        "Utilities": [
            "Electric bill", "Water bill", "Gas bill", "Internet service", "Phone bill",
            "Streaming subscription", "Cloud storage", "Cable TV", "Trash collection",
            "Sewage service", "Home security", "Electricity payment", "Heating bill",
            "Mobile data plan", "Landline phone", "WiFi service", "Solar panel payment"
        ],
        "Entertainment": [
            "Movie ticket", "Concert tickets", "Theater show", "Museum entry", "Bowling",
            "Mini golf", "Arcade games", "Video game purchase", "Book purchase",
            "Magazine subscription", "Spotify premium", "Netflix subscription", "Gaming subscription",
            "Sports event ticket", "Amusement park", "Zoo admission", "Aquarium visit",
            "Comedy show", "Music festival pass", "Art gallery", "Wine tasting", "Escape room"
        ],
        "Healthcare": [
            "Doctor visit copay", "Prescription medication", "Dental cleaning", "Eye exam",
            "Physical therapy", "Lab tests", "X-ray scan", "Emergency room", "Urgent care",
            "Chiropractor", "Massage therapy", "Vitamins and supplements", "First aid supplies",
            "Medical insurance premium", "Pharmacy items", "Contact lenses", "Glasses",
            "Health screening", "Vaccination", "Mental health counseling"
        ],
        "Shopping": [
            "Clothing purchase", "Shoes", "Accessories", "Electronics", "Home decor",
            "Furniture", "Kitchen appliances", "Bedding", "Towels", "Office supplies",
            "Books and stationery", "Toys and games", "Sporting goods", "Jewelry",
            "Cosmetics", "Personal care items", "Pet supplies", "Garden tools",
            "Hardware store", "Art supplies", "Craft materials", "Musical instrument"
        ],
        "Housing": [
            "Rent payment", "Mortgage payment", "Property tax", "Home insurance",
            "HOA fees", "Home repairs", "Plumbing service", "Electrical work",
            "Lawn care", "Pest control", "House cleaning", "Painting service",
            "Roof repair", "HVAC maintenance", "Window cleaning", "Carpet cleaning",
            "Appliance repair", "Lock replacement", "Gutter cleaning"
        ],
        "Education": [
            "Tuition payment", "Textbooks", "School supplies", "Online course",
            "Certification exam", "Workshop fee", "Seminar registration", "Language classes",
            "Music lessons", "Art classes", "Tutoring services", "Educational software",
            "Lab fees", "Student activities", "Academic conference"
        ]
    }
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        random.seed(seed)
        self.receipts = []
        
    def generate_single_item(self) -> Dict[str, str]:
        """Generate a single receipt item with category."""
        category = random.choice(list(self.CATEGORIES.keys()))
        item = random.choice(self.CATEGORIES[category])
        
        # Add some variation to item descriptions
        variations = [
            lambda x: x,
            lambda x: f"{x} - regular",
            lambda x: f"{x} (premium)",
            lambda x: f"Purchase: {x}",
            lambda x: f"{x} item",
        ]
        
        item_text = random.choice(variations)(item)
        
        return {
            "item": item_text,
            "category": category
        }
    
    def generate_receipt(self, num_items: int = None) -> Dict:
        """Generate a complete receipt with multiple items."""
        if num_items is None:
            num_items = random.randint(1, 8)
        
        items = [self.generate_single_item() for _ in range(num_items)]
        
        receipt = {
            "receipt_id": f"RCP{len(self.receipts):06d}",
            "items": items,
            "total_items": num_items
        }
        
        return receipt
    
    def generate_dataset(self, num_samples: int = 1000) -> List[Dict]:
        """Generate a dataset of receipt items."""
        dataset = []
        for _ in range(num_samples):
            item = self.generate_single_item()
            dataset.append(item)
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filepath: Path):
        """Save dataset to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} samples to {filepath}")
    
    def split_dataset(self, dataset: List[Dict], train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, test_ratio: float = 0.15):
        """Split dataset into train, validation, and test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        random.shuffle(dataset)
        n = len(dataset)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return {
            'train': dataset[:train_end],
            'val': dataset[train_end:val_end],
            'test': dataset[val_end:]
        }


def main():
    """Generate and save synthetic receipt data."""
    generator = ReceiptDataGenerator(seed=42)
    
    # Generate 1200 samples
    print("Generating synthetic receipt data...")
    dataset = generator.generate_dataset(num_samples=1200)
    
    # Split dataset
    splits = generator.split_dataset(dataset)
    
    # Save splits
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    generator.save_dataset(splits['train'], data_dir / 'train.json')
    generator.save_dataset(splits['val'], data_dir / 'val.json')
    generator.save_dataset(splits['test'], data_dir / 'test.json')
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Validation: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    print(f"  Total: {len(dataset)} samples")
    
    # Show sample
    print("\nSample items:")
    for i, item in enumerate(splits['train'][:5], 1):
        print(f"  {i}. '{item['item']}' -> {item['category']}")


if __name__ == "__main__":
    main()
