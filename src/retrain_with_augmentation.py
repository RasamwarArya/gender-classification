"""
Quick retraining script that reprocesses data with augmentation and retrains model.
This will improve female voice recognition significantly.
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("RETRAINING MODEL WITH AUGMENTATION FOR BETTER FEMALE DETECTION")
    print("=" * 60)
    print("\nThis will:")
    print("1. Reprocess data with augmentation (may take 20-30 minutes)")
    print("2. Retrain model with balanced dataset")
    print("3. Improve female voice recognition significantly")
    print("\n" + "=" * 60)
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print("\nStep 1: Processing data with augmentation...")
    print("This will take 20-30 minutes. Please be patient...\n")
    
    result1 = subprocess.run([sys.executable, 'prepare_data.py'], 
                            cwd=Path(__file__).parent)
    
    if result1.returncode != 0:
        print("\nError during data preparation!")
        return
    
    print("\n" + "=" * 60)
    print("Step 2: Training improved model...")
    print("=" * 60 + "\n")
    
    result2 = subprocess.run([sys.executable, 'train_gender_classifier.py'],
                            cwd=Path(__file__).parent)
    
    if result2.returncode != 0:
        print("\nError during model training!")
        return
    
    print("\n" + "=" * 60)
    print("SUCCESS! Model retrained with augmentation.")
    print("Female voice recognition should be significantly improved.")
    print("=" * 60)

if __name__ == '__main__':
    main()

