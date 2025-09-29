import numpy as np
import pandas as pd
import torch

from train_extra_feature_only_model import train_feature_lstm


def check_data_balance(csv_path):
    """Check if the dataset is balanced"""
    df = pd.read_csv(csv_path)
    label_counts = df['label'].value_counts()
    print("Dataset balance:")
    total = len(df)
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")
    
    # Check if data is severely imbalanced
    max_percentage = (label_counts.max() / total) * 100
    min_percentage = (label_counts.min() / total) * 100
    
    if max_percentage > 70:
        print(f"⚠️  WARNING: Dataset is imbalanced! Dominant class: {max_percentage:.1f}%")
        return False
    elif max_percentage / min_percentage > 3:
        print(f"⚠️  WARNING: Class imbalance detected. Ratio: {max_percentage/min_percentage:.1f}:1")
        return False
    else:
        print("✅ Dataset appears reasonably balanced")
        return True

    
def train_with_improved_settings(csv_path):
    """Train with improved settings to avoid common issues"""
    # Check data balance first
    is_balanced = check_data_balance(csv_path)
    
    print("\n" + "="*50)
    print("TRAINING WITH IMPROVED SETTINGS")
    print("="*50)
    
    # Improved hyperparameters
    configs = [
        {
            'name': 'Baseline (Conservative)',
            'seq_len': 5,
            'hidden_size': 32,
            'num_layers': 2,
            'learning_rate': 0.0005,  # Lower learning rate
            'num_epochs': 150,
            'test_size': 0.25,
            'batch_size': 16,  # Smaller batch size
        },
        {
            'name': 'Medium Capacity',
            'seq_len': 10,
            'hidden_size': 64,
            'num_layers': 3,
            'learning_rate': 0.001,
            'num_epochs': 200,
            'test_size': 0.25,
            'batch_size': 32,
        },
        {
            'name': 'High Capacity',
            'seq_len': 15,
            'hidden_size': 128,
            'num_layers': 4,
            'learning_rate': 0.0008,
            'num_epochs': 250,
            'test_size': 0.25,
            'batch_size': 16,
        }
    ]
    
    best_config = None
    best_accuracy = 0
    
    for config in configs:
        print(f"\n{'='*20} {config['name']} {'='*20}")
        print(f"Configuration: {config}")
        
        try:
            model, trainer, processor, labels = train_feature_lstm(
                csv_path=csv_path,
                seq_len=config['seq_len'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                learning_rate=config['learning_rate'],
                num_epochs=config['num_epochs'],
                test_size=config['test_size'],
                batch_size=config['batch_size'],
            )
            
            print(f"✅ Training completed successfully for {config['name']}")
            
            # You could add evaluation here to pick the best model
            # For now, we'll consider the last one as potentially the best
            
        except Exception as e:
            print(f"❌ Training failed for {config['name']}: {str(e)}")
            continue
    
    print("\n" + "="*50)
    print("TRAINING RECOMMENDATIONS")
    print("="*50)
    
    if not is_balanced:
        print("\n🔧 DATA ISSUES DETECTED:")
        print("1. Your dataset appears imbalanced. Consider:")
        print("   - Collecting more data for underrepresented classes")
        print("   - Using data augmentation for minority classes")
        print("   - Using class weights in training")
        
    print("\n📝 TESTING")    
    


def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two keypoints."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def calculate_angle(p1, p2, p3):
    """Calculates the angle at p2 between p1-p2 and p2-p3 vectors in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    
    angle = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))
    return np.degrees(angle)    


def quick_test_run(csv_path: str):
    """
    Quick test run with minimal settings for rapid validation
    
    Args:
        csv_path (str): Path to your enhanced CSV dataset
    """
    print("⚡ Quick Test Run - Minimal Settings")
    print("=" * 50)
    
    return train_feature_lstm(
        csv_path=csv_path,
        seq_len=3,              # Short sequences
        hidden_size=16,         # Very small model
        num_layers=1,           # Single layer
        learning_rate=0.01,    # Higher LR for faster training
        num_epochs=20,          # Just 20 epochs
        test_size=0.2,          # Larger test set
        val_size=0.15,           # Larger validation set
        batch_size=128,          # Large batches
        num=0                   # Test model number
    )


