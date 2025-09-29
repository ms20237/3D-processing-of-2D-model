import numpy as np
import pandas as pd
import argparse

from train_extra_feature_only_model import train_feature_lstm
from utils import check_data_balance  


def init():
    parser = argparse.ArgumentParser(description="Run action recognition on a video using YOLO and a custom LSTM model.")
    parser.add_argument("--csv_path", type=str, default="./LSTM_activity_recognization_feature_ext2/dataset/balanced_normalized_features_added_keypoints_activity_v11s.csv", help="path to your csv_file for training lstm.")
    parser.add_argument("--seq_len", type=int, default=10, help="number of sequence that yo want to train model on it.")
    parser.add_argument("--hidden_size", type=int, default=32, help="hidden layers number for lstm model.")
    parser.add_argument("--num_layers", type=int, default=2, help="number of lstm model layers.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="leaning-rate for training lstm model.")
    parser.add_argument("--num_epochs", type=int, default=80, help="number of epochs for training lstm model.")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size for training lstm model.")
    parser.add_argument("--num_train", type=int, default=12, help="order number of training model.")
    
    return parser.parse_args()


def run(
    csv_path: str,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    num_train: int,
):
    check_data_balance(csv_path=csv_path)  
    print("🚀 Enhanced LSTM Training for Activity Recognition")
    print("=" * 200)
    print(f"📁 Dataset: {csv_path}")
    print("=" * 200)
    
    model, trainer, processor, labels = train_feature_lstm(
            csv_path=csv_path,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num=num_train,
            model_save_dir="./model",
            device='auto',
        )
    
    print("\n" + "=" * 100)
    print("🏁 Main execution completed!")
    print("=" * 100)


if __name__ == "__main__":
    
    # CONFIGURATION 
    args = init()
    run(args.csv_path,
        args.seq_len,
        args.hidden_size,
        args.num_layers,
        args.learning_rate,
        args.num_epochs,
        args.batch_size,
        args.num_train)
    