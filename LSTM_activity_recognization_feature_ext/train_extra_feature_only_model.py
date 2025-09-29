import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from model_torch import LSTM, LSTMTrainer


class FeatureDataset(Dataset):
    """
    PyTorch Dataset for engineered features sequences
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeatureDataProcessor:
    """
    Data processor specifically for engineered features (not keypoints)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Define the engineered features based on your CSV header
        self.feature_columns = [
            'left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle',
            'spine_curvature_angle', 'neck_angle', 'body_length', 'tail_length',
            'nose_to_left_wrist_dist', 'nose_to_right_wrist_dist', 'wrist_distance',
            'nose_to_left_ankle_dist', 'nose_to_right_ankle_dist', 'hind_paw_distance',
            'nose_to_left_wrist_ratio', 'nose_to_right_wrist_ratio', 'hind_paw_distance_ratio',
            'tail_angle'
        ]
        
    def load_csv_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and preprocess feature data from CSV
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            X: Feature data (num_samples, num_features)
            y: Encoded labels (num_samples,)
            label_names: Original label names
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Extract only the engineered features
        try:
            X = df[self.feature_columns].values
            print(f"✅ Successfully extracted {len(self.feature_columns)} engineered features")
        except KeyError as e:
            print(f"❌ Error: Some feature columns not found in CSV: {e}")
            # Fallback: try to identify feature columns automatically
            all_cols = df.columns.tolist()
            keypoint_cols = [col for col in all_cols if col.startswith('kp') and ('_x' in col or '_y' in col)]
            feature_cols = [col for col in all_cols if col not in keypoint_cols and col != 'label']
            
            if feature_cols:
                print(f"📋 Found {len(feature_cols)} potential feature columns: {feature_cols}")
                X = df[feature_cols].values
                self.feature_columns = feature_cols
            else:
                raise ValueError("No feature columns found in the dataset")
        
        # Handle missing values
        if np.isnan(X).any():
            print("⚠️  Warning: Found NaN values in features. Filling with column means...")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
        
        # Extract labels
        labels = df['label'].values
        label_names = list(df['label'].unique())
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"📊 Loaded {len(X)} samples with {len(label_names)} classes: {label_names}")
        print(f"📐 Feature dimension: {X.shape[1]} features")
        print(f"📋 Features used: {', '.join(self.feature_columns)}")
        
        return X_scaled, y, label_names
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int, 
                        stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from feature data for temporal modeling
        
        Args:
            X: Input features (num_samples, feature_dim)
            y: Labels (num_samples,)
            seq_len: Length of each sequence
            stride: Step size between sequences
            
        Returns:
            X_seq: Sequence data (num_sequences, seq_len, feature_dim)
            y_seq: Sequence labels (num_sequences,)
        """
        if len(X) < seq_len:
            raise ValueError(f"Not enough data points ({len(X)}) for sequence length {seq_len}")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(0, len(X) - seq_len + 1, stride):
            X_sequences.append(X[i:i + seq_len])
            # Use the label of the last frame in the sequence
            y_sequences.append(y[i + seq_len - 1])
        
        return np.array(X_sequences), np.array(y_sequences)


class FeatureLSTMTrainer(LSTMTrainer):
    """
    Enhanced LSTM trainer specifically for engineered features
    """
    
    def __init__(self, model: LSTM, learning_rate: float = 0.001, device: str = 'cpu'):
        super().__init__(model, learning_rate, device)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform a single training step for feature-based classification
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        outputs, _ = self.model(x)  # (batch_size, seq_len, output_size)
        
        # For classification, use only the last timestep
        final_outputs = outputs[:, -1, :]  # (batch_size, output_size)
        
        # Compute loss
        loss = self.criterion(final_outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on validation/test data
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                outputs, _ = self.model(x)
                final_outputs = outputs[:, -1, :]
                
                loss = self.criterion(final_outputs, y)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(final_outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on input data
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs, _ = self.model(x)
            final_outputs = outputs[:, -1, :]  # (batch_size, output_size)
            
            probabilities = torch.softmax(final_outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
        return predictions.cpu(), probabilities.cpu()


def train_feature_lstm(csv_path: str, seq_len: int = 10, hidden_size: int = 64, 
                      num_layers: int = 2, learning_rate: float = 0.001, 
                      num_epochs: int = 100, test_size: float = 0.2, val_size: float = 0.1, 
                      batch_size: int = 32, num: int = 1, device: str = 'auto', 
                      model_save_dir: str = "./model"):
    """
    Complete training pipeline for feature-based activity recognition using PyTorch LSTM
    
    Args:
        csv_path: Path to CSV file with feature data
        seq_len: Length of input sequences
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        test_size: Fraction of data for testing 
        val_size: Fraction of data for validation
        batch_size: Batch size for training
        num: Model number for saving
        device: Device to use ('cpu', 'cuda', or 'auto')
    """
    
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("-" * 60)
    print(f"🚀 Starting Feature-Based LSTM Model Training 🚀")
    print(f"Using Device: {device.upper()}")
    print("-" * 60)

    # Load and preprocess data
    print("\n📦 Loading and Preprocessing Feature Data...")
    processor = FeatureDataProcessor()
    X, y, label_names = processor.load_csv_data(csv_path)
    
    num_classes = len(label_names)
    num_features = X.shape[1]
    
    print(f"  ✅ Data Loaded Successfully")
    print(f"  📊 Number of Features: {num_features}")
    print(f"  🏷️  Number of Classes: {num_classes}")
    print(f"  📋 Classes: {', '.join(label_names)}")
    
    if num_classes < 2:
        raise ValueError("Error: Need at least 2 classes for classification.")
    
    # Create sequences
    print(f"\n📏 Creating Sequences with length: {seq_len}...")
    X_seq, y_seq = processor.create_sequences(X, y, seq_len)
    print(f"  ✅ Created {len(X_seq)} sequences from {len(X)} samples")
    
    # Check class distribution
    unique, counts = np.unique(y_seq, return_counts=True)
    print("\n📈 Class Distribution in Sequences:")
    for class_idx, count in zip(unique, counts):
        print(f"  - {label_names[class_idx]:<15s}: {count:<6d} sequences ({count/len(y_seq)*100:.1f}%)")
    
    # Split data
    print("\n📊 Splitting Data into Training, Validation, and Test Sets...")
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X_seq, y_seq, test_size=test_size, random_state=42, stratify=y_seq
    )
    val_test_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=val_test_ratio, random_state=42, stratify=y_train_temp
    )
    
    print(f"  Training set:     {len(X_train):<6d} sequences")
    print(f"  Validation set:   {len(X_val):<6d} sequences")
    print(f"  Test set:         {len(X_test):<6d} sequences")
    
    # Create data loaders
    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = num_features  # Number of engineered features
    output_size = num_classes
    
    print(f"\n🧠 Initializing Feature-Based LSTM Model:")
    print(f"  Input features per timestep: {input_size}")
    print(f"  LSTM Hidden size:           {hidden_size}")
    print(f"  Number of LSTM layers:      {num_layers}")
    print(f"  Output classes:             {output_size}")
    
    model = LSTM(input_size, hidden_size, output_size, num_layers, dropout=0.1)
    trainer = FeatureLSTMTrainer(model, learning_rate, device)
    
    # Training loop
    print(f"\n⚙️ Starting Training for {num_epochs} Epochs...")
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        for batch_x, batch_y in train_loader:
            loss = trainer.train_step(batch_x, batch_y)
            epoch_loss += loss
            num_batches += 1
        
        # Evaluation
        train_loss, train_acc = trainer.evaluate(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        #     print(f"Epoch {epoch+1:3d}/{num_epochs}: "
        #           f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} | "
        #           f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f} | "
        #           f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}")
        
        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
          f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} | "
          f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f} | "
          f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}")
    
    # Final evaluation
    print("-" * 60)
    print("\n🏆 Training Complete!")
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    final_test_loss, final_test_acc = trainer.evaluate(test_loader)
    
    print(f"  🎯 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"  🎯 Final Test Accuracy: {final_test_acc:.4f}")

    # Detailed evaluation
    print("\n🔬 Detailed Test Set Evaluation...")
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            predictions, _ = trainer.predict(batch_x)
            all_predictions.extend(predictions.numpy())
            all_true_labels.extend(batch_y.numpy())
    
    # Confusion Matrix
    print("\n📊 Confusion Matrix (True vs. Predicted):")
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(all_true_labels, all_predictions):
        confusion_matrix[true_label, pred_label] += 1
    
    # Print confusion matrix
    print("True\\Pred", end="")
    for label in label_names:
        print(f"{label:>12s}", end="")
    print()
    
    for i, true_label in enumerate(label_names):
        print(f"{true_label:>8s}", end="")
        for j in range(num_classes):
            print(f"{confusion_matrix[i,j]:>12d}", end="")
        print()
    
    # Per-class metrics
    print("\n📋 Per-class Performance Report:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 65)
    
    for i, label in enumerate(label_names):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(confusion_matrix[i, :])
        
        print(f"{label:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10d}")
    
    # Save model
    model_save_path = os.path.join(model_save_dir, f"ext_feature_only_lstm_model{num}.pt")
    os.makedirs(model_save_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'num_layers': num_layers
        },
        'label_names': label_names,
        'scaler': processor.scaler,
        'label_encoder': processor.label_encoder,
        'feature_columns': processor.feature_columns
    }, model_save_path)
    
    print(f"\n💾 Model successfully saved to: {model_save_path}")
    print("-" * 60)
    
    # Plot results
    os.makedirs(f"./LSTM_activity_recognization_feature_ext2/train_infos/train{num}", exist_ok=True)
    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, 
                         confusion_matrix, label_names, counts, num)
    
    return model, trainer, processor, label_names


def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies,
                         confusion_matrix, label_names, class_counts, num):
    """
    Plot comprehensive training results
    """
    plt.style.use('default')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig.suptitle('Feature-Based LSTM Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss Over Time
    ax1 = axes[0, 0]
    ax1.plot(train_losses, label='Training Loss', color='b', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='r', linewidth=2)
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy Over Time
    ax2 = axes[0, 1]
    ax2.plot(train_accuracies, label='Training Accuracy', color='b', linewidth=2)
    ax2.plot(val_accuracies, label='Validation Accuracy', color='r', linewidth=2)
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    ax3 = axes[1, 0]
    im = ax3.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title('Test Set Confusion Matrix')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(label_names))
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(label_names, rotation=45, ha="right")
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(label_names)
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            text_color = "white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black"
            ax3.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center", color=text_color)
    
    # Plot 4: Class Distribution
    ax4 = axes[1, 1]
    bars = ax4.bar(label_names, class_counts, color='lightblue', edgecolor='navy', alpha=0.7)
    ax4.set_title('Class Distribution')
    ax4.set_xlabel('Activity Classes')
    ax4.set_ylabel('Number of Sequences')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(class_counts),
                f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"./LSTM_activity_recognization_feature_ext2/train_infos/train{num}/train{num}.png")
    plt.show()


def load_feature_trained_model(model_path: str, device: str = 'cpu'):
    """
    Load a trained feature-based LSTM model in PyTorch 2.6+
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
        
    Returns:
        model: Loaded LSTM model
        metadata: Model metadata including scaler, label encoder, etc.
    """
    # Load checkpoint with weights_only=False (less secure)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Reconstruct LSTM model
    config = checkpoint['model_config']
    model = LSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size'],
        num_layers=config['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load metadata
    metadata = {
        'label_names': checkpoint['label_names'],
        'scaler': checkpoint['scaler'],
        'label_encoder': checkpoint['label_encoder'],
        'feature_columns': checkpoint['feature_columns']
    }
    return model, metadata
