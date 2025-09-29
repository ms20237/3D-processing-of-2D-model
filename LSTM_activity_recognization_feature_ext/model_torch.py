import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class LSTMCell(nn.Module):
    """
    A single Long Short-Term Memory (LSTM) cell implementation in PyTorch.
    This class implements the forward pass of an LSTM cell, including the forget, input, candidate, and output gates.
    Weights are initialized using Xavier initialization for improved convergence. Biases for the forget gate are
    initialized to ones, while other biases are initialized to zeros.
    
    Attributes:
        input_size (int): The size of the input vector.
        hidden_size (int): The size of the hidden state vector.
        weight_ih (nn.Parameter): Input-to-hidden weights (4 * hidden_size, input_size).
        weight_hh (nn.Parameter): Hidden-to-hidden weights (4 * hidden_size, hidden_size).
        bias_ih (nn.Parameter): Input-to-hidden biases (4 * hidden_size).
        bias_hh (nn.Parameter): Hidden-to-hidden biases (4 * hidden_size).
    
    Methods:
        forward(x, hx): Performs a forward pass through the LSTM cell.
    
    Example:
        lstm_cell = LSTMCell(input_size=10, hidden_size=20)
        h, c = lstm_cell(x, (h_prev, c_prev))
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weights for all gates (forget, input, candidate, output)
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization and set forget gate bias to 1"""
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        
        # Initialize biases
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        
        # Set forget gate bias to 1 (indices 0:hidden_size correspond to forget gate)
        with torch.no_grad():
            self.bias_ih[0:self.hidden_size].fill_(1.0)
            self.bias_hh[0:self.hidden_size].fill_(1.0)
    
    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM cell
        
        Args:
            x: Input at current timestep (batch_size, input_size)
            hx: Tuple of (h_prev, c_prev) where:
                h_prev: Previous hidden state (batch_size, hidden_size)
                c_prev: Previous cell state (batch_size, hidden_size)
                
        Returns:
            h: New hidden state (batch_size, hidden_size)
            c: New cell state (batch_size, hidden_size)
        """
        if hx is None:
            batch_size = x.size(0)
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = hx
        
        # Compute gates using linear transformations
        gi = F.linear(x, self.weight_ih, self.bias_ih)
        gh = F.linear(h_prev, self.weight_hh, self.bias_hh)
        i_f, i_i, i_g, i_o = gi.chunk(4, 1)
        h_f, h_i, h_g, h_o = gh.chunk(4, 1)
        
        # Apply activations
        f = torch.sigmoid(i_f + h_f)  # Forget gate
        i = torch.sigmoid(i_i + h_i)  # Input gate
        g = torch.tanh(i_g + h_g)     # Candidate values
        o = torch.sigmoid(i_o + h_o)  # Output gate
        
        # Update cell state and hidden state
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        
        return h, c


class LSTM(nn.Module):
    """
    A multi-layer Long Short-Term Memory (LSTM) neural network implemented in PyTorch.
    This class supports stacking multiple LSTM layers and provides methods for forward propagation through the network.
    It maintains hidden and cell states across time steps and layers, and includes an output layer for mapping the final
    hidden state to the desired output size.
    
    Attributes:
        input_size (int): Number of input features per time step.
        hidden_size (int): Number of features in the hidden state of each LSTM layer.
        output_size (int): Number of output features per time step.
        num_layers (int): Number of stacked LSTM layers.
        dropout (float): Dropout probability between layers.
        lstm_layers (nn.ModuleList): List of LSTMCell instances, one for each layer.
        dropout_layer (nn.Dropout): Dropout layer applied between LSTM layers.
        output_layer (nn.Linear): Output (fully connected) layer.
    
    Methods:
        reset_states(batch_size: int, device: torch.device):
            Resets the hidden and cell states for all layers to zeros.
        forward(x: torch.Tensor, hx: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
            Performs a forward pass through the LSTM network for a given input sequence.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, dropout: float = 0.0):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.lstm_layers.append(LSTMCell(layer_input_size, hidden_size))
        
        # Dropout layer (applied between LSTM layers, not after the last one)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, hx: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through LSTM network
        
        Args:
            x: Input sequence (batch_size, seq_len, input_size) or (seq_len, batch_size, input_size)
            hx: Initial hidden and cell states for each layer
            
        Returns:
            outputs: Output sequence (batch_size, seq_len, output_size)
            final_states: Final (h, c) states for each layer
        """
        # Handle input dimensions - convert to (seq_len, batch_size, input_size) if needed
        if x.dim() == 3 and x.size(0) != x.size(1):  # Assume (batch_size, seq_len, input_size)
            x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, input_size)
        
        seq_len, batch_size, _ = x.size()
        
        # Initialize hidden states if not provided
        if hx is None:
            hx = [(torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype),
                   torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype))
                  for _ in range(self.num_layers)]
        
        # Store outputs and final states
        outputs = []
        current_states = hx
        
        # Process each time step
        for t in range(seq_len):
            layer_input = x[t]  # (batch_size, input_size)
            new_states = []
            
            # Forward through each LSTM layer
            for i, lstm_layer in enumerate(self.lstm_layers):
                h, c = lstm_layer(layer_input, current_states[i])
                new_states.append((h, c))
                
                # Apply dropout between layers (not after the last layer)
                if i < self.num_layers - 1 and self.dropout_layer is not None:
                    layer_input = self.dropout_layer(h)
                else:
                    layer_input = h
            
            current_states = new_states
            
            # Apply output layer to the final hidden state
            output = self.output_layer(layer_input)  # (batch_size, output_size)
            outputs.append(output)
        
        # Stack outputs: (seq_len, batch_size, output_size)
        outputs = torch.stack(outputs, dim=0)
        
        # Convert back to (batch_size, seq_len, output_size)
        outputs = outputs.transpose(0, 1)
        
        return outputs, current_states


class LossFunction:
    """
    LossFunction provides static methods for common loss calculations used in LSTM training.
    These functions work with PyTorch tensors and support automatic differentiation.
    
    Methods:
        mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor
            Computes the Mean Squared Error (MSE) loss.
        
        cross_entropy_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor
            Computes the cross-entropy loss for classification tasks.
    """
    
    @staticmethod
    def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error loss
        
        Args:
            y_pred: Predicted values (batch_size, seq_len, output_size)
            y_true: True values (batch_size, seq_len, output_size)
            
        Returns:
            loss: Scalar loss value
        """
        return F.mse_loss(y_pred, y_true)
    
    @staticmethod
    def cross_entropy_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss for classification
        
        Args:
            y_pred: Predicted logits (batch_size, seq_len, num_classes)
            y_true: True labels (batch_size, seq_len) as integers
            
        Returns:
            loss: Scalar loss value
        """
        # Reshape for cross entropy: (batch_size * seq_len, num_classes) and (batch_size * seq_len,)
        batch_size, seq_len, num_classes = y_pred.shape
        y_pred_flat = y_pred.view(-1, num_classes)
        y_true_flat = y_true.view(-1).long()
        
        return F.cross_entropy(y_pred_flat, y_true_flat)


# Example usage and training utilities
class LSTMTrainer:
    """
    Utility class for training the PyTorch LSTM model.
    
    Attributes:
        model (LSTM): The LSTM model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        device (torch.device): Device to run training on (CPU or GPU).
    """
    
    def __init__(self, model: LSTM, learning_rate: float = 0.001, device: str = 'cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, loss_fn) -> float:
        """
        Perform a single training step
        
        Args:
            x: Input sequences (batch_size, seq_len, input_size)
            y: Target sequences (batch_size, seq_len, output_size) or (batch_size, seq_len) for classification
            loss_fn: Loss function to use
            
        Returns:
            loss_value: The computed loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        outputs, _ = self.model(x)
        
        # Compute loss
        loss = loss_fn(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor, loss_fn) -> float:
        """
        Evaluate the model on validation data
        
        Args:
            x: Input sequences (batch_size, seq_len, input_size)
            y: Target sequences (batch_size, seq_len, output_size) or (batch_size, seq_len) for classification
            loss_fn: Loss function to use
            
        Returns:
            loss_value: The computed loss value
        """
        self.model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            outputs, _ = self.model(x)
            loss = loss_fn(outputs, y)
            return loss.item()