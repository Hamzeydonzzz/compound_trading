import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import copy
from config import ModelConfig


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model"""
    
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but should be saved)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to the input tensor"""
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head attention and feed-forward network"""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Activation
        self.activation = nn.ReLU()
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        """Process input through transformer encoder block"""
        # Multi-head attention
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)  # Add & Norm
        src = self.norm1(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # Add & Norm
        src = self.norm2(src)
        
        return src


class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, config=None):
        """
        Initialize the Time Series Transformer model for financial prediction
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        
        self.config = config or ModelConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        d_model = self.config.NUM_FEATURES  # Dimension of the model
        nhead = self.config.NUM_HEADS  # Number of attention heads
        dim_feedforward = self.config.FF_DIM  # Feed-forward dimension
        num_layers = self.config.TRANSFORMER_LAYERS  # Number of transformer layers
        dropout = self.config.DROPOUT  # Dropout rate
        self.sequence_length = self.config.SEQUENCE_LENGTH  # Input sequence length
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Global average pooling (similar to keras.layers.GlobalAveragePooling1D)
        self.global_avg_pool = lambda x: torch.mean(x, dim=1)
        
        # Final prediction layers
        self.fc1 = nn.Linear(d_model, dim_feedforward // 2)
        self.fc1_activation = nn.ReLU()
        self.fc1_dropout = nn.Dropout(dropout)
        
        # Output layer
        if self.config.TASK_TYPE == 'classification':
            num_classes = getattr(self.config, 'NUM_CLASSES', 3)  # Default: Up, Down, Neutral
            self.output_layer = nn.Linear(dim_feedforward // 2, num_classes)
            self.output_activation = nn.Softmax(dim=1)
        else:  # regression
            self.output_layer = nn.Linear(dim_feedforward // 2, self.config.PREDICTION_HORIZON)
            self.output_activation = nn.Identity()
        
        # Loss function
        if self.config.TASK_TYPE == 'classification':
            self.loss_fn = nn.CrossEntropyLoss()
        else:  # regression
            self.loss_fn = nn.MSELoss()
        
        # Move model to device
        self.to(self.device)
    
    def forward(self, x):
        """Forward pass through the transformer model"""
        # Input shape: [batch_size, sequence_length, num_features]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer encoder blocks
        for layer in self.transformer_encoder:
            x = layer(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Final prediction layers
        x = self.fc1(x)
        x = self.fc1_activation(x)
        x = self.fc1_dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        x = self.output_activation(x)
        
        return x
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model
        
        Args:
            X_train: Training features (numpy array)
            y_train: Training targets (numpy array)
            X_val: Validation features (numpy array)
            y_val: Validation targets (numpy array)
            **kwargs: Additional arguments
            
        Returns:
            Training history
        """
        # Set model to training mode
        self.train()
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        
        if self.config.TASK_TYPE == 'classification':
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True
        )
        
        # Create validation dataloader if validation data is provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            
            if self.config.TASK_TYPE == 'classification':
                y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
            else:
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.BATCH_SIZE, 
                shuffle=False
            )
        
        # Create optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)
        
        # Learning rate scheduler
        scheduler = None
        if hasattr(self.config, 'LR_PATIENCE') and self.config.LR_PATIENCE:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.LR_PATIENCE,
                min_lr=1e-6,
                verbose=True
            )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }
        
        # Best model state (for early stopping)
        best_model_state = None
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.EPOCHS):
            # Train for one epoch
            train_loss, train_metric = self._train_epoch(train_loader, optimizer)
            
            # Validate if validation data is provided
            val_loss, val_metric = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_metric = self._validate(val_loader)
                
                # Learning rate scheduler step
                if scheduler is not None:
                    scheduler.step(val_loss)
                
                # Early stopping
                if hasattr(self.config, 'EARLY_STOPPING') and self.config.EARLY_STOPPING:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(self.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.config.PATIENCE:
                        print(f"Early stopping at epoch {epoch}")
                        # Restore best model
                        if best_model_state is not None:
                            self.load_state_dict(best_model_state)
                        break
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_metric'].append(train_metric)
            history['val_metric'].append(val_metric)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} - "
                  f"loss: {train_loss:.4f} - "
                  f"{'accuracy' if self.config.TASK_TYPE == 'classification' else 'mae'}: {train_metric:.4f}", 
                  end="")
            
            if val_loader is not None:
                print(f" - val_loss: {val_loss:.4f} - "
                      f"val_{'accuracy' if self.config.TASK_TYPE == 'classification' else 'mae'}: {val_metric:.4f}")
            else:
                print()
        
        # Restore best model if early stopping was used
        if hasattr(self.config, 'EARLY_STOPPING') and self.config.EARLY_STOPPING and best_model_state is not None:
            self.load_state_dict(best_model_state)
        
        return history
    
    def _train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        self.train()
        running_loss = 0.0
        metrics = []
        
        for inputs, targets in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(inputs)
            
            # Calculate loss
            if self.config.TASK_TYPE == 'classification':
                loss = self.loss_fn(outputs, targets)
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                metric = (predicted == targets).float().mean().item()
            else:
                loss = self.loss_fn(outputs, targets)
                # Calculate MAE
                metric = torch.abs(outputs - targets).mean().item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss and metrics
            running_loss += loss.item() * inputs.size(0)
            metrics.append(metric)
        
        # Calculate average loss and metric
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_metric = np.mean(metrics)
        
        return epoch_loss, epoch_metric
    
    def _validate(self, val_loader):
        """Validate the model"""
        self.eval()
        running_loss = 0.0
        metrics = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Forward pass
                outputs = self(inputs)
                
                # Calculate loss
                if self.config.TASK_TYPE == 'classification':
                    loss = self.loss_fn(outputs, targets)
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    metric = (predicted == targets).float().mean().item()
                else:
                    loss = self.loss_fn(outputs, targets)
                    # Calculate MAE
                    metric = torch.abs(outputs - targets).mean().item()
                
                # Update running loss and metrics
                running_loss += loss.item() * inputs.size(0)
                metrics.append(metric)
        
        # Calculate average loss and metric
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_metric = np.mean(metrics)
        
        return epoch_loss, epoch_metric
    
    def predict(self, X):
        """
        Generate predictions
        
        Args:
            X: Input features (numpy array)
            
        Returns:
            Model predictions (numpy array)
        """
        # Set model to evaluation mode
        self.eval()
        
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            predictions = self(X_tensor)
        
        # Convert predictions to numpy array
        if self.config.TASK_TYPE == 'classification':
            # Return class probabilities
            return predictions.cpu().numpy()
        else:
            # Return regression values
            return predictions.cpu().numpy()
    
    def save(self, filepath):
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, filepath)
        
    def load(self, filepath):
        """
        Load the model from disk
        
        Args:
            filepath: Path to the saved model
        """
        # Load model state
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update config if available
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        
        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        
        return self


# Example usage:
if __name__ == "__main__":
    # Sample code to test the model
    import matplotlib.pyplot as plt
    from torchinfo import summary
    
    # Create model instance
    transformer = TimeSeriesTransformer()
    
    # Print model summary
    batch_size = 16
    sequence_length = transformer.config.SEQUENCE_LENGTH
    num_features = transformer.config.NUM_FEATURES
    
    # Print summary using torchinfo if available
    try:
        summary(transformer, input_size=(batch_size, sequence_length, num_features))
    except:
        print(transformer)
    
    # Alternative visualization using matplotlib
    try:
        # Generate random input
        x = torch.randn(batch_size, sequence_length, num_features)
        
        # Get prediction
        y = transformer(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        
        # Plot sample prediction for visualization
        if transformer.config.TASK_TYPE == 'classification':
            plt.figure(figsize=(10, 6))
            plt.bar(range(y.shape[1]), y[0].detach().cpu().numpy())
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Sample Classification Output')
            plt.savefig('transformer_output.png')
            print("Sample output visualization saved to transformer_output.png")
    except Exception as e:
        print(f"Could not generate visualization: {e}")