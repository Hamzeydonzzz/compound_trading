#!/usr/bin/env python3
"""
train.py

This script implements the training loop for the transformer-based trading model.
It handles data loading, feature generation, model training, and evaluation.
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm

# Add project root to path to allow imports from other modules
sys.path.append(str(Path(__file__).parents[1]))

from config import Config
from model import TransformerModel
from feature_engineering import calculate_features
from data_handler import DataHandler
from utils import setup_logging, plot_training_metrics

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the trading model')
    parser.add_argument('--data_dir', type=str, help='Directory containing historical data')
    parser.add_argument('--model_dir', type=str, help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, help='Directory to save training logs')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='15m', help='Candle interval (default: 15m)')
    parser.add_argument('--start_date', type=str, help='Start date for training in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date for training in YYYY-MM-DD format')
    parser.add_argument('--validation_split', type=float, default=0.2, 
                        help='Fraction of data to use for validation (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--sequence_length', type=int, default=96, 
                        help='Sequence length in time steps (default: 96 = 24h for 15m candles)')
    parser.add_argument('--target_horizon', type=int, default=16, 
                        help='Prediction horizon in time steps (default: 16 = 4h for 15m candles)')
    parser.add_argument('--num_features', type=int, default=0, 
                        help='Number of features (default: 0, auto-detect)')
    parser.add_argument('--d_model', type=int, default=64, 
                        help='Transformer model dimension (default: 64)')
    parser.add_argument('--nhead', type=int, default=4, 
                        help='Number of transformer attention heads (default: 4)')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of transformer layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--checkpoint', type=str, 
                        help='Path to model checkpoint to resume training from')
    parser.add_argument('--target_type', type=str, default='binary', 
                        choices=['binary', 'regression', 'multi_class'],
                        help='Type of prediction target (default: binary)')
    parser.add_argument('--threshold', type=float, default=0.0, 
                        help='Threshold for binary classification (default: 0.0)')
    
    return parser.parse_args()

def prepare_data(
    data_handler: DataHandler,
    sequence_length: int,
    target_horizon: int,
    target_type: str,
    threshold: float,
    batch_size: int,
    validation_split: float,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare training and validation data loaders.
    
    Args:
        data_handler: DataHandler instance
        sequence_length: Length of input sequences
        target_horizon: Prediction horizon for target
        target_type: Type of prediction target ('binary', 'regression', or 'multi_class')
        threshold: Threshold for binary classification
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        start_date: Start date for training data
        end_date: End date for training data
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        num_features: Number of features in the data
    """
    # Load and preprocess data
    df = data_handler.load_data(start_date, end_date)
    
    if df is None or len(df) == 0:
        logger.error("No data available for the specified date range")
        sys.exit(1)
    
    logger.info(f"Loaded {len(df)} candles from {df['open_time'].min()} to {df['open_time'].max()}")
    
    # Calculate features
    df = calculate_features(df)
    logger.info(f"Calculated features. Data shape: {df.shape}")
    
    # Prepare sequences and targets
    sequences, targets = data_handler.prepare_sequences(
        df, 
        sequence_length=sequence_length,
        target_horizon=target_horizon,
        target_type=target_type,
        threshold=threshold
    )
    
    logger.info(f"Prepared {len(sequences)} sequences with {sequences[0].shape[1]} features")
    
    # Convert to PyTorch tensors
    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(targets), dtype=torch.float32)
    
    # Get the number of features
    num_features = X.shape[2]
    
    # Split into training and validation
    dataset_size = len(X)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    # Shuffle indices
    indices = np.random.permutation(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets
    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    val_dataset = TensorDataset(X[val_indices], y[val_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    logger.info(f"Created data loaders. Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")
    
    return train_loader, val_loader, num_features

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    target_type: str
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on (CPU or GPU)
        target_type: Type of prediction target
        
    Returns:
        metrics: Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in tqdm(train_loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        outputs = model(X)
        
        # Calculate loss
        loss = criterion(outputs, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * X.size(0)
        
        if target_type == 'binary':
            # For binary classification
            predicted = (outputs > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
        elif target_type == 'multi_class':
            # For multi-class classification
            _, predicted = torch.max(outputs, 1)
            _, y_classes = torch.max(y, 1)
            correct += (predicted == y_classes).sum().item()
            total += y.size(0)
        
    # Calculate average metrics
    avg_loss = total_loss / len(train_loader.dataset)
    metrics = {'loss': avg_loss}
    
    if target_type in ['binary', 'multi_class']:
        accuracy = 100 * correct / total
        metrics['accuracy'] = accuracy
    
    return metrics

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_type: str
) -> Dict[str, float]:
    """
    Validate the model on the validation set.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (CPU or GPU)
        target_type: Type of prediction target
        
    Returns:
        metrics: Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Validation", leave=False):
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            outputs = model(X)
            
            # Calculate loss
            loss = criterion(outputs, y)
            
            # Track metrics
            total_loss += loss.item() * X.size(0)
            
            if target_type == 'binary':
                # For binary classification
                predicted = (outputs > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
            elif target_type == 'multi_class':
                # For multi-class classification
                _, predicted = torch.max(outputs, 1)
                _, y_classes = torch.max(y, 1)
                correct += (predicted == y_classes).sum().item()
                total += y.size(0)
    
    # Calculate average metrics
    avg_loss = total_loss / len(val_loader.dataset)
    metrics = {'loss': avg_loss}
    
    if target_type in ['binary', 'multi_class']:
        accuracy = 100 * correct / total
        metrics['accuracy'] = accuracy
    
    return metrics

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Path
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path
) -> Tuple[nn.Module, torch.optim.Optimizer, int, Dict[str, float]]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        checkpoint_path: Path to checkpoint
        
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
        epoch: Last completed epoch
        metrics: Dictionary of metrics
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    
    return model, optimizer, epoch, metrics

def main():
    """Main function to train the model."""
    args = parse_args()
    
    # Set directories
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parents[1] / "data" / "historical" / args.symbol
    
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = Path(__file__).parents[1] / "models" / args.symbol
    
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(__file__).parents[1] / "logs" / args.symbol
    
    # Create directories if they don't exist
    model_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    # Create data handler
    data_handler = DataHandler(
        data_dir=data_dir,
        symbol=args.symbol,
        interval=args.interval
    )
    
    # Prepare data
    train_loader, val_loader, num_features = prepare_data(
        data_handler=data_handler,
        sequence_length=args.sequence_length,
        target_horizon=args.target_horizon,
        target_type=args.target_type,
        threshold=args.threshold,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        start_date=start_date,
        end_date=end_date
    )
    
    # Use detected number of features if not specified
    if args.num_features == 0:
        args.num_features = num_features
    
    # Determine output size based on target type
    if args.target_type == 'binary':
        output_size = 1
    elif args.target_type == 'regression':
        output_size = 1
    elif args.target_type == 'multi_class':
        output_size = 3  # Assuming 3 classes: up, down, neutral
    
    # Create model
    model = TransformerModel(
        input_dim=args.num_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        output_dim=output_size,
        max_seq_length=args.sequence_length
    ).to(device)
    
    # Define loss function based on target type
    if args.target_type == 'binary':
        criterion = nn.BCEWithLogitsLoss()
    elif args.target_type == 'regression':
        criterion = nn.MSELoss()
    elif args.target_type == 'multi_class':
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Load checkpoint if specified
    start_epoch = 0
    training_history = {'train': [], 'val': []}
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            model, optimizer, start_epoch, metrics = load_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=checkpoint_path
            )
            start_epoch += 1  # Start from the next epoch
    
    # Train the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.symbol}_{args.interval}_{args.target_type}_{timestamp}"
    
    best_val_loss = float('inf')
    best_model_path = model_dir / f"{model_name}_best.pt"
    
    logger.info(f"Starting training from epoch {start_epoch+1}/{args.epochs}")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train and validate
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            target_type=args.target_type
        )
        
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            target_type=args.target_type
        )
        
        # Log metrics
        train_metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        val_metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        
        logger.info(f"Train: {train_metrics_str}")
        logger.info(f"Validation: {val_metrics_str}")
        
        # Save history for plotting
        training_history['train'].append(train_metrics)
        training_history['val'].append(val_metrics)
        
        # Save checkpoint
        checkpoint_path = model_dir / f"{model_name}_epoch_{epoch+1}.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={'train': train_metrics, 'val': val_metrics},
            save_path=checkpoint_path
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'train': train_metrics, 'val': val_metrics},
                save_path=best_model_path
            )
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    # Plot training history
    plot_path = log_dir / f"{model_name}_training_history.png"
    plot_training_metrics(training_history, plot_path)
    logger.info(f"Training history plot saved to {plot_path}")
    
    # Save final model
    final_model_path = model_dir / f"{model_name}_final.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs - 1,
        metrics={'train': train_metrics, 'val': val_metrics},
        save_path=final_model_path
    )
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    logger.info(f"Best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")
    
    # Save model configuration
    config = vars(args)
    config['num_features'] = args.num_features  # In case it was auto-detected
    config['timestamp'] = timestamp
    config['best_val_loss'] = best_val_loss
    
    config_path = model_dir / f"{model_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Model configuration saved to {config_path}")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    logger = logging.getLogger("train")
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)