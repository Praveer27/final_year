"""
Training Module for Gesture Recognition Models
Includes training loop, validation, and model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from tqdm import tqdm
from loguru import logger
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class GestureTrainer:
    """
    Trainer class for gesture recognition models
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: Path,
        num_classes: int,
        class_names: list,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        use_mixed_precision: bool = True
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            num_classes: Number of classes
            class_names: List of class names
            scheduler: Learning rate scheduler
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_classes = num_classes
        self.class_names = class_names
        self.scheduler = scheduler
        self.use_mixed_precision = use_mixed_precision
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Mixed precision training: {use_mixed_precision}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float, Dict]:
        """
        Validate the model
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy, metrics dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for data, labels in pbar:
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        return avg_loss, accuracy, metrics
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 15,
        save_best_only: bool = True
    ) -> Dict:
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, metrics = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                f"LR: {current_lr:.6f}"
            )
            logger.info(
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}"
            )
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                patience_counter = 0
                
                if save_best_only:
                    self.save_checkpoint(epoch, is_best=True, metrics=metrics)
                    logger.info(f"New best model saved! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                
                if not save_best_only:
                    self.save_checkpoint(epoch, is_best=False, metrics=metrics)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save final training history
        self.save_history()
        self.plot_training_history()
        
        logger.info(f"Training completed! Best Val Acc: {self.best_val_acc:.4f}")
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict] = None
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'metrics': metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint.get('epoch', 0)
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {plot_path}")


if __name__ == "__main__":
    # Test trainer
    from src.models.gesture_model import create_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('lstm', input_size=63, num_classes=16)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Made with Bob
