"""
Deep Learning Models for Gesture Recognition
Implements LSTM, GRU, and Transformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger


class LSTMGestureModel(nn.Module):
    """
    LSTM-based model for gesture recognition from keypoint sequences
    """
    
    def __init__(
        self,
        input_size: int = 63,  # 21 keypoints * 3 coordinates
        hidden_size: int = 256,
        num_layers: int = 3,
        num_classes: int = 16,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM model
        
        Args:
            input_size: Size of input features per timestep
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            num_classes: Number of gesture classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMGestureModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        logger.info(f"LSTMGestureModel initialized: input={input_size}, hidden={hidden_size}, classes={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * directions)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size * directions)
        
        # Fully connected layers
        output = self.fc(attended)  # (batch, num_classes)
        
        return output


class GRUGestureModel(nn.Module):
    """
    GRU-based model for gesture recognition
    """
    
    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_classes: int = 16,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(GRUGestureModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        logger.info(f"GRUGestureModel initialized: input={input_size}, hidden={hidden_size}, classes={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        # Use last output
        output = self.fc(gru_out[:, -1, :])
        return output


class TransformerGestureModel(nn.Module):
    """
    Transformer-based model for gesture recognition
    """
    
    def __init__(
        self,
        input_size: int = 63,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        num_classes: int = 16,
        dropout: float = 0.1,
        max_seq_len: int = 200
    ):
        """
        Initialize Transformer model
        
        Args:
            input_size: Size of input features
            d_model: Dimension of model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            num_classes: Number of gesture classes
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super(TransformerGestureModel, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        logger.info(f"TransformerGestureModel initialized: d_model={d_model}, heads={nhead}, classes={num_classes}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Project input to d_model dimension
        x = self.input_projection(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNLSTMGestureModel(nn.Module):
    """
    Hybrid CNN-LSTM model for gesture recognition
    """
    
    def __init__(
        self,
        input_size: int = 63,
        cnn_filters: list = [64, 128, 256],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        num_classes: int = 16,
        dropout: float = 0.3
    ):
        super(CNNLSTMGestureModel, self).__init__()
        
        # 1D CNN for feature extraction
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in cnn_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )
        
        logger.info(f"CNNLSTMGestureModel initialized: filters={cnn_filters}, lstm_hidden={lstm_hidden}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for CNN: (batch, 1, seq_len * input_size)
        batch_size, seq_len, input_size = x.shape
        x = x.view(batch_size, 1, -1)
        
        # CNN feature extraction
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape for LSTM: (batch, new_seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        output = self.fc(lstm_out[:, -1, :])
        
        return output


def create_model(
    model_type: str,
    input_size: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'transformer', 'cnn_lstm')
        input_size: Input feature size
        num_classes: Number of classes
        **kwargs: Additional model-specific parameters
        
    Returns:
        Initialized model
    """
    models = {
        'lstm': LSTMGestureModel,
        'gru': GRUGestureModel,
        'transformer': TransformerGestureModel,
        'cnn_lstm': CNNLSTMGestureModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    model = models[model_type](input_size=input_size, num_classes=num_classes, **kwargs)
    
    logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


if __name__ == "__main__":
    # Test models
    batch_size = 8
    seq_len = 200
    input_size = 63
    num_classes = 16
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Test LSTM
    lstm_model = create_model('lstm', input_size, num_classes)
    lstm_out = lstm_model(x)
    print(f"LSTM output shape: {lstm_out.shape}")
    
    # Test Transformer
    transformer_model = create_model('transformer', input_size, num_classes)
    transformer_out = transformer_model(x)
    print(f"Transformer output shape: {transformer_out.shape}")

# Made with Bob
