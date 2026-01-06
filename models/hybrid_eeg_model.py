"""
Hybrid CNN-RNN-Transformer Model for EEG Lie Detection
Combines CNNs (spatial), RNNs (temporal), and Transformers (attention)
Similar to modern speech recognition architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math

class SpatialCNN(nn.Module):
    """
    Convolutional Neural Network for spatial feature extraction
    Treats EEG channels as spatial dimensions (like topographic maps)
    """

    def __init__(self, n_channels: int, out_channels: int = 64):
        """
        Initialize spatial CNN

        Args:
            n_channels: Number of input EEG channels
            out_channels: Number of output feature channels
        """
        super(SpatialCNN, self).__init__()

        self.spatial_conv1 = nn.Sequential(
            nn.Conv1d(n_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.spatial_conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.out_channels = out_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features

        Args:
            x: Input tensor (batch, n_channels, time)

        Returns:
            Spatial features (batch, out_channels, time)
        """
        x = self.spatial_conv1(x)
        x = self.spatial_conv2(x)
        return x

class TemporalRNN(nn.Module):
    """
    Recurrent Neural Network for temporal sequence modeling
    Uses LSTM/GRU to capture time-varying patterns in EEG signals
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        rnn_type: str = 'LSTM',
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize temporal RNN

        Args:
            input_size: Input feature size (from CNN output)
            hidden_size: Hidden state size
            num_layers: Number of RNN layers
            rnn_type: 'LSTM' or 'GRU'
            bidirectional: Use bidirectional RNN
            dropout: Dropout rate
        """
        super(TemporalRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"RNN type must be 'LSTM' or 'GRU', got {rnn_type}")

        self.output_size = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process temporal sequences

        Args:
            x: Input tensor (batch, time, features)
               Note: RNN expects (batch, seq_len, features)

        Returns:
            output: RNN output (batch, time, hidden_size * directions)
            hidden: Final hidden state
        """

        x = x.transpose(1, 2)

        output, hidden = self.rnn(x)

        return output, hidden

class ChannelAttention(nn.Module):
    """
    Attention mechanism to weight important EEG channels
    Similar to channel attention in CNNs
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        """
        Initialize channel attention

        Args:
            n_channels: Number of channels
            reduction: Reduction ratio for efficiency
        """
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Attention-weighted features (batch, channels, time)
        """

        avg_out = self.avg_pool(x).squeeze(-1)
        max_out = self.max_pool(x).squeeze(-1)

        avg_att = self.fc(avg_out)
        max_att = self.fc(max_out)

        attention = (avg_att + max_att) / 2.0
        attention = attention.unsqueeze(-1)

        return x * attention

class TemporalAttention(nn.Module):
    """
    Attention mechanism to weight important time points
    Uses self-attention (Transformer-style)
    """

    def __init__(self, feature_size: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize temporal attention

        Args:
            feature_size: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(TemporalAttention, self).__init__()

        self.feature_size = feature_size
        self.num_heads = num_heads
        self.head_dim = feature_size // num_heads

        assert feature_size % num_heads == 0, "feature_size must be divisible by num_heads"

        self.query = nn.Linear(feature_size, feature_size)
        self.key = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal self-attention

        Args:
            x: Input tensor (batch, time, features)

        Returns:
            Attention-weighted features (batch, time, features)
        """
        residual = x
        batch_size, seq_len, _ = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.feature_size
        )

        output = self.layer_norm(attn_output + residual)

        return output

class HybridEEGModel(nn.Module):
    """
    Hybrid CNN-RNN-Transformer Model for EEG Lie Detection

    Architecture:
    1. Spatial CNN: Extracts spatial features from EEG channels
    2. Channel Attention: Weights important channels
    3. Temporal RNN: Models temporal sequences (LSTM/GRU)
    4. Temporal Attention: Weights important time points (Transformer-style)
    5. Fusion: Combines all features
    6. Classification: Final prediction
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_timepoints: int = 250,
        n_classes: int = 2,
        cnn_out_channels: int = 64,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_type: str = 'LSTM',
        rnn_bidirectional: bool = True,
        attention_heads: int = 8,
        dropout: float = 0.3
    ):
        """
        Initialize hybrid model

        Args:
            n_channels: Number of EEG channels
            n_timepoints: Number of time points
            n_classes: Number of classes (2 for truth/lie)
            cnn_out_channels: CNN output channels
            rnn_hidden_size: RNN hidden size
            rnn_num_layers: Number of RNN layers
            rnn_type: 'LSTM' or 'GRU'
            rnn_bidirectional: Use bidirectional RNN
            attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(HybridEEGModel, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        self.spatial_cnn = SpatialCNN(n_channels, cnn_out_channels)

        self.channel_attention = ChannelAttention(cnn_out_channels * 2)

        self.temporal_rnn = TemporalRNN(
            input_size=cnn_out_channels * 2,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            bidirectional=rnn_bidirectional,
            dropout=dropout
        )

        rnn_output_size = rnn_hidden_size * 2 if rnn_bidirectional else rnn_hidden_size

        self.temporal_attention = TemporalAttention(
            feature_size=rnn_output_size,
            num_heads=attention_heads,
            dropout=dropout
        )

        fusion_size = (cnn_out_channels * 2) + rnn_output_size

        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, rnn_output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(rnn_output_size)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, rnn_output_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(rnn_output_size // 2, n_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid model

        Args:
            x: Input tensor (batch, n_channels, n_timepoints)

        Returns:
            Classification logits (batch, n_classes)
        """
        batch_size = x.size(0)

        cnn_features = self.spatial_cnn(x)

        cnn_features = self.channel_attention(cnn_features)

        rnn_output, rnn_hidden = self.temporal_rnn(cnn_features)

        attn_features = self.temporal_attention(rnn_output)

        cnn_features_t = cnn_features.transpose(1, 2)

        fused = torch.cat([cnn_features_t, attn_features], dim=-1)

        fused = self.fusion(fused)

        fused = fused.transpose(1, 2)
        pooled = self.global_pool(fused).squeeze(-1)

        output = self.classifier(pooled)

        return output

    def extract_features(self, x: torch.Tensor) -> dict:
        """
        Extract intermediate features for analysis

        Args:
            x: Input tensor (batch, n_channels, n_timepoints)

        Returns:
            Dictionary with intermediate features
        """
        features = {}

        cnn_features = self.spatial_cnn(x)
        features['cnn_features'] = cnn_features

        cnn_attended = self.channel_attention(cnn_features)
        features['channel_attention_features'] = cnn_attended

        rnn_output, rnn_hidden = self.temporal_rnn(cnn_attended)
        features['rnn_output'] = rnn_output
        features['rnn_hidden'] = rnn_hidden

        attn_features = self.temporal_attention(rnn_output)
        features['temporal_attention_features'] = attn_features

        return features

class HybridEEGLieDetector:
    """
    Wrapper class for Hybrid EEG Model with training and inference capabilities
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_timepoints: int = 250,
        device: Optional[torch.device] = None,
        **model_kwargs
    ):
        """
        Initialize hybrid EEG lie detector

        Args:
            n_channels: Number of EEG channels
            n_timepoints: Number of time points per sample
            device: Device for training ('cuda' or 'cpu')
            **model_kwargs: Additional arguments for HybridEEGModel
        """
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = HybridEEGModel(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            **model_kwargs
        ).to(self.device)

        self.is_trained = False

    def prepare_data(self, eeg_data: np.ndarray) -> torch.Tensor:
        """
        Prepare EEG data for model input

        Args:
            eeg_data: EEG data shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)

        Returns:
            Tensor ready for model input
        """
        if eeg_data.ndim == 2:

            eeg_data = eeg_data[np.newaxis, :, :]

        if eeg_data.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {eeg_data.shape[1]}")

        tensor_data = torch.FloatTensor(eeg_data).to(self.device)

        return tensor_data

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Train the hybrid model

        Args:
            X_train: Training data (n_trials, n_channels, n_samples)
            y_train: Training labels (n_trials,)
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        if y_val is not None:
            y_val_encoded = label_encoder.transform(y_val)

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / (len(X_train_tensor) // batch_size + 1)
            train_acc = 100 * train_correct / train_total

            if X_val is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for i in range(0, len(X_val_tensor), batch_size):
                        batch_X = X_val_tensor[i:i+batch_size]
                        batch_y = y_val_tensor[i:i+batch_size]

                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()

                avg_val_loss = val_loss / (len(X_val_tensor) // batch_size + 1)
                val_acc = 100 * val_correct / val_total

                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict().copy()

                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        if X_val is not None and 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

        self.is_trained = True

    def predict(self, eeg_data: np.ndarray) -> dict:
        """
        Predict truth/lie from EEG data

        Args:
            eeg_data: EEG data (n_channels, n_samples) or (n_trials, n_channels, n_samples)

        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train_model() first.")

        self.model.eval()

        tensor_data = self.prepare_data(eeg_data)

        with torch.no_grad():
            outputs = self.model(tensor_data)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        predicted = predicted.cpu().numpy()
        probabilities = probabilities.cpu().numpy()

        label_map = {0: 'truth', 1: 'lie'}

        if len(predicted) == 1:
            return {
                'prediction': label_map[predicted[0]],
                'prediction_int': int(predicted[0]),
                'confidence': float(probabilities[0][predicted[0]]),
                'probabilities': {
                    'truth': float(probabilities[0][0]),
                    'lie': float(probabilities[0][1])
                }
            }
        else:
            return {
                'predictions': [label_map[p] for p in predicted],
                'probabilities': probabilities.tolist()
            }

    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise RuntimeError("No model to save. Train the model first.")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_channels': self.n_channels,
            'n_timepoints': self.n_timepoints,
            'is_trained': self.is_trained
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.n_channels = checkpoint['n_channels']
        self.n_timepoints = checkpoint['n_timepoints']
        self.is_trained = checkpoint['is_trained']
        print(f"Model loaded from {filepath}")
