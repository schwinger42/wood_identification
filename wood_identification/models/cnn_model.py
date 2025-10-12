"""
Deep Learning models for wood identification using Raman spectroscopy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for Raman spectroscopy classification
    """
    
    def __init__(self, input_size, num_classes, num_filters=[64, 128, 256], 
                 kernel_size=5, dropout=0.5):
        """
        Initialize CNN model
        
        Args:
            input_size: Size of input spectrum
            num_classes: Number of wood species to classify
            num_filters: List of filter sizes for each conv layer
            kernel_size: Size of convolutional kernels
            dropout: Dropout probability
        """
        super(CNN1D, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # First convolutional layer
        self.conv_layers.append(nn.Conv1d(1, num_filters[0], kernel_size, padding=kernel_size//2))
        self.bn_layers.append(nn.BatchNorm1d(num_filters[0]))
        
        # Additional convolutional layers
        for i in range(len(num_filters) - 1):
            self.conv_layers.append(nn.Conv1d(num_filters[i], num_filters[i+1], 
                                              kernel_size, padding=kernel_size//2))
            self.bn_layers.append(nn.BatchNorm1d(num_filters[i+1]))
        
        # Calculate size after convolutions
        size_after_conv = input_size // (2 ** len(num_filters))
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters[-1] * size_after_conv, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output logits
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Convolutional layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for 1D CNN
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for Raman spectroscopy classification
    """
    
    def __init__(self, input_size, num_classes, num_blocks=[2, 2, 2, 2], 
                 num_filters=[64, 128, 256, 512], dropout=0.5):
        """
        Initialize ResNet model
        
        Args:
            input_size: Size of input spectrum
            num_classes: Number of wood species to classify
            num_blocks: List of number of residual blocks per stage
            num_filters: List of filter sizes for each stage
            dropout: Dropout probability
        """
        super(ResNet1D, self).__init__()
        
        self.in_channels = num_filters[0]
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, num_filters[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(num_filters[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(num_filters[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(num_filters[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(num_filters[3], num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(num_filters[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output logits
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron for Raman spectroscopy classification
    """
    
    def __init__(self, input_size, num_classes, hidden_sizes=[512, 256, 128], 
                 dropout=0.5):
        """
        Initialize MLP model
        
        Args:
            input_size: Size of input spectrum
            num_classes: Number of wood species to classify
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(SimpleMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output = nn.Linear(prev_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output logits
        """
        for layer, bn in zip(self.layers, self.bn_layers):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.output(x)
        
        return x
