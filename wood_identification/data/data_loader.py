"""
Data loading utilities for Raman spectroscopy data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


class RamanDataset(Dataset):
    """
    PyTorch Dataset for Raman spectroscopy data
    """
    
    def __init__(self, spectra, labels, transform=None):
        """
        Initialize dataset
        
        Args:
            spectra: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
            transform: Optional transform to apply to spectra
        """
        self.spectra = torch.FloatTensor(spectra)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        label = self.labels[idx]
        
        if self.transform:
            spectrum = self.transform(spectrum)
        
        return spectrum, label


def load_csv_data(csv_path: str, 
                  label_column: str = 'species',
                  wavenumber_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Raman spectroscopy data from CSV file
    
    Args:
        csv_path: Path to CSV file
        label_column: Name of the column containing labels
        wavenumber_columns: Optional list of column names to use as wavenumbers
    
    Returns:
        Tuple of (spectra, labels, wavenumbers)
    """
    df = pd.read_csv(csv_path)
    
    # Extract labels
    labels = df[label_column].values
    
    # Extract spectra (all columns except label)
    spectrum_columns = [col for col in df.columns if col != label_column]
    
    if wavenumber_columns is not None:
        spectrum_columns = wavenumber_columns
    
    spectra = df[spectrum_columns].values
    
    # Try to extract wavenumbers from column names
    try:
        wavenumbers = np.array([float(col) for col in spectrum_columns])
    except (ValueError, TypeError):
        wavenumbers = np.arange(len(spectrum_columns))
    
    return spectra, labels, wavenumbers


def load_numpy_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Raman spectroscopy data from numpy file
    
    Args:
        data_path: Path to numpy file (.npy or .npz)
    
    Returns:
        Tuple of (spectra, labels)
    """
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        spectra = data['spectra']
        labels = data['labels']
    else:
        data = np.load(data_path, allow_pickle=True).item()
        spectra = data['spectra']
        labels = data['labels']
    
    return spectra, labels


def create_dataloaders(spectra: np.ndarray,
                       labels: np.ndarray,
                       batch_size: int = 32,
                       train_split: float = 0.7,
                       val_split: float = 0.15,
                       shuffle: bool = True,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        spectra: numpy array of spectra
        labels: numpy array of labels
        batch_size: Batch size for dataloaders
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        shuffle: Whether to shuffle data
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split
    
    # Convert labels to numeric if they are strings
    if labels.dtype == object or labels.dtype.kind == 'U':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(labels)
    
    # First split: separate test set
    test_split = 1.0 - train_split - val_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        spectra, labels, test_size=test_split, random_state=random_state, 
        stratify=labels if len(np.unique(labels)) > 1 else None
    )
    
    # Second split: separate train and validation
    val_size = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state,
        stratify=y_temp if len(np.unique(y_temp)) > 1 else None
    )
    
    # Create datasets
    train_dataset = RamanDataset(X_train, y_train)
    val_dataset = RamanDataset(X_val, y_val)
    test_dataset = RamanDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False)
    
    return train_loader, val_loader, test_loader


def save_processed_data(spectra: np.ndarray,
                        labels: np.ndarray,
                        save_path: str,
                        wavenumbers: Optional[np.ndarray] = None):
    """
    Save preprocessed data to file
    
    Args:
        spectra: numpy array of spectra
        labels: numpy array of labels
        save_path: Path to save file
        wavenumbers: Optional wavenumber array
    """
    save_dict = {
        'spectra': spectra,
        'labels': labels
    }
    
    if wavenumbers is not None:
        save_dict['wavenumbers'] = wavenumbers
    
    np.savez(save_path, **save_dict)


def load_processed_data(load_path: str) -> dict:
    """
    Load preprocessed data from file
    
    Args:
        load_path: Path to load file
    
    Returns:
        Dictionary containing spectra, labels, and optionally wavenumbers
    """
    data = np.load(load_path)
    result = {
        'spectra': data['spectra'],
        'labels': data['labels']
    }
    
    if 'wavenumbers' in data:
        result['wavenumbers'] = data['wavenumbers']
    
    return result
