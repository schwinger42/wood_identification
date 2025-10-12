"""
Visualization utilities for Raman spectroscopy data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def plot_spectrum(wavenumbers, intensities, title="Raman Spectrum", 
                  xlabel="Wavenumber (cm⁻¹)", ylabel="Intensity (a.u.)",
                  figsize=(12, 6), save_path=None):
    """
    Plot a single Raman spectrum
    
    Args:
        wavenumbers: Array of wavenumber values
        intensities: Array of intensity values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    plt.plot(wavenumbers, intensities, linewidth=1.5)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_multiple_spectra(wavenumbers, spectra_list, labels=None, 
                         title="Raman Spectra Comparison",
                         xlabel="Wavenumber (cm⁻¹)", ylabel="Intensity (a.u.)",
                         figsize=(12, 6), save_path=None):
    """
    Plot multiple Raman spectra for comparison
    
    Args:
        wavenumbers: Array of wavenumber values
        spectra_list: List of intensity arrays
        labels: List of labels for each spectrum
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    for i, spectrum in enumerate(spectra_list):
        label = labels[i] if labels else f"Spectrum {i+1}"
        plt.plot(wavenumbers, spectrum, linewidth=1.5, label=label, alpha=0.8)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_preprocessing_comparison(wavenumbers, original, processed, 
                                  title="Preprocessing Effect",
                                  figsize=(12, 8), save_path=None):
    """
    Plot original and preprocessed spectra for comparison
    
    Args:
        wavenumbers: Array of wavenumber values
        original: Original spectrum
        processed: Preprocessed spectrum
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    ax1.plot(wavenumbers, original, linewidth=1.5, color='blue')
    ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax1.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax1.set_title("Original Spectrum", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(wavenumbers, processed, linewidth=1.5, color='red')
    ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax2.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax2.set_title("Preprocessed Spectrum", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix",
                         figsize=(10, 8), normalize=False, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
        save_path: Optional path to save the figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history, metrics=['loss', 'accuracy'],
                         figsize=(15, 5), save_path=None):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            axes[i].plot(history[train_key], label=f'Train {metric}', linewidth=2)
        
        if val_key in history:
            axes[i].plot(history[val_key], label=f'Validation {metric}', linewidth=2)
        
        axes[i].set_xlabel('Epoch', fontsize=12)
        axes[i].set_ylabel(metric.capitalize(), fontsize=12)
        axes[i].set_title(f'{metric.capitalize()} over Epochs', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(importance, feature_names=None, top_n=20,
                           title="Feature Importance", figsize=(10, 8),
                           save_path=None):
    """
    Plot feature importance
    
    Args:
        importance: Array of feature importance values
        feature_names: Optional list of feature names
        top_n: Number of top features to display
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Sort features by importance
    indices = np.argsort(importance)[::-1][:top_n]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importance[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pca_visualization(X_pca, y, class_names=None,
                          title="PCA Visualization", figsize=(10, 8),
                          save_path=None):
    """
    Plot PCA visualization of data
    
    Args:
        X_pca: PCA-transformed data (2D or 3D)
        y: Labels
        class_names: Optional list of class names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    unique_labels = np.unique(y)
    
    if X_pca.shape[1] == 2:
        # 2D plot
        plt.figure(figsize=figsize)
        for label in unique_labels:
            mask = y == label
            label_name = class_names[label] if class_names else f"Class {label}"
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=label_name, alpha=0.6, s=50)
        
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    elif X_pca.shape[1] >= 3:
        # 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        for label in unique_labels:
            mask = y == label
            label_name = class_names[label] if class_names else f"Class {label}"
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                      label=label_name, alpha=0.6, s=50)
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_zlabel('PC3', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, class_names=None,
                   title="ROC Curves", figsize=(10, 8), save_path=None):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        fpr_dict: Dictionary of false positive rates for each class
        tpr_dict: Dictionary of true positive rates for each class
        roc_auc_dict: Dictionary of AUC scores for each class
        class_names: Optional list of class names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    for i in fpr_dict.keys():
        label_name = class_names[i] if class_names else f"Class {i}"
        plt.plot(fpr_dict[i], tpr_dict[i], 
                label=f'{label_name} (AUC = {roc_auc_dict[i]:.2f})',
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
