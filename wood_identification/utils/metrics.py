"""
Evaluation metrics for wood identification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Tuple


def calculate_metrics(y_true, y_pred, average='weighted') -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    return metrics


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred, target_names=None) -> str:
    """
    Generate classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names
    
    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def calculate_roc_auc(y_true, y_pred_proba, num_classes) -> Dict:
    """
    Calculate ROC-AUC for multi-class classification
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        num_classes: Number of classes
    
    Returns:
        Dictionary containing fpr, tpr, and auc for each class
    """
    from sklearn.preprocessing import label_binarize
    
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = roc_auc_score(y_true_bin, y_pred_proba, average="micro")
    
    return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}


def calculate_per_class_accuracy(y_true, y_pred, num_classes) -> Dict[int, float]:
    """
    Calculate accuracy for each class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Dictionary mapping class to accuracy
    """
    per_class_acc = {}
    
    for i in range(num_classes):
        mask = y_true == i
        if mask.sum() > 0:
            per_class_acc[i] = accuracy_score(y_true[mask], y_pred[mask])
        else:
            per_class_acc[i] = 0.0
    
    return per_class_acc


def top_k_accuracy(y_true, y_pred_proba, k=3) -> float:
    """
    Calculate top-k accuracy
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy
    """
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = 0
    
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


class MetricsTracker:
    """
    Track metrics during training
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }
    
    def update(self, train_loss=None, train_acc=None, val_loss=None, val_acc=None):
        """
        Update metrics
        
        Args:
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if train_acc is not None:
            self.history['train_accuracy'].append(train_acc)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_acc is not None:
            self.history['val_accuracy'].append(val_acc)
    
    def get_history(self) -> Dict:
        """
        Get training history
        
        Returns:
            Dictionary of training history
        """
        return self.history
    
    def get_best_epoch(self, metric='val_accuracy', mode='max') -> Tuple[int, float]:
        """
        Get the best epoch based on a metric
        
        Args:
            metric: Metric to use for comparison
            mode: 'max' or 'min'
        
        Returns:
            Tuple of (best_epoch, best_value)
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return 0, 0.0
        
        values = self.history[metric]
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return best_idx + 1, values[best_idx]
    
    def save(self, filepath):
        """
        Save metrics history to file
        
        Args:
            filepath: Path to save file
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath):
        """
        Load metrics history from file
        
        Args:
            filepath: Path to load file
        """
        import json
        with open(filepath, 'r') as f:
            self.history = json.load(f)
