"""
Utils module initialization
"""

from .visualization import (
    plot_spectrum,
    plot_multiple_spectra,
    plot_preprocessing_comparison,
    plot_confusion_matrix,
    plot_training_history,
    plot_feature_importance,
    plot_pca_visualization,
    plot_roc_curves
)

from .metrics import (
    calculate_metrics,
    get_confusion_matrix,
    get_classification_report,
    calculate_roc_auc,
    calculate_per_class_accuracy,
    top_k_accuracy,
    MetricsTracker
)

__all__ = [
    'plot_spectrum',
    'plot_multiple_spectra',
    'plot_preprocessing_comparison',
    'plot_confusion_matrix',
    'plot_training_history',
    'plot_feature_importance',
    'plot_pca_visualization',
    'plot_roc_curves',
    'calculate_metrics',
    'get_confusion_matrix',
    'get_classification_report',
    'calculate_roc_auc',
    'calculate_per_class_accuracy',
    'top_k_accuracy',
    'MetricsTracker',
]
