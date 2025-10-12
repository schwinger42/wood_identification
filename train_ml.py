"""
Training script for ML models
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wood_identification.data import (
    RamanPreprocessor, load_csv_data, load_numpy_data, 
    save_processed_data, extract_features
)
from wood_identification.models import WoodClassifier
from wood_identification.utils import (
    calculate_metrics, get_confusion_matrix, get_classification_report,
    plot_confusion_matrix, plot_feature_importance
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main(args):
    print("=" * 50)
    print("Wood Identification - ML Model Training")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
    if args.data_path.endswith('.csv'):
        spectra, labels, wavenumbers = load_csv_data(args.data_path, args.label_column)
    else:
        data = load_numpy_data(args.data_path)
        spectra = data['spectra']
        labels = data['labels']
        wavenumbers = data.get('wavenumbers', np.arange(spectra.shape[1]))
    
    print(f"Loaded {len(spectra)} spectra with {spectra.shape[1]} features")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Encode labels if they are strings
    le = LabelEncoder()
    if labels.dtype == object or labels.dtype.kind == 'U':
        labels_encoded = le.fit_transform(labels)
        class_names = le.classes_
    else:
        labels_encoded = labels
        class_names = [f"Class {i}" for i in np.unique(labels)]
    
    print(f"Classes: {class_names}")
    
    # Preprocess data
    if args.preprocess:
        print("\nPreprocessing spectra...")
        preprocessor = RamanPreprocessor(normalize_method='standard')
        
        processed_spectra = []
        for spectrum in spectra:
            processed = preprocessor.preprocess_spectrum(
                spectrum,
                baseline_correction=True,
                smoothing=True,
                normalization=True
            )
            processed_spectra.append(processed)
        
        spectra = np.array(processed_spectra)
        print("Preprocessing complete")
    
    # Extract features if requested
    if args.extract_features:
        print("\nExtracting features...")
        features_list = []
        for spectrum in spectra:
            features = extract_features(spectrum, wavenumbers)
            features_list.append(list(features.values()))
        
        spectra = np.array(features_list)
        print(f"Extracted {spectra.shape[1]} features")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        spectra, labels_encoded, test_size=0.3, random_state=args.random_state,
        stratify=labels_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.random_state,
        stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model
    print(f"\nTraining {args.model_type} model...")
    
    model_params = {}
    if args.model_type == 'random_forest':
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'random_state': args.random_state
        }
    elif args.model_type == 'svm':
        model_params = {
            'C': args.C,
            'kernel': args.kernel,
            'random_state': args.random_state
        }
    
    classifier = WoodClassifier(model_type=args.model_type, **model_params)
    classifier.fit(X_train, y_train)
    
    print("Training complete")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_val_pred = classifier.predict(X_val)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = classifier.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(get_classification_report(y_test, y_test_pred, target_names=class_names))
    
    # Confusion matrix
    cm = get_confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    if args.save_model:
        print(f"\nSaving model to {args.save_model}...")
        classifier.save(args.save_model)
    
    # Plot confusion matrix
    if args.plot:
        plot_confusion_matrix(cm, class_names, 
                            save_path='results/confusion_matrix_ml.png')
        
        # Plot feature importance for tree-based models
        if args.model_type in ['random_forest', 'gradient_boosting']:
            try:
                importance = classifier.get_feature_importance()
                plot_feature_importance(importance, 
                                      save_path='results/feature_importance_ml.png')
            except AttributeError:
                pass
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model for wood identification")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data file (CSV or NPZ)')
    parser.add_argument('--label_column', type=str, default='species',
                       help='Name of label column (for CSV)')
    
    # Preprocessing arguments
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply preprocessing to spectra')
    parser.add_argument('--extract_features', action='store_true',
                       help='Extract statistical features')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'gradient_boosting', 'mlp'],
                       help='Type of ML model')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of estimators (for tree-based models)')
    parser.add_argument('--max_depth', type=int, default=None,
                       help='Max depth (for tree-based models)')
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter (for SVM)')
    parser.add_argument('--kernel', type=str, default='rbf',
                       help='Kernel type (for SVM)')
    
    # Training arguments
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    # Output arguments
    parser.add_argument('--save_model', type=str, default='models/saved/ml_model.pkl',
                       help='Path to save trained model')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    # Create output directories
    Path('models/saved').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)
    
    main(args)
