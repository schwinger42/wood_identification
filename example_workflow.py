"""
Complete workflow example for wood identification

This script demonstrates the entire pipeline:
1. Generate synthetic data
2. Preprocess the data
3. Train multiple models
4. Evaluate and compare models
5. Visualize results
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from wood_identification.data import (
    RamanPreprocessor, 
    load_numpy_data,
    save_processed_data
)
from wood_identification.models import WoodClassifier
from wood_identification.utils import (
    calculate_metrics,
    get_confusion_matrix,
    plot_confusion_matrix,
    plot_spectrum,
    plot_multiple_spectra
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Wood Identification - Complete Workflow Example")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic data...")
    from generate_synthetic_data import generate_wood_dataset
    
    spectra, labels, wavenumbers = generate_wood_dataset(
        n_samples_per_class=100,
        n_classes=5,
        output_path='data/processed/example_data.npz'
    )
    
    class_names = ['Oak', 'Pine', 'Maple', 'Birch', 'Beech']
    
    # Step 2: Visualize raw data
    print("\n[Step 2] Visualizing raw spectra...")
    
    # Plot one spectrum from each class
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(5):
        class_idx = np.where(labels == i)[0][0]
        axes[i].plot(wavenumbers, spectra[class_idx])
        axes[i].set_title(f'{class_names[i]} - Raw Spectrum')
        axes[i].set_xlabel('Wavenumber (cm⁻¹)')
        axes[i].set_ylabel('Intensity (a.u.)')
        axes[i].grid(True, alpha=0.3)
    
    axes[5].axis('off')
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/raw_spectra_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: results/raw_spectra_comparison.png")
    plt.close()
    
    # Step 3: Preprocess data
    print("\n[Step 3] Preprocessing spectra...")
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
    
    processed_spectra = np.array(processed_spectra)
    
    # Visualize preprocessing effect
    sample_idx = 0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(wavenumbers, spectra[sample_idx])
    ax1.set_title('Original Spectrum')
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(wavenumbers, processed_spectra[sample_idx])
    ax2.set_title('Preprocessed Spectrum')
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Intensity (a.u.)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_effect.png', dpi=300, bbox_inches='tight')
    print("  Saved: results/preprocessing_effect.png")
    plt.close()
    
    # Step 4: Split data
    print("\n[Step 4] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        processed_spectra, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Step 5: Train multiple models
    print("\n[Step 5] Training models...")
    
    models = {
        'Random Forest': WoodClassifier('random_forest', n_estimators=100),
        'SVM': WoodClassifier('svm', C=10.0),
        'Gradient Boosting': WoodClassifier('gradient_boosting', n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        
        # Validate
        val_pred = model.predict(X_val)
        val_metrics = calculate_metrics(y_val, val_pred)
        
        # Test
        test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, test_pred)
        
        results[name] = {
            'model': model,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'test_pred': test_pred
        }
        
        print(f"    Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"    Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Step 6: Compare models
    print("\n[Step 6] Comparing models...")
    
    comparison_data = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for name, result in results.items():
        comparison_data['Model'].append(name)
        comparison_data['Accuracy'].append(result['test_metrics']['accuracy'])
        comparison_data['Precision'].append(result['test_metrics']['precision'])
        comparison_data['Recall'].append(result['test_metrics']['recall'])
        comparison_data['F1-Score'].append(result['test_metrics']['f1_score'])
    
    # Print comparison table
    print("\n  Model Performance Comparison:")
    print("  " + "-" * 70)
    print(f"  {'Model':<20} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("  " + "-" * 70)
    
    for i in range(len(comparison_data['Model'])):
        print(f"  {comparison_data['Model'][i]:<20} "
              f"{comparison_data['Accuracy'][i]:>12.4f} "
              f"{comparison_data['Precision'][i]:>12.4f} "
              f"{comparison_data['Recall'][i]:>12.4f} "
              f"{comparison_data['F1-Score'][i]:>12.4f}")
    print("  " + "-" * 70)
    
    # Step 7: Visualize best model results
    print("\n[Step 7] Visualizing results...")
    
    best_model_name = max(results.keys(), 
                         key=lambda k: results[k]['test_metrics']['accuracy'])
    best_result = results[best_model_name]
    
    print(f"  Best model: {best_model_name}")
    
    # Confusion matrix
    cm = get_confusion_matrix(y_test, best_result['test_pred'])
    plot_confusion_matrix(cm, class_names, 
                         title=f'Confusion Matrix - {best_model_name}',
                         save_path='results/confusion_matrix_example.png')
    print("  Saved: results/confusion_matrix_example.png")
    
    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_data['Model']))
    width = 0.2
    
    ax.bar(x - 1.5*width, comparison_data['Accuracy'], width, label='Accuracy')
    ax.bar(x - 0.5*width, comparison_data['Precision'], width, label='Precision')
    ax.bar(x + 0.5*width, comparison_data['Recall'], width, label='Recall')
    ax.bar(x + 1.5*width, comparison_data['F1-Score'], width, label='F1-Score')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_data['Model'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.8, 1.0])
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: results/model_comparison.png")
    plt.close()
    
    # Step 8: Save best model
    print("\n[Step 8] Saving best model...")
    Path('models/saved').mkdir(parents=True, exist_ok=True)
    best_result['model'].save('models/saved/best_model_example.pkl')
    print("  Saved: models/saved/best_model_example.pkl")
    
    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/processed/example_data.npz")
    print("  - results/raw_spectra_comparison.png")
    print("  - results/preprocessing_effect.png")
    print("  - results/confusion_matrix_example.png")
    print("  - results/model_comparison.png")
    print("  - models/saved/best_model_example.pkl")


if __name__ == "__main__":
    main()
