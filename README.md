# Wood Identification using Raman Spectroscopy with ML/DL

A comprehensive Python package for wood species identification using Raman spectroscopy data with Machine Learning (ML) and Deep Learning (DL) approaches.

## Features

- **Data Processing**: Complete preprocessing pipeline for Raman spectroscopy data
  - Baseline correction (ALS and polynomial methods)
  - Savitzky-Golay smoothing
  - Normalization (standard and min-max scaling)
  - Cosmic ray removal
  - Feature extraction

- **Machine Learning Models**:
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - Multi-layer Perceptron (MLP)
  - Ensemble methods

- **Deep Learning Models**:
  - 1D Convolutional Neural Network (CNN)
  - 1D ResNet
  - Deep MLP

- **Visualization Tools**:
  - Spectrum plotting
  - Preprocessing comparisons
  - Confusion matrices
  - Training history
  - Feature importance
  - PCA visualization
  - ROC curves

- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Per-class accuracy
  - Top-k accuracy
  - ROC-AUC
  - Confusion matrix
  - Classification report

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.10+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Setup

1. Clone the repository:
```bash
git clone https://github.com/schwinger42/wood_identification.git
cd wood_identification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
wood_identification/
├── wood_identification/          # Main package
│   ├── data/                     # Data processing modules
│   │   ├── preprocessing.py      # Preprocessing utilities
│   │   └── data_loader.py        # Data loading utilities
│   ├── models/                   # Model implementations
│   │   ├── cnn_model.py         # Deep learning models
│   │   └── ml_model.py          # Traditional ML models
│   └── utils/                    # Utility modules
│       ├── visualization.py      # Plotting functions
│       └── metrics.py           # Evaluation metrics
├── data/                         # Data directory
│   ├── raw/                     # Raw data
│   └── processed/               # Preprocessed data
├── models/                       # Saved models directory
│   └── saved/
├── results/                      # Results and plots
├── train_ml.py                  # ML training script
├── train_dl.py                  # DL training script
├── predict.py                   # Prediction script
├── config.yaml                  # Configuration file
├── requirements.txt             # Package dependencies
└── README.md                    # This file
```

## Usage

### 1. Data Preparation

Prepare your Raman spectroscopy data in one of the following formats:

- **CSV format**: Each row is a sample, columns are wavenumbers or features, with a label column
- **NumPy format** (.npz): Dictionary with 'spectra' and 'labels' keys

Example CSV structure:
```csv
species,400.0,401.0,402.0,...
Oak,0.23,0.45,0.67,...
Pine,0.12,0.34,0.56,...
```

### 2. Training Machine Learning Models

```bash
python train_ml.py \
    --data_path data/processed/raman_data.npz \
    --model_type random_forest \
    --preprocess \
    --n_estimators 200 \
    --save_model models/saved/rf_model.pkl \
    --plot
```

Available ML models:
- `random_forest`: Random Forest Classifier
- `svm`: Support Vector Machine
- `gradient_boosting`: Gradient Boosting Classifier
- `mlp`: Multi-layer Perceptron

### 3. Training Deep Learning Models

```bash
python train_dl.py \
    --data_path data/processed/raman_data.npz \
    --model_type cnn \
    --preprocess \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --save_model models/saved/cnn_model.pth \
    --plot
```

Available DL models:
- `cnn`: 1D Convolutional Neural Network
- `resnet`: 1D ResNet
- `mlp`: Deep Multi-layer Perceptron

### 4. Making Predictions

```bash
python predict.py \
    --spectrum_path data/test_spectrum.npy \
    --model_path models/saved/rf_model.pkl \
    --model_framework ml \
    --preprocess \
    --class_names "Oak,Pine,Maple,Birch"
```

### 5. Using the Package in Python

```python
from wood_identification.data import RamanPreprocessor, load_csv_data
from wood_identification.models import WoodClassifier, CNN1D
from wood_identification.utils import plot_spectrum, calculate_metrics

# Load and preprocess data
spectra, labels, wavenumbers = load_csv_data('data.csv', 'species')

preprocessor = RamanPreprocessor(normalize_method='standard')
processed_spectra = [preprocessor.preprocess_spectrum(s) for s in spectra]

# Train ML model
classifier = WoodClassifier(model_type='random_forest', n_estimators=100)
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)
metrics = calculate_metrics(y_test, predictions)

# Visualize spectrum
plot_spectrum(wavenumbers, spectra[0], title="Sample Raman Spectrum")
```

## Data Preprocessing

The package includes comprehensive preprocessing capabilities:

```python
from wood_identification.data import RamanPreprocessor

preprocessor = RamanPreprocessor(normalize_method='standard')

# Complete preprocessing pipeline
processed = preprocessor.preprocess_spectrum(
    spectrum,
    baseline_correction=True,  # Remove baseline
    smoothing=True,            # Apply Savitzky-Golay filter
    normalization=True         # Normalize intensities
)

# Individual preprocessing steps
baseline_corrected = preprocessor.baseline_correction(spectrum, method='als')
smoothed = preprocessor.smooth_spectrum(spectrum, window_length=11)
normalized = preprocessor.normalize_spectrum(spectrum)
```

## Model Architecture

### CNN Model
- Multiple 1D convolutional layers with batch normalization
- Max pooling for feature extraction
- Fully connected layers for classification
- Dropout for regularization

### ResNet Model
- Residual blocks with skip connections
- Deep architecture for complex pattern learning
- Global average pooling
- Suitable for large datasets

### ML Models
- Tree-based: Random Forest, Gradient Boosting
- Kernel-based: SVM with RBF kernel
- Neural network: Scikit-learn MLP

## Configuration

Edit `config.yaml` to customize:
- Data paths and splits
- Preprocessing parameters
- Model hyperparameters
- Training settings
- Output directories

## Evaluation

The package provides comprehensive evaluation tools:

```python
from wood_identification.utils import (
    calculate_metrics,
    get_confusion_matrix,
    plot_confusion_matrix,
    calculate_roc_auc
)

# Calculate metrics
metrics = calculate_metrics(y_true, y_pred)

# Confusion matrix
cm = get_confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, class_names)

# ROC curves
roc_data = calculate_roc_auc(y_true, y_pred_proba, num_classes)
```

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Data loading and preprocessing
- Model training and evaluation
- Visualization techniques
- Feature extraction and analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{wood_identification,
  title = {Wood Identification using Raman Spectroscopy with ML/DL},
  author = {schwinger42},
  year = {2025},
  url = {https://github.com/schwinger42/wood_identification}
}
```

## Contact

For questions and feedback, please open an issue on GitHub.

## Acknowledgments

This project implements methods for Raman spectroscopy analysis and applies modern machine learning and deep learning techniques for wood species identification.
