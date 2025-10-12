# Quick Start Guide

Get started with wood identification using Raman spectroscopy in just a few steps!

## Installation

```bash
# Clone the repository
git clone https://github.com/schwinger42/wood_identification.git
cd wood_identification

# Install dependencies
pip install -r requirements.txt

# Or install the package
pip install -e .
```

## Quick Example

### 1. Generate Synthetic Data (for testing)

```bash
python generate_synthetic_data.py
```

This creates a synthetic dataset with 5 wood species (500 samples total) in `data/processed/synthetic_data.npz`.

### 2. Train a Machine Learning Model

```bash
python train_ml.py \
    --data_path data/processed/synthetic_data.npz \
    --model_type random_forest \
    --preprocess \
    --save_model models/saved/rf_model.pkl \
    --plot
```

### 3. Train a Deep Learning Model

```bash
python train_dl.py \
    --data_path data/processed/synthetic_data.npz \
    --model_type cnn \
    --preprocess \
    --epochs 30 \
    --batch_size 32 \
    --save_model models/saved/cnn_model.pth \
    --plot
```

### 4. Make Predictions

First, save a test spectrum:
```bash
python -c "import numpy as np; data=np.load('data/processed/synthetic_data.npz'); np.save('test_spectrum.npy', data['spectra'][0])"
```

Then predict:
```bash
python predict.py \
    --spectrum_path test_spectrum.npy \
    --model_path models/saved/rf_model.pkl \
    --model_framework ml \
    --preprocess \
    --class_names "Oak,Pine,Maple,Birch,Beech"
```

## Python API Example

```python
import numpy as np
from wood_identification.data import RamanPreprocessor, load_numpy_data
from wood_identification.models import WoodClassifier
from wood_identification.utils import plot_spectrum
from sklearn.model_selection import train_test_split

# Load data
data = load_numpy_data('data/processed/synthetic_data.npz')
spectra = data['spectra']
labels = data['labels']
wavenumbers = data['wavenumbers']

# Preprocess
preprocessor = RamanPreprocessor(normalize_method='standard')
processed_spectra = np.array([
    preprocessor.preprocess_spectrum(s) for s in spectra
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    processed_spectra, labels, test_size=0.2, random_state=42
)

# Train model
classifier = WoodClassifier(model_type='random_forest', n_estimators=100)
classifier.fit(X_train, y_train)

# Evaluate
accuracy = classifier.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
predictions = classifier.predict(X_test[:5])
probabilities = classifier.predict_proba(X_test[:5])

# Visualize a spectrum
plot_spectrum(wavenumbers, spectra[0], 
             title="Sample Raman Spectrum")
```

## Expected Results

With the synthetic dataset:
- **Random Forest**: ~95-99% accuracy
- **SVM**: ~90-95% accuracy
- **CNN**: ~95-98% accuracy
- **ResNet**: ~96-99% accuracy

## Next Steps

1. **Use your own data**: Replace synthetic data with real Raman spectroscopy data
2. **Tune hyperparameters**: Experiment with different model parameters
3. **Try ensemble methods**: Combine multiple models for better performance
4. **Visualize results**: Use the plotting functions to understand your data
5. **Extract features**: Use feature extraction for better interpretability

## Common Issues

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA errors (for deep learning)
If you don't have a GPU, PyTorch will automatically use CPU. To force CPU:
```python
device = torch.device('cpu')
```

### Memory errors
Reduce batch size or use fewer samples:
```bash
python train_dl.py --batch_size 16
```

## Getting Help

- Check the [main README](README.md) for detailed documentation
- Look at examples in the `examples/` directory
- Open an issue on GitHub for bugs or questions

## Citation

If you use this package in your research, please cite it appropriately. See the main README for citation details.
