# Wood Identification Examples

This directory contains example notebooks and scripts demonstrating how to use the wood_identification package.

## Notebooks

1. **basic_usage.ipynb** - Basic usage of preprocessing and model training
2. **visualization_examples.ipynb** - Visualization techniques for Raman spectra
3. **model_comparison.ipynb** - Comparing different ML/DL models
4. **feature_analysis.ipynb** - Feature extraction and analysis

## Running Examples

To run these examples, make sure you have Jupyter installed:

```bash
pip install jupyter
```

Then launch Jupyter:

```bash
jupyter notebook
```

## Generate Synthetic Data

Before running the examples, generate synthetic data:

```bash
python ../generate_synthetic_data.py
```

This will create a synthetic dataset in `data/processed/synthetic_data.npz` that can be used for testing.
