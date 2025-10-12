# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-12

### Added

#### Core Package
- Initial release of wood_identification package
- Complete package structure with proper module organization
- `wood_identification.data` module for data processing
- `wood_identification.models` module for ML/DL models
- `wood_identification.utils` module for utilities

#### Data Processing
- `RamanPreprocessor` class for spectrum preprocessing
  - Baseline correction (ALS and polynomial methods)
  - Savitzky-Golay smoothing
  - Normalization (standard and min-max scaling)
  - Spectrum interpolation
- Cosmic ray removal functionality
- Feature extraction from spectra
- Data loading utilities for CSV and NumPy formats
- PyTorch Dataset and DataLoader creation

#### Machine Learning Models
- `WoodClassifier` wrapper for traditional ML models
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - Gradient Boosting Classifier
  - Multi-layer Perceptron (MLP)
- `EnsembleClassifier` for combining multiple models
- Model saving and loading functionality
- Feature importance extraction

#### Deep Learning Models
- `CNN1D`: 1D Convolutional Neural Network
- `ResNet1D`: 1D ResNet with residual connections
- `SimpleMLP`: Deep Multi-layer Perceptron
- Configurable architecture parameters
- Dropout regularization support

#### Visualization
- Spectrum plotting functions
- Multi-spectrum comparison plots
- Preprocessing effect visualization
- Confusion matrix heatmaps
- Training history plots
- Feature importance plots
- PCA visualization (2D and 3D)
- ROC curve plots

#### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion matrix generation
- Classification reports
- ROC-AUC calculation for multi-class
- Per-class accuracy
- Top-k accuracy
- `MetricsTracker` for training history

#### Scripts and Tools
- `train_ml.py`: Training script for ML models
- `train_dl.py`: Training script for DL models
- `predict.py`: Prediction script for both ML and DL models
- `generate_synthetic_data.py`: Synthetic data generation
- `example_workflow.py`: Complete pipeline demonstration
- `validate_structure.py`: Package structure validation

#### Documentation
- Comprehensive README with installation and usage instructions
- QUICKSTART guide for rapid onboarding
- CONTRIBUTING guide for contributors
- Examples directory with documentation
- Configuration file template (config.yaml)
- MIT License
- This CHANGELOG

#### Project Infrastructure
- Requirements file with all dependencies
- Setup script for package installation
- .gitignore for Python projects
- Directory structure with .gitkeep files
- Proper package initialization files

### Technical Details

#### Dependencies
- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- SciPy >= 1.7.0
- PyTorch >= 1.10.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Additional utilities (tqdm, h5py, joblib, pyyaml, tensorboard)

#### Features
- Support for both traditional ML and deep learning approaches
- Flexible preprocessing pipeline
- Multiple model architectures
- Comprehensive evaluation tools
- Easy-to-use Python API
- Command-line interface for training and prediction
- Synthetic data generation for testing
- Modular and extensible design

### Known Limitations
- Requires manual installation of dependencies
- No built-in hyperparameter tuning
- Limited to Raman spectroscopy data format
- No GUI interface
- Requires separate installation of CUDA for GPU support

### Future Plans
- Add comprehensive unit tests
- Implement hyperparameter optimization
- Add more preprocessing methods
- Support for additional data formats
- Web interface for predictions
- Pre-trained models for common wood species
- Data augmentation techniques
- Model interpretability tools

## [Unreleased]

### Planned
- Unit tests for all modules
- CI/CD pipeline
- Docker container
- Model zoo with pre-trained models
- Interactive web dashboard
- Support for other spectroscopy types
- Transfer learning examples
- Hyperparameter tuning utilities

---

For more details on any release, see the corresponding GitHub release notes.
