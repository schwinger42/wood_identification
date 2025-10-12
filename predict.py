"""
Prediction script for wood identification
"""

import numpy as np
import argparse
from pathlib import Path
import sys
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wood_identification.data import RamanPreprocessor
from wood_identification.models import WoodClassifier, CNN1D, ResNet1D, SimpleMLP
from sklearn.preprocessing import LabelEncoder
import joblib


def load_ml_model(model_path):
    """Load ML model"""
    return joblib.load(model_path)


def load_dl_model(model_path, model_type, input_size, num_classes):
    """Load DL model"""
    if model_type == 'cnn':
        model = CNN1D(input_size, num_classes)
    elif model_type == 'resnet':
        model = ResNet1D(input_size, num_classes)
    elif model_type == 'mlp':
        model = SimpleMLP(input_size, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model


def predict_ml(model, spectrum):
    """Make prediction with ML model"""
    prediction = model.predict([spectrum])[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba([spectrum])[0]
    else:
        probabilities = None
    
    return prediction, probabilities


def predict_dl(model, spectrum, device='cpu'):
    """Make prediction with DL model"""
    spectrum_tensor = torch.FloatTensor(spectrum).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(spectrum_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        prediction = np.argmax(probabilities)
    
    return prediction, probabilities


def main(args):
    print("=" * 50)
    print("Wood Identification - Prediction")
    print("=" * 50)
    
    # Load spectrum
    print("\nLoading spectrum...")
    if args.spectrum_path.endswith('.npy'):
        spectrum = np.load(args.spectrum_path)
    elif args.spectrum_path.endswith('.txt'):
        spectrum = np.loadtxt(args.spectrum_path)
    else:
        raise ValueError("Spectrum file must be .npy or .txt")
    
    print(f"Spectrum shape: {spectrum.shape}")
    
    # Preprocess if needed
    if args.preprocess:
        print("\nPreprocessing spectrum...")
        preprocessor = RamanPreprocessor(normalize_method='standard')
        spectrum = preprocessor.preprocess_spectrum(
            spectrum,
            baseline_correction=True,
            smoothing=True,
            normalization=True
        )
        print("Preprocessing complete")
    
    # Load model and make prediction
    print(f"\nLoading model from {args.model_path}...")
    
    if args.model_framework == 'ml':
        # Load ML model
        clf = WoodClassifier(model_type='random_forest')
        clf.load(args.model_path)
        prediction, probabilities = predict_ml(clf, spectrum)
    
    elif args.model_framework == 'dl':
        # Load DL model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_dl_model(args.model_path, args.model_type, 
                            len(spectrum), args.num_classes)
        model = model.to(device)
        prediction, probabilities = predict_dl(model, spectrum, device)
    
    else:
        raise ValueError(f"Unknown framework: {args.model_framework}")
    
    # Load class names if provided
    if args.class_names:
        class_names = args.class_names.split(',')
        predicted_class = class_names[prediction]
    else:
        predicted_class = f"Class {prediction}"
    
    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"\nPredicted class: {predicted_class}")
    
    if probabilities is not None:
        print("\nClass probabilities:")
        for i, prob in enumerate(probabilities):
            if args.class_names:
                class_name = class_names[i]
            else:
                class_name = f"Class {i}"
            print(f"  {class_name}: {prob:.4f}")
        
        # Show top-3 predictions
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        print("\nTop-3 predictions:")
        for rank, idx in enumerate(top_3_indices, 1):
            if args.class_names:
                class_name = class_names[idx]
            else:
                class_name = f"Class {idx}"
            print(f"  {rank}. {class_name}: {probabilities[idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict wood species from Raman spectrum")
    
    # Input arguments
    parser.add_argument('--spectrum_path', type=str, required=True,
                       help='Path to spectrum file (.npy or .txt)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    
    # Model arguments
    parser.add_argument('--model_framework', type=str, default='ml',
                       choices=['ml', 'dl'],
                       help='Model framework (ml or dl)')
    parser.add_argument('--model_type', type=str, default='cnn',
                       help='DL model type (cnn, resnet, or mlp)')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes (for DL models)')
    
    # Preprocessing arguments
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply preprocessing to spectrum')
    
    # Output arguments
    parser.add_argument('--class_names', type=str, default=None,
                       help='Comma-separated list of class names')
    
    args = parser.parse_args()
    
    main(args)
