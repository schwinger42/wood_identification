"""
Training script for Deep Learning models
"""

import numpy as np
import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wood_identification.data import (
    RamanPreprocessor, load_csv_data, load_numpy_data,
    create_dataloaders
)
from wood_identification.models import CNN1D, ResNet1D, SimpleMLP
from wood_identification.utils import (
    calculate_metrics, get_confusion_matrix, get_classification_report,
    plot_confusion_matrix, plot_training_history, MetricsTracker
)
from sklearn.preprocessing import LabelEncoder


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for spectra, labels in tqdm(train_loader, desc="Training"):
        spectra, labels = spectra.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(spectra)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for spectra, labels in tqdm(val_loader, desc="Validation"):
            spectra, labels = spectra.to(device), labels.to(device)
            
            outputs = model(spectra)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def test_model(model, test_loader, device):
    """Test the model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for spectra, labels in test_loader:
            spectra = spectra.to(device)
            outputs = model(spectra)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds)


def main(args):
    print("=" * 50)
    print("Wood Identification - Deep Learning Training")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
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
    
    # Encode labels if they are strings
    le = LabelEncoder()
    if labels.dtype == object or labels.dtype.kind == 'U':
        labels_encoded = le.fit_transform(labels)
        class_names = le.classes_
    else:
        labels_encoded = labels
        class_names = [f"Class {i}" for i in np.unique(labels)]
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
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
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        spectra, labels_encoded,
        batch_size=args.batch_size,
        train_split=0.7,
        val_split=0.15,
        random_state=args.random_state
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    input_size = spectra.shape[1]
    
    if args.model_type == 'cnn':
        model = CNN1D(input_size, num_classes, dropout=args.dropout)
    elif args.model_type == 'resnet':
        model = ResNet1D(input_size, num_classes, dropout=args.dropout)
    elif args.model_type == 'mlp':
        model = SimpleMLP(input_size, num_classes, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=5)
    
    # Training loop
    print("\nStarting training...")
    metrics_tracker = MetricsTracker()
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        metrics_tracker.update(train_loss, train_acc, val_loss, val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_model:
                torch.save(model.state_dict(), args.save_model)
                print(f"Saved best model with val_acc: {val_acc:.4f}")
    
    # Load best model for testing
    if args.save_model:
        model.load_state_dict(torch.load(args.save_model))
    
    # Test model
    print("\nTesting model...")
    y_test, y_pred = test_model(model, test_loader, device)
    
    test_metrics = calculate_metrics(y_test, y_pred)
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(get_classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = get_confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot results
    if args.plot:
        plot_training_history(metrics_tracker.get_history(),
                            save_path='results/training_history_dl.png')
        plot_confusion_matrix(cm, class_names,
                            save_path='results/confusion_matrix_dl.png')
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DL model for wood identification")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data file (CSV or NPZ)')
    parser.add_argument('--label_column', type=str, default='species',
                       help='Name of label column (for CSV)')
    
    # Preprocessing arguments
    parser.add_argument('--preprocess', action='store_true',
                       help='Apply preprocessing to spectra')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='cnn',
                       choices=['cnn', 'resnet', 'mlp'],
                       help='Type of DL model')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    # Output arguments
    parser.add_argument('--save_model', type=str, default='models/saved/dl_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    
    # Create output directories
    Path('models/saved').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)
    
    main(args)
