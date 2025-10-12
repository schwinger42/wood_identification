"""
Example script to generate synthetic Raman spectroscopy data for testing
"""

import numpy as np
from pathlib import Path


def generate_synthetic_spectrum(wavenumbers, peaks, noise_level=0.02):
    """
    Generate a synthetic Raman spectrum with specified peaks
    
    Args:
        wavenumbers: Array of wavenumber values
        peaks: List of tuples (position, height, width)
        noise_level: Amount of random noise to add
    
    Returns:
        Synthetic spectrum
    """
    spectrum = np.zeros_like(wavenumbers, dtype=float)
    
    # Add Gaussian peaks
    for pos, height, width in peaks:
        spectrum += height * np.exp(-((wavenumbers - pos) / width) ** 2)
    
    # Add baseline
    baseline = 0.1 + 0.0001 * wavenumbers
    spectrum += baseline
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(wavenumbers))
    spectrum += noise
    
    return spectrum


def generate_wood_dataset(n_samples_per_class=100, n_classes=5, output_path='data/processed/synthetic_data.npz'):
    """
    Generate synthetic wood Raman spectroscopy dataset
    
    Args:
        n_samples_per_class: Number of samples per wood species
        n_classes: Number of wood species
        output_path: Path to save the dataset
    """
    # Define wavenumber range (typical for Raman)
    wavenumbers = np.linspace(400, 3000, 1000)
    
    # Define characteristic peaks for different wood species
    # These are simplified fictional patterns for demonstration
    species_patterns = {
        0: [(1100, 0.8, 50), (1600, 1.0, 40), (2900, 0.6, 80)],  # Oak-like
        1: [(1050, 0.7, 45), (1580, 0.9, 45), (2850, 0.7, 75)],  # Pine-like
        2: [(1120, 0.9, 55), (1620, 0.8, 35), (2920, 0.5, 85)],  # Maple-like
        3: [(1080, 0.75, 50), (1590, 0.95, 42), (2880, 0.65, 78)],  # Birch-like
        4: [(1110, 0.85, 48), (1610, 0.85, 38), (2910, 0.55, 82)],  # Beech-like
    }
    
    all_spectra = []
    all_labels = []
    
    for species_id in range(min(n_classes, len(species_patterns))):
        print(f"Generating samples for species {species_id}...")
        peaks = species_patterns[species_id]
        
        for _ in range(n_samples_per_class):
            # Add some variation to peaks
            varied_peaks = []
            for pos, height, width in peaks:
                pos_var = pos + np.random.normal(0, 10)
                height_var = height + np.random.normal(0, 0.1)
                width_var = width + np.random.normal(0, 5)
                varied_peaks.append((pos_var, height_var, width_var))
            
            spectrum = generate_synthetic_spectrum(wavenumbers, varied_peaks)
            all_spectra.append(spectrum)
            all_labels.append(species_id)
    
    # Convert to numpy arrays
    spectra = np.array(all_spectra)
    labels = np.array(all_labels)
    
    # Shuffle data
    indices = np.random.permutation(len(labels))
    spectra = spectra[indices]
    labels = labels[indices]
    
    # Save dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, spectra=spectra, labels=labels, wavenumbers=wavenumbers)
    
    print(f"\nDataset generated successfully!")
    print(f"Shape: {spectra.shape}")
    print(f"Number of classes: {n_classes}")
    print(f"Samples per class: {n_samples_per_class}")
    print(f"Saved to: {output_path}")
    
    return spectra, labels, wavenumbers


if __name__ == "__main__":
    # Generate a synthetic dataset
    spectra, labels, wavenumbers = generate_wood_dataset(
        n_samples_per_class=100,
        n_classes=5,
        output_path='data/processed/synthetic_data.npz'
    )
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(labels)}")
    print(f"Spectrum length: {len(wavenumbers)}")
    print(f"Wavenumber range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹")
    print(f"Class distribution:")
    for i in range(len(np.unique(labels))):
        count = np.sum(labels == i)
        print(f"  Class {i}: {count} samples")
