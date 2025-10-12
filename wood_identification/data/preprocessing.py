"""
Data preprocessing utilities for Raman spectroscopy data
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class RamanPreprocessor:
    """
    Preprocessor for Raman spectroscopy data
    """
    
    def __init__(self, normalize_method='standard'):
        """
        Initialize preprocessor
        
        Args:
            normalize_method: Method for normalization ('standard', 'minmax', or None)
        """
        self.normalize_method = normalize_method
        self.scaler = None
        
        if normalize_method == 'standard':
            self.scaler = StandardScaler()
        elif normalize_method == 'minmax':
            self.scaler = MinMaxScaler()
    
    def baseline_correction(self, spectrum, method='als', **kwargs):
        """
        Apply baseline correction to spectrum
        
        Args:
            spectrum: 1D numpy array of intensity values
            method: Baseline correction method ('als' or 'polynomial')
            **kwargs: Additional parameters for the method
        
        Returns:
            Baseline-corrected spectrum
        """
        if method == 'als':
            # Asymmetric Least Squares Smoothing
            lam = kwargs.get('lam', 1e5)
            p = kwargs.get('p', 0.01)
            baseline = self._als_baseline(spectrum, lam, p)
            return spectrum - baseline
        
        elif method == 'polynomial':
            # Polynomial baseline
            degree = kwargs.get('degree', 3)
            x = np.arange(len(spectrum))
            coeffs = np.polyfit(x, spectrum, degree)
            baseline = np.polyval(coeffs, x)
            return spectrum - baseline
        
        return spectrum
    
    def _als_baseline(self, y, lam, p, niter=10):
        """
        Asymmetric Least Squares baseline correction
        """
        L = len(y)
        D = np.diff(np.eye(L), 2)
        w = np.ones(L)
        
        for _ in range(niter):
            W = np.diag(w)
            Z = W + lam * D.T @ D
            z = np.linalg.solve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        
        return z
    
    def smooth_spectrum(self, spectrum, window_length=11, polyorder=3):
        """
        Apply Savitzky-Golay filter for smoothing
        
        Args:
            spectrum: 1D numpy array of intensity values
            window_length: Length of the filter window
            polyorder: Order of the polynomial
        
        Returns:
            Smoothed spectrum
        """
        if window_length % 2 == 0:
            window_length += 1
        
        return signal.savgol_filter(spectrum, window_length, polyorder)
    
    def normalize_spectrum(self, spectrum):
        """
        Normalize spectrum using configured method
        
        Args:
            spectrum: 1D or 2D numpy array of intensity values
        
        Returns:
            Normalized spectrum
        """
        if self.scaler is None:
            return spectrum
        
        is_1d = spectrum.ndim == 1
        
        if is_1d:
            spectrum = spectrum.reshape(-1, 1)
        
        if not hasattr(self.scaler, 'mean_'):
            # First time - fit and transform
            normalized = self.scaler.fit_transform(spectrum.T).T
        else:
            # Use existing scaler
            normalized = self.scaler.transform(spectrum.T).T
        
        if is_1d:
            normalized = normalized.flatten()
        
        return normalized
    
    def interpolate_spectrum(self, wavenumbers, intensities, target_wavenumbers):
        """
        Interpolate spectrum to target wavenumbers
        
        Args:
            wavenumbers: Original wavenumber array
            intensities: Original intensity array
            target_wavenumbers: Target wavenumber array
        
        Returns:
            Interpolated intensity array
        """
        f = interp1d(wavenumbers, intensities, kind='cubic', 
                     fill_value='extrapolate')
        return f(target_wavenumbers)
    
    def preprocess_spectrum(self, spectrum, wavenumbers=None, 
                           target_wavenumbers=None,
                           baseline_correction=True,
                           smoothing=True,
                           normalization=True):
        """
        Complete preprocessing pipeline
        
        Args:
            spectrum: 1D numpy array of intensity values
            wavenumbers: Optional wavenumber array for interpolation
            target_wavenumbers: Optional target wavenumber array
            baseline_correction: Whether to apply baseline correction
            smoothing: Whether to apply smoothing
            normalization: Whether to apply normalization
        
        Returns:
            Preprocessed spectrum
        """
        processed = spectrum.copy()
        
        # Interpolation if needed
        if wavenumbers is not None and target_wavenumbers is not None:
            processed = self.interpolate_spectrum(wavenumbers, processed, 
                                                  target_wavenumbers)
        
        # Baseline correction
        if baseline_correction:
            processed = self.baseline_correction(processed)
        
        # Smoothing
        if smoothing:
            processed = self.smooth_spectrum(processed)
        
        # Normalization
        if normalization:
            processed = self.normalize_spectrum(processed)
        
        return processed


def remove_cosmic_rays(spectrum, threshold=3.0):
    """
    Remove cosmic ray spikes from spectrum
    
    Args:
        spectrum: 1D numpy array of intensity values
        threshold: Z-score threshold for spike detection
    
    Returns:
        Spectrum with cosmic rays removed
    """
    # Calculate z-scores of differences
    diff = np.diff(spectrum)
    z_scores = np.abs((diff - np.mean(diff)) / np.std(diff))
    
    # Find spikes
    spike_indices = np.where(z_scores > threshold)[0]
    
    # Replace spikes with interpolated values
    cleaned = spectrum.copy()
    for idx in spike_indices:
        if idx > 0 and idx < len(spectrum) - 1:
            cleaned[idx] = (spectrum[idx-1] + spectrum[idx+1]) / 2
    
    return cleaned


def extract_features(spectrum, wavenumbers=None):
    """
    Extract statistical features from spectrum
    
    Args:
        spectrum: 1D numpy array of intensity values
        wavenumbers: Optional wavenumber array
    
    Returns:
        Dictionary of features
    """
    features = {
        'mean': np.mean(spectrum),
        'std': np.std(spectrum),
        'max': np.max(spectrum),
        'min': np.min(spectrum),
        'median': np.median(spectrum),
        'skewness': float(np.mean(((spectrum - np.mean(spectrum)) / np.std(spectrum)) ** 3)),
        'kurtosis': float(np.mean(((spectrum - np.mean(spectrum)) / np.std(spectrum)) ** 4)),
    }
    
    # Peak features
    peaks, properties = signal.find_peaks(spectrum, prominence=0.1)
    features['num_peaks'] = len(peaks)
    
    if len(peaks) > 0:
        features['mean_peak_intensity'] = np.mean(spectrum[peaks])
        features['max_peak_intensity'] = np.max(spectrum[peaks])
        if wavenumbers is not None:
            features['mean_peak_position'] = np.mean(wavenumbers[peaks])
    else:
        features['mean_peak_intensity'] = 0
        features['max_peak_intensity'] = 0
        features['mean_peak_position'] = 0
    
    return features
