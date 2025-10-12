import pickle
import numpy as np
import argparse
import os
from scipy.signal import savgol_filter
from pybaselines.morphological import mor
import xgboost as xgb
import shap

# --- NEW: SCIENTIFIC CONTEXT DICTIONARY ---
# A curated list of the most important bands from your research
KNOWN_BANDS = {
    'Lignin Aromatic Skeleton': (1590, 1615),      # Widened range
    'Cellulose Glycosidic Link': (1090, 1105),      # Widened range
    'Cellulose CH Bending': (1370, 1390),
    'Hardwood (Syringyl) Lignin': (1325, 1335),
    'Softwood (Guaiacyl) Lignin': (1275, 1295),
    'Aliphatic C-H (Extractives)': (2850, 2950),
    'Coniferaldehyde (Softwood)': (1640, 1660)
}

# --- 1. DEFINE ASSET PATHS ---
MODEL_DIR = 'models'
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'wood_classifier_ensemble.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'wood_classifier_xgb.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
WAVENUMBERS_PATH = os.path.join(MODEL_DIR, 'wavenumbers.pkl')

# --- 2. LOAD ASSETS ---
try:
    with open(ENSEMBLE_MODEL_PATH, 'rb') as f:
        MODEL = pickle.load(f)
    with open(XGB_MODEL_PATH, 'rb') as f:
        XGB_MODEL = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        ENCODER = pickle.load(f)
    with open(WAVENUMBERS_PATH, 'rb') as f:
        WAVENUMBERS = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not find a required model file. {e}")
    # ... (rest of the error message)
    exit()


# --- 3. PREPROCESSING FUNCTION (Identical to notebook) ---
def preprocess_spectrum(wavenumbers, intensities):
    smoothed = savgol_filter(intensities, window_length=15, polyorder=2)
    baseline, _ = mor(smoothed)
    baseline_corrected = smoothed - baseline
    min_val = np.min(baseline_corrected)
    shifted = baseline_corrected - min_val
    norm = np.linalg.norm(shifted)
    return shifted / norm if norm > 0 else shifted


# --- 4. PREDICTION & EXPLANATION FUNCTION (ENHANCED) ---
def predict_and_explain(spectrum_filepath):
    if not os.path.exists(spectrum_filepath):
        print(f"Error: File not found at '{spectrum_filepath}'")
        return

    print("-" * 80)
    print(f"Analyzing file: {os.path.basename(spectrum_filepath)}")

    # ... (Loading and Preprocessing code is the same)
    try:
        data = np.loadtxt(spectrum_filepath, delimiter=';')
        intensities = data[:, 1]
        processed_spectrum = preprocess_spectrum(WAVENUMBERS, intensities)
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return
    processed_spectrum = processed_spectrum.reshape(1, -1)

    # ... (Prediction and Confidence code is the same)
    prediction_encoded = MODEL.predict(processed_spectrum)[0]
    prediction_species = ENCODER.inverse_transform([prediction_encoded])[0]
    probabilities = MODEL.predict_proba(processed_spectrum)[0]
    confidence = probabilities[prediction_encoded] * 100

    print("\n--- PREDICTION ---")
    print(f"Predicted Species:  {prediction_species}")
    print(f"Confidence:         {confidence:.2f}%")

    # --- ENHANCED EXPLANATION ---
    print("\n--- EXPLANATION (from XGBoost model) ---")
    explainer = shap.TreeExplainer(XGB_MODEL)
    shap_values = explainer(processed_spectrum)
    class_index = np.where(ENCODER.classes_ == prediction_species)[0][0]
    shap_for_class = shap_values.values[0, :, class_index]
    feature_importance = sorted(zip(WAVENUMBERS, shap_for_class), key=lambda x: x[1], reverse=True)

    print(f"Top 5 spectral regions supporting the '{prediction_species}' prediction:")
    for wavenumber, shap_val in feature_importance[:5]:
        # Check against our dictionary of known bands
        scientific_basis = "N/A"
        for name, (low, high) in KNOWN_BANDS.items():
            if low <= wavenumber <= high:
                scientific_basis = name
                break
        print(f"  - Wavenumber ~{wavenumber:<7.1f} cm⁻¹ (Impact: {shap_val:+.4f})  [Basis: {scientific_basis}]")

    print("-" * 80)


# --- 5. COMMAND-LINE INTERFACE (Identical) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a wood species from a Raman spectrum .txt file.")
    parser.add_argument("filepath", type=str, help="Path to the semicolon-delimited .txt spectrum file.")
    args = parser.parse_args()
    predict_and_explain(args.filepath)