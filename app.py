import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pybaselines.morphological import mor
import xgboost as xgb
import shap

# --- Page Configuration ---
st.set_page_config(page_title="Wood Species Classifier", layout="wide")
st.title("🌳 Raman Wood Species Classifier")
st.write("A machine learning tool to predict Taiwanese wood species from Raman spectra and explain the results.")

# --- 1. LOAD ASSETS (with Caching) ---
# Use st.cache_resource to load models only once
@st.cache_resource
def load_assets():
    MODEL_DIR = 'models'
    try:
        with open(os.path.join(MODEL_DIR, 'wood_classifier_ensemble.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'wood_classifier_xgb.pkl'), 'rb') as f:
            xgb_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'wavenumbers.pkl'), 'rb') as f:
            wavenumbers = pickle.load(f)
        return model, xgb_model, encoder, wavenumbers
    except FileNotFoundError:
        st.error(f"Error: Model asset files not found in the '{MODEL_DIR}/' directory. Please run the training notebook first.")
        return None, None, None, None

MODEL, XGB_MODEL, ENCODER, WAVENUMBERS = load_assets()

# --- KNOWN BANDS DICTIONARY ---
KNOWN_BANDS = {
    'Syringyl  lignin  (S/G ratio)': (360, 380),
    'Glycosidic carbon-oxygen-carbon stretch': (978, 1178),
    'Lignin Aromatic Skeleton': (1570, 1615),
    'Cellulose Glycosidic Link': (1090, 1105),
    'Cellulose CH Bending': (1370, 1390),
    'Hardwood (Syringyl) Lignin': (1325, 1335),
    'Softwood (Guaiacyl) Lignin': (1275, 1295),
    'Aliphatic C-H (Extractives)': (2850, 2950),
    'Coniferaldehyde (Softwood), C=O stretch, C=C stretch': (1640, 1670),
    'Cellulose Crystallinity Band': (1478, 1482),
    'General Polysaccharide Ring Breathing': (995, 1005),
    'Glycosidic β-Link (Cellulose specific)': (894, 896)
}

# --- 2. PREPROCESSING FUNCTION (Identical to notebook) ---
@st.cache_data # Cache the result of this function
def preprocess_spectrum(intensities, wavenumbers):
    smoothed = savgol_filter(intensities, window_length=15, polyorder=2)
    baseline, _ = mor(smoothed)
    baseline_corrected = smoothed - baseline
    min_val = np.min(baseline_corrected)
    shifted = baseline_corrected - min_val
    norm = np.linalg.norm(shifted)
    return shifted / norm if norm > 0 else shifted

# --- 3. UI and APPLICATION LOGIC ---
st.sidebar.header("Upload Spectrum")
uploaded_file = st.sidebar.file_uploader("Choose a .txt Raman spectrum file", type=["txt"])

if uploaded_file is not None and MODEL is not None:
    try:
        # Read the uploaded file
        raw_data = pd.read_csv(uploaded_file, sep=';', header=None, usecols=[0, 1], names=['wavenumber', 'intensity'])
        raw_intensities = raw_data['intensity'].values

        st.header("1. Input Data Visualization")
        processed_intensities = preprocess_spectrum(raw_intensities, WAVENUMBERS)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(WAVENUMBERS, raw_intensities, color='red')
        axes[0].set_title('Raw Spectrum')
        axes[0].set_xlabel('Wavenumber ($cm^{-1}$)')
        axes[0].set_ylabel('Intensity')
        axes[1].plot(WAVENUMBERS, processed_intensities, color='green')
        axes[1].set_title('After Preprocessing')
        axes[1].set_xlabel('Wavenumber ($cm^{-1}$)')
        st.pyplot(fig)

        # --- Prediction ---
        st.header("2. Model Prediction")
        processed_spectrum_reshaped = processed_intensities.reshape(1, -1)
        prediction_encoded = MODEL.predict(processed_spectrum_reshaped)[0]
        prediction_species = ENCODER.inverse_transform([prediction_encoded])[0]
        probabilities = MODEL.predict_proba(processed_spectrum_reshaped)[0]
        confidence = probabilities[prediction_encoded] * 100

        col1, col2 = st.columns(2)
        col1.metric("Predicted Species", prediction_species)
        col2.metric("Confidence", f"{confidence:.2f}%")

        # --- Explanation ---
        st.header("3. Prediction Explanation (via SHAP)")
        st.write("The following spectral regions had the largest positive impact on the prediction for this species, according to the XGBoost model.")

        explainer = shap.TreeExplainer(XGB_MODEL)
        shap_values = explainer(processed_spectrum_reshaped)
        class_index = np.where(ENCODER.classes_ == prediction_species)[0][0]
        shap_for_class = shap_values.values[0, :, class_index]
        feature_importance = sorted(zip(WAVENUMBERS, shap_for_class), key=lambda x: x[1], reverse=True)

        explanation_data = []
        for wavenumber, shap_val in feature_importance[:5]:
            scientific_basis = "N/A"
            for name, (low, high) in KNOWN_BANDS.items():
                if low <= wavenumber <= high:
                    scientific_basis = name
                    break
            explanation_data.append({
                "Wavenumber (cm⁻¹)": f"{wavenumber:.1f}",
                "Impact (SHAP Value)": f"{shap_val:+.4f}",
                "Potential Scientific Basis": scientific_basis
            })

        st.dataframe(explanation_data, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a file in the sidebar to begin analysis.")


