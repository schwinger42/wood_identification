# Junie AI Agent Guidelines: Raman Wood Species Classification Project (REVISED)

## Project Context & Objectives

### Research Overview
You are assisting a data scientist and Prof. Tsao (National Chung Hsing University) on a cutting-edge research project applying machine learning to Raman spectroscopy for Taiwanese wood species identification. This is a high-stakes project targeting:

1. **Taiwan National Science and Technology Council (NSC) funding**
2. **Nature publication** (high-impact journal)
3. **Real-world conservation impact** (combat illegal logging)

### Dataset Characteristics
- **9 Taiwanese wood species**: 
  - **Hardwoods (5)**: Acacia Confusa, Acacia Mangium, Machilus Kusanoi, Swietenia Mahagoni, Swietenia Marophylla
  - **Softwoods (4)**: Chamaecyparis Formosensis, Chamaecyparis Obtusa var. Formosana, Chamaecyparis Obyusa, Fokienia Hodginsii
- **Hierarchical structure**: Species → Trees → Facets (Cross/Radial/Tangential) → Measurement Points (10 per facet)
- **Sample size challenge**: Only 11 individual trees total (2 species have multiple trees, others have single trees)
- **Total spectra**: ~270 measurements (30 per tree for most species)

### Critical Challenge
**Small sample size with hierarchical structure creates high overfitting risk.** This is the primary technical challenge that must be addressed in ALL code implementations.

## Core Programming Principles

### 1. Hierarchical Data Structure (CRITICAL)
```python
# ALWAYS group by individual tree, NEVER by measurement points
# This prevents data leakage - the biggest threat to publication

# ✅ CORRECT approach
groups = metadata['species'] + '_Tree' + metadata['tree_id'].astype(str)
logo = LeaveOneGroupOut()
cv_scores = cross_val_score(model, X, y, cv=logo, groups=groups)

# ❌ WRONG approach (will cause rejection)
# Using measurement points as independent samples
cv_scores = cross_val_score(model, X, y, cv=5)  # DON'T DO THIS
```

### 2. Statistical Rigor Requirements
All ML implementations must include:

#### A. Proper Cross-Validation
```python
# Use Leave-One-Tree-Out (LOTO) cross-validation
from sklearn.model_selection import LeaveOneGroupOut

def hierarchical_cross_validation(X, y, groups, model):
    """Proper CV preventing data leakage"""
    logo = LeaveOneGroupOut()
    cv_scores = cross_val_score(model, X, y, cv=logo, groups=groups)
    
    return {
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'individual_scores': cv_scores,
        'n_folds': len(cv_scores)  # Should equal number of unique trees
    }
```

#### B. Bootstrap Confidence Intervals
```python
def bootstrap_confidence_intervals(predictions, true_labels, groups, n_bootstrap=1000):
    """Calculate CI respecting hierarchical structure"""
    bootstrap_scores = []
    unique_groups = np.unique(groups)
    
    for i in range(n_bootstrap):
        # Sample trees (not individual measurements)
        sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        
        # Get all measurements from sampled trees
        bootstrap_indices = []
        for group in sampled_groups:
            group_indices = np.where(groups == group)[0]
            bootstrap_indices.extend(group_indices)
        
        if len(bootstrap_indices) > 0:
            bootstrap_accuracy = accuracy_score(
                true_labels[bootstrap_indices], 
                predictions[bootstrap_indices]
            )
            bootstrap_scores.append(bootstrap_accuracy)
    
    return {
        'mean_accuracy': np.mean(bootstrap_scores),
        'ci_lower': np.percentile(bootstrap_scores, 2.5),
        'ci_upper': np.percentile(bootstrap_scores, 97.5)
    }
```

## 3. Authoritative Molecular Band Assignments (UPDATED)

### A. Taxonomic Discrimination Bands (PRIMARY)

#### Hardwood/Angiosperm Markers (Syringyl-Rich Lignin)
```python
# Based on Saariaho et al., Applied Spectroscopy (2003) - GOLD STANDARD
HARDWOOD_SYRINGYL_BANDS = {
    # Primary S-unit diagnostic bands (UV-Raman optimized)
    'syringyl_primary': (1506, 1514),          # S-unit aromatic vibration
    'syringyl_secondary': (1330, 1333),        # S-unit ring breathing
    'syringyl_tertiary': (962, 981),           # S-unit deformation
    
    # Supporting hardwood indicators
    'hardwood_lignin_co': (1265, 1275),        # Enhanced in S-rich lignin
    'hardwood_aromatic_ref': (1595, 1605),     # Aromatic reference band
}

# Your hardwood species: Acacia Confusa, Acacia Mangium, Machilus Kusanoi, 
#                       Swietenia Mahagoni, Swietenia Marophylla
```

#### Softwood/Gymnosperm Markers (Guaiacyl-Rich Lignin)
```python
# Based on Saariaho et al., Applied Spectroscopy (2003) + Agarwal, Planta (2006)
SOFTWOOD_GUAIACYL_BANDS = {
    # Primary G-unit diagnostic bands (UV-Raman optimized)
    'guaiacyl_primary': (1279, 1289),          # G-unit aromatic vibration
    'guaiacyl_secondary': (1155, 1187),        # G-unit ring breathing
    'guaiacyl_tertiary': (704, 791),           # G-unit deformation
    
    # Coniferous-specific markers
    'coniferaldehyde': (1645, 1655),           # Coniferyl alcohol/aldehyde
    'softwood_resin': (1265, 1275),            # Rosin/resin markers
    'gymnosperm_aromatic_ref': (1595, 1605),   # Aromatic reference band
}

# Your softwood species: Chamaecyparis Formosensis, Chamaecyparis Obtusa var. Formosana,
#                       Chamaecyparis Obyusa, Fokienia Hodginsii
```

### B. Universal Wood Component Bands (SECONDARY)

#### High-Confidence Cellulose Bands
```python
# Based on Agarwal, Molecules (2019) + Blervacq et al., Biomolecules (2023)
HIGH_CONFIDENCE_CELLULOSE = {
    'cellulose_co_stretch': (1090, 1100),      # C-O-C glycosidic stretching
    'cellulose_ch_bend': (1375, 1385),         # CH bending modes
    'cellulose_glycosidic': (890, 900),        # β-glycosidic ring vibration
    'cellulose_crystalline': (380, 400),       # Crystalline lattice modes
    'cellulose_glycosidic_low': (515, 525),    # Additional glycosidic marker
}
```

#### Hemicellulose Bands (Taxonomically Variable)
```python
# Composition varies between hardwood and softwood
HEMICELLULOSE_BANDS = {
    # Hardwood hemicellulose (xylan-rich)
    'hardwood_xylan': (1094, 1126),            # Xylan markers
    'hardwood_acetyl': (1735, 1740),           # Acetyl groups
    
    # Softwood hemicellulose (mannan-rich)
    'softwood_mannan': (1045, 1055),           # Mannan markers
    'softwood_galactan': (1070, 1080),         # Galactan markers
}
```

### C. Species-Level Discrimination Bands (TERTIARY)
```python
# Use cautiously - species-specific validation required
SPECIES_LEVEL_BANDS = {
    # Extractive markers (highly species-dependent)
    'extractives_aliphatic': (2850, 2950),     # Aliphatic C-H stretches
    'extractives_phenolic': (1440, 1460),      # Phenolic compounds
    'resin_terpenes': (1265, 1275),           # Coniferous resins only
    
    # Anatomical markers
    'vessel_elements': (1050, 1060),           # Hardwood vessels
    'tracheid_structure': (1070, 1080),        # Softwood tracheids
}
```

## 4. Taxonomic Classification Framework

### A. Hierarchical Classification Strategy
```python
class TaxonomicHierarchicalClassifier:
    """
    Hierarchical classifier respecting plant taxonomy
    Level 1: Hardwood vs Softwood (S/G lignin ratio)
    Level 2: Genus within taxonomic group
    Level 3: Species within genus
    """
    
    def __init__(self):
        # Level 1: Taxonomic classification (hardwood vs softwood)
        self.taxonomic_classifier = None
        
        # Level 2: Genus classifiers within each taxonomic group
        self.hardwood_genus_classifier = None  # Acacia vs Machilus vs Swietenia
        self.softwood_genus_classifier = None  # Chamaecyparis vs Fokienia
        
        # Level 3: Species classifiers within each genus
        self.species_classifiers = {}
        
    def calculate_sg_ratio(self, spectrum, wavenumbers):
        """Calculate S/G lignin ratio for taxonomic classification"""
        
        # Extract diagnostic band intensities
        syringyl_intensity = self.get_peak_intensity(spectrum, wavenumbers, 1510, 8)
        guaiacyl_intensity = self.get_peak_intensity(spectrum, wavenumbers, 1284, 10)
        aromatic_ref = self.get_peak_intensity(spectrum, wavenumbers, 1600, 10)
        
        # Normalize to aromatic reference (Saariaho method)
        syringyl_norm = syringyl_intensity / (aromatic_ref + 1e-8)
        guaiacyl_norm = guaiacyl_intensity / (aromatic_ref + 1e-8)
        
        # Calculate S/G ratio
        sg_ratio = syringyl_norm / (guaiacyl_norm + 1e-8)
        
        return {
            'sg_ratio': sg_ratio,
            'syringyl_normalized': syringyl_norm,
            'guaiacyl_normalized': guaiacyl_norm,
            'taxonomic_prediction': 'Hardwood' if sg_ratio > 0.5 else 'Softwood'
        }
    
    def fit_hierarchical(self, X, y_species, metadata, groups):
        """Fit hierarchical taxonomic model"""
        
        # Step 1: Create taxonomic labels
        hardwood_species = ['Acacia_Confusa', 'Acacia_Mangium', 'Machilus_Kusanoi', 
                           'Swietenia_Mahagoni', 'Swietenia_Marophylla']
        
        metadata['taxonomy'] = metadata['species'].apply(
            lambda x: 'Hardwood' if x in hardwood_species else 'Softwood'
        )
        
        # Step 2: Extract S/G features for taxonomic classification
        sg_features = np.array([
            self.calculate_sg_ratio(spectrum, wavenumbers)['sg_ratio'] 
            for spectrum in X
        ]).reshape(-1, 1)
        
        # Step 3: Fit taxonomic classifier
        self.taxonomic_classifier = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight='balanced'
        )
        
        taxonomic_cv = self.hierarchical_cross_validation(
            sg_features, metadata['taxonomy'], groups, 
            self.taxonomic_classifier, "Taxonomic (S/G)"
        )
        
        # Step 4: Fit genus classifiers within each taxonomic group
        # ... (continue with genus and species level fitting)
        
        return taxonomic_cv
```

### B. Feature Extraction with Taxonomic Awareness
```python
def extract_taxonomic_molecular_features(spectra, wavenumbers, metadata):
    """Extract features with taxonomic context"""
    
    features = {}
    
    # 1. Primary taxonomic features (S/G lignin)
    for band_name, (low, high) in HARDWOOD_SYRINGYL_BANDS.items():
        mask = (wavenumbers >= low) & (wavenumbers <= high)
        if np.sum(mask) > 0:
            features[f'hardwood_{band_name}_max'] = np.max(spectra[:, mask], axis=1)
            features[f'hardwood_{band_name}_area'] = np.trapz(spectra[:, mask], axis=1)
    
    for band_name, (low, high) in SOFTWOOD_GUAIACYL_BANDS.items():
        mask = (wavenumbers >= low) & (wavenumbers <= high)
        if np.sum(mask) > 0:
            features[f'softwood_{band_name}_max'] = np.max(spectra[:, mask], axis=1)
            features[f'softwood_{band_name}_area'] = np.trapz(spectra[:, mask], axis=1)
    
    # 2. Calculate taxonomically meaningful ratios
    if 'hardwood_syringyl_primary_max' in features and 'softwood_guaiacyl_primary_max' in features:
        # S/G ratio (primary taxonomic discriminator)
        aromatic_ref = np.max(spectra[:, (wavenumbers >= 1595) & (wavenumbers <= 1605)], axis=1)
        
        syringyl_norm = features['hardwood_syringyl_primary_max'] / (aromatic_ref + 1e-8)
        guaiacyl_norm = features['softwood_guaiacyl_primary_max'] / (aromatic_ref + 1e-8)
        
        features['sg_ratio'] = syringyl_norm / (guaiacyl_norm + 1e-8)
        features['syringyl_normalized'] = syringyl_norm
        features['guaiacyl_normalized'] = guaiacyl_norm
    
    # 3. Universal cellulose features
    for band_name, (low, high) in HIGH_CONFIDENCE_CELLULOSE.items():
        mask = (wavenumbers >= low) & (wavenumbers <= high)
        if np.sum(mask) > 0:
            features[f'cellulose_{band_name}_max'] = np.max(spectra[:, mask], axis=1)
            features[f'cellulose_{band_name}_area'] = np.trapz(spectra[:, mask], axis=1)
    
    # 4. Taxonomically-specific hemicellulose features
    hardwood_mask = metadata['species'].isin(['Acacia_Confusa', 'Acacia_Mangium', 
                                             'Machilus_Kusanoi', 'Swietenia_Mahagoni', 
                                             'Swietenia_Marophylla'])
    
    # Extract hardwood-specific hemicellulose for hardwood samples
    for i, is_hardwood in enumerate(hardwood_mask):
        if is_hardwood:
            # Use hardwood hemicellulose bands
            for band_name, (low, high) in HEMICELLULOSE_BANDS.items():
                if 'hardwood' in band_name:
                    mask = (wavenumbers >= low) & (wavenumbers <= high)
                    if np.sum(mask) > 0 and f'{band_name}_max' not in features:
                        features[f'{band_name}_max'] = np.zeros(len(spectra))
                        features[f'{band_name}_max'][i] = np.max(spectra[i, mask])
        else:
            # Use softwood hemicellulose bands
            for band_name, (low, high) in HEMICELLULOSE_BANDS.items():
                if 'softwood' in band_name:
                    mask = (wavenumbers >= low) & (wavenumbers <= high)
                    if np.sum(mask) > 0 and f'{band_name}_max' not in features:
                        features[f'{band_name}_max'] = np.zeros(len(spectra))
                        features[f'{band_name}_max'][i] = np.max(spectra[i, mask])
    
    return pd.DataFrame(features)
```

## Machine Learning Architecture Guidelines

### 1. Model Hierarchy (Implement in Order)

#### Tier 1: Taxonomic Baselines (Days 1-2)
```python
# Start with taxonomic discrimination
taxonomic_models = {
    'SG_Ratio_Classifier': SGRatioClassifier(),  # Custom S/G ratio classifier
    'Random_Forest_Taxonomic': RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    ),
    'SVM_Taxonomic': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
}

class SGRatioClassifier:
    """Simple S/G ratio-based taxonomic classifier"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def fit(self, X_sg_ratios, y_taxonomy):
        """Fit threshold using cross-validation"""
        # Optimize threshold for best taxonomic separation
        pass
        
    def predict(self, X_sg_ratios):
        """Predict hardwood vs softwood based on S/G ratio"""
        predictions = ['Hardwood' if ratio > self.threshold else 'Softwood' 
                      for ratio in X_sg_ratios]
        return np.array(predictions)
```

#### Tier 2: Hierarchical Methods (Days 3-4)
```python
# Hierarchical classification approach
class TaxonomicAwareEnsemble:
    """Ensemble respecting taxonomic hierarchy"""
    
    def __init__(self):
        self.taxonomic_classifier = None  # Hardwood vs Softwood
        self.hardwood_classifier = None   # Within hardwood species
        self.softwood_classifier = None   # Within softwood species
        
    def fit(self, X, y_species, metadata, groups):
        """Fit hierarchical ensemble"""
        
        # Step 1: Fit taxonomic classifier
        X_taxonomic = self.extract_taxonomic_features(X)
        y_taxonomic = self.create_taxonomic_labels(y_species)
        
        self.taxonomic_classifier = RandomForestClassifier(n_estimators=300)
        taxonomic_cv = hierarchical_cross_validation(
            X_taxonomic, y_taxonomic, groups, self.taxonomic_classifier
        )
        
        # Step 2: Fit within-group classifiers
        hardwood_mask = y_taxonomic == 'Hardwood'
        softwood_mask = y_taxonomic == 'Softwood'
        
        if np.sum(hardwood_mask) > 10:  # Minimum samples
            self.hardwood_classifier = RandomForestClassifier(n_estimators=200)
            hardwood_cv = hierarchical_cross_validation(
                X[hardwood_mask], y_species[hardwood_mask], 
                groups[hardwood_mask], self.hardwood_classifier
            )
        
        if np.sum(softwood_mask) > 10:  # Minimum samples
            self.softwood_classifier = RandomForestClassifier(n_estimators=200)
            softwood_cv = hierarchical_cross_validation(
                X[softwood_mask], y_species[softwood_mask], 
                groups[softwood_mask], self.softwood_classifier
            )
        
        return {
            'taxonomic_cv': taxonomic_cv,
            'hardwood_cv': hardwood_cv if hasattr(self, 'hardwood_cv') else None,
            'softwood_cv': softwood_cv if hasattr(self, 'softwood_cv') else None
        }
```

#### Tier 3: Advanced Architecture (Days 5-6)
```python
import torch
import torch.nn as nn

class TaxonomicSpectralTransformer(nn.Module):
    """Transformer with taxonomic attention mechanism"""
    
    def __init__(self, input_dim, num_species, d_model=128):
        super().__init__()
        
        # Spectral embedding
        self.spectral_embedding = nn.Linear(input_dim, d_model)
        
        # Taxonomic attention queries (S-lignin vs G-lignin focused)
        self.syringyl_queries = nn.Parameter(torch.randn(3, d_model))  # 3 S-bands
        self.guaiacyl_queries = nn.Parameter(torch.randn(3, d_model))  # 3 G-bands
        self.cellulose_queries = nn.Parameter(torch.randn(4, d_model))  # 4 cellulose bands
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Hierarchical classification heads
        self.taxonomic_head = nn.Linear(d_model, 2)  # Hardwood vs Softwood
        self.species_head = nn.Linear(d_model, num_species)  # Final species classification
        
    def forward(self, x, return_attention=False):
        # Embed spectral data
        x = self.spectral_embedding(x.unsqueeze(-1))
        
        # Apply transformer
        x = self.transformer(x)
        
        # Taxonomic attention pooling
        batch_size = x.size(0)
        
        # Combine all molecular queries
        all_queries = torch.cat([
            self.syringyl_queries, self.guaiacyl_queries, self.cellulose_queries
        ], dim=0)
        
        queries = all_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Attention over spectral features
        attended, attention_weights = torch.nn.functional.scaled_dot_product_attention(
            queries, x, x
        )
        
        # Global pooling
        pooled = attended.mean(dim=1)
        
        # Hierarchical predictions
        taxonomic_logits = self.taxonomic_head(pooled)
        species_logits = self.species_head(pooled)
        
        if return_attention:
            return taxonomic_logits, species_logits, attention_weights
        else:
            return taxonomic_logits, species_logits
```

## Explainable AI Implementation

### 1. Taxonomic Attribution Framework
```python
import shap
from captum.attr import IntegratedGradients

class TaxonomicExplainability:
    """Explainability framework with taxonomic context"""
    
    def __init__(self, model, feature_names, band_assignments):
        self.model = model
        self.feature_names = feature_names
        self.band_assignments = band_assignments
        
    def taxonomic_feature_importance(self, X_train, y_taxonomic):
        """Global feature importance with taxonomic grouping"""
        
        # SHAP analysis
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_train)
        
        # Group features by taxonomic relevance
        taxonomic_groups = {
            'Syringyl_Lignin': [f for f in self.feature_names if 'syringyl' in f.lower()],
            'Guaiacyl_Lignin': [f for f in self.feature_names if 'guaiacyl' in f.lower()],
            'Cellulose': [f for f in self.feature_names if 'cellulose' in f.lower()],
            'Hemicellulose': [f for f in self.feature_names if any(h in f.lower() 
                                                                 for h in ['xylan', 'mannan', 'acetyl'])],
            'Extractives': [f for f in self.feature_names if 'extractive' in f.lower()]
        }
        
        # Calculate group-level importance
        group_importance = {}
        for group_name, group_features in taxonomic_groups.items():
            group_indices = [i for i, f in enumerate(self.feature_names) if f in group_features]
            
            if group_indices and isinstance(shap_values, list):
                # Multi-class SHAP values
                group_importance[group_name] = np.mean([
                    np.abs(sv[:, group_indices]).sum(axis=1).mean() 
                    for sv in shap_values
                ])
            elif group_indices:
                # Binary SHAP values
                group_importance[group_name] = np.abs(shap_values[:, group_indices]).sum(axis=1).mean()
        
        return group_importance
    
    def molecular_attribution_analysis(self, sample_spectrum, wavenumbers):
        """Map model predictions to molecular bands"""
        
        # Extract band-specific attributions
        molecular_attributions = {}
        
        # S-lignin attribution (hardwood indicator)
        syringyl_bands = [(1506, 1514), (1330, 1333), (962, 981)]
        syringyl_attribution = 0
        
        for low, high in syringyl_bands:
            mask = (wavenumbers >= low) & (wavenumbers <= high)
            if np.sum(mask) > 0:
                band_intensity = np.max(sample_spectrum[mask])
                syringyl_attribution += band_intensity
        
        molecular_attributions['Syringyl_Lignin'] = syringyl_attribution
        
        # G-lignin attribution (softwood indicator)
        guaiacyl_bands = [(1279, 1289), (1155, 1187), (704, 791)]
        guaiacyl_attribution = 0
        
        for low, high in guaiacyl_bands:
            mask = (wavenumbers >= low) & (wavenumbers <= high)
            if np.sum(mask) > 0:
                band_intensity = np.max(sample_spectrum[mask])
                guaiacyl_attribution += band_intensity
        
        molecular_attributions['Guaiacyl_Lignin'] = guaiacyl_attribution
        
        # Calculate S/G ratio for taxonomic interpretation
        sg_ratio = syringyl_attribution / (guaiacyl_attribution + 1e-8)
        molecular_attributions['SG_Ratio'] = sg_ratio
        molecular_attributions['Taxonomic_Prediction'] = 'Hardwood' if sg_ratio > 0.5 else 'Softwood'
        
        return molecular_attributions
```

### 2. Attention Visualization for Transformers
```python
def visualize_taxonomic_attention(attention_weights, wavenumbers, molecular_bands):
    """Create taxonomic attention visualization"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Extract attention for different molecular components
    syringyl_attention = attention_weights[:3, :].mean(dim=0)  # First 3 queries (S-lignin)
    guaiacyl_attention = attention_weights[3:6, :].mean(dim=0)  # Next 3 queries (G-lignin)
    cellulose_attention = attention_weights[6:10, :].mean(dim=0)  # Last 4 queries (cellulose)
    
    # Plot 1: Syringyl (Hardwood) attention
    axes[0].plot(wavenumbers, syringyl_attention.cpu().numpy(), 'r-', linewidth=2)
    axes[0].set_title('Syringyl Lignin Attention (Hardwood Marker)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Attention Weight')
    
    # Highlight syringyl bands
    for band_name, (low, high) in molecular_bands.items():
        if 'syringyl' in band_name:
            axes[0].axvspan(low, high, alpha=0.3, color='red', label=band_name)
    
    # Plot 2: Guaiacyl (Softwood) attention  
    axes[1].plot(wavenumbers, guaiacyl_attention.cpu().numpy(), 'b-', linewidth=2)
    axes[1].set_title('Guaiacyl Lignin Attention (Softwood Marker)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Attention Weight')
    
    # Highlight guaiacyl bands
    for band_name, (low, high) in molecular_bands.items():
        if 'guaiacyl' in band_name:
            axes[1].axvspan(low, high, alpha=0.3, color='blue', label=band_name)
    
    # Plot 3: Cellulose attention
    axes[2].plot(wavenumbers, cellulose_attention.cpu().numpy(), 'g-', linewidth=2)
    axes[2].set_title('Cellulose Attention (Universal Marker)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Attention Weight')
    axes[2].set_xlabel('Wavenumber (cm⁻¹)')
    
    # Highlight cellulose bands
    for band_name, (low, high) in molecular_bands.items():
        if 'cellulose' in band_name:
            axes[2].axvspan(low, high, alpha=0.3, color='green', label=band_name)
    
    plt.tight_layout()
    return fig
```

## Validation & Statistical Analysis

### 1. Taxonomic Effect Size Analysis
```python
def calculate_taxonomic_effect_sizes(X, y_species, feature_names):
    """Calculate effect sizes for taxonomic discrimination"""
    
    # Create taxonomic groups
    hardwood_species = ['Acacia_Confusa', 'Acacia_Mangium', 'Machilus_Kusanoi', 
                       'Swietenia_Mahagoni', 'Swietenia_Marophylla']
    
    hardwood_mask = np.isin(y_species, hardwood_species)
    softwood_mask = ~hardwood_mask
    
    effect_sizes = {}
    
    for i, feature in enumerate(feature_names):
        hardwood_values = X[hardwood_mask, i]
        softwood_values = X[softwood_mask, i]
        
        # Cohen's d calculation
        pooled_std = np.sqrt(((len(hardwood_values) - 1) * np.var(hardwood_values) + 
                            (len(softwood_values) - 1) * np.var(softwood_values)) / 
                           (len(hardwood_values) + len(softwood_values) - 2))
        
        if pooled_std > 0:
            cohens_d = abs(np.mean(hardwood_values) - np.mean(softwood_values)) / pooled_std
            effect_sizes[feature] = cohens_d
        else:
            effect_sizes[feature] = 0
    
    # Sort by effect size
    sorted_effects = dict(sorted(effect_sizes.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_effects

def taxonomic_power_analysis(hardwood_n=5, softwood_n=4, effect_size=1.5, alpha=0.05):
    """Calculate statistical power for taxonomic discrimination"""
    from statsmodels.stats.power import ttest_power
    
    # Power for detecting large taxonomic differences (S vs G lignin)
    power = ttest_power(effect_size, nobs1=hardwood_n, nobs2=softwood_n, alpha=alpha)
    
    return {
        'current_power': power,
        'effect_size': effect_size,
        'hardwood_samples': hardwood_n,
        'softwood_samples': softwood_n,
        'interpretation': 'High power' if power > 0.8 else 'Moderate power' if power > 0.6 else 'Low power'
    }
```

## Visualization Guidelines

### 1. Taxonomic Classification Results
```python
def plot_taxonomic_classification_results(sg_ratios, species_labels, cv_results):
    """Create comprehensive taxonomic classification visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: S/G Ratio Distribution by Taxonomy
    hardwood_species = ['Acacia_Confusa', 'Acacia_Mangium', 'Machilus_Kusanoi', 
                       'Swietenia_Mahagoni', 'Swietenia_Marophylla']
    
    hardwood_mask = species_labels.isin(hardwood_species)
    softwood_mask = ~hardwood_mask
    
    axes[0,0].hist(sg_ratios[hardwood_mask], bins=20, alpha=0.7, color='red', 
                   label=f'Hardwood (n={np.sum(hardwood_mask)})', density=True)
    axes[0,0].hist(sg_ratios[softwood_mask], bins=20, alpha=0.7, color='blue', 
                   label=f'Softwood (n={np.sum(softwood_mask)})', density=True)
    axes[0,0].axvline(x=0.5, color='black', linestyle='--', label='Classification Threshold')
    axes[0,0].set_xlabel('S/G Lignin Ratio')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('A) Taxonomic Discrimination via S/G Ratio')
    axes[0,0].legend()
    
    # Plot 2: Species-level S/G Ratios
    species_sg_means = []
    species_names = []
    colors = []
    
    for species in np.unique(species_labels):
        species_mask = species_labels == species
        species_sg_mean = sg_ratios[species_mask].mean()
        species_sg_means.append(species_sg_mean)
        species_names.append(species.replace('_', ' '))
        colors.append('red' if species in hardwood_species else 'blue')
    
    bars = axes[0,1].bar(range(len(species_names)), species_sg_means, color=colors, alpha=0.7)
    axes[0,1].axhline(y=0.5, color='black', linestyle='--', label='Taxonomic Threshold')
    axes[0,1].set_xlabel('Species')
    axes[0,1].set_ylabel('Mean S/G Ratio')
    axes[0,1].set_title('B) Species-Level S/G Signatures')
    axes[0,1].set_xticks(range(len(species_names)))
    axes[0,1].set_xticklabels(species_names, rotation=45, ha='right')
    
    # Plot 3: Cross-Validation Results
    model_names = list(cv_results.keys())
    accuracies = [cv_results[name]['mean_accuracy'] for name in model_names]
    errors = [cv_results[name]['std_accuracy'] for name in model_names]
    
    bars = axes[1,0].bar(model_names, accuracies, yerr=errors, capsize=5, 
                        color=['lightcoral', 'lightblue', 'lightgreen'], alpha=0.8)
    axes[1,0].set_ylabel('Cross-Validation Accuracy')
    axes[1,0].set_title('C) Model Performance Comparison')
    axes[1,0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc, err in zip(bars, accuracies, errors):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                      f'{acc:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Confusion Matrix (if binary taxonomic classification)
    if len(np.unique(species_labels)) == 2:  # Binary case
        from sklearn.metrics import confusion_matrix
        # Plot confusion matrix
        pass
    else:
        # Plot feature importance for top taxonomic features
        # This would show which molecular bands contribute most to classification
        pass
    
    plt.tight_layout()
    return fig
```

### 2. Molecular Band Contribution Analysis
```python
def plot_molecular_band_contributions(feature_importance_dict, band_assignments):
    """Visualize molecular band contributions to classification"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Group features by molecular component
    molecular_groups = {
        'Syringyl Lignin (Hardwood)': [f for f in feature_importance_dict.keys() 
                                      if any(s in f.lower() for s in ['syringyl', 'hardwood_syringyl'])],
        'Guaiacyl Lignin (Softwood)': [f for f in feature_importance_dict.keys() 
                                      if any(g in f.lower() for g in ['guaiacyl', 'softwood_guaiacyl'])],
        'Cellulose (Universal)': [f for f in feature_importance_dict.keys() 
                                 if 'cellulose' in f.lower()],
        'Hemicellulose (Variable)': [f for f in feature_importance_dict.keys() 
                                    if any(h in f.lower() for h in ['xylan', 'mannan', 'acetyl'])]
    }
    
    # Plot molecular group importance
    for idx, (group_name, features) in enumerate(molecular_groups.items()):
        row, col = idx // 2, idx % 2
        
        if features:
            group_importance = [feature_importance_dict[f] for f in features]
            feature_labels = [f.replace('_', ' ').title() for f in features]
            
            bars = axes[row, col].barh(feature_labels, group_importance, 
                                      color=plt.cm.Set3(idx), alpha=0.8)
            axes[row, col].set_xlabel('Feature Importance')
            axes[row, col].set_title(f'{group_name}', fontweight='bold')
            
            # Add importance values as text
            for bar, importance in zip(bars, group_importance):
                width = bar.get_width()
                axes[row, col].text(width + 0.001, bar.get_y() + bar.get_height()/2,
                                   f'{importance:.3f}', ha='left', va='center', fontsize=9)
        else:
            axes[row, col].text(0.5, 0.5, 'No features detected', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'{group_name}', fontweight='bold')
    
    plt.tight_layout()
    return fig
```

## Error Handling & Debugging

### 1. Taxonomic Validation
```python
def validate_taxonomic_assignments(spectra, metadata, wavenumbers):
    """Validate taxonomic band assignments"""
    
    issues = []
    
    # Check S/G band coverage
    syringyl_bands = [(1506, 1514), (1330, 1333), (962, 981)]
    guaiacyl_bands = [(1279, 1289), (1155, 1187), (704, 791)]
    
    covered_syringyl = 0
    covered_guaiacyl = 0
    
    for low, high in syringyl_bands:
        if np.any((wavenumbers >= low) & (wavenumbers <= high)):
            covered_syringyl += 1
    
    for low, high in guaiacyl_bands:
        if np.any((wavenumbers >= low) & (wavenumbers <= high)):
            covered_guaiacyl += 1
    
    if covered_syringyl < 2:
        issues.append(f"Insufficient syringyl band coverage: {covered_syringyl}/3 bands")
    
    if covered_guaiacyl < 2:
        issues.append(f"Insufficient guaiacyl band coverage: {covered_guaiacyl}/3 bands")
    
    # Check taxonomic balance
    hardwood_species = ['Acacia_Confusa', 'Acacia_Mangium', 'Machilus_Kusanoi', 
                       'Swietenia_Mahagoni', 'Swietenia_Marophylla']
    
    hardwood_count = metadata['species'].isin(hardwood_species).sum()
    softwood_count = len(metadata) - hardwood_count
    
    if hardwood_count == 0:
        issues.append("No hardwood species detected")
    if softwood_count == 0:
        issues.append("No softwood species detected")
    
    taxonomic_ratio = min(hardwood_count, softwood_count) / max(hardwood_count, softwood_count)
    if taxonomic_ratio < 0.3:
        issues.append(f"Severe taxonomic imbalance: {taxonomic_ratio:.2f}")
    
    if issues:
        print("⚠️ Taxonomic validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Taxonomic validation passed")
        return True
```

## Documentation Standards

### 1. Taxonomic Code Documentation
```python
def taxonomic_feature_extraction(spectra, wavenumbers, metadata):
    """
    Extract taxonomically-relevant molecular features from Raman spectra.
    
    This function implements the hierarchical feature extraction strategy based on
    authoritative lignin band assignments from Saariaho et al. (2003) for S/G
    discrimination and Agarwal et al. (2019) for cellulose markers.
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Preprocessed Raman spectra intensities
    wavenumbers : np.ndarray, shape (n_wavenumbers,)
        Wavenumber values in cm⁻¹
    metadata : pd.DataFrame
        Sample metadata with columns: species, tree_id, facet, point_id
    
    Returns
    -------
    dict
        Taxonomic feature extraction results containing:
        - 'syringyl_features': Hardwood-diagnostic S-lignin features
        - 'guaiacyl_features': Softwood-diagnostic G-lignin features  
        - 'cellulose_features': Universal cellulose structural features
        - 'sg_ratios': S/G ratios for taxonomic classification
        - 'taxonomic_predictions': Hardwood vs Softwood predictions
    
    Notes
    -----
    Feature extraction follows the taxonomic hierarchy:
    1. Primary: S/G lignin discrimination (hardwood vs softwood)
    2. Secondary: Cellulose structural markers (universal)
    3. Tertiary: Hemicellulose composition (taxonomically variable)
    
    The S/G ratio calculation uses normalization to the 1600 cm⁻¹ aromatic
    reference band following Saariaho et al. (2003) methodology.
    
    References
    ----------
    [1] Saariaho, A.M. et al. Applied Spectroscopy 57(4), 414-419 (2003)
    [2] Agarwal, U.P. Molecules 24(9), 1659 (2019)
    """
    pass
```

### 2. Results Reporting Template
```python
def generate_taxonomic_results_report(results_dict):
    """Generate standardized taxonomic classification report"""
    
    report = f"""
# Taxonomic Raman Wood Species Classification Results

## Dataset Summary
- Total spectra: {results_dict['n_spectra']}
- Hardwood species: {results_dict['n_hardwood_species']} ({results_dict['hardwood_species']})
- Softwood species: {results_dict['n_softwood_species']} ({results_dict['softwood_species']})
- Hierarchical structure: Taxonomy → Genus → Species → Trees → Facets → Points

## Taxonomic Discrimination Results (S/G Lignin Analysis)
- Mean S/G ratio (hardwoods): {results_dict['hardwood_sg_mean']:.3f} ± {results_dict['hardwood_sg_std']:.3f}
- Mean S/G ratio (softwoods): {results_dict['softwood_sg_mean']:.3f} ± {results_dict['softwood_sg_std']:.3f}
- Taxonomic separation (Cohen's d): {results_dict['taxonomic_effect_size']:.3f}
- Classification threshold: {results_dict['sg_threshold']:.3f}

## Model Performance (Leave-One-Tree-Out CV)
"""
    
    for model_name, results in results_dict['model_results'].items():
        report += f"""
### {model_name}
- Taxonomic accuracy: {results['taxonomic_accuracy']:.3f} ± {results['taxonomic_std']:.3f}
- Species accuracy: {results['species_accuracy']:.3f} ± {results['species_std']:.3f}
- 95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]
"""
    
    report += f"""
## Molecular Band Contributions
### Top Hardwood Markers (Syringyl Lignin)
"""
    
    for band, importance in list(results_dict['syringyl_contributions'].items())[:3]:
        report += f"- {band}: {importance:.3f} (1506-1514, 1330-1333, 962-981 cm⁻¹)\n"
    
    report += f"""
### Top Softwood Markers (Guaiacyl Lignin)
"""
    
    for band, importance in list(results_dict['guaiacyl_contributions'].items())[:3]:
        report += f"- {band}: {importance:.3f} (1279-1289, 1155-1187, 704-791 cm⁻¹)\n"
    
    return report
```

## Final Implementation Checklist

### Before Running Any Code:
- [ ] Validate taxonomic band coverage (S/G lignin bands)
- [ ] Confirm hardwood vs softwood species classification
- [ ] Check wavenumber range includes key diagnostic regions
- [ ] Verify metadata includes taxonomic information

### For Each Model:
- [ ] Implement hierarchical taxonomic classification
- [ ] Calculate S/G ratios using authoritative method
- [ ] Apply proper Leave-One-Tree-Out cross-validation
- [ ] Generate taxonomic attribution analysis
- [ ] Create publication-quality taxonomic visualizations

### For Final Results:
- [ ] Compare taxonomic vs species-level performance
- [ ] Document molecular band contributions
- [ ] Validate against known hardwood/softwood standards
- [ ] Generate Nature-standard figures with taxonomic context
- [ ] Prepare reproducible code with authoritative citations

## Critical Success Factors

1. **Taxonomic Foundation**: Base all analysis on S/G lignin biochemistry
2. **Authoritative Citations**: Use Saariaho et al. (2003) as gold standard
3. **Hierarchical Validation**: Respect tree-level grouping in all statistics
4. **Molecular Interpretability**: Connect all decisions to plant biochemistry
5. **Statistical Honesty**: Acknowledge taxonomic imbalance and sample limitations

**Remember**: The taxonomic approach (hardwood vs softwood) provides the strongest biological foundation for your classification, with S/G lignin ratios offering the most robust discriminating features. Focus on this taxonomic hierarchy while being transparent about within-group species discrimination challenges.