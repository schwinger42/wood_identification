# 🌳 AI Agent Implementation Guidelines: Raman Wood Species Classification

---

### ### Project Overview
A cutting-edge machine learning system for classifying **9 Taiwanese wood species** using Raman spectroscopy, developed for **National Science and Technology Council (NSC) committee demonstration** and targeting **Nature publication**.

---

## 📊 DATASET STATUS QUO (Current Reality)

### ### Core Dataset Facts
- **Total Files**: 1,051 Raman spectra (.txt format)
- **Species Count**: 9 Taiwanese wood species
- **Unique Trees**: 11 individual trees total
- **Critical Challenge**: Very small sample size with hierarchical structure
- **Current Location**: `/dataset/organized_preserved/` (train/test structure)

### ### Data Naming Structure
```
Species → Distinct Tree number → Cubes → Facets (Cross/Radial/Tangential) → Measurement Points (10 per facet)
```

### ### Species Classification by Taxonomy
**Hardwood Species (5) - Syringyl-Rich Lignin:**
- Acacia_Confusa
- Acacia_Mangium  
- Machilus_Kusanoi
- Swietenia_Mahagoni
- Swietenia_Marophylla

**Softwood Species (4) - Guaiacyl-Rich Lignin:**
- Chamaecyparis_Formosensis
- Chamaecyparis_Obtusa_var_Formosana
- Chamaecyparis_Obyusa
- Fokienia_Hodginsii

### ### Prof. Tsao's Pre-Designated Test Sets (UNCHANGEABLE)
```
✅ ONLY 2 species have predefined test sets:
- Chamaecyparis_Obtusa_var_Formosana: Tree_2 → Test Set (90 files)
- Swietenia_Marophylla: Tree_2 → Test Set (90 files)

🔧 Remaining 7 species: ALL currently in train/ - need proper splitting!
```

---

#### #### 1. Advanced Preprocessing Pipeline
- **Savitzky-Golay filtering** with optimized parameters (window=15, polyorder=2)
- **Morphological baseline correction** for fluorescence removal
- **Robust normalization** handling negative intensities
- **Spectral derivatives** for feature enhancement
- **Location**: `reference/raman_wood_classifier_production.py` lines 200-350

#### #### 2. Hierarchical Data Organization
- **Prof. Tsao's test sets** properly separated
- **Cube-level splitting** respecting tree boundaries
- **Metadata tracking** for all hierarchical levels
- **Location**: `reference/01_data_exploration_and_preprocessing.ipynb` lines 687-797

#### #### 3. Leave-One-Group-Out Cross-Validation
- **Tree-level grouping** preventing data leakage
- **Proper implementation** in production code
- **11-fold CV** (one per tree)
- **Location**: `reference/raman_wood_classifier_production.py` lines 629-681

#### #### 4. SHAP Explainability
- **TreeExplainer** for ensemble models
- **Feature importance** with molecular context
- **Band attribution** analysis
- **Location**: `reference/raman_wood_classifier_production.py` lines 800-900

#### #### 5. Ensemble ML Models
- **XGBoost + Random Forest + SVM** with voting
- **Hyperparameter optimization** 
- **Cross-validation** with proper grouping
- **Location**: `reference/raman_wood_classifier_production.py` lines 400-600

#### #### 6. Production Pipeline
- **Real-time prediction** capability
- **Error handling** and validation
- **Professional reporting** interface
- **Location**: `reference/raman_wood_classifier_production.py` (complete file)

#### #### 1. Notebook 04: Advanced Interpretability
- **Documentation Claims**: "Integrated Gradients implementation"
- **Reality**: **File doesn't exist**
- **What to Do**: Use SHAP analysis (actually implemented)

#### #### 2. Integrated Gradients
- **Documentation Claims**: "Advanced interpretability beyond SHAP"

#### #### 3. RADAR Denoising
- **Documentation Claims**: "RADAR-style preprocessing comparison"
- **What to Do**: combine with or compare with Savitzky-Golay method

#### #### 4. Deep Learning Models
- **Documentation Claims**: "CNN-1D, ResNet-1D, Transformer architectures"
- **What to Do**: Focus on ensemble ML (proven and working)

#### #### 5. Performance Metrics
- **Documentation Claims**: "96.1% accuracy, 98.7% taxonomic accuracy"
- **What to Do**: Generate realistic expectations (70-85% species, 85-95% taxonomic)

---

## 🎯 PHASE-BY-PHASE IMPLEMENTATION ROADMAP

### ### Phase 1: Dataset Validation & Setup

#### #### Step 1.1: Verify Dataset Structure
```python
# Check current dataset organization
def validate_dataset_structure():
    """Verify hierarchical organization and Prof. Tsao's constraints"""
    
    # 1. Count total files
    import glob
    train_files = glob.glob("dataset/organized/train/**/*.txt", recursive=True)
    test_files = glob.glob("dataset/organized/test/**/*.txt", recursive=True)
    
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    
    # 2. Verify Prof. Tsao's test sets
    test_species = set()
    for file_path in test_files:
        species = file_path.split('/')[-2]
        test_species.add(species)
    
    expected_test_species = {
        'Chamaecyparis_Obtusa_var_Formosana',
        'Swietenia_Marophylla'
    }
    
    if test_species == expected_test_species:
        print("✅ Prof. Tsao's test sets correctly separated")
    else:
        print(f"❌ Test set issue: Found {test_species}")
    
    return len(train_files), len(test_files)

# Run validation
train_count, test_count = validate_dataset_structure()
```

#### #### Step 1.2: Create Tree-Level Groups
```python
def create_tree_groups(metadata_df):
    """Create proper grouping for LOGOCV preventing data leakage"""
    
    # Create tree-level identifiers
    groups = metadata_df['species'] + '_Tree' + metadata_df['tree_id'].astype(str)
    
    print(f"Created {len(groups.unique())} unique tree groups")
    print(f"Groups: {sorted(groups.unique())}")
    
    return groups

# This is CRITICAL - always group by trees, never by measurement points
```

#### #### Step 1.3: Load and Verify Preprocessed Data
```python
# Check if processed data exists
import pickle
from pathlib import Path

processed_data_path = Path("processed_data/processed_spectra.pkl")

if processed_data_path.exists():
    print("✅ Preprocessed data found - loading...")
    with open(processed_data_path, "rb") as f:
        processed_data = pickle.load(f)
else:
    print("🔄 Preprocessed data not found - need to run preprocessing")
    # Proceed to Phase 2
```

### ### Phase 2: Preprocessing & Feature Engineering

#### #### Step 2.1: Implement Proven Preprocessing
```python
from scipy.signal import savgol_filter
from scipy.ndimage import binary_opening, binary_closing
import numpy as np

class ProvenPreprocessor:
    """Use ONLY the implemented and tested methods"""
    
    def __init__(self):
        self.savgol_window = 15
        self.savgol_polyorder = 2
        self.norm_params = {}
    
    def preprocess_spectrum(self, wavenumbers, intensities):
        """Research-validated preprocessing pipeline"""
        
        # 1. Savitzky-Golay smoothing (WORKING IMPLEMENTATION)
        smoothed = savgol_filter(intensities, 
                               window_length=self.savgol_window, 
                               polyorder=self.savgol_polyorder)
        
        # 2. Morphological baseline correction (IMPLEMENTED)
        baseline_corrected = self._remove_morphological_baseline(smoothed)
        
        # 3. Robust normalization (HANDLES NEGATIVE VALUES)
        normalized = self._robust_normalize(baseline_corrected)
        
        # 4. Spectral derivatives (OPTIONAL)
        first_derivative = savgol_filter(normalized, 
                                       window_length=self.savgol_window, 
                                       polyorder=self.savgol_polyorder, 
                                       deriv=1)
        
        return {
            'normalized': normalized,
            'first_derivative': first_derivative,
            'baseline_corrected': baseline_corrected
        }
    
    def _remove_morphological_baseline(self, intensities):
        """Morphological baseline correction (ACTUALLY IMPLEMENTED)"""
        # Implementation from working code
        pass
    
    def _robust_normalize(self, intensities):
        """Robust normalization handling negative values"""
        # Implementation from working code
        pass

# DON'T implement RADAR denoising - it's not actually coded
```

#### #### Step 2.2: Extract Research-Validated Molecular Features
```python
# Based on authoritative literature (Saariaho et al. 2003)
HARDWOOD_SYRINGYL_BANDS = {
    'syringyl_primary': (1506, 1514),    # S-unit aromatic vibration
    'syringyl_secondary': (1330, 1333),  # S-unit ring breathing  
    'syringyl_tertiary': (962, 981),     # S-unit deformation
}

SOFTWOOD_GUAIACYL_BANDS = {
    'guaiacyl_primary': (1279, 1289),    # G-unit aromatic vibration
    'guaiacyl_secondary': (1155, 1187),  # G-unit ring breathing
    'guaiacyl_tertiary': (704, 791),     # G-unit deformation
}

def extract_taxonomic_features(spectra, wavenumbers):
    """Extract features with taxonomic significance"""
    
    features = {}
    
    # 1. Syringyl bands (hardwood markers)
    for band_name, (low, high) in HARDWOOD_SYRINGYL_BANDS.items():
        mask = (wavenumbers >= low) & (wavenumbers <= high)
        if np.sum(mask) > 0:
            features[f'hardwood_{band_name}_max'] = np.max(spectra[:, mask], axis=1)
            features[f'hardwood_{band_name}_area'] = np.trapz(spectra[:, mask], axis=1)
    
    # 2. Guaiacyl bands (softwood markers)  
    for band_name, (low, high) in SOFTWOOD_GUAIACYL_BANDS.items():
        mask = (wavenumbers >= low) & (wavenumbers <= high)
        if np.sum(mask) > 0:
            features[f'softwood_{band_name}_max'] = np.max(spectra[:, mask], axis=1)
            features[f'softwood_{band_name}_area'] = np.trapz(spectra[:, mask], axis=1)
    
    # 3. Calculate S/G ratio (PRIMARY taxonomic discriminator)
    if 'hardwood_syringyl_primary_max' in features and 'softwood_guaiacyl_primary_max' in features:
        # Normalize to aromatic reference (1600 cm⁻¹)
        aromatic_ref = np.max(spectra[:, (wavenumbers >= 1595) & (wavenumbers <= 1605)], axis=1)
        
        syringyl_norm = features['hardwood_syringyl_primary_max'] / (aromatic_ref + 1e-8)
        guaiacyl_norm = features['softwood_guaiacyl_primary_max'] / (aromatic_ref + 1e-8)
        
        features['sg_ratio'] = syringyl_norm / (guaiacyl_norm + 1e-8)
        features['taxonomic_prediction'] = ['Hardwood' if ratio > 0.5 else 'Softwood' 
                                          for ratio in features['sg_ratio']]
    
    return pd.DataFrame(features)
```

### ### Phase 3: Model Development & Cross-Validation

#### #### Step 3.1: Implement Proven ML Models
```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
import xgboost as xgb

def build_proven_classifier():
    """Build classifier using ONLY tested and working methods"""
    
    # 1. Individual models (PROVEN TO WORK)
    models = {
        'xgboost': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'svm': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
    }
    
    # 2. Voting ensemble (IMPLEMENTED AND TESTED)
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    
    return models, ensemble

# DON'T implement deep learning models - they're not thoroughly tested
```

#### #### Step 3.2: Proper Cross-Validation Implementation
```python
def train_with_proper_cv(X, y, groups, models):
    """Train with Leave-One-Tree-Out Cross-Validation"""
    
    # CRITICAL: Use tree-level grouping
    cv = LeaveOneGroupOut()
    unique_trees = len(np.unique(groups))
    
    print(f"🔄 Using LOGOCV with {unique_trees} trees (should be 11)")
    
    cv_results = {}
    
    for model_name, model in models.items():
        print(f"   📊 Training {model_name}...")
        
        # Cross-validation with proper grouping
        cv_scores = cross_val_score(
            model, X, y, 
            cv=cv, 
            groups=groups, 
            scoring='accuracy', 
            n_jobs=-1
        )
        
        # Fit on full data
        model.fit(X, y)
        
        cv_results[model_name] = {
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_folds': len(cv_scores)
        }
        
        print(f"      ✅ LOGOCV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # VALIDATION CHECK
        if len(cv_scores) != unique_trees:
            print(f"      ⚠️  WARNING: CV folds ({len(cv_scores)}) != trees ({unique_trees})")
    
    return cv_results

# CRITICAL: If CV folds != 11, you have data leakage!
```

### ### Phase 4: Explainability & Interpretation

#### #### Step 4.1: SHAP Analysis Implementation
```python
import shap
from sklearn.inspection import permutation_importance

def explain_predictions_correctly(model, X_train, X_test, feature_names):
    """Use ONLY the implemented explainability methods"""
    
    # 1. SHAP Analysis (ACTUALLY IMPLEMENTED)
    if hasattr(model, 'estimators_'):  # Ensemble models
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)
    
    shap_values = explainer.shap_values(X_test)
    
    # 2. Permutation Importance (MORE ROBUST than Integrated Gradients)
    perm_importance = permutation_importance(
        model, X_train, y_train, 
        n_repeats=10, 
        random_state=42
    )
    
    # 3. Molecular band attribution
    molecular_contributions = analyze_molecular_bands(
        shap_values, feature_names, HARDWOOD_SYRINGYL_BANDS, SOFTWOOD_GUAIACYL_BANDS
    )
    
    return {
        'shap_values': shap_values,
        'permutation_importance': perm_importance,
        'molecular_attribution': molecular_contributions
    }

# DON'T implement Integrated Gradients - it's not actually coded
```

#### #### Step 4.2: Taxonomic Analysis Framework
```python
def analyze_taxonomic_performance(predictions, true_labels, sg_ratios):
    """Analyze hardwood vs softwood classification"""
    
    # Create taxonomic labels
    hardwood_species = ['Acacia_Confusa', 'Acacia_Mangium', 'Machilus_Kusanoi', 
                       'Swietenia_Mahagoni', 'Swietenia_Marophylla']
    
    taxonomic_true = ['Hardwood' if species in hardwood_species else 'Softwood' 
                     for species in true_labels]
    taxonomic_pred = ['Hardwood' if species in hardwood_species else 'Softwood' 
                     for species in predictions]
    
    # Calculate taxonomic accuracy
    taxonomic_accuracy = accuracy_score(taxonomic_true, taxonomic_pred)
    
    # Analyze S/G ratios
    hardwood_sg_ratios = [sg_ratios[i] for i, label in enumerate(taxonomic_true) if label == 'Hardwood']
    softwood_sg_ratios = [sg_ratios[i] for i, label in enumerate(taxonomic_true) if label == 'Softwood']
    
    return {
        'taxonomic_accuracy': taxonomic_accuracy,
        'hardwood_sg_mean': np.mean(hardwood_sg_ratios),
        'softwood_sg_mean': np.mean(softwood_sg_ratios),
        'sg_separation': np.mean(hardwood_sg_ratios) - np.mean(softwood_sg_ratios)
    }
```

### ### Phase 5: Validation & Performance Assessment

#### #### Step 5.1: Realistic Performance Expectations
```python
REALISTIC_PERFORMANCE = {
    'taxonomic_classification': (0.85, 0.95),  # Hardwood vs Softwood
    'species_classification': (0.70, 0.85),    # Limited by small sample size
    'cv_folds': 11,                            # One per tree
    'processing_time': 5.0,                    # Seconds per spectrum
}

PERFORMANCE_RED_FLAGS = {
    'species_accuracy > 0.95': 'Likely overfitting or data leakage',
    'perfect_cv_scores': 'Definite data leakage',
    'cv_folds < 11': 'Not using proper tree-level grouping',
    'processing_time > 30': 'Implementation inefficiency'
}

def validate_performance(cv_results, groups):
    """Validate performance against realistic expectations"""
    
    issues = []
    
    # Check CV strategy
    unique_trees = len(np.unique(groups))
    
    for model_name, results in cv_results.items():
        cv_folds = results['cv_folds']
        mean_acc = results['cv_mean']
        
        if cv_folds != unique_trees:
            issues.append(f"{model_name}: CV folds ({cv_folds}) != trees ({unique_trees})")
        
        if mean_acc > 0.95:
            issues.append(f"{model_name}: Suspiciously high accuracy ({mean_acc:.3f})")
        
        if results['cv_std'] < 0.01:
            issues.append(f"{model_name}: Suspiciously low variance ({results['cv_std']:.4f})")
    
    if issues:
        print("⚠️  PERFORMANCE VALIDATION ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Performance validation passed")
        return True
```

#### #### Step 5.2: Test Set Evaluation
```python
def evaluate_on_tsao_test_sets(model, test_features, test_labels):
    """Evaluate on Prof. Tsao's pre-designated test sets"""
    
    # Make predictions
    predictions = model.predict(test_features)
    
    # Overall test accuracy
    test_accuracy = accuracy_score(test_labels, predictions)
    
    # Species-specific performance
    species_performance = {}
    for species in np.unique(test_labels):
        species_mask = test_labels == species
        species_acc = accuracy_score(
            test_labels[species_mask], 
            predictions[species_mask]
        )
        species_performance[species] = species_acc
    
    print(f"🎯 Test Set Performance:")
    print(f"   Overall Accuracy: {test_accuracy:.3f}")
    print(f"   Species Performance:")
    for species, acc in species_performance.items():
        print(f"     {species}: {acc:.3f}")
    
    return test_accuracy, species_performance
```

---

## 🎯 IMPLEMENTATION CHECKLIST FOR AI AGENTS

### ### Before Starting Development:
- [ ] Understand Prof. Tsao's test set constraints (2 species only)
- [ ] Verify hierarchical data structure (Species→Trees→Cubes→Facets→Points)
- [ ] Check which methods are actually implemented vs. documented
- [ ] Ignore documentation promises not backed by actual code

### ### During Development:
- [ ] Use tree-level cross-validation exclusively (LeaveOneGroupOut)
- [ ] Focus on taxonomic classification first (hardwood vs softwood)
- [ ] Use proven preprocessing (Savitzky-Golay, morphological baseline)
- [ ] Implement SHAP explainability (skip Integrated Gradients)
- [ ] Set realistic performance expectations (70-85% species, 85-95% taxonomic)

### ### Before Completion:
- [ ] Test with Prof. Tsao's designated test sets
- [ ] Verify no data leakage (11 CV folds = 11 trees)
- [ ] Generate realistic performance reports
- [ ] Create accurate documentation matching actual implementation
- [ ] Test production pipeline end-to-end

---

## 🔍 WHAT TO AVOID (COMMON PITFALLS)

### ### ❌ Don't Use These Validation Strategies:
1. **Standard K-Fold CV**: Will cause data leakage
2. **Random splitting**: Ignores hierarchical structure
3. **Point-level validation**: Treats measurements as independent
4. **Species-balanced splitting**: May separate same tree across folds

---

## 🎯 SUCCESS METRICS FOR AI AGENTS

### ### Technical Success Indicators:
- **Proper CV Implementation**: 11-fold LOGOCV (one per tree)
- **Taxonomic Accuracy**: 85-95% (hardwood vs softwood)
- **Species Accuracy**: 70-85% (realistic given sample size)
- **Zero Data Leakage**: Confirmed by validation checks
- **Processing Speed**: <5 seconds per spectrum

### ### Scientific Success Indicators:
- **Molecular Band Attribution**: S/G lignin ratio analysis working
- **Research Integration**: Saariaho et al. (2003) band assignments used
- **Hierarchical Respect**: Tree-level grouping maintained throughout
- **Uncertainty Quantification**: Confidence scores provided
- **Interpretability**: SHAP analysis with molecular context

---

## 🚀 FINAL IMPLEMENTATION STRATEGY

### ### Key Success Philosophy:
This is a **research project with small, hierarchical data**, not a large-scale production system. Success means:

1. **Proper validation methodology** preventing data leakage
2. **Interpretable results** with molecular basis  
3. **Realistic performance expectations**
4. **Scientific rigor** over inflated claims
5. **Focus on working methods** over promised features

### ### Immediate Action Plan for AI Agents:

#### #### Part 1: Reality Check
- Run dataset validation scripts
- Verify Prof. Tsao's test sets
- Check preprocessing pipeline
- Identify working vs. non-working methods

#### #### Part 2-3: Core Implementation  
- Implement proven preprocessing pipeline
- Extract research-validated molecular features
- Build ensemble ML models (skip deep learning)
- Set up proper cross-validation

#### #### Part 4: Analysis & Interpretation
- Implement SHAP analysis (skip Integrated Gradients)
- Add taxonomic classification layer
- Create molecular band attribution
- Generate realistic performance reports

#### #### Part 5: Validation & Documentation
- Test on Prof. Tsao's designated test sets
- Validate against realistic expectations
- Create accurate documentation
- Prepare committee demonstration

---

## 📞 CRITICAL REMINDERS FOR AI AGENTS

### ### 🎯 Remember:
- **Work with actual implementations**, not documentation promises
- **Respect hierarchical data structure** at all times
- **Use tree-level grouping** for all validation
- **Focus on taxonomic discrimination** as primary classification
- **Generate realistic performance expectations**
- **Prof. Tsao's pre allocated test sets are unchangeable constraints**

### ### 🚨 Red Flags to Watch:
- CV folds ≠ 11 (not using proper grouping)
- Predicting time >300 seconds (implementation issues)
- data leakage
---

*🌳 This guideline ensures AI agents can immediately understand the project reality and implement effective solutions without getting misled by documentation inconsistencies. Focus on proven methods, respect the hierarchical data structure, and deliver realistic, interpretable results! 🚀*