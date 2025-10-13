#!/usr/bin/env python3
"""
Create Average Spectrum Plot from One Tree's All Dataset


This script creates an average spectrum plot using all spectra from a specific tree,
following the hierarchical data structure: Species → Trees → Cubes → Facets → Points
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import signal
from scipy.ndimage import grey_opening
import glob

warnings.filterwarnings('ignore')

class TreeAverageAnalyzer:
    """
    Analyzer for creating average spectra from all data points of a specific tree.
    """
    
    def __init__(self, dataset_path: str = "dataset/organized"):
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.test_path = self.dataset_path / "test"
        
        # Species classification for context
        self.hardwood_species = [
            'Acacia_Confusa', 'Acacia_Mangium', 'Machilus_Kusanoi', 
            'Swietenia_Mahagoni', 'Swietenia_Marophylla'
        ]
        
        self.softwood_species = [
            'Chamaecyparis_Formosensis', 'Chamaecyparis_Obtusa_var_Formosana',
            'Chamaecyparis_Obyusa', 'Fokienia_Hodginsii'
        ]
        
    def load_spectrum(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single Raman spectrum from .txt file."""
        try:
            # Try different delimiters
            for delimiter in [';', '\t', ',']:
                try:
                    data = np.loadtxt(file_path, delimiter=delimiter)
                    if data.shape[1] == 2:
                        return data[:, 0], data[:, 1]  # wavenumbers, intensities
                except:
                    continue
            
            print(f"⚠️ Could not load {file_path}")
            return None, None
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None, None
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Parse hierarchical information from filename."""
        try:
            name = filename.replace('.txt', '')
            parts = name.split('-')
            
            if len(parts) >= 7:
                return {
                    'power': parts[0],
                    'date': parts[1], 
                    'species': parts[2],
                    'tree': parts[3],
                    'cube': parts[4],
                    'facet': parts[5],
                    'point': parts[6]
                }
            else:
                # Handle species names with underscores
                species_part = '-'.join(parts[2:-4])
                return {
                    'power': parts[0],
                    'date': parts[1],
                    'species': species_part,
                    'tree': parts[-4],
                    'cube': parts[-3], 
                    'facet': parts[-2],
                    'point': parts[-1]
                }
        except Exception:
            return {'species': 'unknown', 'tree': '1', 'cube': '1', 'facet': 'C', 'point': '1'}
    
    def preprocess_spectrum(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply preprocessing pipeline to spectrum.
        
        1. Savitzky-Golay smoothing
        2. Morphological baseline correction
        3. Robust normalization
        """
        try:
            # Step 1: Savitzky-Golay smoothing
            smoothed = signal.savgol_filter(intensities, window_length=15, polyorder=2)
            
            # Step 2: Morphological baseline correction
            structure_element = np.ones(50)
            baseline = grey_opening(smoothed, structure=structure_element)
            baseline_corrected = smoothed - baseline
            baseline_corrected = np.maximum(baseline_corrected, 0.1 * np.mean(smoothed))
            
            # Step 3: Robust normalization (SNV)
            normalized = (baseline_corrected - np.mean(baseline_corrected)) / (np.std(baseline_corrected) + 1e-8)
            
            return {
                'wavenumbers': wavenumbers,
                'raw': intensities,
                'smoothed': smoothed,
                'baseline_corrected': baseline_corrected,
                'normalized': normalized
            }
            
        except Exception as e:
            print(f"⚠️ Preprocessing failed: {e}")
            # Fallback: simple normalization
            normalized = (intensities - np.mean(intensities)) / (np.std(intensities) + 1e-8)
            return {
                'wavenumbers': wavenumbers,
                'raw': intensities,
                'smoothed': intensities,
                'baseline_corrected': intensities,
                'normalized': normalized
            }
    
    def find_tree_files(self, species: str, tree_id: str) -> List[Path]:
        """Find all spectrum files for a specific species and tree."""
        files = []
        
        # Check both train and test directories
        for base_path in [self.train_path, self.test_path]:
            species_path = base_path / species
            
            if species_path.exists():
                # Find files matching the tree ID
                pattern = f"*-{species}-{tree_id}-*.txt"
                matching_files = list(species_path.glob(pattern))
                files.extend(matching_files)
        
        return sorted(files)
    
    def create_tree_average(self, species: str, tree_id: str, use_preprocessing: bool = True) -> Dict:
        """
        Create average spectrum from all data points of a specific tree.
        
        Args:
            species: Species name (e.g., 'Acacia_Confusa')
            tree_id: Tree identifier (e.g., '1' or '2')
            use_preprocessing: Whether to apply preprocessing pipeline
            
        Returns:
            Dictionary with average spectrum data and metadata
        """
        print(f"🔄 Creating average spectrum for {species} Tree {tree_id}...")
        
        # Find all files for this tree
        tree_files = self.find_tree_files(species, tree_id)
        
        if not tree_files:
            print(f"❌ No files found for {species} Tree {tree_id}")
            return None
        
        print(f"   📁 Found {len(tree_files)} spectrum files")
        
        # Load and process all spectra
        all_wavenumbers = None
        raw_spectra = []
        processed_spectra = []
        metadata_records = []
        
        successful_loads = 0
        
        for file_path in tree_files:
            wavenumbers, intensities = self.load_spectrum(file_path)
            
            if wavenumbers is not None:
                # Store reference wavenumbers from first successful load
                if all_wavenumbers is None:
                    all_wavenumbers = wavenumbers
                
                # Parse metadata from filename
                parsed = self.parse_filename(file_path.name)
                metadata_records.append({
                    'filename': file_path.name,
                    'species': species,
                    'tree': tree_id,
                    'cube': parsed['cube'],
                    'facet': parsed['facet'],
                    'point': parsed['point'],
                    'filepath': str(file_path)
                })
                
                # Store raw spectrum
                raw_spectra.append(intensities)
                
                # Apply preprocessing if requested
                if use_preprocessing:
                    processed = self.preprocess_spectrum(wavenumbers, intensities)
                    processed_spectra.append(processed['normalized'])
                else:
                    processed_spectra.append(intensities)
                
                successful_loads += 1
        
        if successful_loads == 0:
            print(f"❌ No spectra could be loaded successfully")
            return None
        
        print(f"   ✅ Successfully loaded {successful_loads} spectra")
        
        # Convert to numpy arrays
        raw_spectra = np.array(raw_spectra)
        processed_spectra = np.array(processed_spectra)
        
        # Calculate averages and statistics
        raw_mean = np.mean(raw_spectra, axis=0)
        raw_std = np.std(raw_spectra, axis=0)
        
        processed_mean = np.mean(processed_spectra, axis=0)
        processed_std = np.std(processed_spectra, axis=0)
        
        # Taxonomic classification
        taxonomic_group = 'Hardwood' if species in self.hardwood_species else 'Softwood'
        
        # Count different measurement types
        cube_counts = pd.DataFrame(metadata_records)['cube'].value_counts()
        facet_counts = pd.DataFrame(metadata_records)['facet'].value_counts()
        
        result = {
            'species': species,
            'tree_id': tree_id,
            'taxonomic_group': taxonomic_group,
            'wavenumbers': all_wavenumbers,
            'raw_spectra': raw_spectra,
            'processed_spectra': processed_spectra,
            'raw_mean': raw_mean,
            'raw_std': raw_std,
            'processed_mean': processed_mean,
            'processed_std': processed_std,
            'n_spectra': successful_loads,
            'metadata': metadata_records,
            'cube_counts': cube_counts.to_dict(),
            'facet_counts': facet_counts.to_dict(),
            'use_preprocessing': use_preprocessing
        }
        
        return result
    
    def plot_tree_average(self, tree_data: Dict, save_path: Optional[str] = None, 
                         show_individual: bool = False, max_individual: int = 20) -> plt.Figure:
        """
        Create comprehensive visualization of tree average spectrum.
        
        Args:
            tree_data: Result from create_tree_average()
            save_path: Optional path to save the plot
            show_individual: Whether to show individual spectra
            max_individual: Maximum number of individual spectra to show
            
        Returns:
            matplotlib Figure object
        """
        if tree_data is None:
            print("❌ No data to plot")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'{tree_data["species"].replace("_", " ")} Tree {tree_data["tree_id"]} - Average Spectrum Analysis\n'
            f'{tree_data["taxonomic_group"]} | {tree_data["n_spectra"]} Spectra',
            fontsize=16, fontweight='bold'
        )
        
        wavenumbers = tree_data['wavenumbers']
        
        # Plot 1: Raw spectra overview
        ax1 = axes[0, 0]
        
        if show_individual:
            n_show = min(max_individual, len(tree_data['raw_spectra']))
            for i in range(n_show):
                ax1.plot(wavenumbers, tree_data['raw_spectra'][i], 
                        alpha=0.3, color='gray', linewidth=0.5)
        
        # Plot raw average with error bands
        ax1.plot(wavenumbers, tree_data['raw_mean'], 'b-', linewidth=1, label='Raw Average')
        ax1.fill_between(wavenumbers, 
                        tree_data['raw_mean'] - tree_data['raw_std'],
                        tree_data['raw_mean'] + tree_data['raw_std'],
                        alpha=0.3, color='blue')
        
        ax1.set_title('Raw Spectra with Average')
        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Intensity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Processed spectra overview
        ax2 = axes[0, 1]
        
        if show_individual:
            n_show = min(max_individual, len(tree_data['processed_spectra']))
            for i in range(n_show):
                ax2.plot(wavenumbers, tree_data['processed_spectra'][i], 
                        alpha=0.3, color='gray', linewidth=0.5)
        
        # Plot processed average with error bands
        ax2.plot(wavenumbers, tree_data['processed_mean'], 'r-', linewidth=1, label='Processed Average')
        ax2.fill_between(wavenumbers, 
                        tree_data['processed_mean'] - tree_data['processed_std'],
                        tree_data['processed_mean'] + tree_data['processed_std'],
                        alpha=0.3, color='red')
        
        ax2.set_title('Processed Spectra with Average')
        ax2.set_xlabel('Wavenumber (cm⁻¹)')
        ax2.set_ylabel('Normalized Intensity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Comparison of raw vs processed average
        ax3 = axes[1, 0]
        ax3.plot(wavenumbers, tree_data['raw_mean'], 'b-', linewidth=1, label='Raw Average', alpha=0.7)
        ax3.plot(wavenumbers, tree_data['processed_mean'], 'r-', linewidth=1, label='Processed Average')
        
        ax3.set_title('Raw vs Processed Average Comparison')
        ax3.set_xlabel('Wavenumber (cm⁻¹)')
        ax3.set_ylabel('Intensity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Variability analysis (coefficient of variation)
        ax4 = axes[1, 1]
        
        # Calculate coefficient of variation
        cv_raw = tree_data['raw_std'] / (np.abs(tree_data['raw_mean']) + 1e-8)
        cv_processed = tree_data['processed_std'] / (np.abs(tree_data['processed_mean']) + 1e-8)
        
        ax4.plot(wavenumbers, cv_raw, 'b-', linewidth=1, label='Raw CV', alpha=0.7)
        ax4.plot(wavenumbers, cv_processed, 'r-', linewidth=1, label='Processed CV')
        
        ax4.set_title('Coefficient of Variation (Stability)')
        ax4.set_xlabel('Wavenumber (cm⁻¹)')
        ax4.set_ylabel('CV (std/mean)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, np.percentile(cv_processed, 95))  # Limit y-axis for better visibility
        
        plt.tight_layout()
        
        # Add metadata text box
        metadata_text = f"""Sample Composition:
        • Total Spectra: {tree_data['n_spectra']}
        • Cubes: {', '.join([f"{k}({v})" for k, v in tree_data['cube_counts'].items()])}
        • Facets: {', '.join([f"{k}({v})" for k, v in tree_data['facet_counts'].items()])}
        • Preprocessing: {'Applied' if tree_data['use_preprocessing'] else 'Not Applied'}
        """
        
        fig.text(0.02, 0.02, metadata_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        
        return fig
    
    def compare_multiple_trees(self, species_tree_pairs: List[Tuple[str, str]], 
                              use_preprocessing: bool = True, save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare average spectra from multiple trees.
        
        Args:
            species_tree_pairs: List of (species, tree_id) tuples
            use_preprocessing: Whether to use preprocessed spectra
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        fig.suptitle('Tree Average Spectra Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(species_tree_pairs)))
        
        for i, (species, tree_id) in enumerate(species_tree_pairs):
            print(f"🔄 Processing {species} Tree {tree_id}...")
            
            tree_data = self.create_tree_average(species, tree_id, use_preprocessing)
            
            if tree_data is not None:
                wavenumbers = tree_data['wavenumbers']
                
                if use_preprocessing:
                    spectrum = tree_data['processed_mean']
                    std = tree_data['processed_std']
                else:
                    spectrum = tree_data['raw_mean']
                    std = tree_data['raw_std']
                
                label = f"{species.replace('_', ' ')} T{tree_id} (n={tree_data['n_spectra']})"
                
                # Plot average spectrum
                ax1.plot(wavenumbers, spectrum, linewidth=1, color=colors[i], label=label)
                ax1.fill_between(wavenumbers, spectrum - std, spectrum + std, 
                               alpha=0.2, color=colors[i])
                
                # Plot taxonomic classification
                taxonomic_group = tree_data['taxonomic_group']
                ax2.plot(wavenumbers, spectrum, linewidth=1, color=colors[i], 
                        label=f"{taxonomic_group}: {label}", 
                        linestyle='-' if taxonomic_group == 'Hardwood' else '--')
        
        # Format first subplot
        ax1.set_title('All Tree Averages')
        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Normalized Intensity' if use_preprocessing else 'Intensity')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Format second subplot (taxonomic view)
        ax2.set_title('Taxonomic Classification View')
        ax2.set_xlabel('Wavenumber (cm⁻¹)')
        ax2.set_ylabel('Normalized Intensity' if use_preprocessing else 'Intensity')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comparison plot saved to: {save_path}")
        
        return fig

def main():
    """
    Example usage of the TreeAverageAnalyzer.
    """
    # Initialize analyzer
    analyzer = TreeAverageAnalyzer()
    
    # Example 1: Create average for a single tree
    print("=" * 60)
    print("🌳 EXAMPLE 1: Single Tree Average")
    print("=" * 60)
    
    # Create average for Acacia_Confusa Tree 1
    tree_data = analyzer.create_tree_average('Acacia_Confusa', '1', use_preprocessing=True)
    
    if tree_data:
        print(f"\n📊 Summary for {tree_data['species']} Tree {tree_data['tree_id']}:")
        print(f"   • Taxonomic Group: {tree_data['taxonomic_group']}")
        print(f"   • Total Spectra: {tree_data['n_spectra']}")
        print(f"   • Cubes: {tree_data['cube_counts']}")
        print(f"   • Facets: {tree_data['facet_counts']}")
        
        # Create visualization
        fig = analyzer.plot_tree_average(
            tree_data, 
            save_path="tree_average_plot.png",
            show_individual=True,
            max_individual=10
        )
        
        plt.show()
    
    # Example 2: Compare multiple trees
    print("\n" + "=" * 60)
    print("🌳 EXAMPLE 2: Multiple Trees Comparison")
    print("=" * 60)
    
    # Compare trees from different species
    species_tree_pairs = [
        ('Acacia_Confusa', '1'),
        ('Chamaecyparis_Obtusa_var_Formosana', '1'),
        ('Chamaecyparis_Obtusa_var_Formosana', '2'),  # Test set
        ('Swietenia_Marophylla', '1'),
    ]
    
    fig_comparison = analyzer.compare_multiple_trees(
        species_tree_pairs,
        use_preprocessing=True,
        save_path="trees_comparison.png"
    )
    
    plt.show()
    
    print("\n✅ Analysis complete!")
    print("📁 Plots saved as 'tree_average_plot.png' and 'trees_comparison.png'")

if __name__ == "__main__":
    main()