"""
Simple test script to validate package structure and imports
Run this to ensure all modules are properly structured
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test main package
        import wood_identification
        print("✓ wood_identification package imported")
        
        # Test data module
        from wood_identification.data import (
            RamanPreprocessor,
            remove_cosmic_rays,
            extract_features,
            RamanDataset,
            load_csv_data,
            load_numpy_data,
            create_dataloaders,
            save_processed_data,
            load_processed_data
        )
        print("✓ wood_identification.data module imported")
        
        # Test models module
        from wood_identification.models import (
            CNN1D,
            ResNet1D,
            SimpleMLP,
            WoodClassifier,
            EnsembleClassifier
        )
        print("✓ wood_identification.models module imported")
        
        # Test utils module
        from wood_identification.utils import (
            plot_spectrum,
            plot_confusion_matrix,
            calculate_metrics,
            MetricsTracker
        )
        print("✓ wood_identification.utils module imported")
        
        print("\nAll imports successful! ✓")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    print("\nTesting basic functionality...")
    
    try:
        # This test doesn't require numpy/torch to be installed
        # It just checks that the module structure is correct
        
        import wood_identification
        print(f"✓ Package version: {wood_identification.__version__}")
        print(f"✓ Package author: {wood_identification.__author__}")
        
        # Check __all__ exports
        expected_exports = [
            'preprocessing', 'data_loader', 
            'cnn_model', 'ml_model',
            'visualization', 'metrics'
        ]
        
        for export in expected_exports:
            if export in wood_identification.__all__:
                print(f"✓ {export} in __all__")
        
        print("\nBasic functionality tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"\n✗ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_file_structure():
    """Check that all expected files exist"""
    print("\nChecking file structure...")
    
    expected_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'config.yaml',
        '.gitignore',
        'LICENSE',
        'train_ml.py',
        'train_dl.py',
        'predict.py',
        'generate_synthetic_data.py',
        'example_workflow.py',
        'wood_identification/__init__.py',
        'wood_identification/data/__init__.py',
        'wood_identification/data/preprocessing.py',
        'wood_identification/data/data_loader.py',
        'wood_identification/models/__init__.py',
        'wood_identification/models/cnn_model.py',
        'wood_identification/models/ml_model.py',
        'wood_identification/utils/__init__.py',
        'wood_identification/utils/visualization.py',
        'wood_identification/utils/metrics.py',
    ]
    
    missing_files = []
    
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n✗ {len(missing_files)} files missing")
        return False
    else:
        print("\nAll expected files exist! ✓")
        return True


def check_syntax():
    """Check Python syntax for all Python files"""
    print("\nChecking Python syntax...")
    
    import py_compile
    
    python_files = [
        'train_ml.py',
        'train_dl.py',
        'predict.py',
        'generate_synthetic_data.py',
        'example_workflow.py',
        'setup.py',
        'wood_identification/__init__.py',
        'wood_identification/data/__init__.py',
        'wood_identification/data/preprocessing.py',
        'wood_identification/data/data_loader.py',
        'wood_identification/models/__init__.py',
        'wood_identification/models/cnn_model.py',
        'wood_identification/models/ml_model.py',
        'wood_identification/utils/__init__.py',
        'wood_identification/utils/visualization.py',
        'wood_identification/utils/metrics.py',
    ]
    
    syntax_errors = []
    
    for file_path in python_files:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"✓ {file_path}")
        except py_compile.PyCompileError as e:
            print(f"✗ {file_path} - SYNTAX ERROR")
            syntax_errors.append((file_path, e))
    
    if syntax_errors:
        print(f"\n✗ {len(syntax_errors)} files have syntax errors")
        for file_path, error in syntax_errors:
            print(f"  {file_path}: {error}")
        return False
    else:
        print("\nAll Python files have valid syntax! ✓")
        return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Wood Identification Package - Structure Validation")
    print("=" * 60)
    
    results = []
    
    # Test file structure
    results.append(("File Structure", check_file_structure()))
    
    # Test syntax
    results.append(("Python Syntax", check_syntax()))
    
    # Note: Imports test requires dependencies installed
    print("\n" + "=" * 60)
    print("Note: Import and functionality tests require dependencies.")
    print("Install with: pip install -r requirements.txt")
    print("=" * 60)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All validation tests passed!")
        print("Package structure is correct.")
        return 0
    else:
        print("\n✗ Some validation tests failed.")
        print("Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
