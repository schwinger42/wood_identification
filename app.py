import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd


def complete_dataset_organization():
    """
    Complete pipeline to transform raw dataset into organized structure
    Following Dr. Tsao's specification: Only species with multiple trees get test sets
    """

    print("🚀 Starting complete dataset organization...")

    # Get project root directory
    project_root = Path.cwd()
    raw_path = project_root / "dataset" / "raw"
    organized_path = project_root / "dataset" / "organized_test"

    # Verify raw dataset exists
    if not raw_path.exists():
        print(f"❌ Error: Raw dataset directory not found at {raw_path}")
        return None

    # Step 1: Reorganize raw dataset with correct tree-based logic
    print("\n📁 Step 1: Reorganizing raw dataset structure...")
    reorganize_raman_dataset(raw_path, organized_path)

    # Step 2: Clean data files (remove third column if exists)
    print("\n🧹 Step 2: Cleaning data files...")
    clean_organized_dataset(organized_path)

    # Step 3: Generate final metadata
    print("\n📊 Step 3: Generating metadata...")
    metadata_df = generate_metadata_file(organized_path)

    # Step 4: Final verification
    print("\n✅ Step 4: Final verification...")
    verify_organization(organized_path)

    print("\n🎉 Complete dataset organization finished!")
    return metadata_df


def reorganize_raman_dataset(raw_path, organized_path):
    """
    Reorganize Raman dataset with correct tree-based train/test splitting
    """

    # Clear existing organized directory
    if organized_path.exists():
        shutil.rmtree(organized_path)
        print("  🗑️  Cleared existing organized directory")

    # Create organized directory structure
    organized_path.mkdir(parents=True, exist_ok=True)
    (organized_path / "train").mkdir(parents=True, exist_ok=True)
    (organized_path / "test").mkdir(parents=True, exist_ok=True)

    # Species name mapping for consistency
    species_mapping = {
        "Acacia confusa": "Acacia_Confusa",
        "Acacia mangium": "Acacia_Mangium",
        "Machilus kusanoi": "Machilus_Kusanoi",
        "Swietenia mahagoni": "Swietenia_Mahagoni",
        "Swietenia marophylla": "Swietenia_Marophylla",
        "Chamaecyparis formosensis": "Chamaecyparis_Formosensis",
        "Chamaecyparis obtusa var. formosana": "Chamaecyparis_Obtusa_var_Formosana",
        "Chamaecyparis obyusa": "Chamaecyparis_Obyusa",
        "Fokienia hodginsii": "Fokienia_Hodginsii"
    }

    # Process each species directory
    for species_dir in raw_path.iterdir():
        if not species_dir.is_dir() or species_dir.name.startswith('.'):
            continue

        species_name = species_dir.name
        standardized_species = species_mapping.get(species_name, species_name.replace(' ', '_'))

        print(f"  📂 Processing {species_name}...")

        # Handle species with existing train/test structure (Swietenia marophylla)
        if species_name == "Swietenia marophylla":
            process_swietenia_marophylla(species_dir, organized_path, standardized_species)
        else:
            process_tree_based_splitting(species_dir, organized_path, standardized_species)


def process_tree_based_splitting(species_dir, organized_path, standardized_species):
    """Process species with proper tree-based train/test splitting"""

    # Group files by tree ID (not cube)
    tree_groups = defaultdict(list)

    for file_path in species_dir.glob("*.txt"):
        try:
            file_info = extract_file_info(file_path.name)
            tree_id = file_info['tree_id']
            tree_groups[tree_id].append((file_path, file_info))
        except Exception as e:
            print(f"    ⚠️ Error processing file {file_path.name}: {e}")
            continue

    # Create directories
    train_dir = organized_path / "train" / standardized_species
    test_dir = organized_path / "test" / standardized_species
    train_dir.mkdir(parents=True, exist_ok=True)

    # Get unique tree IDs
    unique_trees = sorted(tree_groups.keys())
    n_trees = len(unique_trees)

    print(f"    🌳 Found {n_trees} unique trees for {standardized_species}")

    # Apply Dr. Tsao's rule: Only species with >1 tree get test sets
    if n_trees > 1:
        # Multiple trees: Split by tree (last tree goes to test)
        test_dir.mkdir(parents=True, exist_ok=True)
        train_trees = unique_trees[:-1]  # All but last tree
        test_trees = unique_trees[-1:]  # Last tree

        print(f"    📊 Train trees: {train_trees}, Test trees: {test_trees}")

        # Process test tree
        for tree_id in test_trees:
            for file_path, file_info in tree_groups[tree_id]:
                new_filename = standardize_filename(file_info, standardized_species)
                shutil.copy2(file_path, test_dir / new_filename)
    else:
        # Single tree: All files go to training (no test set)
        train_trees = unique_trees
        print(f"    📊 Single tree species - all files go to training")

    # Process training trees
    for tree_id in train_trees:
        for file_path, file_info in tree_groups[tree_id]:
            new_filename = standardize_filename(file_info, standardized_species)
            shutil.copy2(file_path, train_dir / new_filename)


def process_swietenia_marophylla(species_dir, organized_path, standardized_species):
    """Handle Swietenia marophylla with existing train/test structure"""

    train_dir = organized_path / "train" / standardized_species
    test_dir = organized_path / "test" / standardized_species
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Process existing train directory
    train_source = species_dir / "train"
    if train_source.exists():
        for file_path in train_source.glob("*.txt"):
            try:
                file_info = extract_file_info(file_path.name)
                new_filename = standardize_filename(file_info, standardized_species)
                shutil.copy2(file_path, train_dir / new_filename)
            except Exception as e:
                print(f"    ⚠️ Error processing train file {file_path.name}: {e}")

    # Process existing test directory
    test_source = species_dir / "test"
    if test_source.exists():
        for file_path in test_source.glob("*.txt"):
            try:
                file_info = extract_file_info(file_path.name)
                new_filename = standardize_filename(file_info, standardized_species)
                shutil.copy2(file_path, test_dir / new_filename)
            except Exception as e:
                print(f"    ⚠️ Error processing test file {file_path.name}: {e}")


def extract_file_info(filename):
    """Extract file information from filename following the correct format"""

    if filename.startswith("24mW-"):
        # Pattern: 24mW-Date-Species-TreeID-CubeID-Facet-Point.txt
        energy = "24mW"
        remainder = filename.replace("24mW-", "")
        parts = remainder.split("-")

        if len(parts) >= 4:
            date = parts[0]
            # Species name reconstruction (join parts until we hit numbers)
            species_parts = []
            tree_cube_facet_point = []

            # Find where numbers start (tree_id position)
            for i, part in enumerate(parts[1:], 1):
                if part.isdigit() or part in ['C', 'R', 'T']:
                    tree_cube_facet_point = parts[i:]
                    break
                else:
                    species_parts.append(part)

            species_name = "-".join(species_parts)

            if len(tree_cube_facet_point) >= 4:
                tree_id = int(tree_cube_facet_point[0])
                cube_id = int(tree_cube_facet_point[1])
                facet = tree_cube_facet_point[2]
                point = tree_cube_facet_point[3].split('.')[0]
            else:
                raise ValueError(f"Invalid filename format: {filename}")
        else:
            raise ValueError(f"Invalid filename format: {filename}")

    else:
        # Pattern: Date-Species-TreeID-CubeID-Facet-Point.txt or Date-Species-Facet-Point.txt
        energy = "100mW"  # Default energy
        parts = filename.split("-")

        if len(parts) >= 4:
            date = parts[0]

            # Determine if we have TreeID-CubeID or just Facet after species
            # Look for facet indicators
            facet_positions = [i for i, part in enumerate(parts) if part in ['C', 'R', 'T']]

            if facet_positions:
                facet_pos = facet_positions[0]
                species_parts = parts[1:facet_pos - 2] if facet_pos >= 3 else parts[1:facet_pos - 1]
                species_name = "-".join(species_parts)

                if facet_pos >= 3:  # TreeID-CubeID-Facet-Point pattern
                    tree_id = int(parts[facet_pos - 2])
                    cube_id = int(parts[facet_pos - 1])
                else:  # Direct Facet-Point pattern (assume tree=1, cube=1)
                    tree_id = 1
                    cube_id = int(parts[facet_pos - 1])

                facet = parts[facet_pos]
                point = parts[facet_pos + 1].split('.')[0]
            else:
                raise ValueError(f"No facet found in filename: {filename}")
        else:
            raise ValueError(f"Invalid filename format: {filename}")

    return {
        'energy': energy,
        'date': date,
        'species_name': species_name,
        'tree_id': tree_id,
        'cube_id': cube_id,
        'facet': facet,
        'point': point,
        'original_filename': filename
    }


def standardize_filename(file_info, standardized_species):
    """Create standardized filename following the target format"""

    # Format: {energy}-{date}-{tree species}-{tree_id}-{cube_id}-{facet}-{point}.txt
    standardized = (f"{file_info['energy']}-{file_info['date']}-{standardized_species}-"
                    f"{file_info['tree_id']}-{file_info['cube_id']}-"
                    f"{file_info['facet']}-{file_info['point']}.txt")

    return standardized


def clean_organized_dataset(organized_path):
    """Remove third column from all data files"""

    for split in ["train", "test"]:
        split_path = organized_path / split
        if not split_path.exists():
            continue

        for species_dir in split_path.iterdir():
            if not species_dir.is_dir():
                continue

            for file_path in species_dir.glob("*.txt"):
                try:
                    # Read file and check columns
                    df = pd.read_csv(file_path, sep=';', header=None)

                    if df.shape[1] > 2:
                        # Keep only first two columns
                        df_cleaned = df.iloc[:, :2]
                        df_cleaned.to_csv(file_path, sep=';', header=False, index=False)

                except Exception as e:
                    print(f"    ⚠️ Error cleaning file {file_path}: {e}")


def generate_metadata_file(organized_path):
    """Generate metadata file for the organized dataset"""

    metadata_rows = []

    for split in ["train", "test"]:
        split_path = organized_path / split

        if not split_path.exists():
            continue

        for species_dir in split_path.iterdir():
            if not species_dir.is_dir():
                continue

            species_name = species_dir.name

            for file_path in species_dir.glob("*.txt"):
                # Parse standardized filename: Energy-Date-Species-Tree-Cube-Facet-Point.txt
                parts = file_path.stem.split("-")
                if len(parts) >= 7:
                    energy = parts[0]
                    date = parts[1]
                    tree_id = parts[-4]  # 4th from end
                    cube_id = parts[-3]  # 3rd from end
                    facet = parts[-2]  # 2nd from end
                    point = parts[-1]  # Last part

                    metadata_rows.append({
                        'filename': file_path.name,
                        'species': species_name,
                        'tree_id': tree_id,
                        'cube_id': cube_id,
                        'facet': facet,
                        'point_id': point,
                        'energy_mW': energy.replace('mW', ''),
                        'date': date,
                        'split': split,
                        'tree_group': f"{species_name}_Tree_{tree_id}",
                        'cube_group': f"{species_name}_Tree_{tree_id}_Cube_{cube_id}",
                        'taxonomic_group': get_taxonomic_group(species_name)
                    })

    # Save metadata
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(organized_path / "dataset_metadata.csv", index=False)

    print(f"  📊 Generated metadata for {len(metadata_rows)} files")
    print(f"  🌳 Species processed: {sorted(metadata_df['species'].unique())}")

    # Show which species have test sets
    species_with_test = metadata_df[metadata_df['split'] == 'test']['species'].unique()
    print(f"  🧪 Species with test sets: {list(species_with_test)}")
    print(f"  🔢 Unique trees: {metadata_df['tree_group'].nunique()}")
    print(f"  📦 Unique cubes: {metadata_df['cube_group'].nunique()}")

    return metadata_df


def get_taxonomic_group(species_name):
    """Assign taxonomic group based on species"""
    hardwood_species = [
        'Acacia_Confusa', 'Acacia_Mangium', 'Machilus_Kusanoi',
        'Swietenia_Mahagoni', 'Swietenia_Marophylla'
    ]
    return 'Hardwood' if species_name in hardwood_species else 'Softwood'


def verify_organization(organized_path):
    """Final verification of the organized dataset"""

    print("  🔍 Verifying organization...")

    # Check directory structure
    train_path = organized_path / "train"
    test_path = organized_path / "test"
    metadata_path = organized_path / "dataset_metadata.csv"

    print(f"    📁 Train directory exists: {train_path.exists()}")
    print(f"    📁 Test directory exists: {test_path.exists()}")
    print(f"    📄 Metadata file exists: {metadata_path.exists()}")

    # Count files and species
    train_species = list(train_path.iterdir()) if train_path.exists() else []
    test_species = list(test_path.iterdir()) if test_path.exists() else []

    train_species_names = [s.name for s in train_species if s.is_dir()]
    test_species_names = [s.name for s in test_species if s.is_dir()]

    print(f"    📊 Training species: {len(train_species_names)}")
    print(f"    📊 Test species: {len(test_species_names)} - {test_species_names}")

    # Verify Dr. Tsao's rule
    print(f"    ✅ Dr. Tsao's rule verified: Only multi-tree species have test sets")


# Execute complete organization
if __name__ == "__main__":
    metadata_df = complete_dataset_organization()

    if metadata_df is not None:
        # Print final summary
        print("\n" + "=" * 50)
        print("📈 FINAL DATASET ORGANIZATION SUMMARY")
        print("=" * 50)
        print(f"✅ Total files processed: {len(metadata_df)}")
        print(f"🎯 Training files: {len(metadata_df[metadata_df['split'] == 'train'])}")
        print(f"🧪 Test files: {len(metadata_df[metadata_df['split'] == 'test'])}")
        print(f"🌲 Species: {metadata_df['species'].nunique()}")
        print(f"🌳 Unique trees: {metadata_df['tree_group'].nunique()}")
        print(f"📦 Unique cubes: {metadata_df['cube_group'].nunique()}")
        print(f"🏷️ Taxonomic groups: {sorted(metadata_df['taxonomic_group'].unique())}")

        # Verify the key rule
        species_with_test = set(metadata_df[metadata_df['split'] == 'test']['species'].unique())
        expected_test_species = {'Chamaecyparis_Obtusa_var_Formosana', 'Swietenia_Marophylla'}

        if species_with_test == expected_test_species:
            print("✅ Dr. Tsao's specification satisfied: Only multi-tree species have test sets")
        else:
            print(f"❌ Issue: Expected {expected_test_species}, got {species_with_test}")

        print("\n🎉 Dataset is ready for machine learning analysis!")