import shutil
from pathlib import Path
import random
import math
from collections import defaultdict
from tqdm import tqdm
import pandas as pd


def create_and_split_dataset(
        raw_source_dir: str,
        output_base_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
):
    """
    Performs a full data preparation pipeline:
    1. Reads raw files with arbitrary string names.
    2. Renames them sequentially into a new 'full/raw' directory.
    3. Creates a mapping.xlsx file.
    4. Performs a stratified train/val/test split.
    5. Updates mapping.xlsx with the split information.

    Args:
        raw_source_dir (str): Path to the folder with original, unsplit files.
        output_base_dir (str): Path for the final processed dataset structure.
        train_ratio (float): Proportion for the training set.
        val_ratio (float): Proportion for the validation set.
    """
    source_path = Path(raw_source_dir)
    base_path = Path(output_base_dir)
    renamed_unsplit_path = base_path / "full" / "raw"
    excel_path = base_path / "mapping.xlsx"

    # --- Initial Checks and Cleanup ---
    if not source_path.is_dir():
        print(f"Error: Raw source directory not found at '{source_path.resolve()}'")
        return

    base_path.mkdir(parents=True, exist_ok=True)
    print(f"Created new output directory: '{base_path.resolve()}'")

    # === STEP 1: RENAME FILES and CREATE INITIAL MAPPING ===
    print("\nStep 1: Renaming files and creating initial mapping...")

    all_raw_files = sorted(list(source_path.rglob("*.txt")), key=lambda p: p.name)
    if not all_raw_files:
        print(f"Error: No .txt files found in '{source_path.resolve()}'")
        return

    mapping_data = []
    file_counter = 1
    for raw_file_path in tqdm(all_raw_files, desc="Renaming files"):
        new_filename = f"{file_counter}.txt"
        label = raw_file_path.parent.name

        dest_dir = renamed_unsplit_path / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(raw_file_path, dest_dir / new_filename)

        mapping_data.append({
            "Original Name": raw_file_path.name,
            "New Name": new_filename,
            "Label": label,
        })
        file_counter += 1

    df_mapping = pd.DataFrame(mapping_data)
    df_mapping.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Initial mapping.xlsx created with {len(df_mapping)} entries.")

    # === STEP 2: PERFORM STRATIFIED SPLIT ===
    print("\nStep 2: Performing 80/10/10 stratified split...")

    output_paths = {
        "train": base_path / "train" / "raw",
        "validation": base_path / "validation" / "raw",
        "test": base_path / "test" / "raw",
    }
    for path in output_paths.values():
        path.mkdir(parents=True)

    file_to_split_map = {}
    total_counts = defaultdict(int)
    label_dirs = [d for d in renamed_unsplit_path.iterdir() if d.is_dir()]

    for label_dir in tqdm(label_dirs, desc="Splitting labels"):
        label_name = label_dir.name
        files = list(label_dir.glob("*.txt"))
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        split_files = {
            "train": files[:n_train],
            "validation": files[n_train: n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, file_list in split_files.items():
            destination_dir = output_paths[split_name] / label_name
            destination_dir.mkdir(exist_ok=True)
            total_counts[split_name] += len(file_list)
            for file_path in file_list:
                file_to_split_map[file_path.name] = split_name
                shutil.copy(file_path, destination_dir)

    # === STEP 3: UPDATE MAPPING FILE WITH SPLIT INFO ===
    print("\nStep 3: Updating mapping.xlsx with split information...")
    df_mapping['split'] = df_mapping['New Name'].map(file_to_split_map)
    df_mapping.to_excel(excel_path, index=False, engine='openpyxl')

    print("\nProcessing complete!")
    print("\nFinal distribution:")
    for split, count in total_counts.items():
        print(f"  - {split.capitalize()} set: {count} files")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # 1. The folder with your original .txt files ("Joao001.txt", etc.)
    #    IMPORTANT: This folder must exist and contain your data.
    RAW_DATA_FOLDER = "data/datasets/STF_HC/original"

    # 2. The base folder where the final, processed dataset will be created.
    #    This folder will contain train/, validation/, test/, and mapping.xlsx.
    FINAL_DATASET_FOLDER = "data/datasets/STF_HC"

    # --- EXECUTION ---
    # Run the full pipeline
    create_and_split_dataset(
        raw_source_dir=RAW_DATA_FOLDER,
        output_base_dir=FINAL_DATASET_FOLDER
    )