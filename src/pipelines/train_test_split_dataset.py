import logging
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
    from pathlib import Path
    import shutil, random
    import pandas as pd
    from collections import defaultdict
    from tqdm import tqdm

    source_path = Path(raw_source_dir)
    base_path = Path(output_base_dir)
    renamed_unsplit_path = base_path / "full" / "raw"
    excel_path = base_path / "mapping.xlsx"

    if not source_path.is_dir():
        logging.info(f"Error: Raw source directory not found at '{source_path.resolve()}'")
        return

    base_path.mkdir(parents=True, exist_ok=True)

    # === Step 1: Rename files ===
    all_raw_files = sorted(list(source_path.rglob("*.txt")), key=lambda p: p.name)
    if not all_raw_files:
        logging.info("No .txt files found!")
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
    df_mapping.to_excel(excel_path, index=False, engine="openpyxl")
    logging.info(f"Initial mapping.xlsx created with {len(df_mapping)} entries.")

    # === Step 2: Split ===
    output_paths = {
        "train": base_path / "train" / "raw",
        "validation": base_path / "validation" / "raw",
        "test": base_path / "test" / "raw",
    }
    for p in output_paths.values():
        p.mkdir(parents=True, exist_ok=True)

    file_to_split_map = {}
    total_counts = defaultdict(int)

    for label_dir in tqdm(list(renamed_unsplit_path.iterdir()), desc="Splitting labels"):
        if not label_dir.is_dir():
            continue
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
            dest_dir = output_paths[split_name] / label_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            total_counts[split_name] += len(file_list)

            for file_path in file_list:
                # ✅ use label+filename as unique key
                key = f"{label_name}/{file_path.name}"
                file_to_split_map[key] = split_name
                shutil.copy(file_path, dest_dir / file_path.name)

    # === Step 3: Update mapping.xlsx ===
    df_mapping["split"] = df_mapping.apply(
        lambda row: file_to_split_map.get(f"{row['Label']}/{row['New Name']}"), axis=1
    )
    df_mapping.to_excel(excel_path, index=False, engine="openpyxl")

    logging.info("\nProcessing complete!")
    for split, count in total_counts.items():
        logging.info(f"  - {split.capitalize()} set: {count} files")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # 1. The folder with your original .txt files ("Joao001.txt", etc.)
    #    IMPORTANT: This folder must exist and contain your data.
    RAW_DATA_FOLDER = "data/datasets/STF_HC_Voto_Relatorio/original"

    # 2. The base folder where the final, processed dataset will be created.
    #    This folder will contain train/, validation/, test/, and mapping.xlsx.
    FINAL_DATASET_FOLDER = "data/datasets/STF_HC_Voto_Relatorio"

    # --- EXECUTION ---
    # Run the full pipeline
    create_and_split_dataset(
        raw_source_dir=RAW_DATA_FOLDER,
        output_base_dir=FINAL_DATASET_FOLDER
    )