import os
import torch
import random
import numpy as np
import logging
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from torch.amp import GradScaler, autocast

from sklearn.utils import compute_class_weight
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, PreTrainedTokenizerBase
from torch.optim import AdamW
from transformers import logging as hf_logging

from src.data.preprocessing import preprocessing_legal_pt_voto_relatorio

# Set transformers logging to error to avoid excessive warnings
hf_logging.set_verbosity_error()


# --- Core Functions ---

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: str) -> None:
    """Configure logging to file and console."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


# --- Scalable Dataset Class (Lazy Loading) ---

class TextFolderDataset(Dataset):
    """
    A scalable PyTorch Dataset that loads text files lazily from a folder structure.
    It processes long documents by creating sliding window chunks.

    Structure expected:
    - root_dir/
      - class_A/
        - doc1.txt
        - doc2.txt
      - class_B/
        - doc3.txt
        ...
    """

    def __init__(self,
                 root_dir: str,
                 tokenizer: PreTrainedTokenizerBase,
                 label_encoder: LabelEncoder,
                 max_length: int = 4096,
                 stride: int = 512,
                 max_files_per_class: int = None):

        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.stride = stride

        self.samples = []
        self.labels = []

        root = Path(root_dir)
        label_names = sorted([d.name for d in root.iterdir() if d.is_dir()])

        logging.info(f"Building dataset from {root_dir}...")
        for label_name in tqdm.tqdm(label_names, desc="Scanning files"):
            label_dir = root / label_name
            fpaths = list(label_dir.glob("*.txt"))
            if max_files_per_class:
                fpaths = fpaths[:max_files_per_class]

            for fpath in fpaths:
                # Add a sample for each chunk that can be made from the file
                # This initial tokenization is just to determine the number of chunks
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Use encode to get token IDs directly
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)

                # Create chunk offsets
                for i in range(0, len(token_ids), self.max_length - self.stride):
                    self.samples.append((str(fpath), i))
                    self.labels.append(label_name)

        self.encoded_labels = self.label_encoder.transform(self.labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath, chunk_start_idx = self.samples[idx]
        label = self.encoded_labels[idx]

        # Lazy loading: read and tokenize the file only when requested
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()

        text = preprocessing_legal_pt_voto_relatorio(text)

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=False,  # We handle truncation manually by chunking
            padding=False,
            return_tensors=None  # Return list of ints
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Extract the specific chunk
        chunk_end_idx = chunk_start_idx + self.max_length
        chunk_input_ids = input_ids[chunk_start_idx:chunk_end_idx]
        chunk_attention_mask = attention_mask[chunk_start_idx:chunk_end_idx]

        # Manually pad the chunk to max_length
        padding_length = self.max_length - len(chunk_input_ids)
        if padding_length > 0:
            pad_token_id = self.tokenizer.pad_token_id
            chunk_input_ids += [pad_token_id] * padding_length
            chunk_attention_mask += [0] * padding_length

        return {
            'input_ids': torch.tensor(chunk_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(chunk_attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# --- Training & Evaluation Logic ---
def train(model: torch.nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler,
          device: torch.device,
          class_weights: torch.Tensor,
          max_epochs: int,
          patience: int,
          checkpoint_path: str,
          label_encoder: LabelEncoder) -> torch.nn.Module:
    best_f1 = 0.0
    patience_counter = 0
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Initialize the GradScaler
    scaler = GradScaler()

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm.tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            # Use autocast for the forward pass
            with autocast(device_type=device.type):
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                labels=batch['labels'])
                loss = outputs.loss

            # Scale the loss and call backward
            scaler.scale(loss).backward()

            # scaler.step() un-scales the gradients and calls optimizer.step()
            scaler.step(optimizer)

            # Update the scale for next iteration
            scaler.update()

            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Average Training Loss for Epoch {epoch + 1}: {avg_train_loss:.4f}")

        # --- Validation with Full Metrics ---
        # Pass the label_encoder to get a detailed report every epoch
        val_report = evaluate(model, val_loader, device, label_encoder, phase="Validation")
        macro_f1 = val_report['macro avg']['f1-score']

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            logging.info(f"New best validation F1: {best_f1:.4f}. Saving model to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    logging.info("Loading best model checkpoint for final evaluation.")
    model.load_state_dict(torch.load(checkpoint_path))

    return model


def evaluate(model: torch.nn.Module,
             data_loader: DataLoader,
             device: torch.device,
             label_encoder: LabelEncoder,
             phase: str = "Test") -> Dict[str, Any]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in tqdm.tqdm(data_loader, desc=f"Evaluating [{phase}]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    # Ensure target_names are available for the report
    target_names = label_encoder.classes_ if label_encoder else [str(i) for i in range(len(np.unique(all_labels)))]

    # Generate both the dictionary and the text-formatted report
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, digits=3, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=target_names, digits=3, output_dict=False)

    logging.info(f"\n--- {phase} Set Classification Report ---\n{report_text}")

    return report_dict


# --- Main Execution ---
def main(args: argparse.Namespace):

    setup_logging(args.log_file)
    set_seed(args.seed)

    logging.info(f"Args: {args}")

    logging.info("=== Starting Longformer training ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. Tokenizer and Label Encoder
    logging.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    all_labels = sorted([p.name for p in Path(args.train_dir).iterdir() if p.is_dir()])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_labels = len(label_encoder.classes_)

    # 2. Datasets and DataLoaders
    logging.info("Loading datasets...")
    train_ds = TextFolderDataset(args.train_dir, tokenizer, label_encoder, args.max_length, args.stride,
                                 args.max_files_per_class)
    val_ds = TextFolderDataset(args.val_dir, tokenizer, label_encoder, args.max_length, args.stride)
    test_ds = TextFolderDataset(args.test_dir, tokenizer, label_encoder, args.max_length, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, num_workers=4)

    # 3. Class Weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_ds.encoded_labels),
        y=train_ds.encoded_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    logging.info(f"Using class weights: {class_weights.cpu().numpy().tolist()}")

    # 4. Model Initialization
    logging.info(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        # The model automatically uses the weights when labels are passed
    )

    if args.freeze_base_model:
        logging.info("Freezing base model layers and training only the classifier.")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    else:
        logging.info("Fine-tuning all model layers.")

    model.to(device)

    # 5. Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 6. Training
    logging.info("Starting training...")
    model = train(
        model, train_loader, val_loader, optimizer, scheduler, device,
        class_weights, args.epochs, args.patience, args.checkpoint_path, label_encoder
    )

    # 7. Final Evaluation
    logging.info("Evaluating on the test set...")
    report = evaluate(model, test_loader, device, label_encoder, phase="Test")

    with open(args.results_path, 'w') as f:
        json.dump(report, f, indent=4)
    logging.info(f"Test results saved to {args.results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Transformer for Text Classification")

    # Paths
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the training data directory.")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to the validation data directory.")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to the test data directory.")
    parser.add_argument('--log_file', type=str, default="logs/longformer_training.log", help="Path to save the log file.")
    parser.add_argument('--checkpoint_path', type=str, default="models/long_former_best_model.pt",
                        help="Path to save the best model checkpoint.")
    parser.add_argument('--results_path', type=str, default="results/test_report.json",
                        help="Path to save the final classification report.")

    # Model & Tokenizer
    parser.add_argument('--model_name', type=str, default="allenai/longformer-base-4096",
                        help="Name of the pre-trained model from Hugging Face Hub.")
    parser.add_argument('--max_length', type=int, default=4096, help="Maximum sequence length for the model.")
    parser.add_argument('--stride', type=int, default=512, help="Stride for sliding window chunking of long documents.")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay for the AdamW optimizer.")
    parser.add_argument('--num_warmup_steps', type=int, default=50, help="Number of warmup steps for the scheduler.")
    parser.add_argument('--patience', type=int, default=3, help="Patience for early stopping.")
    parser.add_argument('--freeze_base_model', action='store_true', help="If set, only train the classifier head.")

    # Data Handling
    parser.add_argument('--max_files_per_class', type=int, default=None,
                        help="Maximum number of files to load per class for debugging.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Create directories for outputs if they don't exist
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)

    main(args)