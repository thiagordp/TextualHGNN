import os
import torch
import random
import numpy as np
import logging

from sklearn.utils import compute_class_weight
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from transformers import logging as hf_logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/bert_training.log"),
        logging.StreamHandler()
    ]
)

hf_logging.set_verbosity_error()


# Fix seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Dataset class
class TextFolderDataset(Dataset):
    def __init__(self, root_dir, tokenizer, label_encoder, max_length=4096, stride=512):
        self.samples = []
        self.labels = []
        self.label_names = sorted(os.listdir(root_dir))
        self.label_encoder = label_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        for label in self.label_names:
            label_dir = os.path.join(root_dir, label)

            fnames = list(os.listdir(label_dir))
            for fname in fnames[:5000]:
                fpath = os.path.join(label_dir, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    chunks = self.chunk_text(text)
                    for chunk in chunks:
                        self.samples.append(chunk)
                        self.labels.append(label)

        self.encoded_labels = self.label_encoder.transform(self.labels)

    def chunk_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), self.max_length - self.stride):
            chunk = tokens[i:i + self.max_length]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
            chunks.append(chunk_text)
        return chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.samples[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.encoded_labels[idx])
        return item


# Training loop
def train(model, train_loader, val_loader, optimizer, scheduler, device, class_weights, patience=3, max_epochs=10):
    best_f1 = 0
    patience_counter = 0
    best_model = None

    for epoch in range(max_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.inference_mode():  # instead of torch.no_grad()
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        logging.info(f"Validation Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

    model.load_state_dict(best_model)
    return model


# Evaluation
def evaluate(model, test_loader, device, label_encoder):
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    target_names = label_encoder.classes_
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=3)
    logging.info("\nTest Set Classification Report:\n")
    logging.info(f"\n{report}")


# Main entry point
def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info("Loading tokenizer and encoding labels...")
    model_name = "allenai/longformer-base-4096"
    # model_name = "joelito/legal-xlm-longformer-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_labels = sorted(os.listdir(args.train_dir))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    logging.info("Loading datasets...")
    train_ds = TextFolderDataset(args.train_dir, tokenizer, label_encoder)
    val_ds = TextFolderDataset(args.val_dir, tokenizer, label_encoder)
    test_ds = TextFolderDataset(args.test_dir, tokenizer, label_encoder)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4, num_workers=8, pin_memory=True) 
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 4, num_workers=8, pin_memory=True)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_ds.encoded_labels),
        y=train_ds.encoded_labels
    )

    logging.info(f"Using class weights: {class_weights}")
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    logging.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_encoder.classes_))

    # Freeze all layers except classifier
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    model.to(device)

    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    logging.info(f"Trainable parameters: {trainable_params}")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    logging.info("Starting training...")
    model = train(model, train_loader, val_loader, optimizer, scheduler, device, class_weights, patience=args.patience,
                  max_epochs=args.epochs)

    logging.info("Evaluating on test set...")
    evaluate(model, test_loader, device, label_encoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--patience', type=int, default=1)
    args = parser.parse_args()

    main(args)
