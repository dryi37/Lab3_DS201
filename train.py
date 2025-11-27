import os
import yaml
import argparse
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from build_dataset.VSFC_dataset import VSFCDataset, VocabVSFC, collate_fn
from model.lstm import LSTMClassifier
from model.gru import GRUClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    return parser.parse_args()


def build_model(cfg, vocab):
    m = cfg["model"]
    name = m["name"].lower()
    if name == "lstm":
        return LSTMClassifier(
            vocab_size=vocab.vocab_size,
            hidden_size=m["hidden_size"],
            n_layers=m["num_layers"],
            n_labels=vocab.num_labels,
            dropout=m.get("dropout", 0.0),
        )
    elif name == "gru":
        return GRUClassifier(
            vocab_size=vocab.vocab_size,
            hidden_size=m["hidden_size"],
            n_layers=m["num_layers"],
            n_labels=vocab.num_labels,
            dropout=m.get("dropout", 0.0),
        )
    else:
        raise ValueError("Invalid model name in config")

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    all_preds, all_labels = [], []
    losses = []

    for batch in tqdm(loader, desc="train"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = sum(losses) / len(losses)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, f1


def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    losses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            losses.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = sum(losses) / len(losses)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, f1

# def compute_class_weights(train_file, vocab, device):

#     with open(train_file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     labels = [vocab.encode_label(item["topic"]).item() for item in data]

#     counter = Counter(labels)
#     num_classes = vocab.num_labels
#     total = sum(counter.values())

#     weights = [total / (num_classes * counter[i]) for i in range(num_classes)]
#     weights = torch.tensor(weights, dtype=torch.float).to(device)

#     print("Class weights:", weights)

#     return weights

def main():
    args = get_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = VocabVSFC(path=cfg["data"]["train"])

    print("Vocab size:", vocab.vocab_size)
    print("Num labels:", vocab.num_labels)

    train_dataset = VSFCDataset(cfg["data"]["train"], vocab)
    dev_dataset = VSFCDataset(cfg["data"]["dev"], vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Build model
    model = build_model(cfg, vocab).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    # class_weights = compute_class_weights(cfg["data"]["train"], vocab, device)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    no_improve = 0
    patience = cfg["train"]["patience"]
    save_path = cfg["train"]["save_path"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_f1 = eval_epoch(model, dev_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Dev Loss:   {dev_loss:.4f} | Dev F1:   {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            no_improve = 0

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            print(f"[INFO] New best model saved! F1 = {best_f1:.4f}")

        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n[INFO] Early stopping! Best F1 = {best_f1:.4f}")
            break


if __name__ == "__main__":
    main()
