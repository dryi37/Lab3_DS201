import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report

from build_dataset.PhoNER_dataset import PhoNERDataset, VocabNER, ner_collate_fn
from model.bilstm_ner import BiLSTM_NER


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device, vocab):
    model.train()

    losses = []
    all_preds = []
    all_golds = []

    for batch in tqdm(loader, desc="train"):
        input_ids = batch["input_ids"].to(device)
        gold_tags = batch["tags"].to(device)
        mask = batch["mask"].to(device)

        logits = model(input_ids)      # (B, T, num_labels)
        logits = logits.view(-1, logits.size(-1))
        gold_tags_flat = gold_tags.view(-1)

        loss = criterion(logits, gold_tags_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        preds = torch.argmax(model(input_ids), dim=-1).cpu()
        golds = gold_tags.cpu()

        for p, g, m in zip(preds, golds, mask):
            L = m.sum().item()
            p = p[:L].tolist()
            g = g[:L].tolist()

            all_preds.append([vocab.id2tag[i] for i in p])
            all_golds.append([vocab.id2tag[i] for i in g])

    avg_loss = sum(losses) / len(losses)
    f1 = f1_score(all_golds, all_preds, average="macro")

    return avg_loss, f1


def eval_epoch(model, loader, criterion, device, vocab):
    model.eval()

    losses = []
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            input_ids = batch["input_ids"].to(device)
            gold_tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            logits = model(input_ids)
            logits_flat = logits.view(-1, logits.size(-1))
            gold_flat = gold_tags.view(-1)

            loss = criterion(logits_flat, gold_flat)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=-1).cpu()
            golds = gold_tags.cpu()

            for p, g, m in zip(preds, golds, mask):
                L = m.sum().item()
                p = p[:L].tolist()
                g = g[:L].tolist()

                all_preds.append([vocab.id2tag[i] for i in p])
                all_golds.append([vocab.id2tag[i] for i in g])

    avg_loss = sum(losses) / len(losses)
    f1 = f1_score(all_golds, all_preds, average="macro")
    return avg_loss, f1


def main():
    args = get_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = VocabNER(path=cfg["data"]["train"])
    print("Vocab size:", vocab.vocab_size)
    print("Num labels:", vocab.num_labels)

    train_set = PhoNERDataset(cfg["data"]["train"], vocab)
    dev_set = PhoNERDataset(cfg["data"]["dev"], vocab)

    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=ner_collate_fn
    )
    dev_loader = DataLoader(
        dev_set, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=ner_collate_fn
    )

    m = cfg["model"]
    model = BiLSTM_NER(
        vocab_size=vocab.vocab_size,
        hidden_size=m["hidden_size"],
        n_layers=m["num_layers"],
        n_labels=vocab.num_labels,
        dropout=m["dropout"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg["train"]["lr"], 
        weight_decay=cfg["train"]["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_f1 = 0
    patience = cfg["train"]["patience"]
    no_improve = 0
    save_path = cfg["train"]["save_path"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, vocab)
        dev_loss, dev_f1 = eval_epoch(model, dev_loader, criterion, device, vocab)

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Dev   Loss: {dev_loss:.4f} | Dev   F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            print(f"[SAVE] Best model saved, F1 = {best_f1:.4f}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print("\n[STOP] Early stopping triggered!")
            print(f"Best F1 = {best_f1:.4f}")
            break


if __name__ == "__main__":
    main()
