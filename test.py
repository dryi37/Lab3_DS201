import json
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from build_dataset.VSFC_dataset import VSFCDataset, VocabVSFC, collate_fn

from model.lstm import LSTMClassifier
from model.gru import GRUClassifier

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
        raise ValueError(f"Model '{name}' chưa được hỗ trợ")

def load_model(cfg, vocab, device):
    model = build_model(cfg, vocab).to(device)

    ckpt = torch.load(cfg["train"]["save_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model

def predict_dataset(model, loader, device):
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)

            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return true_labels, pred_labels

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (VSFC Test Set)")
    plt.tight_layout()
    plt.show()


def main():
    cfg = yaml.safe_load(open("config/lstm.yaml", "r", encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = VocabVSFC(path=cfg["data"]["train"])
    print("Vocab size:", vocab.vocab_size)
    print("Num labels:", vocab.num_labels)

    test_dataset = VSFCDataset(cfg["data"]["test"], vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = load_model(cfg, vocab, device)

    true_labels, pred_labels = predict_dataset(model, test_loader, device)

    f1 = f1_score(true_labels, pred_labels, average="weighted")
    print("Test F1:", f1)
    print("\n")

    class_names = [vocab.id2label[i] for i in range(vocab.num_labels)]
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    plot_confusion_matrix(true_labels, pred_labels, class_names)


if __name__ == "__main__":
    main()
