import json
import torch
import yaml

from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, classification_report

from build_dataset.PhoNER_dataset import VocabNER, PhoNERDataset, ner_collate_fn
from model.bilstm_ner import BiLSTM_NER


def load_model(cfg, vocab, device):
    m = cfg["model"]
    model = BiLSTM_NER(
        vocab_size=vocab.vocab_size,
        hidden_size=m["hidden_size"],
        n_layers=m["num_layers"],
        n_labels=vocab.num_labels,
        dropout=m["dropout"],
    ).to(device)

    ckpt = torch.load(cfg["train"]["save_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


def predict_dataset(model, loader, device, vocab):
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            gold_tags = batch["tags"]
            mask = batch["mask"]

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1).cpu()

            for p, g, m in zip(preds, gold_tags, mask):
                real_len = m.sum().item()
                p = p[:real_len].tolist()
                g = g[:real_len].tolist()

                all_preds.append([vocab.id2tag[i] for i in p])
                all_golds.append([vocab.id2tag[i] for i in g])

    return all_golds, all_preds


def main():
    cfg = yaml.safe_load(open("config/bilstm_ner.yaml", "r", encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    vocab = VocabNER(path=cfg["data"]["train"])
    print("Vocab size:", vocab.vocab_size)
    print("Num labels:", vocab.num_labels)

    test_dataset = PhoNERDataset(cfg["data"]["test"], vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=ner_collate_fn,
    )

    model = load_model(cfg, vocab, device)

    gold, pred = predict_dataset(model, test_loader, device, vocab)

    f1 = f1_score(gold, pred)
    print("\nTest F1:", f"{f1:.4f}\n")

    print(classification_report(gold, pred))


if __name__ == "__main__":
    main()
