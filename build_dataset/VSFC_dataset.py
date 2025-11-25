import json
import torch
import re
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import Dataset

class VocabVSFC:
    def __init__(self, path, min_freq=2):
        counter = Counter()
        labels = set()

        data = json.load(open(path, "r", encoding="utf-8"))
        for item in data:
            sentence = self.preprocess_sentence(item["sentence"])
            counter.update(sentence.split())
            labels.add(item["topic"])

        self.pad = "<pad>"
        self.unk = "<unk>"
        self.bos = "<s>"

        self.w2i = {
            self.pad: 0,
            self.unk: 1,
            self.bos: 2,
        }

        for word, freq in counter.items():
            if freq >= min_freq:
                self.w2i[word] = len(self.w2i)

        self.i2w = {i: w for w, i in self.w2i.items()}

        self.label2id = {l: i for i, l in enumerate(sorted(labels))}
        self.id2label = {i: l for l, i in self.label2id.items()}
    
    def preprocess_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub(r"[^0-9a-zA-Zà-ỹÀ-Ỹ ]", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        return sentence

    def encode_sentence(self, sentence):
        sentence = self.preprocess_sentence(sentence)
        words = [self.bos] + sentence.split()
        ids = [self.w2i.get(w, self.w2i[self.unk]) for w in words]
        return torch.tensor(ids).long()

    def encode_label(self, label):
        return torch.tensor(self.label2id[label]).long()

    def decode_label(self, idx):
        return self.id2label[idx]
    
    @property
    def vocab_size(self):
        return len(self.w2i)
    
    @property
    def num_labels(self):
        return len(self.label2id)

class VSFCDataset(Dataset):
    def __init__(self, path, vocab):
        self.data = json.load(open(path, "r", encoding="utf-8"))
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        input_ids = self.vocab.encode_sentence(item["sentence"])
        label_id = self.vocab.encode_label(item["topic"])

        return {
            "input_ids": input_ids,
            "label": label_id
        }


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch], dim=0)

    max_len = max(len(x) for x in input_ids)
    padded = []

    for seq in input_ids:
        padded.append(F.pad(seq, (0, max_len - len(seq)), value=0).unsqueeze(0))

    input_ids = torch.cat(padded, dim=0)

    return {
        "input_ids": input_ids,
        "label": labels
    }
