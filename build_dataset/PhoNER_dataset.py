import json
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class VocabNER:
    def __init__(self, path, min_freq=1):

        word_counter = Counter()
        tag_set = set()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    word_counter.update(item["words"])
                    tag_set.update(item["tags"])

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.word2id = {
            self.pad_token: 0,
            self.unk_token: 1
        }
        for w, c in word_counter.items():
            if c >= min_freq:
                self.word2id[w] = len(self.word2id)

        self.id2word = {i: w for w, i in self.word2id.items()}

        self.tag2id = {t: i for i, t in enumerate(sorted(list(tag_set)))}
        self.id2tag = {i: t for t, i in self.tag2id.items()}

    @property
    def vocab_size(self):
        return len(self.word2id)

    @property
    def num_labels(self):
        return len(self.tag2id)

    def encode_words(self, words):
        ids = [self.word2id.get(w, self.word2id[self.unk_token]) for w in words]
        return torch.tensor(ids)

    def encode_tags(self, tags):
        ids = [self.tag2id[t] for t in tags]
        return torch.tensor(ids, dtype=torch.long)

    def decode_tags(self, tag_ids):
        return [self.id2tag[i] for i in tag_ids]

class PhoNERDataset(Dataset):
    def __init__(self, json_path, vocab):
        self.vocab = vocab
        self.data = []

        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        words = item["words"]
        tags = item["tags"]

        input_ids = self.vocab.encode_words(words)
        tag_ids = self.vocab.encode_tags(tags)

        return {"input_ids": input_ids, "tags": tag_ids}

def ner_collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    tags = [b["tags"] for b in batch]

    max_len = max(len(seq) for seq in input_ids)

    padded_inputs = []
    padded_tags = []
    masks = []

    for inp, tag in zip(input_ids, tags):
        pad_len = max_len - len(inp)

        padded_inputs.append(
            F.pad(inp, (0, pad_len), value=0) 
        )

        padded_tags.append(
            F.pad(tag, (0, pad_len), value=-100) 
        )

        masks.append(
            torch.tensor([1] * len(inp) + [0] * pad_len)
        )

    return {
        "input_ids": torch.stack(padded_inputs),  # (B, T)
        "tags": torch.stack(padded_tags),         # (B, T)
        "mask": torch.stack(masks)                # (B, T)
    }
