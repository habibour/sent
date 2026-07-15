"""BERT+GRU sentiment model, ported from Codes/2-class/BERT_w_GRU.ipynb.

The original notebook uses the legacy torchtext.data (Field/TabularDataset/
BucketIterator) API, which was removed from modern torchtext and has no
Python 3.12 wheel. This module keeps the model architecture and training
hyperparameters identical to the notebook, but replaces the data pipeline
with a plain torch.utils.data.Dataset/DataLoader so it runs on current
torch/transformers (e.g. on Kaggle).
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

SEED = 1234
MAX_INPUT_LENGTH = 400
BATCH_SIZE = 32
HIDDEN_DIM = 300
OUTPUT_DIM = 2
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-6
N_EPOCHS = 20
BERT_MODEL_NAME = 'bert-base-multilingual-cased'


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_valid_split(df, split_ratio: float = 0.85, seed: int = SEED):
    """Equivalent to the notebook's train_data.split(split_ratio=0.85, ...)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * split_ratio)
    train_idx, valid_idx = idx[:cut], idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)


class SentimentDataset(Dataset):
    """Replicates the notebook's tokenize_and_cut + [CLS]...[SEP] encoding."""

    def __init__(self, texts, labels, tokenizer, max_input_length: int = MAX_INPUT_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.tokenize(str(self.texts[idx]))
        tokens = tokens[:self.max_input_length - 2]
        ids = ([self.tokenizer.cls_token_id]
               + self.tokenizer.convert_tokens_to_ids(tokens)
               + [self.tokenizer.sep_token_id])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def make_collate_fn(pad_token_id: int):
    """Dynamic per-batch padding, replacing BucketIterator's batching."""

    def collate_fn(batch):
        ids, labels = zip(*batch)
        padded = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
        return padded, torch.stack(labels)

    return collate_fn


class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, batch_first=True,
                           dropout=0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.out(hidden)


def freeze_bert(model: BERTGRUSentiment) -> None:
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rounded_preds = torch.max(preds, 1)[1]
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)


def train_epoch(model, loader, optimizer, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.train()
    for text, labels in loader:
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, loader, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    all_predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for text, labels in loader:
            text, labels = text.to(device), labels.to(device)
            predictions = model(text)
            rounded = torch.max(predictions, 1)[1]
            all_predictions.extend(rounded.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader), all_predictions, targets


def epoch_time(start_time: float, end_time: float):
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)
    return mins, secs
