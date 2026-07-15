"""BanglaBERT + CNN-BiLSTM hybrid architecture -- the accuracy/novelty model.

Unlike src/model.py's frozen-mBERT baseline (kept as-is for the paper
comparison row), this fine-tunes the encoder end-to-end and fuses two views
of its output:

  - the [CLS] pooled representation (global sentence-level signal)
  - a Conv1D -> BiLSTM branch over the full token sequence (local n-gram +
    sequential signal), adapted from the CNN-BiLSTM head described in
    Islam & Alam's BangDSA paper, but applied here to fine-tuned contextual
    embeddings instead of a frozen document vector.

Trained with class-weighted cross-entropy, discriminative learning rates
(lower for the encoder, higher for the head), linear warmup+decay, fp16,
and early stopping on validation macro-F1 (not loss/accuracy), so the
minority class can't be silently ignored.
"""

import copy

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.utils.data import Dataset
from transformers import AutoModel, get_linear_schedule_with_warmup

DEFAULT_MODEL_NAME = 'csebuetnlp/banglabert'
MAX_LEN = 128


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BanglaBertHybrid(nn.Module):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, num_labels: int = 3,
                 conv_filters: int = 128, conv_kernel: int = 3,
                 lstm_hidden: int = 128, fusion_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.conv = nn.Conv1d(hidden_size, conv_filters, kernel_size=conv_kernel,
                               padding=conv_kernel // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(conv_filters, lstm_hidden, num_layers=1,
                             bidirectional=True, batch_first=True)

        fused_dim = hidden_size + lstm_hidden * 2
        self.fc1 = nn.Linear(fused_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state          # [B, L, H]
        cls_repr = seq_output[:, 0, :]                    # [B, H]

        x = seq_output.permute(0, 2, 1)                   # [B, H, L]
        x = self.relu(self.conv(x))                       # [B, C, L]
        x = self.pool(x)                                  # [B, C, L//2]
        x = x.permute(0, 2, 1)                             # [B, L//2, C]
        _, (h_n, _) = self.lstm(x)                         # h_n: [2, B, lstm_hidden]
        lstm_repr = torch.cat((h_n[-2], h_n[-1]), dim=1)   # [B, lstm_hidden*2]

        fused = torch.cat((cls_repr, lstm_repr), dim=1)
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        return self.classifier(x)


def build_optimizer(model: BanglaBertHybrid, encoder_lr: float, head_lr: float,
                     weight_decay: float = 0.01):
    """Discriminative-LR AdamW: encoder gets a small LR (it's already
    pretrained), the new hybrid head gets a much larger one (it starts from
    random init). bias/LayerNorm params are excluded from weight decay,
    standard practice for transformer fine-tuning."""
    no_decay = ['bias', 'LayerNorm.weight']
    encoder_params = list(model.encoder.named_parameters())
    head_params = [(n, p) for n, p in model.named_parameters()
                   if not n.startswith('encoder.')]

    groups = [
        {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in head_params if not any(nd in n for nd in no_decay)],
         'lr': head_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in head_params if any(nd in n for nd in no_decay)],
         'lr': head_lr, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(groups)


def train_epoch(model, loader, optimizer, scheduler, criterion, device,
                 scaler=None, grad_clip: float = 1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_names=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'report': classification_report(all_labels, all_preds,
                                         target_names=label_names, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'predictions': all_preds,
        'targets': all_labels,
    }


def train_with_early_stopping(model, train_loader, val_loader, criterion, device, *,
                               encoder_lr: float = 2e-5, head_lr: float = 1e-3,
                               weight_decay: float = 0.01, warmup_ratio: float = 0.06,
                               epochs: int = 15, patience: int = 3,
                               grad_clip: float = 1.0, use_fp16: bool = True,
                               label_names=None):
    """Full training loop with early stopping on validation macro-F1.
    Returns (model_with_best_weights_loaded, best_val_macro_f1)."""
    optimizer = build_optimizer(model, encoder_lr, head_lr, weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if (use_fp16 and device.type == 'cuda') else None

    best_macro_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                  criterion, device, scaler, grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, device, label_names)

        print(f"Epoch {epoch + 1:02d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_metrics['loss']:.4f} | "
              f"val_acc {val_metrics['accuracy'] * 100:.2f}% | "
              f"val_macro_f1 {val_metrics['macro_f1']:.4f}")

        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping: no val macro-F1 improvement in {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_macro_f1
