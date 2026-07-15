"""BanglaBERT + CNN-BiLSTM hybrid architecture -- the accuracy/novelty model.

Unlike src/model.py's frozen-mBERT baseline (kept as-is for the paper
comparison row), this fine-tunes the encoder end-to-end and fuses two views
of its output:

  - the [CLS] pooled representation (global sentence-level signal)
  - a Conv1D -> BiLSTM branch over the full token sequence (local n-gram +
    sequential signal), adapted from the CNN-BiLSTM head described in
    Islam & Alam's BangDSA paper, but applied here to fine-tuned contextual
    embeddings instead of a frozen document vector.

Trained with class-weighted (+ optionally label-smoothed) cross-entropy,
discriminative learning rates (lower for the encoder, higher for the head),
optional partial encoder freezing, linear warmup+decay, fp16, and early
stopping on validation macro-F1 (not loss/accuracy), so the minority class
can't be silently ignored.

Round-1 Kaggle results (full fine-tune, head_lr=1e-3, dropout=0.3, no
freezing) showed clear overfitting: train loss collapsed from 0.91->0.20
over 6 epochs while val loss rose from 0.87->2.11, and even the best
val-macro-F1 checkpoint (epoch 3, 0.65) generalized poorly to the held-out
test set (0.50 macro-F1, worse than the paper's 60% accuracy on 3-class).
2-class showed a milder version of the same pattern. The changes in this
version (lower head/encoder LR, label smoothing via the criterion,
higher dropout, optional layer freezing, an LR-decay horizon decoupled
from the early-stopping cap, and per-epoch history for plotting) target
that overfitting directly.
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
                 dropout: float = 0.4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.conv = nn.Conv1d(hidden_size, conv_filters, kernel_size=conv_kernel,
                               padding=conv_kernel // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(conv_filters, lstm_hidden, num_layers=1,
                             bidirectional=True, batch_first=True)

        # Extra dropout on each branch before fusion, on top of the dropout
        # before the classifier -- cheap additional regularization against
        # the overfitting seen in round 1.
        self.branch_dropout = nn.Dropout(dropout)

        fused_dim = hidden_size + lstm_hidden * 2
        self.fc1 = nn.Linear(fused_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state          # [B, L, H]
        cls_repr = self.branch_dropout(seq_output[:, 0, :])   # [B, H]

        x = seq_output.permute(0, 2, 1)                   # [B, H, L]
        x = self.relu(self.conv(x))                       # [B, C, L]
        x = self.pool(x)                                  # [B, C, L//2]
        x = x.permute(0, 2, 1)                             # [B, L//2, C]
        _, (h_n, _) = self.lstm(x)                         # h_n: [2, B, lstm_hidden]
        lstm_repr = torch.cat((h_n[-2], h_n[-1]), dim=1)   # [B, lstm_hidden*2]
        lstm_repr = self.branch_dropout(lstm_repr)

        fused = torch.cat((cls_repr, lstm_repr), dim=1)
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        return self.classifier(x)


def freeze_encoder_layers(model: BanglaBertHybrid, num_layers: int) -> None:
    """Freeze the embeddings + the bottom `num_layers` transformer layers of
    the encoder, leaving the top layers + hybrid head trainable.

    Full end-to-end fine-tuning of all 12 BanglaBERT layers on a ~9-12k
    example dataset is prone to overfitting (round-1 results). Freezing the
    lower layers -- which mostly encode generic syntax/morphology rather
    than task-specific sentiment signal -- reduces effective trainable
    capacity without giving up fine-tuning entirely.
    """
    for param in model.encoder.embeddings.parameters():
        param.requires_grad = False
    layers = model.encoder.encoder.layer
    for layer in layers[:num_layers]:
        for param in layer.parameters():
            param.requires_grad = False
    print(f"[freeze] froze embeddings + bottom {num_layers}/{len(layers)} encoder layers")


def build_optimizer(model: BanglaBertHybrid, encoder_lr: float, head_lr: float,
                     weight_decay: float = 0.01):
    """Discriminative-LR AdamW: encoder gets a small LR (it's already
    pretrained), the new hybrid head gets a larger one (it starts from
    random init). bias/LayerNorm params are excluded from weight decay,
    standard practice for transformer fine-tuning. Frozen params (see
    freeze_encoder_layers) are skipped entirely so they don't sit in the
    optimizer doing nothing."""
    no_decay = ['bias', 'LayerNorm.weight']
    encoder_params = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_params = [(n, p) for n, p in model.named_parameters()
                   if not n.startswith('encoder.') and p.requires_grad]

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
    # Drop empty groups (e.g. if everything in one bucket was frozen).
    groups = [g for g in groups if len(g['params']) > 0]
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
                               encoder_lr: float = 1e-5, head_lr: float = 3e-4,
                               weight_decay: float = 0.01, warmup_ratio: float = 0.06,
                               epochs: int = 15, lr_decay_epochs: int = 8,
                               patience: int = 3, grad_clip: float = 1.0,
                               use_fp16: bool = True, label_names=None):
    """Full training loop with early stopping on validation macro-F1.

    `lr_decay_epochs` (not `epochs`) sizes the linear warmup+decay schedule.
    Round-1 runs stopped at epoch 5-6 while the scheduler assumed a 15-epoch
    horizon, so the LR never fully decayed by the time the best checkpoint
    was hit -- decoupling the two keeps the LR curve realistic even though
    `epochs` is still the hard cap the loop can run to if early stopping
    doesn't trigger.

    Returns (model_with_best_weights_loaded, best_val_macro_f1, history), where
    history is a list of per-epoch dicts (epoch, train_loss, val_loss, val_acc,
    val_macro_f1) suitable for plotting.
    """
    optimizer = build_optimizer(model, encoder_lr, head_lr, weight_decay)
    total_steps = len(train_loader) * lr_decay_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if (use_fp16 and device.type == 'cuda') else None

    best_macro_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                  criterion, device, scaler, grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, device, label_names)

        print(f"Epoch {epoch + 1:02d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_metrics['loss']:.4f} | "
              f"val_acc {val_metrics['accuracy'] * 100:.2f}% | "
              f"val_macro_f1 {val_metrics['macro_f1']:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1'],
        })

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
    return model, best_macro_f1, history


def train_fixed_epochs(model, train_loader, criterion, device, *,
                        encoder_lr: float = 1e-5, head_lr: float = 3e-4,
                        weight_decay: float = 0.01, warmup_ratio: float = 0.06,
                        epochs: int, grad_clip: float = 1.0, use_fp16: bool = True):
    """Phase 2: train for a fixed epoch count with NO held-out validation set
    and NO early stopping -- used for the final full-data run once the val
    split (train_with_early_stopping) has already told us roughly which
    epoch the model peaks at.

    Held-out val is a methodology tool for choosing when to stop, not a
    resource the final model should be denied at test time -- once the
    epoch count is fixed from earlier val-based runs, folding val back into
    train gives the model ~5-15% more labeled examples for the same
    train/test comparison (test.csv is never touched here or anywhere else
    in training). `epochs` has no default: it must be chosen deliberately
    from a prior train_with_early_stopping run's history (see the
    'full-data' notebook section), not left at some arbitrary number.

    Since there's no early stopping to potentially cut the run short, the
    warmup+decay schedule is sized to the full `epochs` directly (no
    `lr_decay_epochs` decoupling needed here).

    Returns (model, history), where history is a list of per-epoch dicts
    with only `epoch`/`train_loss` (no val_* keys -- there is no val set in
    this mode).
    """
    optimizer = build_optimizer(model, encoder_lr, head_lr, weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if (use_fp16 and device.type == 'cuda') else None

    history = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                  criterion, device, scaler, grad_clip)
        print(f"Epoch {epoch + 1:02d}/{epochs} | train_loss {train_loss:.4f}")
        history.append({'epoch': epoch + 1, 'train_loss': train_loss})

    return model, history


def plot_history(history, title: str = ''):
    """Plot train/val loss and val accuracy/macro-F1 per epoch.

    Kept as a plain matplotlib helper (not saved to disk) so it renders
    inline in a Kaggle/Jupyter notebook right after training.
    """
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    val_macro_f1 = [h['val_macro_f1'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, marker='o', label='train loss')
    axes[0].plot(epochs, val_loss, marker='o', label='val loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title(f'{title} -- loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_acc, marker='o', label='val accuracy')
    axes[1].plot(epochs, val_macro_f1, marker='o', label='val macro-F1')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('score')
    axes[1].set_title(f'{title} -- val accuracy / macro-F1')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
