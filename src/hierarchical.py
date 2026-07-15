"""Two-stage hierarchical classifier for the 3-class task: Neutral-vs-rest
first, then Positive-vs-Negative on whatever survives stage 1.

Motivation (Phase 3): the flat 3-way BanglaBertHybrid's confusion matrix
showed Neutral getting absorbed into Negative -- Neutral is both the
minority class and semantically the "boundary" between the other two, so a
single softmax has to carve a 3-way decision through territory that's
genuinely ambiguous. Splitting the decision into "is this Neutral at all?"
and, only for rows that survive, "Positive or Negative?" isolates that
specific confusion into its own dedicated binary head instead of asking one
flat classifier to solve both problems at once.

Both stages reuse BanglaBertHybrid unchanged (num_labels=2) and the same
train_with_early_stopping/evaluate loop from hybrid_model.py -- this module
only adds the label remapping for each stage and the two-model cascade at
inference time, so it's directly comparable to the flat classifier under
the same data/training recipe.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

from src.data_utils import clean_light, dedup, remove_overlap, stratified_split

# Matches Dataset/*.csv's Sentiment encoding (see CLAUDE.md): 0=Neutral,
# 1=Positive, 2=Negative -- NOT the usual 0/1/2-in-order convention.
NEUTRAL, POSITIVE, NEGATIVE = 0, 1, 2
LABEL_NAMES_3CLASS = ['Neutral', 'Positive', 'Negative']


def to_stage1_labels(df: pd.DataFrame, sentiment_col: str = 'Sentiment') -> pd.DataFrame:
    """Stage 1 label: 1 if Neutral, 0 if Positive or Negative ("rest")."""
    df = df.copy()
    df['label'] = (df[sentiment_col] == NEUTRAL).astype(int)
    return df


def to_stage2_labels(df: pd.DataFrame, sentiment_col: str = 'Sentiment') -> pd.DataFrame:
    """Stage 2 label: Neutral rows dropped, remaining rows mapped
    {Negative -> 0, Positive -> 1} -- identical to data_utils.to_2class_labels,
    since stage 2 only ever runs on rows stage 1 has already ruled out as
    non-Neutral.
    """
    df = df[df[sentiment_col] != NEUTRAL].reset_index(drop=True)
    df['label'] = df[sentiment_col].map({NEGATIVE: 0, POSITIVE: 1})
    return df


def load_and_prepare_hierarchical(train_path: str, test_path: str, *,
                                   val_ratio: float = 0.05, seed: int = 42,
                                   text_col: str = 'Data',
                                   sentiment_col: str = 'Sentiment'):
    """Data prep for the two-stage pipeline. Shares data_utils's
    clean/dedup/overlap-removal with the flat 3-class task (so both pipelines
    see the same underlying rows), then produces:

      - stage1_train/val: Neutral (1) vs rest (0), full training set
      - stage2_train/val: Positive/Negative only (Neutral dropped)
      - test_df: 3-class-labeled (0/1/2) test set, for scoring the cascade
        end-to-end against the same labels the flat classifier is scored on

    Returns (stage1_train, stage1_val, stage2_train, stage2_val, test_df).
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"[load] raw train: {train_df.shape}, raw test: {test_df.shape}")

    train_df = clean_light(train_df, text_col)
    test_df = clean_light(test_df, text_col)

    train_df = dedup(train_df, text_col)
    train_df = remove_overlap(train_df, test_df, text_col)

    stage1_full = to_stage1_labels(train_df, sentiment_col)
    stage2_full = to_stage2_labels(train_df, sentiment_col)

    stage1_train, stage1_val = stratified_split(stage1_full, 'label', val_ratio, seed)
    stage2_train, stage2_val = stratified_split(stage2_full, 'label', val_ratio, seed)

    test_df = test_df.copy()
    test_df['label'] = test_df[sentiment_col].astype(int)

    print(f"[hierarchical] stage1 (Neutral-vs-rest) train: {len(stage1_train)}, "
          f"val: {len(stage1_val)}")
    print(f"[hierarchical] stage1 train label counts:\n"
          f"{stage1_train['label'].value_counts().sort_index()}")
    print(f"[hierarchical] stage2 (Positive-vs-Negative) train: {len(stage2_train)}, "
          f"val: {len(stage2_val)}")
    print(f"[hierarchical] stage2 train label counts:\n"
          f"{stage2_train['label'].value_counts().sort_index()}")
    print(f"[hierarchical] test (3-class, for cascade scoring): {len(test_df)}")

    return stage1_train, stage1_val, stage2_train, stage2_val, test_df


@torch.no_grad()
def predict_hierarchical(stage1_model, stage2_model, loader, device):
    """Cascade stage1 -> stage2 over a loader built from the 3-class-labeled
    test_df (labels 0/1/2), returning predictions remapped back into that
    same 0=Neutral/1=Positive/2=Negative scheme -- directly comparable to
    the flat classifier's evaluate() output.
    """
    stage1_model.eval()
    stage2_model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        stage1_pred = torch.argmax(stage1_model(input_ids, attention_mask), dim=1)
        is_neutral = stage1_pred == 1

        preds = torch.full_like(labels, NEUTRAL)

        rest_idx = (~is_neutral).nonzero(as_tuple=True)[0]
        if len(rest_idx) > 0:
            stage2_logits = stage2_model(input_ids[rest_idx], attention_mask[rest_idx])
            stage2_pred = torch.argmax(stage2_logits, dim=1)  # 0=Negative, 1=Positive
            preds[rest_idx] = torch.where(
                stage2_pred == 1,
                torch.full_like(stage2_pred, POSITIVE),
                torch.full_like(stage2_pred, NEGATIVE),
            )

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'report': classification_report(all_labels, all_preds,
                                         labels=[NEUTRAL, POSITIVE, NEGATIVE],
                                         target_names=LABEL_NAMES_3CLASS,
                                         zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds,
                                              labels=[NEUTRAL, POSITIVE, NEGATIVE]),
        'predictions': all_preds,
        'targets': all_labels,
    }


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    """Recall per class from a confusion matrix (rows=true, cols=pred),
    shared by the class-weight sweep and the flat-vs-hierarchical
    comparison so both read recall off the same definition."""
    cm = np.asarray(cm, dtype=float)
    return cm.diagonal() / cm.sum(axis=1)
