"""Shared data utilities for the BanglaBERT hybrid pipeline.

Deliberately separate from src/preprocess.py, which is the faithful port of
the original paper's cleaning pipeline (digit conversion, punctuation
stripping, Bangla stemming) used by the frozen-mBERT baseline in
src/model.py. Stemming text before feeding it to a subword-tokenizing
pretrained model like BanglaBERT throws away exactly the word forms
BanglaBERT was pretrained on, so the cleaning here is deliberately lighter
and skips stemming entirely.

Verified against the real Dataset/train.csv + Dataset/test.csv locally:
  3class -> train 12398 / val 2189 / test 3000
  2class -> train 9160  / val 1617  / test 2123
"""

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

_URL_RE = re.compile(r'(https?://\S+|www\.\S+)')
_WS_RE = re.compile(r'\s+')

# Zero-width space (U+200B) and word joiner (U+2060) -- the only two
# invisible-Unicode characters BanglaBERT's own normalizer (csebuetnlp/
# normalizer, UNICODE_REPLACEMENTS) treats as noise to strip. Narrowed from
# an earlier version that also stripped ZWNJ (U+200C) and ZWJ (U+200D):
# those are *not* noise in Bengali -- they control conjunct/non-conjunct
# glyph formation and are legitimate characters BanglaBERT's subword
# vocabulary expects intact. An audit of Dataset/train.csv found only 4 rows
# with genuine ZWSP noise, versus 135 rows (442 occurrences) with ZWNJ and
# 94 rows (131 occurrences) with ZWJ that the old blanket regex was
# incorrectly stripping -- over 98% of what it "cleaned" was real text.
_ZW_RE = re.compile('[​⁠]')

try:
    from normalizer import normalize as _bnorm
    _HAS_NORMALIZER = True
except ImportError:
    _HAS_NORMALIZER = False
    print("[data_utils] 'normalizer' package not found -- skipping BanglaBERT's "
          "official text normalization step. Install with:\n"
          "  pip install git+https://github.com/csebuetnlp/normalizer.git")


def verify_normalizer() -> bool:
    """Explicit, unambiguous check of whether BanglaBERT's official
    normalizer is active -- call this once at the top of a training run so
    it's impossible to miss in the logs (relying on the absence of the
    import-failure warning above is easy to overlook).
    """
    if not _HAS_NORMALIZER:
        print("[verify_normalizer] NOT ACTIVE -- 'normalizer' package is not "
              "installed. Text normalization is being skipped.")
        return False
    sample = "আমি  ভালো​আছি।।।"  # deliberately messy: double space + ZWSP + repeated danda
    out = _bnorm(sample)
    changed = out != sample
    print(f"[verify_normalizer] ACTIVE -- sample {sample!r} -> {out!r} "
          f"(changed={changed})")
    return True


def bangla_normalize(text: str) -> str:
    """Apply BanglaBERT authors' recommended Unicode/punctuation normalizer.

    Falls back to the raw text if the `normalizer` package isn't installed,
    so the pipeline still runs (just slightly less faithfully preprocessed).
    """
    if _HAS_NORMALIZER:
        return _bnorm(text)
    return text


def clean_light(df: pd.DataFrame, text_col: str = 'Data') -> pd.DataFrame:
    """Minimal cleaning for transformer fine-tuning: strip URLs and
    zero-width Unicode artifacts, collapse whitespace, apply the BanglaBERT
    normalizer. Keeps punctuation such as '!'/'?' since it carries sentiment
    signal, and does not stem.
    """
    df = df.copy().reset_index(drop=True)
    texts = df[text_col].astype(str)

    n_zw = texts.str.contains(_ZW_RE, regex=True).sum()
    if n_zw:
        print(f"[clean_light] stripping zero-width Unicode artifacts from {n_zw} rows")

    texts = texts.str.replace(_URL_RE, ' ', regex=True)
    texts = texts.str.replace(_ZW_RE, '', regex=True)
    texts = texts.apply(bangla_normalize)
    texts = texts.str.replace(_WS_RE, ' ', regex=True).str.strip()
    df[text_col] = texts
    return df[df[text_col].str.len() > 0].reset_index(drop=True)


def dedup(df: pd.DataFrame, text_col: str = 'Data') -> pd.DataFrame:
    """Drop exact-duplicate rows on the text column (run after clean_light,
    so rows that only differed by whitespace are also caught)."""
    before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[dedup] dropped {dropped} duplicate rows ({before} -> {len(df)})")
    return df


def remove_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    text_col: str = 'Data') -> pd.DataFrame:
    """Drop any train rows whose text also appears in the test set, closing
    the small train/test leakage found in the raw CSVs (4 rows)."""
    overlap = set(train_df[text_col]) & set(test_df[text_col])
    if overlap:
        print(f"[overlap] removing {len(overlap)} train rows that also appear in test")
        train_df = train_df[~train_df[text_col].isin(overlap)].reset_index(drop=True)
    return train_df


def to_3class_labels(df: pd.DataFrame, sentiment_col: str = 'Sentiment') -> pd.DataFrame:
    """3-class task: Sentiment is already 0=Neutral/1=Positive/2=Negative --
    just make the model-facing label column explicit."""
    df = df.copy()
    df['label'] = df[sentiment_col].astype(int)
    return df


def to_2class_labels(df: pd.DataFrame, sentiment_col: str = 'Sentiment') -> pd.DataFrame:
    """2-class task: drop Neutral (Sentiment==0), remap {2: Negative -> 0,
    1: Positive -> 1}, matching the paper's own 2-class construction and
    src/preprocess.py's to_two_class()."""
    df = df[df[sentiment_col] != 0].reset_index(drop=True)
    df['label'] = df[sentiment_col].map({2: 0, 1: 1})
    return df


def stratified_split(df: pd.DataFrame, label_col: str = 'label',
                      val_ratio: float = 0.15, seed: int = 42):
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=seed, stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_class_weights(labels, num_labels: int, multipliers=None):
    """Inverse-frequency class weights for nn.CrossEntropyLoss(weight=...).

    `multipliers`, if given, is a dict {label: factor} or a length-`num_labels`
    sequence applied on top of the balanced weights -- e.g. to push a
    chronically under-recalled class (Neutral in 3class, Positive in 2class)
    harder than plain inverse-frequency balancing does. This trades away some
    majority-class recall for minority-class recall, so it should be swept
    and checked against both, not just macro-F1.
    """
    import torch
    classes = np.arange(num_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes,
                                    y=np.asarray(labels))
    if multipliers is not None:
        if isinstance(multipliers, dict):
            mult = np.array([multipliers.get(c, 1.0) for c in classes], dtype=float)
        else:
            mult = np.asarray(multipliers, dtype=float)
        weights = weights * mult
    return torch.tensor(weights, dtype=torch.float)


def load_and_prepare(train_path: str, test_path: str, task: str, *,
                      val_ratio: float = 0.15, seed: int = 42,
                      text_col: str = 'Data', sentiment_col: str = 'Sentiment'):
    """End-to-end data prep for one task ('3class' or '2class'):
    load -> light clean -> dedup -> remove train/test overlap -> label ->
    stratified train/val split. Returns (train_df, val_df, test_df).
    """
    assert task in ('3class', '2class'), f"task must be '3class' or '2class', got {task!r}"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"[load] raw train: {train_df.shape}, raw test: {test_df.shape}")

    train_df = clean_light(train_df, text_col)
    test_df = clean_light(test_df, text_col)

    train_df = dedup(train_df, text_col)
    train_df = remove_overlap(train_df, test_df, text_col)

    label_fn = to_3class_labels if task == '3class' else to_2class_labels
    train_df = label_fn(train_df, sentiment_col)
    test_df = label_fn(test_df, sentiment_col)

    train_df, val_df = stratified_split(train_df, 'label', val_ratio, seed)

    print(f"[{task}] train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"[{task}] train label counts:\n{train_df['label'].value_counts().sort_index()}")

    return train_df, val_df, test_df
