# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A re-benchmarking of the ICCIT 2020 paper *Sentiment analysis in Bengali via transfer learning using
multi-lingual BERT* (original implementation: github.com/KhondokerIslam/Bengali_Sentiment). The paper's
`BERT_BSA`+GRU model — a **frozen** `bert-base-multilingual-cased` feature extractor feeding a trainable
GRU head — reports 71% accuracy (2-class) and 60% accuracy (3-class) on the dataset in `Dataset/`. This
repo fine-tunes a Bengali-specific transformer end-to-end with a hybrid head, targeting 85-90%+ (2-class)
and 75-82% (3-class) accuracy, for a thesis + IEEE conference paper submission.

There are two independent pipelines in this repo, intentionally not sharing preprocessing:

1. **Baseline reproduction** (`src/model.py` + `src/preprocess.py`) — a torchtext-free port of the
   original notebook's frozen-mBERT+GRU model, kept only as the literature-comparison row. It is not
   being retrained or improved; its preprocessing (`clean()`) applies Bangla stemming and aggressive
   punctuation stripping, appropriate for the static word2vec/fastText-style embeddings the original paper
   also compared against, but not used by the new model.
2. **Accuracy/novelty model** (`src/hybrid_model.py` + `src/data_utils.py`) — fine-tuned
   `csebuetnlp/banglabert` fused with a Conv1D→BiLSTM branch. Its preprocessing (`clean_light()`)
   deliberately skips stemming: stemming Bengali text before a subword-tokenizing pretrained model throws
   away exactly the word forms the model was pretrained on.

Do not merge these two preprocessing paths or "clean up" the duplication between `preprocess.py` and
`data_utils.py` — the difference is deliberate and load-bearing.

## Data

`Dataset/train.csv` / `Dataset/test.csv` are the original paper's raw 3-class data (17,852 rows: `Data`
text column, `Sentiment` column where **2=Negative, 1=Positive, 0=Neutral**). This is not the usual
0/1/2-in-order encoding — always check `src/data_utils.py`'s `to_2class_labels`/`to_3class_labels` for the
exact mapping rather than assuming.

- 3-class task: use `Sentiment` as-is.
- 2-class task: drop `Sentiment==0` (Neutral) rows, remap `{2: Negative -> 0, 1: Positive -> 1}`. This
  reproduces the paper's own 13,120-row 2-class dataset construction.

On Kaggle, the same raw data is read from `/kaggle/input/datasets/reversedthoutgts/bangla-dataset/{train_,test_}.csv`
rather than from `Dataset/` directly — both notebooks clone this repo and read from the Kaggle input path,
not from the repo's own `Dataset/` folder.

`train.csv` contains 135 exact-duplicate rows (261 once cleaned/whitespace-normalized) and 4 rows whose
text also appears in `test.csv`. `src/data_utils.py`'s `load_and_prepare()` handles both (dedup +
overlap removal) before splitting; `src/preprocess.py`'s baseline pipeline does not, since it reproduces
the original paper's methodology as-is.

## Running things

There is no test suite, linter, or build step in this repo — it is experiment notebooks plus the shared
`src/` modules they import. Development loop:

- **Local sanity-check of the data pipeline** (no GPU/transformers needed, just pandas/scikit-learn):
  ```python
  from src.data_utils import load_and_prepare
  train_df, val_df, test_df = load_and_prepare('Dataset/train.csv', 'Dataset/test.csv', task='3class')
  ```
  Swap `task='2class'` for the binary task. This is the fastest way to verify a preprocessing change
  before burning Kaggle GPU time on it.
- **Actual training** happens on Kaggle, not locally — this environment has no GPU. The two notebooks:
  - `Codes/2-class/BERT_w_GRU_Kaggle.ipynb` — baseline reproduction.
  - `Codes/BanglaBERT_Hybrid/BanglaBERT_Hybrid_Kaggle.ipynb` — the hybrid model; runs both the 3-class
    and 2-class experiments in one execution via `run_experiment(task)`, then prints a comparison table
    against the paper's 60%/71%.
  Both clone `https://github.com/habibour/sent.git` into `/kaggle/working/sent` as their first cell, so
  local changes to `src/` must be pushed to GitHub before they'll show up in a Kaggle run.
  Before a full run, lower `CONFIG['epochs']` to 1-2 in the hybrid notebook (or `N_EPOCHS` for the
  baseline) as a smoke test.

## Architecture of the hybrid model (`src/hybrid_model.py`)

`BanglaBertHybrid` fine-tunes the encoder end-to-end (unlike the baseline, which freezes BERT) and fuses
two views of its output before classifying:

- the `[CLS]` pooled embedding (global sentence-level signal), and
- a `Conv1D → MaxPool1D → BiLSTM` branch over the full token sequence (local n-gram + sequential signal),
  adapted from the CNN-BiLSTM head in Islam & Alam's BangDSA paper but applied to fine-tuned contextual
  embeddings rather than a frozen document vector.

Training uses discriminative learning rates (`build_optimizer`: low LR for the pretrained encoder, high
LR for the randomly-initialized head, bias/LayerNorm excluded from weight decay), class-weighted
cross-entropy for the label imbalance, linear warmup+decay, fp16, and early stopping on **validation
macro-F1** — not loss or raw accuracy, so the minority Neutral class can't be silently ignored by
checkpoint selection.

## Git

Commits in this repo are authored as MD HABIBOUR RAHMAN (`habibourrahmanm@gmail.com`); keep that identity
when committing here rather than a generic default. `origin` is `github.com/habibour/sent.git`.
