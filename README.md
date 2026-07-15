# Bengali Sentiment Analysis — modernizing the ICCIT 2020 benchmark

Base paper: *Sentiment analysis in Bengali via transfer learning using
multi-lingual BERT* (ICCIT 2020), original implementation at
[KhondokerIslam/Bengali_Sentiment](https://github.com/KhondokerIslam/Bengali_Sentiment).
Its `BERT_BSA`+GRU model, a frozen `bert-base-multilingual-cased` feature
extractor feeding a trainable GRU head, reports **71% accuracy (2-class)**
and **60% accuracy (3-class)** on the dataset in `Dataset/`.

This repo re-benchmarks that dataset with a fine-tuned, language-specific
model and a hybrid architecture, targeting **85-90%+ (2-class)** and
**75-82% (3-class)** accuracy — see project memory / thesis notes for the
full literature review and target rationale.

## Layout

- `Dataset/` — the original paper's raw 3-class data (`train.csv`/`test.csv`,
  17,852 rows total: 2=Negative, 1=Positive, 0=Neutral). The 2-class task is
  derived from this by dropping Neutral rows (matches the paper's reported
  13,120-row 2-class set).
- `src/preprocess.py` + `src/model.py` — faithful port of the original
  frozen-mBERT+GRU baseline (torchtext-free, runs on current
  torch/transformers). Used only as the literature comparison row, not
  retrained.
- `src/data_utils.py` — data pipeline for the new model: light Bangla
  cleaning (no stemming — stemming would fight BanglaBERT's subword
  pretraining), dedup, train/test overlap removal, stratified split, 2-class
  label derivation.
- `src/hybrid_model.py` — the accuracy/novelty model: fine-tuned
  `csebuetnlp/banglabert` fused with a Conv1D→BiLSTM branch over the token
  sequence, class-weighted loss, discriminative-LR AdamW, early stopping on
  validation macro-F1.
- `Codes/2-class/BERT_w_GRU_Kaggle.ipynb` — Kaggle runner for the baseline
  reproduction.
- `Codes/BanglaBERT_Hybrid/` — Kaggle runner for the new hybrid model, both
  tasks.

## Data

Kaggle dataset (raw 3-class CSVs): `reversedthoutgts/bangla-dataset`, read
from `/kaggle/input/datasets/reversedthoutgts/bangla-dataset/{train_,test_}.csv`.
