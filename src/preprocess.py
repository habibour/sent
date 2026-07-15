"""Bengali text preprocessing pipeline, ported from Codes/PreProcess.ipynb.

The original notebook is a scratchpad of ad-hoc cell executions against
Google-Drive-only files with no single linear pipeline. This module extracts
the actual cleaning logic (digit conversion, punctuation/junk stripping,
short-text filtering, stemming) into reusable functions, plus the 3-class ->
2-class label filter that recreates the paper's 2-class dataset (13,120 rows)
from the raw 3-class annotated data.
"""

import contextlib
import io

import pandas as pd
from bnlp import BasicTokenizer  # bnlp_toolkit >= 4: BasicTokenizer moved to top-level
from bangla_stemmer.stemmer import stemmer

basic_t = BasicTokenizer()
stmr = stemmer.BanglaStemmer()

_DIGIT_MAP = str.maketrans("1234567890", "১২৩৪৫৬৭৮৯০")

# Ported verbatim from PreProcess.ipynb's punc_conv()
_REM_C = ['#', '$', '&', '(', ')', ';', '<', '=', '>', '@', '[', ']', '^', '_',
          '`', '{', '}', '~', ':', '"', "'", '–', '—', '‘', '’', '“', '”',
          '•', '…', '৯৷']
_KEEP_C = ['!', '?', '|', chr(2404), '%', '*', '+', 'ঃ']
_GONE = ['‪১', '‰', '⚽️', '✌', '✌✌✌', '￰জীবনযাপন', '৷', '৷এবার',
         '৷জাতীয়', '৷শুভেচ্ছা', '৷সে', '৷২', '৷৩৷', '‌কিন্তু', '‌কেন',
         '‌কোথায়', '‌তাই', '‌দে‌খে', '‌নোয়াখালী',
         '‌ফিজিওর', '‌বিএন‌পি‌', '‍', '‍আগেই',
         '‍আমরা', '‍আশাবাদী', '‍উচিত', '‍উজ্জল',
         '‍উদ্দেশ্যমূলকভাবে', '‍উৎপাদনকারী', '‍এক্ষেত্রে',
         '‍এবং', '‍নির্ভরশীল', '‍মিঃ', '‍যার', '‍যিনি',
         '‍সত্যই', '‍সব', '‍সী', '‍সুষ্ঠ']

# Ported verbatim from PreProcess.ipynb's process()
_PROCESS_REM_C = "!\"#$%&'(),-./:;=<>?@[]^_`{|}~¥§©’‚“”‪™−√∝∞৷" + chr(2404)


def eng_dig_conv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Latin digits to Bengali digit glyphs in the Data column."""
    df = df.copy()
    df['Data'] = df['Data'].astype(str).str.translate(_DIGIT_MAP)
    return df


def punc_conv(df: pd.DataFrame) -> pd.DataFrame:
    """Strip punctuation/quote tokens and drop rows containing junk artifacts."""
    df = df.reset_index(drop=True)
    for i, text in enumerate(df['Data']):
        tokens = basic_t.tokenize(text)
        out = []
        j = 0
        while j < len(tokens):
            t = tokens[j]
            if (t in _KEEP_C and j + 1 < len(tokens) and tokens[j + 1] == t) or \
               t in _REM_C or t in _KEEP_C:
                j += 1
                continue
            out.append(t)
            j += 1
        df.at[i, 'Data'] = ' '.join(out)

    drop_idx = [i for i, text in enumerate(df['Data'])
                if any(tok in _GONE for tok in basic_t.tokenize(text))]
    df = df.drop(drop_idx).reset_index(drop=True)
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Second punctuation pass: replace punctuation characters with spaces."""
    df = df.reset_index(drop=True)
    for i, text in enumerate(df['Data']):
        tokens = basic_t.tokenize(text)
        out = []
        for t in tokens:
            out.append(' ' if t in _PROCESS_REM_C else t)
        df.at[i, 'Data'] = ' '.join(out)
    return df


def stemming(df: pd.DataFrame) -> pd.DataFrame:
    """Bangla stemming via bnlp tokenizer + bangla-stemmer.

    bangla-stemmer prints an "applied Nth rules.." line per token stemmed;
    suppressed here since it would otherwise flood output over 10k+ rows.
    """
    df = df.reset_index(drop=True)
    with contextlib.redirect_stdout(io.StringIO()):
        for i, text in enumerate(df['Data']):
            tokens = basic_t.tokenize(text)
            stemmed = stmr.stem(tokens)
            df.at[i, 'Data'] = ' '.join(stemmed)
    return df


def filter_short_text(df: pd.DataFrame, min_tokens: int = 3) -> pd.DataFrame:
    """Drop rows whose Data has fewer than min_tokens whitespace tokens."""
    df = df.reset_index(drop=True)
    keep = df['Data'].astype(str).str.split().str.len() >= min_tokens
    return df[keep].reset_index(drop=True)


def drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows left blank/whitespace-only by earlier cleaning steps."""
    df = df.reset_index(drop=True)
    keep = df['Data'].astype(str).str.strip().str.len() > 0
    return df[keep].reset_index(drop=True)


def to_two_class(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Neutral (Sentiment==0) and remap to a binary label column.

    Sentiment 2 = Negative -> label 0
    Sentiment 1 = Positive -> label 1
    """
    df = df[df['Sentiment'] != 0].reset_index(drop=True)
    df['label'] = df['Sentiment'].map({2: 0, 1: 1})
    return df


def clean(df: pd.DataFrame, apply_stemming: bool = True) -> pd.DataFrame:
    """Run the full cleaning pipeline (digits -> punctuation x2 -> [stemming] -> short-text filter)."""
    df = eng_dig_conv(df)
    df = punc_conv(df)
    df = process(df)
    if apply_stemming:
        df = stemming(df)
    df = drop_empty(df)
    df = filter_short_text(df)
    return df


def run_pipeline(train_path: str, test_path: str, apply_stemming: bool = True):
    """Load raw 3-class data, clean it, and filter down to the 2-class task.

    Returns (train_df, test_df), each with columns ['Data', 'Sentiment', 'label'].
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"[raw] train: {train_df.shape}, test: {test_df.shape}")
    print(f"[raw] train Sentiment counts:\n{train_df['Sentiment'].value_counts()}")

    train_df = clean(train_df, apply_stemming=apply_stemming)
    test_df = clean(test_df, apply_stemming=apply_stemming)
    print(f"[cleaned] train: {train_df.shape}, test: {test_df.shape}")

    train_df = to_two_class(train_df)
    test_df = to_two_class(test_df)

    total = len(train_df) + len(test_df)
    print(f"[2-class] train: {train_df.shape}, test: {test_df.shape}, total: {total}")
    print(f"[2-class] train label counts:\n{train_df['label'].value_counts()}")
    print(f"[2-class] test label counts:\n{test_df['label'].value_counts()}")
    print("Paper's reported 2-class dataset size: 13,120")

    return train_df, test_df
