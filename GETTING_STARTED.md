# Getting started (VS Code)

## Opening the project

Open this folder (`sent`) directly in VS Code — it's already a git repo tracking
`github.com/habibour/sent.git`. Install the **Python** and **Jupyter** extensions if you don't have them,
so you can run/step through the `.ipynb` files locally when you're just editing code (not training).

## Local environment

Training itself happens on Kaggle (no local GPU needed), but a local Python environment is still useful
for editing `src/` and sanity-checking data logic before pushing:

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`torch`/`transformers` in `requirements.txt` are only needed if you want to actually import
`src/hybrid_model.py` or `src/model.py` locally (e.g. in a debugger). If you just want to check the data
pipeline, `pandas` + `scikit-learn` are enough:

```python
from src.data_utils import load_and_prepare
train_df, val_df, test_df = load_and_prepare("Dataset/train.csv", "Dataset/test.csv", task="3class")
```

## Day-to-day workflow

1. Edit `src/*.py` locally in VS Code.
2. Sanity-check with the snippet above (or open the relevant notebook and run cells) before pushing —
   catches bugs without burning Kaggle GPU quota.
3. Commit and push to `origin` — the Kaggle notebooks `git clone`/`git pull` this repo as their first
   cell, so changes to `src/` only reach Kaggle after a push.
4. Run/re-run on Kaggle (GPU + internet enabled on the notebook), pulling from
   `/kaggle/input/datasets/reversedthoutgts/bangla-dataset/{train_,test_}.csv`.
5. Bring results back — `results_summary.csv` and the printed comparison table — into the thesis/paper
   write-up.

## Current status

- Baseline reproduction (`Codes/2-class/BERT_w_GRU_Kaggle.ipynb`, frozen mBERT+GRU) — coded, not yet run;
  its result is only needed as the literature-comparison row, so it doesn't need re-running unless you
  want to double check the paper's reported 71%.
- Hybrid model (`Codes/BanglaBERT_Hybrid/BanglaBERT_Hybrid_Kaggle.ipynb`) — coded and locally verified
  (data pipeline + syntax only; the actual fine-tuning run has not happened yet, since this dev
  environment has no GPU/internet access to Hugging Face). **Next step: run it on Kaggle**, smoke-testing
  with `CONFIG['epochs'] = 2` first.
- Not started yet: imbalance-loss ablation, error analysis / confusion matrices from a real run, IEEE
  paper draft, defense slides (see the task list for the full breakdown).

## Reminders

- Git commits in this repo use the identity `MD HABIBOUR RAHMAN <habibourrahmanm@gmail.com>` — already
  configured in this environment's local clone, but set it yourself (`git config user.name`/`user.email`)
  if working from a fresh machine.
- `src/preprocess.py` (stemming, for the frozen baseline) and `src/data_utils.py` (no stemming, for the
  fine-tuned hybrid model) are intentionally different pipelines — see `CLAUDE.md` for why.
