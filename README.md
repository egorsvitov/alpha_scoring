# alpha_scoring

Compact final pipeline for the Alfa credit scoring solution.

The repository intentionally contains only the last useful modelling layer:

```text
raw parquet
-> sequence cache
-> Alpha-GRU + Payment Transformer
-> test prediction
-> rank blend with base full100 id-prior submission
-> final submission
```

The final leaderboard-improving submission was:

```text
0.839 * submission_full100_checkpoint_average_idprior.csv
+ 0.161 * Alpha-GRU epoch 9 prediction
```

Leaderboard ROC-AUC: `0.786643`.

## Inputs

Place files under `data/`:

- `train_data.parquet`
- `test_data.parquet`
- `train_target.csv`
- `sample_submission.csv`
- `submission_full100_checkpoint_average_idprior.csv`

The last file is the previous best full100 checkpoint-averaged ensemble with
the `id` target prior. This repository keeps the final incremental layer small;
it does not include the older exploratory CatBoost/LightGBM/Transformer code.

## Install

Python 3.10+ is expected.

For CUDA 11.8:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

## Run

```bash
bash scripts/run_final.sh
```

This creates:

- `sequence_cache/`
- `runs/full100_alpha_gru_payment/epoch_9.pt`
- `runs/full100_alpha_gru_payment/submission_epoch_9.csv`
- `submissions/submission_alpha_gru161.csv`

## Model

The final model is `AlphaGRUClassifier`:

```text
payment codes
-> Payment Transformer
-> product embedding
-> position embedding by rn
-> GRU over credit products
-> mean/max/attention pooling over GRU outputs
+ mean/max/attention pooling over input product embeddings
+ projected final GRU hidden states
-> MLP
```

Training uses:

- `OneCycleLR`;
- linear sample weights by normalized `id`;
- full-train checkpoint at epoch 9;
- rank blend weight `0.161`.

