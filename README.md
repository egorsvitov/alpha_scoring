# alpha_scoring

Финальный воспроизводимый пайплайн для задачи кредитного скоринга Альфа-Банка.

Leaderboard ROC-AUC: `0.786643`.

## Входные файлы

Положите в `data/`:

- `train_data.parquet`
- `test_data.parquet`
- `train_target.csv`
- `sample_submission.csv`

пайплайн обучает все модели с нуля и сохраняет финальный файл в
`submissions/submission_alpha_gru161.csv`.

## Установка

Ожидается Python `3.10+` и CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```


## Запуск

```bash
bash scripts/run_final.sh
```

Основные артефакты:

- `features/`
- `tabular_cache/`
- `sequence_cache/`
- `sequence_runs/`
- `submissions/submission_alpha_gru161.csv`

Финальный сабмит:

```text
submissions/submission_alpha_gru161.csv
```

## Схема ансамбля

Все смешивания выполняются по rank-normalized предсказаниям: перед сложением каждая ветка переводится в ранги через `rank01`.

```text
tabular =
  0.30 * CatBoost V4
+ 0.20 * CatBoost V5
+ 0.50 * LightGBM V5

polished_sequence =
  0.218196 * full100_pooling_seed42, epochs 5/6/7
+ 0.250000 * full100_pooling_seed137, epochs 4/5/6
+ 0.241164 * full100_product_transformer, epochs 5/6/7
+ 0.114840 * tabular
+ 0.085800 * full100_dual_pooling, epochs 2/3/4
+ 0.090000 * full100_late_fusion, epochs 2/3/4

payment_sequence =
  0.75 * polished_sequence
+ 0.25 * full100_payment_transformer, epochs 6/7/8

hierarchical_sequence =
  0.79 * payment_sequence
+ 0.21 * full100_hierarchical_transformer, epochs 5/6/7

temporal_pooling_two_seed =
  mean(
    temporal_full_seed271, epochs 7/8/9/10,
    temporal_full_seed42,  epochs 7/8/9/10
  )

base_sequence =
  0.696 * hierarchical_sequence
+ 0.304 * temporal_pooling_two_seed

base_with_idprior =
  0.979 * base_sequence
+ 0.021 * id_target_prior

final_submission =
  0.839 * base_with_idprior
+ 0.161 * Alpha-GRU + Payment Transformer, epoch 9
```

`id_target_prior`: `bins=400`, `sigma=3.0`, `smoothing=0.0`.

## Запуск на сервере

```bash
cp server.env.example server.env
```

Заполните `REMOTE_HOST`, `REMOTE_PORT`, `REMOTE_USER`, `REMOTE_DIR`.
Если нужен пароль, добавьте `REMOTE_PASSWORD`.

```bash
bash scripts/server/sync_to_server.sh
bash scripts/server/upload_inputs_to_server.sh
bash scripts/server/run_on_server.sh
bash scripts/server/fetch_submission_from_server.sh
```

Если на сервере нет passwordless `sudo`, swap можно подготовить вручную:

```bash
sudo SWAP_SIZE_GB=64 MIN_SWAP_GB=60 bash scripts/server/ensure_swap.sh
```

## Структура

```text
src/alpha_scoring/features/          # признаки и кеши
src/alpha_scoring/models/tabular/    # CatBoost V4/V5, LightGBM V5
src/alpha_scoring/models/sequence/   # sequence-модели и Alpha-GRU
src/alpha_scoring/ensembling/        # averaging, id-prior, rank-blend
scripts/                             # локальный запуск
scripts/server/                      # серверный запуск
```
