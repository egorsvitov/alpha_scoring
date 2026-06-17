# alpha_scoring

Минимальный финальный пайплайн для задачи кредитного скоринга Альфа-Банка.

Репозиторий намеренно содержит только последний полезный слой решения:

```text
исходные parquet
-> tabular features
-> CatBoost V4/V5 + LightGBM V5
-> sequence cache
-> full100 sequence ensemble
-> id-prior
-> Alpha-GRU + Payment Transformer
-> финальный submission
```

Leaderboard ROC-AUC: `0.786643`.

## Входные файлы

Данные лежат в каталоге `data/`:

- `train_data.parquet` — тренировочные кредитные истории;
- `test_data.parquet` — тестовые кредитные истории;
- `train_target.csv` — целевая переменная для train;
- `sample_submission.csv` — эталонный порядок `id`.

Готовые сабмиты, предвычисленные кеши, checkpoints и OOF-файлы не нужны для
запуска. Все модели обучаются с нуля на `train_data.parquet` и
`train_target.csv`, затем предсказывают `test_data.parquet`.

Данные, checkpoints, кеши и сабмиты игнорируются git-ом.

## Установка

Ожидается Python `3.10+`.

Пример установки для CUDA 11.8:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Если PyTorch уже установлен в окружении, последнюю команду можно пропустить.

## Запуск

Полный запуск:

```bash
bash scripts/run_final.sh
```

Скрипт выполняет четыре шага:

1. строит tabular features;
2. обучает CatBoost V4, CatBoost V5 и LightGBM V5;
3. строит sequence cache;
4. обучает full100 sequence ensemble;
5. собирает checkpoint-averaged base ensemble и добавляет `id-prior`;
6. обучает финальную `Alpha-GRU` ветку;
7. делает финальный rank-blend.

На выходе создаются:

- `sequence_cache/`;
- `features/`;
- `tabular_cache/`;
- `sequence_runs/`;
- `sequence_runs/blends/submission_full100_checkpoint_average_idprior.csv`;
- `sequence_runs/full100_alpha_gru_payment/submission_epoch_9.csv`;
- `submissions/submission_alpha_gru161.csv`.

Финальный файл для отправки:

```text
submissions/submission_alpha_gru161.csv
```

Финальная формула последнего шага:

```text
0.839 * sequence_runs/blends/submission_full100_checkpoint_average_idprior.csv
+ 0.161 * sequence_runs/full100_alpha_gru_payment/submission_epoch_9.csv
```

## Запуск на сервере

Серверные скрипты лежат в `scripts/`.

1. Создайте локальный конфиг:

```bash
cp server.env.example server.env
```

2. Отредактируйте `server.env`: укажите `REMOTE_HOST`, `REMOTE_PORT`,
   `REMOTE_USER`, `REMOTE_DIR`. Если используется пароль, добавьте
   `REMOTE_PASSWORD`; файл `server.env` не коммитится.

3. Синхронизируйте код:

```bash
bash scripts/sync_to_server.sh
```

4. При необходимости загрузите входные файлы из `data/`:

```bash
bash scripts/upload_inputs_to_server.sh
```

5. На сервере заранее создайте окружение и поставьте зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

6. Проверьте swap. Табличные бустинги могут занимать до `60 ГБ` ОЗУ, поэтому
   перед запуском нужен активный swap. Серверный запуск делает эту проверку
   автоматически через `scripts/ensure_swap.sh`. Если у пользователя нет
   passwordless `sudo`, выполните один раз вручную:

```bash
sudo SWAP_SIZE_GB=64 MIN_SWAP_GB=60 bash scripts/ensure_swap.sh
```

По умолчанию создается `/swapfile_alpha_scoring` и добавляется запись в
`/etc/fstab`.

7. Запустите полный пайплайн на сервере:

```bash
bash scripts/run_on_server.sh
```

8. Заберите готовый сабмит:

```bash
bash scripts/fetch_submission_from_server.sh
```

## Модель

Финальная модель называется `AlphaGRUClassifier`.

Схема:

```text
payment codes
-> Payment Transformer
-> product embedding
-> position embedding по rn
-> GRU по кредитным продуктам клиента
-> mean/max/attention pooling по GRU outputs
+ mean/max/attention pooling по входным product embeddings
+ projected final GRU hidden states
-> MLP
-> default score
```

Особенности обучения:

- scheduler: `OneCycleLR`;
- sample weights: линейный вес по нормализованному `id`;
- обучение на полном train;
- используется checkpoint `epoch_9`;
- вес в финальном rank-blend: `0.161`.

## Структура

```text
scripts/run_final.sh                 # полный self-contained запуск
scripts/ensure_swap.sh               # проверка/создание swap-файла на сервере
build_tabular_features.py            # построение табличных признаков
catboost_*.py / train_*submission.py # табличные модели
build_sequence_cache.py              # sequence cache
train_sequence_model.py              # sequence-модели и Alpha-GRU
assemble_checkpoint_average_submission.py
blend_prediction_files.py
scripts/sync_to_server.sh            # синхронизация кода на сервер
scripts/run_on_server.sh             # запуск команды на сервере
```

## Примечания

- Исходные данные, кеши, checkpoints и сабмиты не коммитятся.
- На вход полного запуска не подаются готовые prediction/submission-файлы.
- Для обучения требуется CUDA GPU.
- Порядок `id` в финальном файле проверяется по `sample_submission.csv`.
