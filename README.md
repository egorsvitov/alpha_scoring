# alpha_scoring

Минимальный финальный пайплайн для задачи кредитного скоринга Альфа-Банка.

Репозиторий намеренно содержит только последний полезный слой решения:

```text
исходные parquet
-> sequence cache
-> Alpha-GRU + Payment Transformer
-> предсказание на test
-> rank-blend с базовым full100 id-prior submission
-> финальный submission
```

Итоговая формула последнего успешного сабмита:

```text
0.839 * submission_full100_checkpoint_average_idprior.csv
+ 0.161 * Alpha-GRU epoch 9 prediction
```

Leaderboard ROC-AUC: `0.786643`.

## Входные файлы

Данные лежат в каталоге `data/`:

- `train_data.parquet` — тренировочные кредитные истории;
- `test_data.parquet` — тестовые кредитные истории;
- `train_target.csv` — целевая переменная для train;
- `sample_submission.csv` — эталонный порядок `id`;
- `submission_full100_checkpoint_average_idprior.csv` — предыдущий лучший
  full100 ensemble submission с `id` target prior.

Последний файл нужен как базовая сильная модель. Этот репозиторий хранит только
финальный инкрементальный слой `Alpha-GRU`, поэтому старый исследовательский код
для CatBoost, LightGBM, Transformer-ансамблей и подбора id-prior сюда не входит.

Готовые сабмиты для сверки лежат в `submissions/`:

- `submission_full100_checkpoint_average_idprior_alpha_gru161.csv` — лучший
  отправленный вариант, LB `0.786643`;
- `submission_full100_checkpoint_average_idprior_alpha_gru150.csv` — запасной
  более консервативный blend.

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

1. строит sequence cache из `train_data.parquet` и `test_data.parquet`;
2. обучает `AlphaGRUClassifier` на полном train до `epoch_9`;
3. считает test prediction для checkpoint `epoch_9.pt`;
4. делает rank-blend с базовым `submission_full100_checkpoint_average_idprior.csv`.

На выходе создаются:

- `sequence_cache/`;
- `runs/full100_alpha_gru_payment/epoch_9.pt`;
- `runs/full100_alpha_gru_payment/submission_epoch_9.csv`;
- `submissions/submission_alpha_gru161.csv`.

Финальный файл для отправки:

```text
submissions/submission_alpha_gru161.csv
```

Если нужно только пересобрать финальный blend из уже готового Alpha-GRU
prediction, можно запустить:

```bash
PYTHONPATH=src python -m alpha_scoring.blend \
  --base data/submission_full100_checkpoint_average_idprior.csv \
  --alpha runs/full100_alpha_gru_payment/submission_epoch_9.csv \
  --sample data/sample_submission.csv \
  --alpha-weight 0.161 \
  --output submissions/submission_alpha_gru161.csv
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

6. Запустите полный пайплайн на сервере:

```bash
bash scripts/run_on_server.sh
```

7. Заберите готовый сабмит:

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
src/alpha_scoring/cache.py   # построение sequence_cache
src/alpha_scoring/model.py   # Payment Transformer + Alpha-GRU
src/alpha_scoring/train.py   # train / predict CLI
src/alpha_scoring/blend.py   # финальный rank-blend
scripts/run_final.sh         # полный воспроизводящий запуск
scripts/sync_to_server.sh    # синхронизация кода на сервер
scripts/run_on_server.sh     # запуск команды на сервере
```

## Примечания

- Исходные данные, кеши, checkpoints и сабмиты не коммитятся.
- Для обучения требуется CUDA GPU.
- Порядок `id` в финальном файле проверяется по `sample_submission.csv`.
