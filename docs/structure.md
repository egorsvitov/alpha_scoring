# Project structure

Проект разделен на четыре рабочих слоя.

```text
src/alpha_scoring/features/
```

Строит производные данные: табличные признаки, sequence cache и tabular cache.
Эти модули читают `data/` и создают локальные артефакты для моделей.

```text
src/alpha_scoring/models/
```

Содержит все модели, участвующие в итоговом решении:

- `tabular/` — CatBoost V4/V5, LightGBM V5 и их block-validation helpers;
- `sequence/` — sequence ensemble, temporal ветки и Alpha-GRU.

```text
src/alpha_scoring/ensembling/
```

Собирает предсказания моделей: checkpoint averaging, sequence averaging,
`id-prior` и финальный rank-blend.

```text
scripts/
```

Оркестрирует воспроизводимый запуск. Главный локальный entrypoint:
`scripts/run_final.sh`. Серверная обвязка лежит отдельно в `scripts/server/`.
