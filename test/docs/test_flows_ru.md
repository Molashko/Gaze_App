# Конфигурация потоков автоматизированных тестов

Файл конфигурации: `test/test_flow_config.json`

## Доступные потоки

## `smoke`

- Назначение: быстрый контроль базовой работоспособности.
- Аргументы pytest: `-m smoke -q`

## `full`

- Назначение: полный прогон всех модульных тестов.
- Аргументы pytest: `-q`

## `calibration_only`

- Назначение: проверка только калибратора.
- Аргументы pytest: `tests/test_calibration_module.py -q`

## `gaze_only`

- Назначение: проверка только извлечения признаков взгляда.
- Аргументы pytest: `tests/test_gaze_features_module.py -q`

## `head_pose_only`

- Назначение: проверка только оценки позы головы.
- Аргументы pytest: `tests/test_head_pose_module.py -q`

## `tracker_only`

- Назначение: проверка только orchestration-логики трекера.
- Аргументы pytest: `tests/test_tracker_module.py -q`

## Способы запуска

CLI:

```bash
python test/run_tests.py --flow full
python test/run_tests.py --flow smoke
python test/run_tests.py --flow tracker_only
```

UI:

```bash
python test/ui/test_runner_ui.py
```

В интерфейсе flow выбирается через выпадающий список и запускается кнопкой.
