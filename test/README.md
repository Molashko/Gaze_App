# Модуль В — тестирование модели

Эта папка содержит полный комплект артефактов для инвариантного тестирования:

- 4 модульных теста (`pytest`)
- конфигурацию потоков автозапуска
- UI-лаунчер на `PyQt5`
- отчетные документы

## Установка зависимостей

```bash
python -m pip install pytest PyQt5
```

## Запуск из CLI

Полный набор:

```bash
python -m pytest test/tests -q
```

Через flow-конфиг:

```bash
python test/run_tests.py --flow full
python test/run_tests.py --flow smoke
python test/run_tests.py --flow tracker_only
```

## Запуск UI

```bash
python test/ui/test_runner_ui.py
```

В UI можно:

- выбрать модули галочками
- запустить flow из списка
- смотреть live-лог
- экспортировать лог в файл
