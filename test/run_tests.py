import argparse
import json
from pathlib import Path
from typing import List

import pytest


BASE_DIR = Path(__file__).resolve().parent
FLOW_CONFIG_PATH = BASE_DIR / "test_flow_config.json"


def load_flow_config() -> dict:
    with FLOW_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_pytest_args(flow: str, modules: List[str]) -> List[str]:
    common = ["-c", str(BASE_DIR / "pytest.ini")]
    if modules:
        return common + [str(BASE_DIR / "tests" / name) for name in modules] + ["-q"]
    if flow:
        config = load_flow_config()
        flow_cfg = config["flows"].get(flow)
        if flow_cfg is None:
            available = ", ".join(config["flows"].keys())
            raise ValueError(f"Unknown flow '{flow}'. Available flows: {available}")
        return common + [str(BASE_DIR / "tests")] + flow_cfg["pytest_args"]
    return common + [str(BASE_DIR / "tests"), "-q"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Запуск модульных тестов проекта.")
    parser.add_argument("--flow", type=str, default="full", help="Имя потока из test_flow_config.json")
    parser.add_argument(
        "--modules",
        nargs="*",
        default=[],
        help="Список модулей внутри test/tests, например: test_tracker_module.py",
    )
    args = parser.parse_args()

    pytest_args = build_pytest_args(args.flow, args.modules)
    exit_code = pytest.main(pytest_args)
    if exit_code == 0:
        print("Итог: Все хорошо, тесты прошли успешно.")
    else:
        print("Итог: Есть проблемы, часть тестов завершилась с ошибкой.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
