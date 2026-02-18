"""Публичный API пакета трекера взгляда."""

from .ui.app import main as _ui_main

__version__ = "0.1.1"


def run_app() -> None:
    """Запустить GUI-приложение трекера взгляда."""
    _ui_main()


def main() -> None:
    """Алиас точки входа для обратной совместимости."""
    run_app()


__all__ = ["run_app", "main", "__version__"]
