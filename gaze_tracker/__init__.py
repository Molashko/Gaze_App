"""Публичный API пакета трекера взгляда."""

from .ui.app import main as run_app

__version__ = "0.1.1"
main = run_app

__all__ = ["run_app", "main", "__version__"]
