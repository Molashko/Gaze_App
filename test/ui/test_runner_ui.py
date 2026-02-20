import json
import subprocess
import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets


BASE_DIR = Path(__file__).resolve().parents[1]
FLOW_CONFIG = BASE_DIR / "test_flow_config.json"
TEST_DIR = BASE_DIR / "tests"

MODULE_MAP = {
    "Калибровка": "test_calibration_module.py",
    "Признаки взгляда": "test_gaze_features_module.py",
    "Поза головы": "test_head_pose_module.py",
    "Трекер": "test_tracker_module.py",
}

FLOW_LABELS = {
    "smoke": "Быстрый (smoke)",
    "full": "Полный",
    "calibration_only": "Только калибровка",
    "gaze_only": "Только признаки взгляда",
    "head_pose_only": "Только поза головы",
    "tracker_only": "Только трекер",
}


class TestRunnerWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Запуск тестов - Gaze Tracker")
        self.resize(980, 700)

        self.process: QtCore.QProcess | None = None
        self.module_boxes: dict[str, QtWidgets.QCheckBox] = {}
        self.flow_combo = QtWidgets.QComboBox()
        self.log_output = QtWidgets.QPlainTextEdit()
        self.status_label = QtWidgets.QLabel("Готово")
        self.last_exit_code: int | None = None
        self.flow_key_by_label: dict[str, str] = {}

        self._build_ui()
        self._load_flows()

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)

        title = QtWidgets.QLabel("Панель тестирования (4 модуля)")
        title.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        layout.addWidget(title)

        module_group = QtWidgets.QGroupBox("Выбор тестовых модулей")
        module_layout = QtWidgets.QGridLayout(module_group)
        for i, name in enumerate(MODULE_MAP):
            box = QtWidgets.QCheckBox(name)
            box.setChecked(True)
            self.module_boxes[name] = box
            module_layout.addWidget(box, i // 2, i % 2)
        layout.addWidget(module_group)

        flow_group = QtWidgets.QGroupBox("Поток автозапуска")
        flow_layout = QtWidgets.QHBoxLayout(flow_group)
        flow_layout.addWidget(QtWidgets.QLabel("Сценарий:"))
        flow_layout.addWidget(self.flow_combo)
        layout.addWidget(flow_group)

        buttons_layout = QtWidgets.QHBoxLayout()
        self.run_selected_btn = QtWidgets.QPushButton("Запустить выбранные модули")
        self.run_flow_btn = QtWidgets.QPushButton("Запустить flow")
        self.clear_btn = QtWidgets.QPushButton("Очистить лог")
        self.export_btn = QtWidgets.QPushButton("Экспорт лога")
        buttons_layout.addWidget(self.run_selected_btn)
        buttons_layout.addWidget(self.run_flow_btn)
        buttons_layout.addWidget(self.clear_btn)
        buttons_layout.addWidget(self.export_btn)
        layout.addLayout(buttons_layout)

        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Здесь появится поток вывода pytest...")
        layout.addWidget(self.log_output, 1)

        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(QtWidgets.QLabel("Статус:"))
        status_row.addWidget(self.status_label, 1)
        layout.addLayout(status_row)

        self.run_selected_btn.clicked.connect(self.run_selected_modules)
        self.run_flow_btn.clicked.connect(self.run_selected_flow)
        self.clear_btn.clicked.connect(self.log_output.clear)
        self.export_btn.clicked.connect(self.export_log)

    def _load_flows(self) -> None:
        with FLOW_CONFIG.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for flow_name in data.get("flows", {}).keys():
            label = FLOW_LABELS.get(flow_name, flow_name)
            self.flow_key_by_label[label] = flow_name
            self.flow_combo.addItem(label)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        cursor = self.log_output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.run_selected_btn.setEnabled(enabled)
        self.run_flow_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)
        for box in self.module_boxes.values():
            box.setEnabled(enabled)
        self.flow_combo.setEnabled(enabled)

    def _start_process(self, args: list[str], status_prefix: str) -> None:
        if self.process is not None:
            QtWidgets.QMessageBox.warning(self, "Занято", "Тесты уже выполняются.")
            return

        self._set_controls_enabled(False)
        self.status_label.setText(f"{status_prefix}...")
        self._append_log(f"\n>>> {' '.join(args)}\n")

        process = QtCore.QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(args)
        process.setWorkingDirectory(str(BASE_DIR))
        process.setProcessChannelMode(QtCore.QProcess.SeparateChannels)

        process.readyReadStandardOutput.connect(
            lambda: self._append_log(bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace"))
        )
        process.readyReadStandardError.connect(
            lambda: self._append_log(bytes(process.readAllStandardError()).decode("utf-8", errors="replace"))
        )
        process.finished.connect(self._on_process_finished)
        process.start()
        self.process = process

    def _on_process_finished(self, exit_code: int, _exit_status: QtCore.QProcess.ExitStatus) -> None:
        self.last_exit_code = int(exit_code)
        if exit_code == 0:
            self.status_label.setText("Успешно: все выбранные тесты прошли")
            self._append_log(">>> Итог: Все хорошо, тесты прошли успешно.\n")
        else:
            self.status_label.setText(f"Ошибка: pytest завершился с кодом {exit_code}")
            self._append_log(">>> Итог: Есть проблемы, часть тестов завершилась с ошибкой.\n")
        self._append_log(f"\n>>> Завершено. Код выхода: {exit_code}\n")
        self.process = None
        self._set_controls_enabled(True)

    def run_selected_modules(self) -> None:
        selected = [
            MODULE_MAP[name] for name, box in self.module_boxes.items() if box.isChecked()
        ]
        if not selected:
            QtWidgets.QMessageBox.information(self, "Нет выбора", "Выберите хотя бы один модуль.")
            return
        args = ["-m", "pytest"] + [str(TEST_DIR / name) for name in selected] + ["-q"]
        self._start_process(args=args, status_prefix="Запуск выбранных модулей")

    def run_selected_flow(self) -> None:
        flow_label = self.flow_combo.currentText().strip()
        flow_name = self.flow_key_by_label.get(flow_label, flow_label)
        with FLOW_CONFIG.open("r", encoding="utf-8") as f:
            data = json.load(f)
        flow = data.get("flows", {}).get(flow_name)
        if not flow:
            QtWidgets.QMessageBox.warning(self, "Flow не найден", f"Не найден flow: {flow_name}")
            return
        args = ["-m", "pytest"] + flow.get("pytest_args", [])
        self._start_process(args=args, status_prefix=f"Запуск flow '{flow_name}'")

    def export_log(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить лог",
            str(BASE_DIR / "last_test_run.log"),
            "Log files (*.log);;Text files (*.txt)",
        )
        if not path:
            return
        Path(path).write_text(self.log_output.toPlainText(), encoding="utf-8")
        QtWidgets.QMessageBox.information(self, "Сохранено", f"Лог сохранен:\n{path}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = TestRunnerWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
