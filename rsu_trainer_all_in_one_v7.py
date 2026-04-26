#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from vanet_ids_rsu_core import DEFAULT_SEQ_LEN, DEFAULT_WINDOW_SIZE, RSUMultiHeadTrainer


def run_training_cli(
    *,
    csv_path: str,
    out_dir: str,
    train_family: str,
    window_size: int,
    seq_len: int,
) -> Dict[str, Any]:
    trainer = RSUMultiHeadTrainer(
        train_family=train_family,
        window_size=window_size,
        seq_len=seq_len,
    )
    result = trainer.fit_csv(csv_path, output_dir=out_dir)
    summary = {
        "output_dir": result.output_dir,
        "manifest_path": result.manifest_path,
        "report_path": result.report_path,
        "enabled_heads": result.enabled_heads,
        "threshold": result.threshold,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RSU trainer wrapper for the thesis-aligned multi-head release pipeline.\n"
            "Without arguments it launches the GUI.\n"
            "With --csv and --out-dir it trains directly from the terminal."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--csv", help="Labeled training CSV that already includes attack_id for attacks.")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "release_v3"))
    parser.add_argument(
        "--train-family",
        default="all",
        choices=["binary", "all", "pos_speed", "replay_stale", "dos", "sybil"],
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--gui", action="store_true", help="Force GUI mode even if CLI arguments are present.")
    return parser


def launch_gui() -> None:
    from PyQt5.QtCore import QObject, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    class TrainWorker(QObject):
        log = pyqtSignal(str)
        finished = pyqtSignal(dict)
        failed = pyqtSignal(str)

        def __init__(self, csv_path: str, out_dir: str, family: str, window_size: int, seq_len: int):
            super().__init__()
            self.csv_path = csv_path
            self.out_dir = out_dir
            self.family = family
            self.window_size = int(window_size)
            self.seq_len = int(seq_len)

        def run(self) -> None:
            try:
                self.log.emit(f"[I] csv={self.csv_path}")
                self.log.emit(f"[I] out_dir={self.out_dir}")
                self.log.emit(f"[I] family={self.family} window={self.window_size} seq={self.seq_len}")
                summary = run_training_cli(
                    csv_path=self.csv_path,
                    out_dir=self.out_dir,
                    train_family=self.family,
                    window_size=self.window_size,
                    seq_len=self.seq_len,
                )
                self.finished.emit(summary)
            except Exception:
                self.failed.emit(traceback.format_exc())

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("RSU Trainer — All-in-One v7")
            self.resize(900, 560)
            self.thread: Optional[QThread] = None
            self.worker: Optional[TrainWorker] = None

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            form = QFormLayout()

            self.csv_edit = QLineEdit()
            browse_csv = QPushButton("Browse")
            browse_csv.clicked.connect(self.pick_csv)
            row_csv = QWidget()
            row_csv_layout = QHBoxLayout(row_csv)
            row_csv_layout.setContentsMargins(0, 0, 0, 0)
            row_csv_layout.addWidget(self.csv_edit)
            row_csv_layout.addWidget(browse_csv)
            form.addRow("Training CSV", row_csv)

            self.out_edit = QLineEdit(os.path.join(os.path.dirname(__file__), "release_v3"))
            browse_out = QPushButton("Browse")
            browse_out.clicked.connect(self.pick_out_dir)
            row_out = QWidget()
            row_out_layout = QHBoxLayout(row_out)
            row_out_layout.setContentsMargins(0, 0, 0, 0)
            row_out_layout.addWidget(self.out_edit)
            row_out_layout.addWidget(browse_out)
            form.addRow("Output Dir", row_out)

            self.family_combo = QComboBox()
            self.family_combo.addItems(["all", "binary", "pos_speed", "replay_stale", "dos", "sybil"])
            form.addRow("Training Family", self.family_combo)

            self.window_spin = QSpinBox()
            self.window_spin.setRange(5, 100)
            self.window_spin.setValue(DEFAULT_WINDOW_SIZE)
            form.addRow("Window Size", self.window_spin)

            self.seq_spin = QSpinBox()
            self.seq_spin.setRange(5, 100)
            self.seq_spin.setValue(DEFAULT_SEQ_LEN)
            form.addRow("Replay Seq Len", self.seq_spin)

            layout.addLayout(form)

            self.start_btn = QPushButton("Start Training")
            self.start_btn.clicked.connect(self.start_training)
            layout.addWidget(self.start_btn)

            layout.addWidget(QLabel("Log"))
            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            layout.addWidget(self.log_view)

        def pick_csv(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Choose training CSV", "", "CSV Files (*.csv)")
            if path:
                self.csv_edit.setText(path)

        def pick_out_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Choose output directory")
            if path:
                self.out_edit.setText(path)

        def append_log(self, text: str) -> None:
            self.log_view.appendPlainText(text)

        def start_training(self) -> None:
            csv_path = self.csv_edit.text().strip()
            out_dir = self.out_edit.text().strip()
            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(self, "Missing CSV", "Choose a valid training CSV.")
                return
            if not out_dir:
                QMessageBox.warning(self, "Missing Output", "Choose an output directory.")
                return

            self.start_btn.setEnabled(False)
            self.log_view.clear()

            self.thread = QThread()
            self.worker = TrainWorker(
                csv_path=csv_path,
                out_dir=out_dir,
                family=self.family_combo.currentText(),
                window_size=self.window_spin.value(),
                seq_len=self.seq_spin.value(),
            )
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.log.connect(self.append_log)
            self.worker.finished.connect(self.on_done)
            self.worker.failed.connect(self.on_fail)
            self.worker.finished.connect(lambda _: self.thread.quit())
            self.worker.failed.connect(lambda _: self.thread.quit())
            self.thread.finished.connect(lambda: self.start_btn.setEnabled(True))
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()

        def on_done(self, summary: Dict[str, Any]) -> None:
            self.append_log(json.dumps(summary, indent=2))
            QMessageBox.information(self, "Training Complete", f"Artifacts saved in:\n{summary['output_dir']}")

        def on_fail(self, err: str) -> None:
            self.append_log(err)
            QMessageBox.critical(self, "Training Failed", "See the log for details.")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.gui or not args.csv:
        launch_gui()
        return
    run_training_cli(
        csv_path=args.csv,
        out_dir=args.out_dir,
        train_family=args.train_family,
        window_size=args.window_size,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
