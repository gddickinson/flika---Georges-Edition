# -*- coding: utf-8 -*-
"""Benchmark Runner dialog for evaluating analysis pipelines."""
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Signal

from .benchmarks import BENCHMARKS, run_benchmark
from .evaluation import format_benchmark_report, BenchmarkResult


class BenchmarkWorker(QtCore.QThread):
    """Run benchmarks in a background thread."""
    sig_progress = Signal(str)
    sig_result = Signal(object)
    sig_finished = Signal(list)
    sig_error = Signal(str)

    def __init__(self, benchmark_names, parent=None):
        super().__init__(parent)
        self._names = benchmark_names

    def run(self):
        results = []
        try:
            for name in self._names:
                self.sig_progress.emit(f"Running {name}...")
                result = run_benchmark(name)
                results.append(result)
                self.sig_result.emit(result)
            self.sig_finished.emit(results)
        except Exception as e:
            self.sig_error.emit(str(e))


class BenchmarkDialog(QtWidgets.QDialog):
    """Dialog for running and viewing benchmark results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Benchmarks")
        self.setMinimumSize(700, 500)
        self._worker = None
        self._results = []

        layout = QtWidgets.QVBoxLayout(self)

        # Benchmark selection
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Benchmarks:"))

        self._select_all = QtWidgets.QPushButton("Select All")
        self._select_all.clicked.connect(self._on_select_all)
        self._select_none = QtWidgets.QPushButton("Select None")
        self._select_none.clicked.connect(self._on_select_none)
        top.addWidget(self._select_all)
        top.addWidget(self._select_none)
        top.addStretch()
        layout.addLayout(top)

        # Benchmark list with checkboxes, grouped by category
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Benchmark", "Category"])
        self._tree.setRootIsDecorated(True)
        self._category_items = {}

        for name, info in sorted(BENCHMARKS.items(),
                                  key=lambda x: x[1]['category']):
            cat = info['category']
            if cat not in self._category_items:
                cat_item = QtWidgets.QTreeWidgetItem([cat.upper(), ""])
                cat_item.setFlags(cat_item.flags() |
                                  QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                cat_item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                self._tree.addTopLevelItem(cat_item)
                self._category_items[cat] = cat_item

            item = QtWidgets.QTreeWidgetItem([name, cat])
            item.setFlags(item.flags() |
                          QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self._category_items[cat].addChild(item)

        self._tree.expandAll()
        layout.addWidget(self._tree)

        # Results table
        layout.addWidget(QtWidgets.QLabel("Results:"))
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Benchmark", "Category", "Key Metric", "Value", "Time (s)"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        # Status and buttons
        bottom = QtWidgets.QHBoxLayout()
        self._status = QtWidgets.QLabel("Ready")
        bottom.addWidget(self._status, 1)

        self._btn_run = QtWidgets.QPushButton("Run Selected")
        self._btn_run.clicked.connect(self._on_run)
        bottom.addWidget(self._btn_run)

        self._btn_export = QtWidgets.QPushButton("Export Report")
        self._btn_export.clicked.connect(self._on_export)
        self._btn_export.setEnabled(False)
        bottom.addWidget(self._btn_export)

        self._btn_close = QtWidgets.QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        bottom.addWidget(self._btn_close)

        layout.addLayout(bottom)

    def _get_selected_benchmarks(self):
        """Get list of checked benchmark names."""
        names = []
        for cat_item in self._category_items.values():
            for i in range(cat_item.childCount()):
                child = cat_item.child(i)
                if child.checkState(0) == QtCore.Qt.CheckState.Checked:
                    names.append(child.text(0))
        return names

    def _on_select_all(self):
        for cat_item in self._category_items.values():
            for i in range(cat_item.childCount()):
                cat_item.child(i).setCheckState(
                    0, QtCore.Qt.CheckState.Checked)

    def _on_select_none(self):
        for cat_item in self._category_items.values():
            for i in range(cat_item.childCount()):
                cat_item.child(i).setCheckState(
                    0, QtCore.Qt.CheckState.Unchecked)

    def _on_run(self):
        names = self._get_selected_benchmarks()
        if not names:
            QtWidgets.QMessageBox.information(
                self, "No Benchmarks", "Select at least one benchmark.")
            return

        self._btn_run.setEnabled(False)
        self._table.setRowCount(0)
        self._results = []

        self._worker = BenchmarkWorker(names, self)
        self._worker.sig_progress.connect(
            lambda msg: self._status.setText(msg))
        self._worker.sig_result.connect(self._on_result)
        self._worker.sig_finished.connect(self._on_finished)
        self._worker.sig_error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result):
        """Add a single result row to the table."""
        self._results.append(result)
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(result.name))
        self._table.setItem(row, 1,
                            QtWidgets.QTableWidgetItem(result.category))

        # Pick the most informative metric to show
        key_metric = _pick_key_metric(result)
        val = result.metrics.get(key_metric, 'N/A')
        if isinstance(val, float):
            val_str = f'{val:.4f}'
        else:
            val_str = str(val)

        self._table.setItem(row, 2, QtWidgets.QTableWidgetItem(key_metric))
        self._table.setItem(row, 3, QtWidgets.QTableWidgetItem(val_str))
        self._table.setItem(
            row, 4,
            QtWidgets.QTableWidgetItem(f'{result.elapsed_seconds:.2f}'))

    def _on_finished(self, results):
        self._btn_run.setEnabled(True)
        self._btn_export.setEnabled(True)
        n = len(results)
        self._status.setText(f"Done: {n} benchmarks completed.")

    def _on_error(self, msg):
        self._btn_run.setEnabled(True)
        self._status.setText("Error!")
        QtWidgets.QMessageBox.critical(self, "Benchmark Error", msg)

    def _on_export(self):
        if not self._results:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Benchmark Report", "benchmark_report.txt",
            "Text files (*.txt);;JSON files (*.json)")
        if not path:
            return

        if path.endswith('.json'):
            import json
            data = [r.to_dict() for r in self._results]
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            report = format_benchmark_report(self._results)
            with open(path, 'w') as f:
                f.write(report)

        self._status.setText(f"Report saved to {path}")


def _pick_key_metric(result):
    """Choose the most informative metric to display."""
    priority = ['f1', 'MOTA', 'event_f1', 'jaccard', 'dice',
                'precision', 'recall']
    for key in priority:
        if key in result.metrics:
            return key
    return next(iter(result.metrics), 'N/A')
