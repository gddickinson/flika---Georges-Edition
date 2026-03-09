# -*- coding: utf-8 -*-
"""
Results Table — spreadsheet view for SPT localization data.

Provides a sortable, filterable, exportable table (like ImageJ
ThunderSTORM results) backed by :class:`ParticleData`.
"""
import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt, Signal

from ..logger import logger
from ..utils.singleton import DockSingleton


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------

class ParticleTableModel(QtCore.QAbstractTableModel):
    """Virtualised table model wrapping a pandas DataFrame.

    Only the visible cells are formatted on demand so performance is
    acceptable for 100k–1M+ rows.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._columns = []

    def set_dataframe(self, df):
        """Replace the backing DataFrame."""
        self.beginResetModel()
        self._df = df
        self._columns = list(df.columns)
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._df)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    return ''
                return f'{value:.4g}'
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._columns):
                return self._columns[section]
        else:
            return str(section)
        return None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if column < 0 or column >= len(self._columns):
            return
        col_name = self._columns[column]
        ascending = (order == Qt.SortOrder.AscendingOrder)
        self.beginResetModel()
        self._df = self._df.sort_values(col_name, ascending=ascending,
                                         ignore_index=True)
        self.endResetModel()

    @property
    def dataframe(self):
        return self._df


# ---------------------------------------------------------------------------
# Results table dock widget
# ---------------------------------------------------------------------------

class ResultsTableWidget(DockSingleton):
    """Singleton dockable results table for SPT localization data.

    Features:
    - Sortable column headers
    - Filter bar with pandas query() expressions
    - Column visibility dialog
    - CSV export (flika format + ThunderSTORM format)
    - Context menu: copy rows, jump to frame
    """

    sigRowSelected = Signal(int, float, float)  # frame, x, y

    def __init__(self, parent=None):
        super().__init__("SPT Results", parent)
        self.setObjectName("SPTResultsTable")
        self.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumHeight(200)
        self.setMinimumWidth(400)

        self._particle_data = None
        self._full_df = pd.DataFrame()
        self._build_ui()
        self._connect()

    def _build_ui(self):
        main = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(main)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Filter bar
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Filter:"))
        self._filter_edit = QtWidgets.QLineEdit()
        self._filter_edit.setPlaceholderText(
            "e.g. track_id == 5  or  intensity > 100")
        filter_row.addWidget(self._filter_edit)
        self._apply_btn = QtWidgets.QPushButton("Apply")
        filter_row.addWidget(self._apply_btn)
        self._clear_btn = QtWidgets.QPushButton("Clear")
        filter_row.addWidget(self._clear_btn)
        layout.addLayout(filter_row)

        # Status row
        status_row = QtWidgets.QHBoxLayout()
        self._row_count_label = QtWidgets.QLabel("0 rows")
        status_row.addWidget(self._row_count_label)
        status_row.addStretch()
        self._selected_label = QtWidgets.QLabel("Selected: 0")
        status_row.addWidget(self._selected_label)
        layout.addLayout(status_row)

        # Table view
        self._model = ParticleTableModel(self)
        self._table = QtWidgets.QTableView()
        self._table.setModel(self._model)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        layout.addWidget(self._table)

        # Button row
        btn_row = QtWidgets.QHBoxLayout()
        self._export_csv_btn = QtWidgets.QPushButton("Export CSV...")
        btn_row.addWidget(self._export_csv_btn)
        self._export_ts_btn = QtWidgets.QPushButton("Export TS CSV...")
        self._export_ts_btn.setToolTip("Export in ThunderSTORM format")
        btn_row.addWidget(self._export_ts_btn)
        self._columns_btn = QtWidgets.QPushButton("Columns...")
        btn_row.addWidget(self._columns_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.setWidget(main)

    def _connect(self):
        self._apply_btn.clicked.connect(self._apply_filter)
        self._clear_btn.clicked.connect(self._clear_filter)
        self._filter_edit.returnPressed.connect(self._apply_filter)
        self._export_csv_btn.clicked.connect(self._export_flika_csv)
        self._export_ts_btn.clicked.connect(self._export_thunderstorm_csv)
        self._columns_btn.clicked.connect(self._show_column_dialog)
        self._table.customContextMenuRequested.connect(self._context_menu)
        self._table.selectionModel().selectionChanged.connect(
            self._on_selection_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_particle_data(self, particle_data):
        """Set the ParticleData to display."""
        self._particle_data = particle_data
        self._full_df = particle_data.df.copy()
        self._model.set_dataframe(self._full_df)
        self._update_row_count()

    def set_dataframe(self, df):
        """Set a raw DataFrame (when no ParticleData is available)."""
        self._particle_data = None
        self._full_df = df.copy()
        self._model.set_dataframe(self._full_df)
        self._update_row_count()

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def _apply_filter(self):
        expr = self._filter_edit.text().strip()
        if not expr:
            self._clear_filter()
            return
        try:
            filtered = self._full_df.query(expr).reset_index(drop=True)
            self._model.set_dataframe(filtered)
            self._update_row_count()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Filter Error",
                f"Invalid filter expression:\n{exc}")

    def _clear_filter(self):
        self._filter_edit.clear()
        self._model.set_dataframe(self._full_df)
        self._update_row_count()

    def _update_row_count(self):
        total = len(self._full_df)
        shown = self._model.rowCount()
        if shown == total:
            self._row_count_label.setText(f"{total:,} rows")
        else:
            self._row_count_label.setText(
                f"{shown:,} of {total:,} rows (filtered)")

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        indices = self._table.selectionModel().selectedRows()
        n = len(indices)
        self._selected_label.setText(f"Selected: {n}")

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        copy_action = menu.addAction("Copy selected rows")
        jump_action = menu.addAction("Jump to frame")
        select_track_action = menu.addAction("Select track")

        action = menu.exec(self._table.viewport().mapToGlobal(pos))

        if action == copy_action:
            self._copy_selected_rows()
        elif action == jump_action:
            self._jump_to_frame()
        elif action == select_track_action:
            self._select_track()

    def _copy_selected_rows(self):
        indices = self._table.selectionModel().selectedRows()
        if not indices:
            return
        rows = sorted(idx.row() for idx in indices)
        df = self._model.dataframe
        subset = df.iloc[rows]
        text = subset.to_csv(index=False)
        QtWidgets.QApplication.clipboard().setText(text)

    def _jump_to_frame(self):
        indices = self._table.selectionModel().selectedRows()
        if not indices:
            return
        row = indices[0].row()
        df = self._model.dataframe
        if 'frame' not in df.columns:
            return
        frame = int(df.iat[row, df.columns.get_loc('frame')])
        try:
            from .. import global_vars as g
            win = getattr(g, 'win', None)
            if win is not None and hasattr(win, 'setIndex'):
                win.setIndex(frame)
        except Exception:
            pass

        # Emit signal with position
        if 'x' in df.columns and 'y' in df.columns:
            x = float(df.iat[row, df.columns.get_loc('x')])
            y = float(df.iat[row, df.columns.get_loc('y')])
            self.sigRowSelected.emit(frame, x, y)

    def _select_track(self):
        indices = self._table.selectionModel().selectedRows()
        if not indices:
            return
        row = indices[0].row()
        df = self._model.dataframe
        if 'track_id' not in df.columns:
            return
        tid = int(df.iat[row, df.columns.get_loc('track_id')])
        if tid < 0:
            return
        self._filter_edit.setText(f"track_id == {tid}")
        self._apply_filter()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_flika_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", "results.csv",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            df = self._model.dataframe
            df.to_csv(path, index=False)
            logger.info("Exported %d rows to %s", len(df), path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Export Error", f"Export failed:\n{exc}")

    def _export_thunderstorm_csv(self):
        if self._particle_data is not None:
            # Use ParticleData for proper conversion
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export ThunderSTORM CSV", "results_thunderstorm.csv",
                "CSV Files (*.csv);;All Files (*)")
            if not path:
                return
            try:
                # Ask for pixel size
                pixel_size, ok = QtWidgets.QInputDialog.getDouble(
                    self, "Pixel Size",
                    "Pixel size (nm/px):", 108.0, 1.0, 10000.0, 1)
                if not ok:
                    return
                self._particle_data.to_thunderstorm_csv(path, pixel_size)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Export Error", f"Export failed:\n{exc}")
        else:
            # Fallback: use spt_formats writer
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export ThunderSTORM CSV", "results_thunderstorm.csv",
                "CSV Files (*.csv);;All Files (*)")
            if not path:
                return
            try:
                pixel_size, ok = QtWidgets.QInputDialog.getDouble(
                    self, "Pixel Size",
                    "Pixel size (nm/px):", 108.0, 1.0, 10000.0, 1)
                if not ok:
                    return
                from ..spt.io.spt_formats import write_localizations
                write_localizations(self._model.dataframe, path,
                                     format='thunderstorm',
                                     pixel_size=pixel_size)
                logger.info("Exported ThunderSTORM CSV to %s", path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Export Error", f"Export failed:\n{exc}")

    # ------------------------------------------------------------------
    # Column visibility
    # ------------------------------------------------------------------

    def _show_column_dialog(self):
        """Show a dialog to toggle column visibility."""
        if self._full_df.empty:
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Column Visibility")
        layout = QtWidgets.QVBoxLayout(dlg)

        checkboxes = {}
        for col in self._full_df.columns:
            cb = QtWidgets.QCheckBox(col)
            cb.setChecked(not self._table.isColumnHidden(
                list(self._full_df.columns).index(col)))
            checkboxes[col] = cb
            layout.addWidget(cb)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            columns = list(self._model.dataframe.columns)
            for col, cb in checkboxes.items():
                if col in columns:
                    idx = columns.index(col)
                    self._table.setColumnHidden(idx, not cb.isChecked())

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    # closeEvent is handled by DockSingleton (calls cleanup + clears _instance)
