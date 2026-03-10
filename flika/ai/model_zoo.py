"""BioImage.IO Model Zoo browser for flika.

Browse, download and run models from the BioImage.IO model collection
(https://bioimage.io) directly within flika.
"""
from __future__ import annotations

import os
import json
from typing import List, Optional

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui

from flika.logger import logger
import flika.global_vars as g


COLLECTION_URL = 'https://bioimage-io.github.io/collection-bioimage-io/collection.json'
MODELS_DIR = os.path.join(os.path.expanduser('~'), '.FLIKA', 'models')


def _model_cache_dir(model_id: str) -> str:
    """Return the local cache directory for a given model id."""
    safe_name = model_id.replace('/', '_').replace('\\', '_')
    return os.path.join(MODELS_DIR, safe_name)


def _is_model_cached(model_id: str) -> bool:
    """Check if a model has been downloaded locally."""
    cache_dir = _model_cache_dir(model_id)
    return os.path.isdir(cache_dir) and os.path.isfile(
        os.path.join(cache_dir, '.downloaded')
    )


class FetchCollectionWorker(QtCore.QThread):
    """Fetch the BioImage.IO collection JSON in a background thread."""
    finished = QtCore.Signal(list)
    error = QtCore.Signal(str)

    def run(self):
        try:
            import requests
            resp = requests.get(COLLECTION_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Filter for model resources
            models = [r for r in data.get('collection', []) if r.get('type') == 'model']
            self.finished.emit(models)
        except Exception as e:
            self.error.emit(str(e))


class DownloadModelWorker(QtCore.QThread):
    """Download a model from BioImage.IO to local cache."""
    finished = QtCore.Signal(str)  # model_id that was downloaded
    progress = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, model_id: str, parent=None):
        super().__init__(parent)
        self.model_id = model_id

    def run(self):
        try:
            import bioimageio.spec
            cache_dir = _model_cache_dir(self.model_id)
            os.makedirs(cache_dir, exist_ok=True)
            self.progress.emit(f'Downloading {self.model_id}...')
            # load_description fetches and caches the model spec + weights
            model_desc = bioimageio.spec.load_description(self.model_id)
            # Persist a marker so we know this model is locally available,
            # and store the rdf_source for reloading without network.
            meta = {
                'model_id': self.model_id,
                'name': str(getattr(model_desc, 'name', self.model_id)),
            }
            # If the model description has a local root path, record it
            if hasattr(model_desc, 'root'):
                meta['root'] = str(model_desc.root)
            with open(os.path.join(cache_dir, '.downloaded'), 'w') as f:
                json.dump(meta, f)
            self.finished.emit(self.model_id)
        except Exception as e:
            self.error.emit(str(e))


class RunModelWorker(QtCore.QThread):
    """Run a BioImage.IO model on the current window data."""
    finished = QtCore.Signal(object)  # result array
    progress = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, model_id: str, data: np.ndarray, parent=None):
        super().__init__(parent)
        self.model_id = model_id
        self.data = data

    @staticmethod
    def _pad_to_divisible(arr, divisor=16):
        """Pad spatial dimensions (last 2) to be divisible by *divisor*."""
        pads = []
        for i, s in enumerate(arr.shape):
            if i >= arr.ndim - 2:  # spatial dims
                remainder = s % divisor
                pads.append((0, (divisor - remainder) % divisor))
            else:
                pads.append((0, 0))
        if all(p[1] == 0 for p in pads):
            return arr, None
        return np.pad(arr, pads, mode='reflect'), tuple(s for s in arr.shape)

    def run(self):
        try:
            import bioimageio.core
            import bioimageio.spec
            self.progress.emit('Loading model...')
            # load_description will use its internal cache if already downloaded
            model_desc = bioimageio.spec.load_description(self.model_id)
            self.progress.emit('Running prediction...')
            # Prepare input — add batch dimension if needed
            inp = self.data
            if inp.ndim == 2:
                inp = inp[np.newaxis, np.newaxis, ...]  # (1,1,H,W)
            elif inp.ndim == 3:
                inp = inp[np.newaxis, ...]  # (1,T,H,W)
            # Pad spatial dims to be divisible by 16 (common U-Net requirement)
            inp, orig_shape = self._pad_to_divisible(inp, divisor=16)
            # bioimageio.core.predict() uses keyword-only args,
            # returns Sample(members={MemberId: Tensor}, ...)
            sample = bioimageio.core.predict(model=model_desc, inputs=inp)
            # Extract first output tensor from Sample.members dict
            result = next(iter(sample.members.values()))
            result = np.squeeze(np.asarray(result))
            # Crop back to original spatial dimensions
            if orig_shape is not None:
                crop = tuple(slice(0, s) for s in orig_shape)
                # Match trailing spatial dims
                while len(crop) < result.ndim:
                    crop = (slice(None),) + crop
                result = result[crop[-result.ndim:]]
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ModelZooBrowser(QtWidgets.QDialog):
    """Dialog for browsing and running BioImage.IO models."""

    _COL_NAME = 0
    _COL_DESC = 1
    _COL_TAGS = 2
    _COL_STATUS = 3
    _COL_DOWNLOADS = 4
    _NUM_COLS = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('BioImage.IO Model Zoo')
        self.resize(850, 500)
        self._models: List[dict] = []
        self._filtered_models: List[dict] = []
        self._worker = None
        self._setup_ui()
        self._fetch_collection()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Search bar and filter
        top_layout = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText('Search models...')
        self.search_edit.textChanged.connect(self._filter_models)
        top_layout.addWidget(self.search_edit)

        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.addItems(['All Tasks', 'segmentation', 'denoising', 'classification', 'detection', 'restoration'])
        self.task_combo.currentTextChanged.connect(self._filter_models)
        top_layout.addWidget(self.task_combo)
        layout.addLayout(top_layout)

        # Model table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(self._NUM_COLS)
        self.table.setHorizontalHeaderLabels(['Name', 'Description', 'Tags', 'Status', 'Downloads'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self._COL_NAME, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self._COL_DESC, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(self._COL_TAGS, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self._COL_STATUS, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self._COL_DOWNLOADS, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table)

        # Status label
        self.status_label = QtWidgets.QLabel('Fetching model collection...')
        layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.download_btn = QtWidgets.QPushButton('Download')
        self.download_btn.clicked.connect(self._download_selected)
        self.download_btn.setEnabled(False)
        btn_layout.addWidget(self.download_btn)

        self.run_btn = QtWidgets.QPushButton('Run on Current Window')
        self.run_btn.clicked.connect(self._run_selected)
        self.run_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)

        btn_layout.addStretch()
        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _fetch_collection(self):
        self._fetch_worker = FetchCollectionWorker()
        self._fetch_worker.finished.connect(self._on_collection_fetched)
        self._fetch_worker.error.connect(self._on_fetch_error)
        self._fetch_worker.start()

    def _on_collection_fetched(self, models):
        self._models = models
        self._filter_models()
        self.status_label.setText(f'{len(models)} models available')
        self.download_btn.setEnabled(True)
        self.run_btn.setEnabled(True)

    def _on_fetch_error(self, msg):
        self.status_label.setText(f'Error fetching collection: {msg}')

    def _model_id(self, model_dict: dict) -> str:
        return model_dict.get('id', model_dict.get('name', ''))

    def _populate_table(self, models):
        self._filtered_models = models
        self.table.setRowCount(len(models))
        for i, m in enumerate(models):
            mid = self._model_id(m)
            name = m.get('name', mid)
            desc = m.get('description', '')[:100]
            tags = ', '.join(m.get('tags', [])[:5])
            downloads = str(m.get('download_count', m.get('downloads', '')))
            cached = _is_model_cached(mid)

            self.table.setItem(i, self._COL_NAME, QtWidgets.QTableWidgetItem(str(name)))
            self.table.setItem(i, self._COL_DESC, QtWidgets.QTableWidgetItem(str(desc)))
            self.table.setItem(i, self._COL_TAGS, QtWidgets.QTableWidgetItem(str(tags)))
            self.table.setItem(i, self._COL_DOWNLOADS, QtWidgets.QTableWidgetItem(str(downloads)))

            status_item = QtWidgets.QTableWidgetItem('Downloaded' if cached else '')
            if cached:
                status_item.setForeground(QtGui.QBrush(QtGui.QColor(0, 150, 0)))
            self.table.setItem(i, self._COL_STATUS, status_item)

    def _filter_models(self):
        text = self.search_edit.text().lower()
        task = self.task_combo.currentText()
        filtered = []
        for m in self._models:
            name = str(m.get('name', m.get('id', ''))).lower()
            desc = str(m.get('description', '')).lower()
            tags = [str(t).lower() for t in m.get('tags', [])]
            if text and text not in name and text not in desc and not any(text in t for t in tags):
                continue
            if task != 'All Tasks' and task.lower() not in tags:
                continue
            filtered.append(m)
        self._populate_table(filtered)

    def _get_selected_model(self):
        """Return the model dict for the selected row, or None."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        if 0 <= row < len(self._filtered_models):
            return self._filtered_models[row]
        return None

    def _on_selection_changed(self):
        m = self._get_selected_model()
        if m is None:
            self.download_btn.setText('Download')
            return
        mid = self._model_id(m)
        if _is_model_cached(mid):
            self.download_btn.setText('Re-download')
        else:
            self.download_btn.setText('Download')

    def _download_selected(self):
        m = self._get_selected_model()
        if m is None:
            g.alert('Please select a model first.')
            return
        mid = self._model_id(m)
        self.status_label.setText(f'Downloading {mid}...')
        self.download_btn.setEnabled(False)
        worker = DownloadModelWorker(mid, self)
        worker.finished.connect(self._on_download_finished)
        worker.error.connect(self._on_download_error)
        worker.start()
        self._worker = worker

    def _on_download_finished(self, model_id):
        self.download_btn.setEnabled(True)
        self.status_label.setText(f'{model_id} downloaded successfully')
        # Refresh the table to update status column
        self._filter_models()
        self._on_selection_changed()

    def _on_download_error(self, msg):
        self.download_btn.setEnabled(True)
        self.status_label.setText(f'Download error: {msg}')

    def _run_selected(self):
        m = self._get_selected_model()
        if m is None:
            g.alert('Please select a model first.')
            return
        mid = self._model_id(m)
        if not _is_model_cached(mid):
            g.alert('Please download the model first before running.')
            return
        if g.win is None:
            g.alert('No window selected. Open an image first.')
            return
        data = g.win.image
        self.status_label.setText(f'Running {mid}...')
        self.run_btn.setEnabled(False)
        worker = RunModelWorker(mid, data, self)
        worker.finished.connect(self._on_run_finished)
        worker.progress.connect(lambda msg: self.status_label.setText(msg))
        worker.error.connect(self._on_run_error)
        worker.start()
        self._worker = worker

    def _on_run_finished(self, result):
        from flika.window import Window
        self.run_btn.setEnabled(True)
        self.status_label.setText('Prediction complete')
        Window(result, 'Model Zoo Result')

    def _on_run_error(self, msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText(f'Run error: {msg}')
