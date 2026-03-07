from ..logger import logger
logger.debug("Started 'reading app/settings_editor.py'")

import numpy as np
from qtpy import QtWidgets, QtGui
from multiprocessing import cpu_count
from ..utils.misc import setConsoleVisible
from ..utils.BaseProcess import BaseDialog, BaseProcess, ColorSelector, ComboBox
from .. import global_vars as g


__all__ = ['SettingsEditor', 'rectSettings', 'pointSettings', 'pencilSettings', 'csSettings']

data_types = ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

class SettingsEditor(BaseDialog):
    gui = None
    def __init__(self):
        from pyqtgraph import ComboBox

        old_dtype=g.settings['internal_data_type']
        dataDrop = ComboBox(items=data_types, default=old_dtype)
        showCheck = QtWidgets.QCheckBox()
        showCheck.setChecked(g.settings['show_windows'])
        multipleTracesCheck = QtWidgets.QCheckBox()
        multipleTracesCheck.setChecked(g.settings['multipleTraceWindows'])
        multiprocessing = QtWidgets.QCheckBox()
        multiprocessing.setChecked(g.settings['multiprocessing'])
        nCores = QtWidgets.QComboBox()
        debug_check = QtWidgets.QCheckBox(checked=g.settings['debug_mode'])
        debug_check.toggled.connect(setConsoleVisible)
        for i in np.arange(cpu_count())+1:
            nCores.addItem(str(i))
        nCores.setCurrentIndex(g.settings['nCores']-1)
        random_color_check = QtWidgets.QCheckBox()
        random_color_check.setChecked(g.settings['roi_color'] == 'random')
        roi_color = ColorSelector()
        roi_color.setEnabled(g.settings['roi_color'] != 'random')
        roi_color.color=g.settings['roi_color'] if g.settings['roi_color'] != 'random' else '#ffff00'

        default_roi = QtWidgets.QCheckBox()
        default_roi.setChecked(g.settings['default_roi_on_click'])

        apply_to_all_planes = QtWidgets.QCheckBox()
        apply_to_all_planes.setChecked(g.settings.get('apply_to_all_planes', False))

        default_axis_order = ComboBox()
        for order in ['Auto', 'TXYZ', 'TXYC', 'TZXY']:
            default_axis_order.addItem(order)
        current_order = g.settings.get('default_axis_order', 'Auto')
        idx = default_axis_order.findText(current_order)
        if idx >= 0:
            default_axis_order.setCurrentIndex(idx)

        api_key_edit = QtWidgets.QLineEdit()
        api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        api_key_edit.setPlaceholderText("sk-ant-...")
        existing_key = g.settings.d.get('anthropic_api_key', '')
        if existing_key:
            api_key_edit.setText(existing_key)

        accel_device = QtWidgets.QComboBox()
        for dev in ['Auto', 'CPU', 'CUDA', 'MPS']:
            accel_device.addItem(dev)
        current_dev = g.settings.get('acceleration_device', 'Auto') or 'Auto'
        idx_dev = accel_device.findText(current_dev)
        if idx_dev >= 0:
            accel_device.setCurrentIndex(idx_dev)

        gpu_mem_limit = QtWidgets.QSpinBox()
        gpu_mem_limit.setRange(0, 65536)
        gpu_mem_limit.setSuffix(" MB")
        gpu_mem_limit.setSpecialValueText("No limit")
        gpu_mem_limit.setValue(g.settings.get('gpu_memory_limit', 0) or 0)

        items = []
        items.append({'name': 'random_color_check', 'string': 'Randomly color ROIs', 'object': random_color_check})
        items.append({'name': 'roi_color', 'string': 'Default ROI Color', 'object': roi_color})
        items.append({'name': 'internal_data_type', 'string': 'Internal Data Type', 'object': dataDrop})
        items.append({'name': 'show_windows', 'string': 'Show Windows', 'object': showCheck})
        items.append({'name': 'multipleTraceWindows', 'string': 'Multiple Trace Windows', 'object': multipleTracesCheck})
        items.append({'name': 'multiprocessing', 'string': 'Multiprocessing On', 'object': multiprocessing})
        items.append({'name': 'nCores', 'string': 'Number of cores to use when multiprocessing', 'object': nCores})
        items.append({'name': 'debug_mode', 'string': 'Debug Mode', 'object': debug_check})
        items.append({'name': 'default_roi_on_click', 'string': 'Enable default ROI on right click', 'object': default_roi})
        items.append({'name': 'apply_to_all_planes', 'string': 'Apply filters to all Z/C planes', 'object': apply_to_all_planes})
        items.append({'name': 'default_axis_order', 'string': 'Default axis order for TIFF import', 'object': default_axis_order})
        items.append({'name': 'acceleration_device', 'string': 'Acceleration Device', 'object': accel_device})
        items.append({'name': 'gpu_memory_limit', 'string': 'GPU Memory Limit (0=no limit)', 'object': gpu_mem_limit})
        pixel_size = QtWidgets.QDoubleSpinBox()
        pixel_size.setRange(1.0, 100000.0)
        pixel_size.setDecimals(1)
        pixel_size.setSuffix(" nm")
        pixel_size.setValue(g.settings.get('pixel_size', 108.0) or 108.0)

        frame_interval = QtWidgets.QDoubleSpinBox()
        frame_interval.setRange(0.0001, 1000.0)
        frame_interval.setDecimals(4)
        frame_interval.setSuffix(" s")
        frame_interval.setValue(g.settings.get('frame_interval', 0.05) or 0.05)

        items.append({'name': 'pixel_size', 'string': 'Default Pixel Size', 'object': pixel_size})
        items.append({'name': 'frame_interval', 'string': 'Default Frame Interval', 'object': frame_interval})
        items.append({'name': 'anthropic_api_key', 'string': 'Anthropic API Key (for AI features)', 'object': api_key_edit})

        def update():
            g.settings['internal_data_type'] = str(dataDrop.currentText())
            g.settings['show_windows'] = showCheck.isChecked()
            g.settings['multipleTraceWindows'] = multipleTracesCheck.isChecked()
            g.settings['multiprocessing']=multiprocessing.isChecked()
            g.settings['nCores']=int(nCores.itemText(nCores.currentIndex()))
            g.settings['debug_mode'] = debug_check.isChecked()
            if not random_color_check.isChecked() and roi_color.color == 'random':
                roi_color.color = "#ffff00"
            g.settings['roi_color'] = roi_color.value() if not random_color_check.isChecked() else "random"
            roi_color.setEnabled(g.settings['roi_color'] != 'random')
            g.settings['default_roi_on_click'] = default_roi.isChecked()
            g.settings['apply_to_all_planes'] = apply_to_all_planes.isChecked()
            g.settings['default_axis_order'] = str(default_axis_order.currentText())
            g.settings['acceleration_device'] = str(accel_device.currentText())
            g.settings['gpu_memory_limit'] = gpu_mem_limit.value()
            g.settings['pixel_size'] = pixel_size.value()
            g.settings['frame_interval'] = frame_interval.value()
            key_text = api_key_edit.text().strip()
            if key_text:
                g.settings['anthropic_api_key'] = key_text
            elif 'anthropic_api_key' in g.settings.d:
                del g.settings.d['anthropic_api_key']


        super(SettingsEditor, self).__init__(items, 'flika settings', None)
        self.accepted.connect(update)
        self.changeSignal.connect(update)
        g.dialogs.append(self)

    @staticmethod
    def show():
        if SettingsEditor.gui == None:
            SettingsEditor.gui = SettingsEditor()
        BaseDialog.show(SettingsEditor.gui)


def pencilSettings(pencilButton):
    """
    default points color
    default points size
    currentWindow points color
    currentWindow points size
    clear current points

    """
    pencil_value = QtWidgets.QSpinBox()
    pencil_value.setRange(0, 1000)
    v = g.settings['pencil_value']
    if v is None:
        v = 0
    pencil_value.setValue(v)
    items = []
    items.append({'name': 'pencil_value', 'string': 'Pencil Value', 'object': pencil_value})
    def update_final():
        g.settings['pencil_value'] = pencil_value.value()
    dialog = BaseDialog(items, 'Points Settings', '')
    g.dialogs.append(dialog)
    dialog.accepted.connect(update_final)
    dialog.show()

def pointSettings(pointButton):
    """
    default points color
    default points size
    currentWindow points color
    currentWindow points size
    clear current points

    """
    point_color = ColorSelector()
    point_color.color=g.settings['point_color']
    point_color.label.setText(point_color.color)
    point_size = QtWidgets.QSpinBox()
    point_size.setRange(1, 50)
    point_size.setValue(g.settings['point_size'])
    show_all_points = QtWidgets.QCheckBox()
    show_all_points.setChecked(g.settings['show_all_points'])
    delete_all_points = QtWidgets.QCheckBox()
    delete_all_points.setChecked(False)
    
    update_current_points_check = QtWidgets.QCheckBox()
    update_current_points_check.setChecked(False)
    
    items = []
    items.append({'name': 'point_color', 'string': 'Default Point Color', 'object': point_color})
    items.append({'name': 'point_size', 'string': 'Default Point Size', 'object': point_size})
    items.append({'name': 'show_all_points', 'string': 'Show points from all frames', 'object': show_all_points})
    items.append({'name': 'update_current_points_check', 'string': 'Update already plotted points', 'object': update_current_points_check})
    items.append({'name': 'delete_all_points', 'string': 'Delete all points', 'object': delete_all_points})

    def update():
        from pyqtgraph import mkBrush
        win = g.win
        g.settings['point_color'] = point_color.value()
        g.settings['point_size'] = point_size.value()
        g.settings['show_all_points'] = show_all_points.isChecked()
        if win is not None and update_current_points_check.isChecked() == True:
            color = QtGui.QColor(point_color.value())
            size = point_size.value()
            for t in np.arange(win.mt):
                for i in np.arange(len(win.scatterPoints[t])):
                    win.scatterPoints[t][i][2] = color
                    win.scatterPoints[t][i][3] = size
            win.updateindex()
        if win is not None:
            if g.settings['show_all_points']:
                pts = []
                for t in np.arange(win.mt):
                    pts.extend(win.scatterPoints[t])
                point_sizes = [pt[3] for pt in pts]
                brushes = [mkBrush(*pt[2].getRgb()) for pt in pts]
                win.scatterPlot.setData(pos=pts, size=point_sizes, brush=brushes)
            else:
                win.updateindex()

    def update_final():
        win = g.win
        if win is not None and delete_all_points.isChecked():
            for t in np.arange(win.mt):
                win.scatterPoints[t] = []
            win.updateindex()
        update()

    dialog = BaseDialog(items, 'Points Settings', '')
    g.dialogs.append(dialog) 
    dialog.accepted.connect(update_final)
    dialog.changeSignal.connect(update)
    dialog.show()

def rectSettings(rectButton):
    """
    default rect height
    default rect width
    """
    rect_width = QtWidgets.QSpinBox()
    rect_width.setRange(1,500)
    rect_width.setValue(g.settings['rect_width'])
    rect_height = QtWidgets.QSpinBox()
    rect_height.setRange(1,500)
    rect_height.setValue(g.settings['rect_height'])

    items = []
    items.append({'name': 'rect_width', 'string': 'Default Rectangle Width', 'object': rect_width})
    items.append({'name': 'rect_height', 'string': 'Default Rectangle Height', 'object': rect_height})
    
    def update():
        win = g.win
        g.settings['rect_width'] = rect_width.value()
        g.settings['rect_height'] = rect_height.value()
        
    dialog = BaseDialog(items, 'Rectangle Settings', '')
    dialog.accepted.connect(update)
    dialog.changeSignal.connect(update)
    g.dialogs.append(dialog)
    dialog.show()

def csSettings(csButton):
    """Settings dialog for Center-Surround ROI tool (right-click on CS button)."""
    shape_combo = QtWidgets.QComboBox()
    for s in ['circle', 'ellipse', 'square']:
        shape_combo.addItem(s)
    current_shape = g.settings.get('cs_shape', 'circle') or 'circle'
    idx = shape_combo.findText(current_shape)
    if idx >= 0:
        shape_combo.setCurrentIndex(idx)

    inner_ratio = QtWidgets.QDoubleSpinBox()
    inner_ratio.setRange(0.05, 0.95)
    inner_ratio.setSingleStep(0.05)
    inner_ratio.setDecimals(2)
    inner_ratio.setValue(g.settings.get('cs_inner_ratio', 0.5) or 0.5)

    items = []
    items.append({'name': 'cs_shape', 'string': 'Shape', 'object': shape_combo})
    items.append({'name': 'cs_inner_ratio', 'string': 'Inner / Outer Ratio', 'object': inner_ratio})

    def update():
        g.settings['cs_shape'] = shape_combo.currentText()
        g.settings['cs_inner_ratio'] = inner_ratio.value()

    dialog = BaseDialog(items, 'Center-Surround Settings', '')
    dialog.accepted.connect(update)
    dialog.changeSignal.connect(update)
    g.dialogs.append(dialog)
    dialog.show()

logger.debug("Completed 'reading app/settings_editor.py'")
