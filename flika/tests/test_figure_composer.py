import numpy as np
import pytest
import os
import tempfile
from flika import global_vars as g
from flika.window import Window


class TestFigurePanel:
    def test_panel_defaults(self):
        from ..viewers.figure_composer import FigurePanel
        panel = FigurePanel()
        assert panel.source_window is None
        assert panel.frame_index == 0
        assert panel.title == ''
        assert panel.show_scale_bar is False
        assert panel.show_colorbar is False
        assert panel.row == 0
        assert panel.col == 0

    def test_panel_assignment(self):
        from ..viewers.figure_composer import FigurePanel
        win = Window(np.random.random((64, 64)).astype(np.float32))
        panel = FigurePanel()
        panel.source_window = win
        panel.title = 'Test'
        panel.row = 1
        panel.col = 2
        assert panel.source_window is win
        assert panel.title == 'Test'
        assert panel.row == 1
        assert panel.col == 2
        win.close()


class TestFigureScene:
    def setup_method(self):
        self.im1 = np.random.random((64, 64)).astype(np.float32)
        self.im2 = np.random.random((64, 64)).astype(np.float32)
        self.win1 = Window(self.im1)
        self.win2 = Window(self.im2)

    def teardown_method(self):
        if not self.win1.closed:
            self.win1.close()
        if not self.win2.closed:
            self.win2.close()

    def test_scene_creation(self):
        from ..viewers.figure_composer import FigureScene, FigurePanel
        scene = FigureScene()
        panels = []
        for r in range(2):
            for c in range(2):
                p = FigurePanel()
                p.row = r
                p.col = c
                panels.append(p)
        panels[0].source_window = self.win1
        panels[1].source_window = self.win2
        scene.set_layout(2, 2, panels, 150, 'White', 10, 'Arial', 12)
        scene.render_panels()
        # Scene should have items
        assert len(scene.items()) > 0

    def test_png_export(self):
        from ..viewers.figure_composer import FigureScene, FigurePanel
        scene = FigureScene()
        p = FigurePanel()
        p.source_window = self.win1
        scene.set_layout(1, 1, [p], 72, 'White', 5, 'Arial', 12)
        scene.render_panels()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            scene.export_png(path, 72)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
