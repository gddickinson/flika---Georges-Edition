import numpy as np
import pytest
from .. import global_vars as g
from ..window import Window


class TestColormaps:
    def test_all_colormaps_exist(self):
        from ..viewers.channel_compositor import COLORMAPS
        expected = ['Green', 'Red', 'Blue', 'Cyan', 'Magenta', 'Yellow', 'Gray', 'Fire', 'Ice']
        for name in expected:
            assert name in COLORMAPS, f"Missing colormap: {name}"

    def test_colormap_shape_and_dtype(self):
        from ..viewers.channel_compositor import COLORMAPS
        for name, lut in COLORMAPS.items():
            assert lut.shape == (256, 4), f"{name} shape is {lut.shape}"
            assert lut.dtype == np.uint8, f"{name} dtype is {lut.dtype}"

    def test_colormap_alpha_channel(self):
        from ..viewers.channel_compositor import COLORMAPS
        for name, lut in COLORMAPS.items():
            assert np.all(lut[:, 3] == 255), f"{name} alpha channel not all 255"


class TestChannelLayer:
    def setup_method(self):
        self.im2d = np.random.random((64, 64)).astype(np.float32)
        self.im3d = np.random.random((10, 64, 64)).astype(np.float32)
        self.win2d = Window(self.im2d)
        self.win3d = Window(self.im3d)

    def teardown_method(self):
        if not self.win2d.closed:
            self.win2d.close()
        if not self.win3d.closed:
            self.win3d.close()

    def test_create_layer(self):
        from ..viewers.channel_compositor import ChannelLayer
        layer = ChannelLayer(self.win2d, 'Green')
        assert layer.colormap_name == 'Green'
        assert layer.visible is True
        assert layer.opacity == 1.0

    def test_get_current_frame_2d(self):
        from ..viewers.channel_compositor import ChannelLayer
        layer = ChannelLayer(self.win2d, 'Green')
        frame = layer.get_current_frame()
        assert frame.ndim == 2
        assert frame.shape == (64, 64)

    def test_get_current_frame_3d(self):
        from ..viewers.channel_compositor import ChannelLayer
        layer = ChannelLayer(self.win3d, 'Red')
        frame = layer.get_current_frame()
        assert frame.ndim == 2
        assert frame.shape == (64, 64)

    def test_set_colormap(self):
        from ..viewers.channel_compositor import ChannelLayer
        layer = ChannelLayer(self.win2d, 'Green')
        layer.set_colormap('Magenta')
        assert layer.colormap_name == 'Magenta'

    def test_set_levels(self):
        from ..viewers.channel_compositor import ChannelLayer
        layer = ChannelLayer(self.win2d, 'Green')
        layer.set_levels(0.2, 0.8)
        assert layer.min_level == 0.2
        assert layer.max_level == 0.8

    def test_set_opacity(self):
        from ..viewers.channel_compositor import ChannelLayer
        layer = ChannelLayer(self.win2d, 'Green')
        layer.set_opacity(0.5)
        assert layer.opacity == 0.5


class TestChannelCompositor:
    def setup_method(self):
        self.host_im = np.random.random((10, 64, 64)).astype(np.float32)
        self.ch1_im = np.random.random((10, 64, 64)).astype(np.float32)
        self.ch2_im = np.random.random((10, 64, 64)).astype(np.float32)
        self.host = Window(self.host_im)
        self.ch1 = Window(self.ch1_im)
        self.ch2 = Window(self.ch2_im)

    def teardown_method(self):
        for w in [self.host, self.ch1, self.ch2]:
            if not w.closed:
                w.close()

    def test_add_channel(self):
        from ..viewers.channel_compositor import ChannelCompositor
        comp = ChannelCompositor(self.host)
        layer = comp.add_channel(self.ch1, 'Green')
        assert len(comp.layers) == 1
        assert layer.source_window is self.ch1
        comp.cleanup()

    def test_remove_channel(self):
        from ..viewers.channel_compositor import ChannelCompositor
        comp = ChannelCompositor(self.host)
        layer = comp.add_channel(self.ch1, 'Green')
        comp.remove_channel(layer)
        assert len(comp.layers) == 0
        comp.cleanup()

    def test_export_composite_rgb(self):
        from ..viewers.channel_compositor import ChannelCompositor
        comp = ChannelCompositor(self.host)
        comp.add_channel(self.ch1, 'Green')
        comp.add_channel(self.ch2, 'Magenta')
        rgb = comp.export_composite_rgb()
        assert rgb.ndim == 3
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8
        comp.cleanup()

    def test_auto_colormap(self):
        from ..viewers.channel_compositor import ChannelCompositor, AUTO_COLORMAP_ORDER
        comp = ChannelCompositor(self.host)
        layer1 = comp.add_channel(self.ch1)
        layer2 = comp.add_channel(self.ch2)
        assert layer1.colormap_name == AUTO_COLORMAP_ORDER[0]
        assert layer2.colormap_name == AUTO_COLORMAP_ORDER[1]
        comp.cleanup()

    def test_cleanup_restores_host(self):
        from ..viewers.channel_compositor import ChannelCompositor
        comp = ChannelCompositor(self.host)
        comp.add_channel(self.ch1, 'Green')
        comp.cleanup()
        assert len(comp.layers) == 0
