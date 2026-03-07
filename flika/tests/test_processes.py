import sys, os

from ..process import *
from .. import global_vars as g
from ..window import Window
from ..roi import makeROI
import numpy as np
import time
import pytest
import warnings
warnings.filterwarnings("ignore")


def _get_zproject_types():
    """Get the list of zproject projection types (requires Qt app)."""
    zproject.gui()
    obj = [i for i in zproject.items if i['name'] == 'projection_type'][0]['object']
    result = [obj.itemText(i) for i in range(obj.count())]
    zproject.ui.close()
    return result


def _get_operand_types():
    """Get the list of image_calculator operations (requires Qt app)."""
    image_calculator.gui()
    obj = [i for i in image_calculator.items if i['name'] == 'operation'][0]['object']
    result = [obj.itemText(i) for i in range(obj.count())]
    image_calculator.ui.close()
    return result


# Lazy-initialized after Qt app is running
ZPROJECTS = None
OPERANDS = None


@pytest.fixture(autouse=True, scope='module')
def _init_process_options(fa):
    """Initialize process options after the Qt app is ready."""
    global ZPROJECTS, OPERANDS
    g.settings['multiprocessing'] = False
    ZPROJECTS = _get_zproject_types()
    OPERANDS = _get_operand_types()



# Representative dtypes and shapes — covers the important combinations
# without the excessive memory usage of 55 images × 20+ test methods.
# Shapes: 3D movie, 4D RGB movie, 2D still, 2D RGB still
# Dtypes: uint8, int32, float32, float64 (plus binary for morphology tests)
@pytest.mark.parametrize("img", [
    # 3D movies (binary — needed for morphology tests)
    (np.random.randint(2, size=[10, 20, 20]).astype("uint8")),
    (np.random.randint(2, size=[10, 20, 20]).astype("float32")),
    # 3D movies (continuous)
    ((np.random.random([10, 20, 20]) * 10).astype("uint8")),
    ((np.random.random([10, 20, 20]) * 10).astype("int32")),
    (np.random.random([10, 20, 20]).astype("float32")),
    (np.random.random([10, 20, 20]).astype("float64")),
    # 4D RGB movies
    ((np.random.random([10, 20, 20, 3]) * 10).astype("uint8")),
    (np.random.random([10, 20, 20, 3]).astype("float32")),
    # 2D stills
    ((np.random.random([20, 20]) * 10).astype("uint8")),
    ((np.random.random([20, 20]) * 10).astype("int32")),
    (np.random.random([20, 20]).astype("float32")),
    (np.random.random([20, 20]).astype("float64")),
    # 2D RGB stills
    ((np.random.random([20, 20, 3]) * 10).astype("uint8")),
    (np.random.random([20, 20, 3]).astype("float32")),
])
class ProcessTest:
	def teardown_method(self):
		import gc
		from qtpy.QtWidgets import QApplication
		from ..core.undo import undo_stack
		if g.m is not None:
			g.m.clear()
		undo_stack.clear()
		gc.collect()
		QApplication.processEvents()


class TestBinary(ProcessTest):
	
	def test_threshold(self, img, fa):
		w1 = Window(img)
		w = threshold(.5)
		

	def test_adaptive_threshold(self, img, fa):
		w1 = Window(img)
		w = adaptive_threshold(.5, 3)
		

	def test_canny_edge_detector(self, img):
		if img.ndim == 4:
			return
		w1 = Window(img)
		w = canny_edge_detector(.5)
		
	
	def test_binary_dilation(self, img):
		if img.ndim == 4 or not ((img==0)|(img==1)).all():
			return
		w1 = Window(img)
		w = binary_dilation(2, 3, 1)
		
	
	def test_binary_erosion(self, img):
		if img.ndim == 4 or not ((img==0)|(img==1)).all():
			return
		w1 = Window(img)
		w = binary_erosion(2, 3, 1)
		

	def test_generate_rois(self, img):
		if img.ndim == 4 or not ((img==0)|(img==1)).all():
			return
		w1 = Window(img)
		w = generate_rois(.5, 10)
		

	def test_remove_small_blobs(self, img):
		if img.ndim == 4 or not ((img==0)|(img==1)).all():
			return
		w1 = Window(img)
		w = threshold(.5)
		


class TestFilters(ProcessTest):
	def test_gaussian_blur(self, img):
		w1 = Window(img)
		w = gaussian_blur(.5)
		

	def test_butterworth_filter(self, img):
		w1 = Window(img)
		w = butterworth_filter(1, .2, .6)
		

	def test_mean_filter(self, img):
		w1 = Window(img)
		w = mean_filter(5)
		

	def test_median_filter(self, img):
		w1 = Window(img)
		w = median_filter(5)
		

	def test_fourier_filter(self, img):
		w1 = Window(img)
		w = fourier_filter(3, .2, .6, False)
		

	def test_difference_filter(self, img):
		w1 = Window(img)
		w = difference_filter()
		

	def test_boxcar_differential_filter(self, img):
		w1 = Window(img)
		w = boxcar_differential_filter(2, 3)
		

	def test_wavelet_filter(self, img):
		if img.ndim != 3:
			return
		w1 = Window(img)
		w = wavelet_filter(2, 3)
		

	def test_bilateral_filter(self, img):
		w1 = Window(img)
		w = bilateral_filter(True, 30, 10, .05, 100) # soft filter
		w2 = bilateral_filter(False, 30, 10, .05, 100) # hard filter
		

class TestMath(ProcessTest):
	def test_subtract(self, img):
		w1 = Window(img)
		subtract(2)
		

	def test_subtract_trace(self, img):
		if img.ndim != 3 or img.shape[-1] == 3:
			return  # trace operations require a 3D movie (not RGB stills)
		w1 = Window(img)
		roi1 = makeROI('rectangle', [[3, 3], [5, 6]])
		tr = roi1.plot()
		if tr:
			subtract_trace()


	def test_divide_trace(self, img):
		if img.ndim != 3 or img.shape[-1] == 3:
			return  # trace operations require a 3D movie (not RGB stills)
		w1 = Window(img)
		roi1 = makeROI('rectangle', [[3, 3], [5, 6]])
		tr = roi1.plot()
		if tr:
			divide_trace()
		

	def test_multiply(self, img):
		w1 = Window(img)
		multiply(2.4)
		

	def test_power(self, img):
		w1 = Window(img)
		power(2)
		

	def test_ratio(self, img):
		if img.ndim != 3 or img.shape[-1] == 3:
			return  # ratio requires a 3D movie (not RGB stills)
		w1 = Window(img)
		ratio(2, 6, 'average')
		ratio(2, 6, 'standard deviation')
		

	def test_absolute_value(self, img):
		w1 = Window(img)
		absolute_value()
		

class TestOverlay(ProcessTest):
	def test_time_stamp(self, img):
		w1 = Window(img)
		time_stamp(2)
		

	def test_background(self, img):
		w1 = Window(img)
		w2 = Window(img/2)
		background(w1, w2, .5, True)
		

	def test_scale_bar(self, img):
		w1 = Window(img)
		scale_bar.gui()
		scale_bar(30, 5, 12, 'White', 'None','Lower Left')
		

class TestColor(ProcessTest):
	def test_split_channels(self, img):
		if img.ndim == 4 or (img.ndim == 3 and img.shape[2] == 3):
			w1 = Window(img)
			split_channels()
		

class TestROIProcess(ProcessTest):
	def test_set_value(self, img):
		w1 = Window(img)
		roi = makeROI('rectangle', [[3, 3], [4, 5]])
		set_value(2, 2, 5)
		

class TestStacks(ProcessTest):
	def test_deinterleave(self, img):
		w1 = Window(img)
		deinterleave(2)
		

	def test_trim(self, img):
		w1 = Window(img)
		trim(2, 6, 2)
		

	def test_zproject(self, img):
		w1 = Window(img)
		
		for i in ZPROJECTS:
			w3 = zproject(2, 6, i, True)
			if isinstance(w3, Window):
				w3.close()
			w1.setAsCurrentWindow()
		

	def test_image_calculator(self, img):
		if img.ndim == 4:
			return  # image_calculator Average doesn't support 4D RGB
		w1 = Window(img)
		w2 = Window(img)
		for i in OPERANDS:
			w3 = image_calculator(w1, w2, i, True)
			if isinstance(w3, Window):
				w3.close()
		

	def test_pixel_binning(self, img):
		w1 = Window(img)
		pixel_binning(2)
		

	def test_frame_binning(self, img):
		w1 = Window(img)
		frame_binning(2)
		

	def test_resize(self, img):
		w1 = Window(img)
		resize(2)
		
