import numpy as np
import skimage
import skimage.filters
from qtpy import QtWidgets, QtGui, QtCore
from ..logger import logger
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, SliderLabel, SliderLabelOdd, CheckBox
from .progress_bar import ProgressBar
from ..utils.ndim import per_plane
from scipy.ndimage import uniform_filter1d, median_filter as nd_median_filter
from scipy.fft import fft as sp_fft, ifft as sp_ifft, fftfreq as sp_fftfreq

__all__ = ['gaussian_blur', 'difference_of_gaussians', 'mean_filter', 'variance_filter', 'median_filter', 'butterworth_filter', 'boxcar_differential_filter','wavelet_filter','difference_filter', 'fourier_filter', 'bilateral_filter', 'sobel_filter', 'laplacian_filter', 'gaussian_laplace_filter', 'gaussian_gradient_magnitude_filter', 'sato_tubeness', 'meijering_neuriteness', 'hessian_filter', 'gabor_filter', 'maximum_filter', 'minimum_filter', 'percentile_filter', 'tv_denoise', 'flash_remover']
###############################################################################
##################   SPATIAL FILTERS       ####################################
###############################################################################


@per_plane
def _gaussian_blur_impl(tif, sigma, norm_edges=False):
    mode = 'constant' if norm_edges else 'nearest'
    result = np.zeros(tif.shape)
    if tif.ndim == 3:
        for i in range(len(tif)):
            result[i] = skimage.filters.gaussian(tif[i].astype(np.float64), sigma, mode=mode)
    elif tif.ndim == 2:
        result = skimage.filters.gaussian(tif.astype(np.float64), sigma, mode=mode)
    return result


class Gaussian_blur(BaseProcess):
    """ gaussian_blur(sigma, norm_edges=False, keepSourceWindow=False)

    This applies a spatial gaussian_blur to every frame of your stack.

    Args:
        sigma (float): The width of the gaussian
        norm_edges (bool): If true, this reduces the values of the pixels near the edges so they have the same standard deviation as the rest of the image

    Returns:
        flika.window.Window

    """
    def __init__(self):
        super().__init__()

    def gui(self):
        logger.debug("Started 'running process.filters.gaussian_blur.gui()'")
        self.gui_reset()
        sigma=SliderLabel(2)
        sigma.setRange(0,100)
        sigma.setValue(1)
        norm_edges=CheckBox()
        norm_edges.setChecked(False)
        preview=CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'sigma', 'string': 'Sigma (pixels)', 'object': sigma})
        self.items.append({'name': 'norm_edges', 'string': 'Normalize Edges', 'object': norm_edges})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()
        logger.debug("Completed 'running process.filters.gaussian_blur.gui()'")

    def __call__(self, sigma, norm_edges=False, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if sigma > 0:
            self.newtif = _gaussian_blur_impl(self.tif, sigma, norm_edges)
            self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        else:
            self.newtif = self.tif
        self.newname = self.oldname + ' - Gaussian Blur sigma=' + str(sigma)
        return self.end()
    def preview(self):
        logger.debug("Started 'running process.filters.gaussian_blur.preview()'")
        norm_edges = self.getValue('norm_edges')
        if norm_edges:
            mode = 'constant'
        else:
            mode = 'nearest'
        sigma=self.getValue('sigma')
        preview=self.getValue('preview')
        if preview:
            if len(g.win.image.shape)==3:
                testimage=g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape)==2:
                testimage=g.win.image.astype(np.float64)
            if sigma>0:
                testimage=skimage.filters.gaussian(testimage,sigma, mode=mode)
            g.win.imageview.setImage(testimage,autoLevels=False)
        else:
            g.win.reset()
        logger.debug("Completed 'running process.filters.gaussian_blur.preview()'")
gaussian_blur = Gaussian_blur()


@per_plane
def _dog_impl(tif, sigma1, sigma2):
    result = np.zeros(tif.shape)
    if tif.ndim == 3:
        for i in range(len(tif)):
            result[i] = skimage.filters.gaussian(tif[i].astype(np.float64), sigma1, mode='nearest') - \
                         skimage.filters.gaussian(tif[i].astype(np.float64), sigma2, mode='nearest')
    elif tif.ndim == 2:
        result = skimage.filters.gaussian(tif.astype(np.float64), sigma1, mode='nearest') - \
                 skimage.filters.gaussian(tif.astype(np.float64), sigma2, mode='nearest')
    return result


class Difference_of_Gaussians(BaseProcess):
    """gaussian_blur(sigma1, sigma2, keepSourceWindow=False)

    This subtracts one gaussian blurred image from another to spatially bandpass filter.

    Args:
        sigma1 (float): The width of the first gaussian
        sigma2 (float): The width of the first gaussian

    Returns:
        flika.window.Window

    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        sigma1 = SliderLabel(2)
        sigma1.setRange(0, 100)
        sigma1.setValue(1)
        sigma2 = SliderLabel(2)
        sigma2.setRange(0, 100)
        sigma2.setValue(2)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'sigma1', 'string': 'Sigma 1 (pixels)', 'object': sigma1})
        self.items.append({'name': 'sigma2', 'string': 'Sigma 2 (pixels)', 'object': sigma2})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, sigma1, sigma2, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if sigma1 > 0 and sigma2 > 0:
            self.newtif = _dog_impl(self.tif, sigma1, sigma2)
            self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        else:
            self.newtif = self.tif
        self.newname = self.oldname + ' - Difference of Gaussians ({} {})'.format(sigma1, sigma2)
        return self.end()

    def preview(self):
        sigma1 = self.getValue('sigma1')
        sigma2 = self.getValue('sigma2')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            if sigma1 > 0 and sigma2 > 0:
                testimage = skimage.filters.gaussian(testimage, sigma1, mode='nearest') - skimage.filters.gaussian(testimage, sigma2, mode='nearest')
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()


difference_of_gaussians = Difference_of_Gaussians()
###############################################################################
##################   TEMPORAL FILTERS       ###################################
###############################################################################
from scipy.signal import butter, filtfilt


@per_plane
def _butterworth_impl(tif, filter_order, low, high, framerate, makeButterFilter):
    if g.settings['multiprocessing']:
        return butterworth_filter_multi(filter_order, low/(framerate/2), high/(framerate/2), tif)
    else:
        b, a, padlen = makeButterFilter(filter_order, low/(framerate/2), high/(framerate/2))
        # Vectorized: filtfilt supports axis parameter
        result = filtfilt(b, a, tif, axis=0, padlen=padlen)
        return result.astype(g.settings.d['internal_data_type'])


class Butterworth_filter(BaseProcess):
    """ butterworth_filter(filter_order, low, high, framerate, keepSourceWindow=False)

    This filters a stack in time.

    Parameters:
        filter_order (int): The order of the butterworth filter (higher order -> steeper cutoff).
        low (float): The low frequency cutoff.  Must be between 0 and 1 and must be below high.
        high (float): The high frequency cutoff.  Must be between 0 and 1 and must be above low.
        framerate (float): The framerate in Hz. If set to zero, a framerate of 2 Hz will be used, so as to set the Nyquist frequency to 1. Default is 0.
    Returns:
        newWindow
    """
    def __init__(self):
        super().__init__()

    def framerate_adjusted(self, rate):
        high = self.items[2]
        assert high['name'] == 'high'
        if rate == 0:
            rate = 2
        high['object'].setMaximum(rate / 2)


    def gui(self):
        self.gui_reset()
        filter_order=QtWidgets.QSpinBox()
        filter_order.setRange(1,10)
        low=SliderLabel(5)
        low.setRange(0,1)
        low.setValue(0)
        high=SliderLabel(5)
        high.setRange(0,1)
        high.setValue(1)
        framerate = SliderLabel(2)
        framerate.setRange(0, 1000)
        low.valueChanged.connect(lambda low: high.setMinimum(low))
        high.valueChanged.connect(lambda high: low.setMaximum(high))
        framerate.valueChanged.connect(self.framerate_adjusted)
        preview=CheckBox()
        preview.setChecked(True)
        self.items.append({'name':'filter_order','string':'Filter Order','object':filter_order})
        self.items.append({'name':'low','string':'Low Cutoff Frequency','object':low})
        self.items.append({'name':'high','string':'High Cutoff Frequency','object':high})
        self.items.append({'name': 'framerate', 'string': 'Frame rate (Hz)', 'object': framerate})
        self.items.append({'name':'preview','string':'Preview','object':preview})
        super().gui()
        if g.win is None:
            self.roi = None
        else:
            self.roi=g.win.currentROI
        if self.roi is not None:
            self.ui.rejected.connect(self.roi.redraw_trace)
            self.ui.accepted.connect(self.roi.redraw_trace)
        if self.roi is None or g.currentTrace is None:
            preview.setChecked(False)
            preview.setEnabled(False)

    def __call__(self, filter_order, low, high, framerate=0, keepSourceWindow=False):
        if framerate == 0:
            framerate = 2
        if low == 0 and high == framerate/2:
            return
        self.start(keepSourceWindow)
        if self.tif.ndim < 3:
            g.alert("Butterworth filter only works on 3+ dimensional movies.")
            return
        self.newtif = _butterworth_impl(self.tif, filter_order, low, high, framerate, self.makeButterFilter)
        self.newname = self.oldname+' - Butter Filtered'
        return self.end()

    def preview(self):
        if g.currentTrace is not None:
            framerate = self.getValue('framerate')
            if framerate == 0:
                framerate = 2
            filter_order = self.getValue('filter_order')
            low = self.getValue('low')
            high = self.getValue('high')
            preview = self.getValue('preview')
            if self.roi is not None:
                if preview:
                    if (low == 0 and high == framerate/2) or (low == 0 and high == 0):
                        self.roi.onRegionChangeFinished() #redraw roi without filter
                    else:
                        b, a, padlen = self.makeButterFilter(filter_order, low/(framerate/2), high/(framerate/2))
                        trace = self.roi.getTrace()
                        trace = filtfilt(b,a, trace, padlen=padlen)
                        roi_index = g.currentTrace.get_roi_index(self.roi)
                        g.currentTrace.update_trace_full(roi_index,trace) #update_trace_partial may speed it up
                else:
                    self.roi.redraw_trace()
    def makeButterFilter(self, filter_order, low, high):
        padlen = 0
        if high == 1:
            if low == 0: #if there is no temporal filter at all,
                return None,None,None
            else: #if only high pass temporal filter
                [b, a] = butter(filter_order, low, btype='highpass')
                padlen = 3
        else:
            if low == 0:
                [b,a] = butter(filter_order, high, btype='lowpass')
            else:
                [b,a] = butter(filter_order, [low, high], btype='bandpass')
            padlen = 6
        return b, a, padlen

butterworth_filter=Butterworth_filter()


def butterworth_filter_multi(filter_order, low, high, tif):
    nThreads = g.settings['nCores']
    mt, mx, my = tif.shape
    block_ends = np.linspace(0, mx, nThreads+1).astype(int)
    data = [tif[:, block_ends[i]:block_ends[i+1], :] for i in np.arange(nThreads)] #split up data along x axis. each thread will get one.
    args = (filter_order, low, high)
    progress = ProgressBar(butterworth_filter_multi_inner, data, args, nThreads, msg='Performing Butterworth Filter')
    if progress.results is None or any(r is None for r in progress.results):
        result = None
    else:
        result = np.concatenate(progress.results,axis=1).astype(g.settings['internal_data_type'])
    return result


def butterworth_filter_multi_inner(q_results, q_progress, q_status, child_conn, args):
    data = child_conn.recv()
    status = q_status.get(True) #this blocks the process from running until all processes are launched
    if status == 'Stop':
        q_results.put(None)

    def makeButterFilter(filter_order, low, high):
        padlen = 0
        if high == 1:
            if low == 0: #if there is no temporal filter at all,
                return None,None,None
            else: #if only high pass temporal filter
                [b,a] = butter(filter_order,low,btype='highpass')
                padlen = 3
        else:
            if low == 0:
                [b,a]= butter(filter_order, high, btype='lowpass')
            else:
                [b,a]=butter(filter_order, [low,high], btype='bandpass')
            padlen = 6
        return b, a, padlen

    filter_order,low,high = args
    b, a, padlen = makeButterFilter(filter_order,low,high)
    mt, mx, my = data.shape
    result = np.zeros(data.shape,g.settings['internal_data_type'])
    nPixels = mx*my
    pixel = 0
    percent = 0
    for x in np.arange(mx):
        for y in np.arange(my):
            if not q_status.empty():
                stop = q_status.get(False)
                q_results.put(None)
                return
            pixel += 1
            if percent < int(100*pixel/nPixels):
                percent = int(100*pixel/nPixels)
                q_progress.put(percent)
            result[:, x, y] = filtfilt(b,a, data[:, x, y], padlen=padlen)
    q_results.put(result)

from scipy.ndimage import convolve

@per_plane
def _mean_filter_impl(tif, nFrames):
    return convolve(tif, weights=np.full((nFrames, 1, 1), 1.0 / nFrames))


class Mean_filter(BaseProcess):
    """ mean_filter(nFrames, keepSourceWindow=False)

    This filters a stack in time.

    Parameters:
        nFrames (int): Number of frames to average
    Returns:
        newWindow
    """
    def __init__(self):
        super().__init__()
    def gui(self):
        self.gui_reset()
        nFrames=SliderLabel(0)
        nFrames.setRange(1,100)
        preview=CheckBox()
        preview.setChecked(True)
        self.items.append({'name':'nFrames','string':'nFrames','object':nFrames})
        self.items.append({'name':'preview','string':'Preview','object':preview})
        super().gui()
        self.roi=g.win.currentROI
        if self.roi is not None:
            self.ui.rejected.connect(self.roi.redraw_trace)
            self.ui.accepted.connect(self.roi.redraw_trace)
        else:
            preview.setChecked(False)
            preview.setEnabled(False)

    def __call__(self,nFrames,keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Mean Filter does not support float16 type arrays")
            return
        if self.tif.ndim < 3:
            g.alert("Mean Filter requires at least 3-dimensional movies.")
            return
        self.newtif = _mean_filter_impl(self.tif, nFrames)
        self.newname=self.oldname+' - Mean Filtered'
        return self.end()
    def preview(self):
        nFrames=self.getValue('nFrames')
        preview=self.getValue('preview')
        if self.roi is not None:
            if preview:
                if nFrames==1:
                    self.roi.redraw_trace() #redraw roi without filter
                else:
                    trace=self.roi.getTrace()
                    trace=convolve(trace,weights=np.full((nFrames),1.0/nFrames))
                    roi_index=g.currentTrace.get_roi_index(self.roi)
                    g.currentTrace.update_trace_full(roi_index,trace) #update_trace_partial may speed it up
            else:
                self.roi.redraw_trace()
mean_filter=Mean_filter()

def varfilt(trace, nFrames):
    result = np.zeros_like(trace)
    mt = len(trace)
    for i in np.arange(mt):
        i0 = int(i-nFrames/2)
        i1 = int(i+nFrames/2)
        if i0 < 0:
            i0 = 0
        if i1 > len(trace):
            i1 = len(trace)
        result[i] = np.var(trace[i0:i1])
    return result

@per_plane
def _variance_filter_impl(tif, nFrames):
    # Vectorized: Var(X) = E[X²] - E[X]² via uniform_filter1d
    tif_f = tif.astype(np.float64)
    mean = uniform_filter1d(tif_f, size=nFrames, axis=0, mode='nearest')
    mean_sq = uniform_filter1d(tif_f ** 2, size=nFrames, axis=0, mode='nearest')
    result = np.maximum(mean_sq - mean ** 2, 0)
    return result


class Variance_filter(BaseProcess):
    """ variance_filter(nFrames, keepSourceWindow=False)

    This filters a stack in time.

    Parameters:
        nFrames (int): Number of frames to take teh variance of
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        nFrames = SliderLabel(0)
        nFrames.setRange(1, 100)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'nFrames', 'string': 'nFrames', 'object': nFrames})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.roi = g.win.currentROI
        if self.roi is not None:
            self.ui.rejected.connect(self.roi.redraw_trace)
            self.ui.accepted.connect(self.roi.redraw_trace)
        else:
            preview.setChecked(False)
            preview.setEnabled(False)

    def __call__(self, nFrames, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Variance filter does not support float16 type arrays")
            return
        if self.tif.ndim < 3:
            g.alert("Variance filter requires at least 3-dimensional movies.")
            return
        self.newtif = _variance_filter_impl(self.tif, nFrames)
        self.newname = self.oldname + ' - Variance Filtered'
        return self.end()

    def preview(self):
        nFrames = self.getValue('nFrames')
        preview = self.getValue('preview')
        if self.roi is not None:
            if preview:
                if nFrames == 1:
                    self.roi.redraw_trace()  # redraw roi without filter
                else:
                    trace = self.roi.getTrace()
                    trace = varfilt(trace, nFrames)
                    roi_index = g.currentTrace.get_roi_index(self.roi)
                    g.currentTrace.update_trace_full(roi_index, trace)  # update_trace_partial may speed it up
            else:
                self.roi.redraw_trace()


variance_filter = Variance_filter()

from scipy.signal import medfilt

@per_plane
def _median_filter_impl(tif, nFrames):
    # Vectorized: scipy.ndimage.median_filter with temporal-only kernel
    return nd_median_filter(tif.astype(np.float64), size=(nFrames, 1, 1))


class Median_filter(BaseProcess):
    """ median_filter(nFrames, keepSourceWindow=False)

    This filters a stack in time.

    Parameters:
        nFrames (int): Number of frames to average.  This must be an odd number
    Returns:
        newWindow
    """
    def __init__(self):
        super().__init__()
    def gui(self):
        self.gui_reset()
        nFrames=SliderLabelOdd()
        nFrames.setRange(1,100)
        preview=CheckBox()
        preview.setChecked(True)
        self.items.append({'name':'nFrames','string':'nFrames','object':nFrames})
        self.items.append({'name':'preview','string':'Preview','object':preview})
        super().gui()
        self.roi=g.win.currentROI
        if self.roi is not None:
            self.ui.rejected.connect(self.roi.redraw_trace)
            self.ui.accepted.connect(self.roi.redraw_trace)
        else:
            preview.setChecked(False)
            preview.setEnabled(False)

    def __call__(self, nFrames, keepSourceWindow=False):
        if nFrames%2 == 0: #if value is even:
            g.alert('median_filter only takes odd numbers.  Operation cancelled')
            return None
        self.start(keepSourceWindow)
        if self.tif.ndim < 3:
            g.alert("Median filter requires at least 3 dimensions. %d < 3" % self.tif.ndim)
            return
        self.newtif = _median_filter_impl(self.tif, nFrames)
        self.newname=self.oldname+' - Median Filtered'
        return self.end()
    def preview(self):
        nFrames=self.getValue('nFrames')
        preview=self.getValue('preview')
        if self.roi is not None:
            if preview:
                logger.debug(nFrames)
                if nFrames==1:
                    self.roi.redraw_trace() #redraw roi without filter
                elif nFrames%2==0: #if value is even
                    return None
                else:
                    trace=self.roi.getTrace()
                    trace=medfilt(trace,kernel_size=nFrames)
                    roi_index=g.currentTrace.get_roi_index(self.roi)
                    g.currentTrace.update_trace_full(roi_index,trace) #update_trace_partial may speed it up
            else:
                self.roi.redraw_trace()

median_filter=Median_filter()




from scipy.fftpack import fft, ifft, fftfreq

@per_plane
def _fourier_filter_impl(tif, frame_rate, low, high):
    mt = tif.shape[0]
    W = sp_fftfreq(mt, d=1.0 / frame_rate)
    filt = np.ones(mt)
    filt[np.abs(W) < low] = 0
    filt[np.abs(W) > high] = 0

    # Try GPU-accelerated FFT for large arrays
    tif_f = tif.astype(np.float64)
    try:
        from ..utils.accel import should_use_gpu
        if should_use_gpu(tif_f):
            import torch
            from ..utils.accel import get_torch_device
            pref = g.settings.get('acceleration_device', 'Auto')
            device = get_torch_device(pref)
            tensor = torch.from_numpy(tif_f).to(device)
            filt_t = torch.from_numpy(filt).to(device).reshape(-1, 1, 1)
            f_signal = torch.fft.fft(tensor, dim=0)
            f_signal *= filt_t
            result = torch.fft.ifft(f_signal, dim=0).real.cpu().numpy()
            return result
    except Exception:
        pass  # Fall back to CPU path

    # CPU vectorized batch FFT
    filt_3d = filt[:, np.newaxis, np.newaxis]
    f_signal = sp_fft(tif_f, axis=0)
    result = np.real(sp_ifft(f_signal * filt_3d, axis=0))
    return result


class Fourier_filter(BaseProcess):
    """ fourier_filter(frame_rate, low, high, loglogPreview, keepSourceWindow=False)

    I'm going to eventually plot the trace in the frequency domain inside this box so you can see where the power is.

    Parameters:
        frame_rate (int): Frame Rate in Hz
        low (float): Low cutoff frequency for the fourier filter
        high (float): High cutoff frequency for fourier filter
        loglogPreview (boolean): whether or not to plot frequency spectrum on log log axes
    """
    def __init__(self):
        super().__init__()
    def gui(self):
        self.gui_reset()
        frame_rate=QtWidgets.QDoubleSpinBox()
        frame_rate.setRange(.01,1000)
        frame_rate.setValue(200)
        low=SliderLabel(5)
        low.setRange(0,1)
        low.setValue(0)
        high=SliderLabel(5)
        high.setRange(0,1)
        high.setValue(1)
        frame_rate.valueChanged.connect(self.frame_rate_changed)
        low.valueChanged.connect(lambda low: high.setMinimum(low))
        high.valueChanged.connect(lambda high: low.setMaximum(high))
        preview=CheckBox()
        preview.setChecked(True)
        loglogPreview=CheckBox()
        self.items.append({'name':'frame_rate','string':'Frame Rate (Hz)','object':frame_rate})
        self.items.append({'name':'low','string':'Low Cutoff Frequency','object':low})
        self.items.append({'name':'high','string':'High Cutoff Frequency','object':high})
        self.items.append({'name':'loglogPreview','string':'Plot frequency spectrum on log log axes','object':loglogPreview})
        self.items.append({'name':'preview','string':'Preview','object':preview})
        super().gui()
        self.roi=g.win.currentROI
        if self.roi is not None:
            self.ui.rejected.connect(self.roi.redraw_trace)
            self.ui.accepted.connect(self.roi.redraw_trace)
        else:
            preview.setChecked(False)
            preview.setEnabled(False)
            loglogPreview.setEnabled(False)

    def __call__(self, frame_rate, low, high, loglogPreview, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Fourier transform does not support float16 movies.")
            return
        if self.tif.ndim < 3:
            g.alert('Fourier transform requires at least 3 dimensional movies.')
            return
        if low==0 and high==frame_rate/2.0:
            return
        self.newtif = _fourier_filter_impl(self.tif, frame_rate, low, high)
        self.newname=self.oldname+' - Fourier Filtered'
        return self.end()
    def preview(self):
        frame_rate=self.getValue('frame_rate')
        low=self.getValue('low')
        high=self.getValue('high')
        loglogPreview=self.getValue('loglogPreview')
        preview=self.getValue('preview')
        if self.roi is not None:
            if preview:
                if (low==0 and high==frame_rate/2.0) or (low==0 and high==0):
                    self.roi.redraw_trace() #redraw roi without filter
                else:
                    trace=self.roi.getTrace()
                    W = fftfreq(len(trace), d=1.0/frame_rate)
                    f_signal = fft(trace)
                    f_signal[(np.abs(W)<low)] = 0
                    f_signal[(np.abs(W)>high)] = 0
                    cut_signal=np.real(ifft(f_signal))
                    roi_index=g.currentTrace.get_roi_index(self.roi)
                    g.currentTrace.update_trace_full(roi_index,cut_signal) #update_trace_partial may speed it up
            else:
                self.roi.redraw_trace()

    def frame_rate_changed(self):
        low=[item for item in self.items if item['name']=='low'][0]['object']
        high=[item for item in self.items if item['name']=='high'][0]['object']
        frame_rate=[item for item in self.items if item['name']=='frame_rate'][0]['object']
        f=frame_rate.value()
        low.setRange(0.0,f/2.0)
        high.setRange(0.0,f/2.0)
        low.setValue(0)
        high.setValue(f/2.0)
""" This is demo code for plotting in the frequency domain, something I hoepfully will get around to implementing

from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange

def plotSpectrum(y,Fs):
    n = len(y) # length of the signal
    k = arange(n)
    W = fftfreq(len(y), d=1/Fs)

    Y = fft(y) # fft computing and normalization
    f_signal[(np.abs(W)>10)] = 0
    plot(abs(f_signal)[0:N/2])
    cut_signal=np.real(ifft(f_signal))
    p=plot(trace)
    p.plot(cut_signal,pen=pg.mkPen('r'))

    plot(frq,abs(f_signal),'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')"""

fourier_filter=Fourier_filter()



class Difference_filter(BaseProcess):
    """ difference_filter(keepSourceWindow=False)

    Subtracts each frame from the preceeding frame

    Returns:
        newWindow
    """
    def __init__(self):
        super().__init__()
    def gui(self):
        self.gui_reset()
        if super().gui()==False:
            return False
    def __call__(self,keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif=np.zeros(self.tif.shape)
        for i in np.arange(1,len(self.newtif)):
            self.newtif[i]=self.tif[i]-self.tif[i-1]
        self.newname=self.oldname+' - Difference Filtered'
        return self.end()
difference_filter=Difference_filter()


class Boxcar_differential_filter(BaseProcess):
    """ boxcar_differential_filter(minNframes, maxNframes, keepSourceWindow=False)

    Applies a Boxcar differential filter by comparing each frameat index I to the frames in range [I+minNframes, I+maxNframes]

    Parameters:
        minNframes (int): The starting point of your boxcar window.
        maxNframes (int): The ending point of your boxcar window.
    Returns:
        newWindow
    """
    def __init__(self):
        super().__init__()
    def gui(self):
        self.gui_reset()
        minNframes=SliderLabel(0)
        minNframes.setRange(1,100)
        maxNframes=SliderLabel(0)
        maxNframes.setRange(2,101)
        minNframes.valueChanged.connect(lambda minn: maxNframes.setMinimum(minn+1))
        maxNframes.valueChanged.connect(lambda maxx: minNframes.setMaximum(maxx-1))
        preview=CheckBox()
        preview.setChecked(True)
        self.items.append({'name':'minNframes','string':'Minimum Number of Frames','object':minNframes})
        self.items.append({'name':'maxNframes','string':'Maximum Number of Frames','object':maxNframes})
        self.items.append({'name':'preview','string':'Preview','object':preview})
        if super().gui()==False:
            return False
        self.roi=g.win.currentROI
        if self.roi is not None:
            self.ui.rejected.connect(self.roi.redraw_trace)
            self.ui.accepted.connect(self.roi.redraw_trace)
        else:
            preview.setChecked(False)
            preview.setEnabled(False)
    def __call__(self,minNframes,maxNframes,keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif=np.zeros(self.tif.shape)
        for i in np.arange(maxNframes,len(self.newtif)):
            self.newtif[i]=self.tif[i]-np.min(self.tif[i-maxNframes:i-minNframes],0)
        self.newname=self.oldname+' - Boxcar Differential Filtered'
        return self.end()
    def preview(self):
        minNframes=self.getValue('minNframes')
        maxNframes=self.getValue('maxNframes')
        preview=self.getValue('preview')
        if self.roi is not None:
            if preview:
                self.tif=self.roi.window.image
                (mt,mx,my)=self.tif.shape
                cnt=np.array([np.array([np.array([p[1],p[0]])]) for p in self.roi.pts ])
                mask=np.zeros(self.tif[0,:,:].shape,np.uint8)
                cv2.drawContours(mask,[cnt],0,255,-1)
                mask=mask.reshape(mx*my).astype(bool)
                tif=self.tif.reshape((mt,mx*my))
                tif=tif[:,mask]
                newtrace=np.zeros(mt)
                for i in np.arange(maxNframes,mt):
                    newtrace[i]=np.mean(tif[i]-np.min(tif[i-maxNframes:i-minNframes],0))
                roi_index=g.currentTrace.get_roi_index(self.roi)
                g.currentTrace.update_trace_full(roi_index,newtrace) #update_trace_partial may speed it up
            else:
                self.roi.redraw_trace()
boxcar_differential_filter=Boxcar_differential_filter()


try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False

@per_plane
def _wavelet_filter_impl(tif, low, high):
    mt, my, mx = tif.shape
    widths = np.arange(low, high)
    # Flatten spatial dims to a single loop for better cache locality
    flat = tif.reshape(mt, -1)
    result_flat = np.zeros_like(flat)
    for k in range(flat.shape[1]):
        cwtmatr, _ = pywt.cwt(flat[:, k], widths, 'mexh')
        result_flat[:, k] = np.mean(cwtmatr, 0)
    return result_flat.reshape(tif.shape)


class Wavelet_filter(BaseProcess):
    ''' wavelet_filter(low, high, keepSourceWindow=False)

    ***Warning!! This function is extremely slow.***

    Parameters:
        low (int): The starting point of your boxcar window.
        high (int): The ending point of your boxcar window.
    Returns:
        newWindow
    '''
    def __init__(self):
        super().__init__()
    def gui(self):
        self.gui_reset()
        if not _HAS_PYWT:
            g.alert("Wavelet filter requires PyWavelets. Install with: pip install PyWavelets")
            return
        low=SliderLabel(0)
        low.setRange(1,50)
        high=SliderLabel(0)
        high.setRange(2,50)
        low.valueChanged.connect(lambda minn: high.setMinimum(minn+1))
        high.valueChanged.connect(lambda maxx: low.setMaximum(maxx-1))
        preview=CheckBox()
        preview.setChecked(True)
        self.items.append({'name':'low','string':'Low Frequency Threshold','object':low})
        self.items.append({'name':'high','string':'High Frequency Threshold','object':high})
        self.items.append({'name':'preview','string':'Preview','object':preview})
        super().gui()
        self.roi=g.win.currentROI
        if self.roi is not None:
            self.ui.rejected.connect(self.roi.redraw_trace)
            self.ui.accepted.connect(self.roi.redraw_trace)
        else:
            preview.setChecked(False)
            preview.setEnabled(False)
    def __call__(self,low,high,keepSourceWindow=False):
        self.start(keepSourceWindow)
        if not _HAS_PYWT:
            g.alert("Wavelet filter requires PyWavelets. Install with: pip install PyWavelets")
            return
        if self.tif.ndim < 3:
            g.alert("Wavelet filter requires at least 3 dimensional movies")
            return
        self.newtif = _wavelet_filter_impl(self.tif, low, high)
        self.newname=self.oldname+' - Wavelet Filtered'
        return self.end()
    def preview(self):
        if not _HAS_PYWT:
            return
        low=self.getValue('low')
        high=self.getValue('high')
        preview=self.getValue('preview')
        if self.roi is not None:
            if preview:
                trace=self.roi.getTrace()
                widths = np.arange(low, high)
                cwtmatr, _ = pywt.cwt(trace, widths, 'mexh')
                newtrace=np.mean(cwtmatr,0)
                roi_index=g.currentTrace.get_roi_index(self.roi)
                g.currentTrace.update_trace_full(roi_index,newtrace) #update_trace_partial may speed it up
            else:
                self.roi.redraw_trace()
wavelet_filter=Wavelet_filter()


@per_plane
def _bilateral_filter_impl(tif, soft, beta, width, stoptol, maxiter):
    if g.settings['multiprocessing']:
        return bilateral_filter_multi(soft, beta, width, stoptol, maxiter, tif)
    else:
        result = np.zeros(tif.shape)
        mt, mx, my = tif.shape
        for i in np.arange(mx):
            for j in np.arange(my):
                result[:, i, j] = bilateral_smooth(soft, beta, width, stoptol, maxiter, tif[:, i, j])
        return result


class Bilateral_filter(BaseProcess):
    """bilateral_filter( keepSourceWindow=False)

    Parameters:
        soft (bool): True for guassian, False for hard filter
        beta (float): beta of kernel
        width (float): width of kernel
        stoptol (float): tolerance for convergence
        maxiter (int): maximum number of iterations
    Returns:
        newWindow
    """
    def __init__(self):
        super().__init__()
    def gui(self):
        self.gui_reset()

        soft=CheckBox()
        soft.setChecked(True)
        beta=SliderLabel(2)
        beta.setRange(1,500)
        beta.setValue(200)
        width=SliderLabel(2)
        width.setRange(1,50)
        width.setValue(8)
        stoptol=SliderLabel(4)
        stoptol.setRange(0,.02)
        stoptol.setValue(.001)
        maxiter=SliderLabel(0)
        maxiter.setRange(1,100)
        maxiter.setValue(10)
        preview=CheckBox()
        preview.setChecked(True)
        self.items.append({'name':'soft','string':'soft','object':soft})
        self.items.append({'name':'beta','string':'beta','object':beta})
        self.items.append({'name':'width','string':'width','object':width})
        self.items.append({'name':'stoptol','string':'stop tolerance','object':stoptol})
        self.items.append({'name':'maxiter','string':'Maximum Iterations','object':maxiter})
        self.items.append({'name':'preview','string':'Preview','object':preview})
        super().gui()
        self.roi=g.win.currentROI
        if self.roi is not None:
            self.ui.rejected.connect(self.roi.redraw_trace)
            self.ui.accepted.connect(self.roi.redraw_trace)
        else:
            preview.setChecked(False)
            preview.setEnabled(False)

    def __call__(self,soft, beta, width, stoptol, maxiter, keepSourceWindow=False):

        self.start(keepSourceWindow)
        if self.tif.ndim < 3:
            g.alert("Bilateral filter requires at least 3-dimensional image.")
            return
        self.newtif = _bilateral_filter_impl(self.tif, soft, beta, width, stoptol, maxiter)
        self.newname=self.oldname+' - Bilateral Filtered'
        return self.end()

    def preview(self):
        soft=self.getValue('soft')
        beta=self.getValue('beta')
        width=self.getValue('width')
        stoptol=self.getValue('stoptol')
        maxiter=self.getValue('maxiter')
        preview=self.getValue('preview')

        if self.roi is not None:
            if preview:
                trace=self.roi.getTrace()
                trace=bilateral_smooth(soft,beta,width,stoptol,maxiter,trace)
                roi_index=g.currentTrace.get_roi_index(self.roi)
                g.currentTrace.update_trace_full(roi_index,trace) #update_trace_partial may speed it up
            else:
                self.roi.redraw_trace()


def bilateral_filter_multi(soft,beta,width,stoptol,maxiter,tif):
    nThreads= g.settings['nCores']
    mt,mx,my=tif.shape
    block_ends=np.linspace(0,mx,nThreads+1).astype(int)
    data=[tif[:, block_ends[i]:block_ends[i+1],:] for i in np.arange(nThreads)] #split up data along x axis. each thread will get one.
    args=(soft,beta,width,stoptol,maxiter)
    progress = ProgressBar(bilateral_filter_inner, data, args, nThreads, msg='Performing Bilateral Filter')
    if progress.results is None or any(r is None for r in progress.results):
        result=None
    else:
        result=np.concatenate(progress.results,axis=1)
    return result


def bilateral_filter_inner(q_results, q_progress, q_status, child_conn, args):
    data=child_conn.recv() # unfortunately this step takes a long time
    percent=0  # This is the variable we send back which displays our progress
    status=q_status.get(True) #this blocks the process from running until all processes are launched
    if status=='Stop':
        q_results.put(None) # if the user presses stop, return None


    # Here is the meat of the inner_func.
    soft,beta,width,stoptol,maxiter=args #unpack all the variables inside the args tuple
    result=np.zeros(data.shape)
    tt,xx,yy=data.shape
    for x in np.arange(xx):
        for y in np.arange(yy):
            result[:,x,y]=bilateral_smooth(soft,beta,width,stoptol,maxiter,data[:,x,y])
            if not q_status.empty(): #check if the stop button has been pressed
                stop=q_status.get(False)
                q_results.put(None)
                return
        if percent<int(100*x/xx):
            percent=int(100*x/xx)
            q_progress.put(percent)

    # finally, when we've finished with our calculation, we send back the result
    q_results.put(result)

def bilateral_smooth(soft,beta,width,stoptol,maxiter,y):
    display=False       # 1 to report iteration values

    y=np.array(y[:])
    N=np.size(y,0)
    w=np.zeros((N,N))
    j=np.arange(0,N)

    #construct initial bilateral kernel
    for i in np.arange(0,N):
        w[i,np.arange(0,N)]=(abs(i-j) <= width)

    #initial guess from input signal
    xold=np.copy(y)

    #new matrix for storing distances
    d=np.zeros((N,N))

    #fig1 = plt.plot(y)

    if (display):
        if (soft):
            logger.debug('Soft kernel')
        else:
            logger.debug('Hard kernel')
        logger.debug('Kernel parameters beta= %d, W= %d' % (beta,width))
        logger.debug('Iter# Change')

    #start iteration
    iterate=1
    gap=np.inf

    while (iterate < maxiter):

        if (display):
            logger.debug('%d %f'% (iterate,gap))

        # calculate paiwise distances for all points
        for i in np.arange(0,N):
            d[:,i] = (0.5 * (xold - xold[i]) ** 2)

        #create kernel
        if (soft):
            W=np.multiply(np.exp(-beta*d),w)

        else:
            W=np.multiply((d <= beta ** 2),w)

        #apply kernel to get weighted mean shift
        xnew1=np.sum(np.multiply(np.transpose(W),xold), axis=1)
        xnew2=np.sum(W, axis=1)
        xnew=np.divide(xnew1,xnew2)

        #plt.plot(xnew)

        #check for convergence
        gap=np.sum(np.square(xold-xnew))

        if (gap < stoptol):
            if (display):
                logger.debug('Converged in %d iterations' % iterate)
            break

        xold=np.copy(xnew)
        iterate+=1
    return xold

bilateral_filter=Bilateral_filter()


###############################################################################
##################   EDGE / GRADIENT FILTERS   ################################
###############################################################################
from scipy.ndimage import gaussian_laplace, gaussian_gradient_magnitude
from scipy.ndimage import maximum_filter as nd_maximum_filter
from scipy.ndimage import minimum_filter as nd_minimum_filter
from scipy.ndimage import percentile_filter as nd_percentile_filter
import skimage.restoration


@per_plane
def _sobel_impl(tif):
    if tif.ndim == 2:
        return skimage.filters.sobel(tif.astype(np.float64))
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = skimage.filters.sobel(tif[i].astype(np.float64))
        return result


class Sobel_Filter(BaseProcess):
    """ sobel_filter(keepSourceWindow=False)

    Applies a Sobel edge detection filter to every frame of your stack.

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _sobel_impl(self.tif)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Sobel Filter'
        return self.end()

    def preview(self):
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            testimage = skimage.filters.sobel(testimage)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

sobel_filter = Sobel_Filter()


@per_plane
def _laplacian_impl(tif, ksize):
    if tif.ndim == 2:
        return skimage.filters.laplace(tif.astype(np.float64), ksize=ksize)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = skimage.filters.laplace(tif[i].astype(np.float64), ksize=ksize)
        return result


class Laplacian_Filter(BaseProcess):
    """ laplacian_filter(ksize, keepSourceWindow=False)

    Applies a Laplacian filter to every frame of your stack.

    Args:
        ksize (int): Size of the discrete Laplacian operator (must be odd)

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        ksize = SliderLabelOdd()
        ksize.setRange(3, 15)
        ksize.setValue(3)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'ksize', 'string': 'Kernel Size', 'object': ksize})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, ksize=3, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _laplacian_impl(self.tif, ksize)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Laplacian ksize=' + str(ksize)
        return self.end()

    def preview(self):
        ksize = self.getValue('ksize')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            testimage = skimage.filters.laplace(testimage, ksize=ksize)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

laplacian_filter = Laplacian_Filter()


@per_plane
def _gaussian_laplace_impl(tif, sigma):
    if tif.ndim == 2:
        return gaussian_laplace(tif.astype(np.float64), sigma=sigma)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = gaussian_laplace(tif[i].astype(np.float64), sigma=sigma)
        return result


class Gaussian_Laplace_Filter(BaseProcess):
    """ gaussian_laplace_filter(sigma, keepSourceWindow=False)

    Applies a Laplacian of Gaussian (LoG) filter to every frame of your stack.

    Args:
        sigma (float): Standard deviation of the Gaussian

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        sigma = SliderLabel(2)
        sigma.setRange(0.5, 50)
        sigma.setValue(1)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'sigma', 'string': 'Sigma (pixels)', 'object': sigma})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, sigma, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _gaussian_laplace_impl(self.tif, sigma)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Gaussian Laplace sigma=' + str(sigma)
        return self.end()

    def preview(self):
        sigma = self.getValue('sigma')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            testimage = gaussian_laplace(testimage, sigma=sigma)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

gaussian_laplace_filter = Gaussian_Laplace_Filter()


@per_plane
def _gaussian_gradient_magnitude_impl(tif, sigma):
    if tif.ndim == 2:
        return gaussian_gradient_magnitude(tif.astype(np.float64), sigma=sigma)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = gaussian_gradient_magnitude(tif[i].astype(np.float64), sigma=sigma)
        return result


class Gaussian_Gradient_Magnitude(BaseProcess):
    """ gaussian_gradient_magnitude_filter(sigma, keepSourceWindow=False)

    Applies a Gaussian gradient magnitude filter to every frame of your stack.

    Args:
        sigma (float): Standard deviation of the Gaussian

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        sigma = SliderLabel(2)
        sigma.setRange(0.5, 50)
        sigma.setValue(1)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'sigma', 'string': 'Sigma (pixels)', 'object': sigma})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, sigma, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _gaussian_gradient_magnitude_impl(self.tif, sigma)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Gaussian Gradient Magnitude sigma=' + str(sigma)
        return self.end()

    def preview(self):
        sigma = self.getValue('sigma')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            testimage = gaussian_gradient_magnitude(testimage, sigma=sigma)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

gaussian_gradient_magnitude_filter = Gaussian_Gradient_Magnitude()


###############################################################################
##################   RIDGE / TUBENESS FILTERS   ###############################
###############################################################################


@per_plane
def _sato_impl(tif, sigma_min, sigma_max, black_ridges):
    sigmas = np.linspace(sigma_min, sigma_max, max(int(sigma_max - sigma_min) + 1, 2))
    if tif.ndim == 2:
        return skimage.filters.sato(tif.astype(np.float64), sigmas=sigmas, black_ridges=black_ridges)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = skimage.filters.sato(tif[i].astype(np.float64), sigmas=sigmas, black_ridges=black_ridges)
        return result


class Sato_Tubeness(BaseProcess):
    """ sato_tubeness(sigma_min, sigma_max, black_ridges, keepSourceWindow=False)

    Applies the Sato tubeness filter for detecting tube-like structures.

    Args:
        sigma_min (float): Minimum sigma for the range of scales
        sigma_max (float): Maximum sigma for the range of scales
        black_ridges (bool): If True, detect black ridges; otherwise white ridges

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        sigma_min = SliderLabel(2)
        sigma_min.setRange(0.5, 50)
        sigma_min.setValue(1)
        sigma_max = SliderLabel(2)
        sigma_max.setRange(0.5, 50)
        sigma_max.setValue(5)
        black_ridges = CheckBox()
        black_ridges.setChecked(False)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'sigma_min', 'string': 'Sigma Min', 'object': sigma_min})
        self.items.append({'name': 'sigma_max', 'string': 'Sigma Max', 'object': sigma_max})
        self.items.append({'name': 'black_ridges', 'string': 'Black Ridges', 'object': black_ridges})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, sigma_min, sigma_max, black_ridges=False, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _sato_impl(self.tif, sigma_min, sigma_max, black_ridges)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Sato Tubeness'
        return self.end()

    def preview(self):
        sigma_min = self.getValue('sigma_min')
        sigma_max = self.getValue('sigma_max')
        black_ridges = self.getValue('black_ridges')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            sigmas = np.linspace(sigma_min, sigma_max, max(int(sigma_max - sigma_min) + 1, 2))
            testimage = skimage.filters.sato(testimage, sigmas=sigmas, black_ridges=black_ridges)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

sato_tubeness = Sato_Tubeness()


@per_plane
def _meijering_impl(tif, sigma_min, sigma_max, black_ridges):
    sigmas = np.linspace(sigma_min, sigma_max, max(int(sigma_max - sigma_min) + 1, 2))
    if tif.ndim == 2:
        return skimage.filters.meijering(tif.astype(np.float64), sigmas=sigmas, black_ridges=black_ridges)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = skimage.filters.meijering(tif[i].astype(np.float64), sigmas=sigmas, black_ridges=black_ridges)
        return result


class Meijering_Neuriteness(BaseProcess):
    """ meijering_neuriteness(sigma_min, sigma_max, black_ridges, keepSourceWindow=False)

    Applies the Meijering neuriteness filter for detecting neurite-like structures.

    Args:
        sigma_min (float): Minimum sigma for the range of scales
        sigma_max (float): Maximum sigma for the range of scales
        black_ridges (bool): If True, detect black ridges; otherwise white ridges

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        sigma_min = SliderLabel(2)
        sigma_min.setRange(0.5, 50)
        sigma_min.setValue(1)
        sigma_max = SliderLabel(2)
        sigma_max.setRange(0.5, 50)
        sigma_max.setValue(5)
        black_ridges = CheckBox()
        black_ridges.setChecked(False)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'sigma_min', 'string': 'Sigma Min', 'object': sigma_min})
        self.items.append({'name': 'sigma_max', 'string': 'Sigma Max', 'object': sigma_max})
        self.items.append({'name': 'black_ridges', 'string': 'Black Ridges', 'object': black_ridges})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, sigma_min, sigma_max, black_ridges=False, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _meijering_impl(self.tif, sigma_min, sigma_max, black_ridges)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Meijering Neuriteness'
        return self.end()

    def preview(self):
        sigma_min = self.getValue('sigma_min')
        sigma_max = self.getValue('sigma_max')
        black_ridges = self.getValue('black_ridges')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            sigmas = np.linspace(sigma_min, sigma_max, max(int(sigma_max - sigma_min) + 1, 2))
            testimage = skimage.filters.meijering(testimage, sigmas=sigmas, black_ridges=black_ridges)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

meijering_neuriteness = Meijering_Neuriteness()


@per_plane
def _hessian_impl(tif, sigma_min, sigma_max, black_ridges):
    sigmas = np.linspace(sigma_min, sigma_max, max(int(sigma_max - sigma_min) + 1, 2))
    if tif.ndim == 2:
        return skimage.filters.hessian(tif.astype(np.float64), sigmas=sigmas, black_ridges=black_ridges)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = skimage.filters.hessian(tif[i].astype(np.float64), sigmas=sigmas, black_ridges=black_ridges)
        return result


class Hessian_Filter(BaseProcess):
    """ hessian_filter(sigma_min, sigma_max, black_ridges, keepSourceWindow=False)

    Applies the Hessian filter for detecting ridge-like structures.

    Args:
        sigma_min (float): Minimum sigma for the range of scales
        sigma_max (float): Maximum sigma for the range of scales
        black_ridges (bool): If True, detect black ridges; otherwise white ridges

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        sigma_min = SliderLabel(2)
        sigma_min.setRange(0.5, 50)
        sigma_min.setValue(1)
        sigma_max = SliderLabel(2)
        sigma_max.setRange(0.5, 50)
        sigma_max.setValue(5)
        black_ridges = CheckBox()
        black_ridges.setChecked(False)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'sigma_min', 'string': 'Sigma Min', 'object': sigma_min})
        self.items.append({'name': 'sigma_max', 'string': 'Sigma Max', 'object': sigma_max})
        self.items.append({'name': 'black_ridges', 'string': 'Black Ridges', 'object': black_ridges})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, sigma_min, sigma_max, black_ridges=False, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _hessian_impl(self.tif, sigma_min, sigma_max, black_ridges)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Hessian Filter'
        return self.end()

    def preview(self):
        sigma_min = self.getValue('sigma_min')
        sigma_max = self.getValue('sigma_max')
        black_ridges = self.getValue('black_ridges')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            sigmas = np.linspace(sigma_min, sigma_max, max(int(sigma_max - sigma_min) + 1, 2))
            testimage = skimage.filters.hessian(testimage, sigmas=sigmas, black_ridges=black_ridges)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

hessian_filter = Hessian_Filter()


###############################################################################
##################   GABOR FILTER   ###########################################
###############################################################################


@per_plane
def _gabor_impl(tif, frequency, theta):
    if tif.ndim == 2:
        real, _ = skimage.filters.gabor(tif.astype(np.float64), frequency=frequency, theta=theta)
        return real
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            real, _ = skimage.filters.gabor(tif[i].astype(np.float64), frequency=frequency, theta=theta)
            result[i] = real
        return result


class Gabor_Filter(BaseProcess):
    """ gabor_filter(frequency, theta, keepSourceWindow=False)

    Applies a Gabor filter to every frame of your stack. Returns the real component.

    Args:
        frequency (float): Spatial frequency of the harmonic function (cycles/pixel)
        theta (float): Orientation of the filter in degrees (converted to radians internally)

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        frequency = SliderLabel(2)
        frequency.setRange(0.01, 1.0)
        frequency.setValue(0.1)
        theta = SliderLabel(2)
        theta.setRange(0, 180)
        theta.setValue(0)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'frequency', 'string': 'Frequency (cycles/pixel)', 'object': frequency})
        self.items.append({'name': 'theta', 'string': 'Theta (degrees)', 'object': theta})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, frequency, theta, keepSourceWindow=False):
        self.start(keepSourceWindow)
        theta_rad = np.deg2rad(theta)
        self.newtif = _gabor_impl(self.tif, frequency, theta_rad)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Gabor freq={} theta={}'.format(frequency, theta)
        return self.end()

    def preview(self):
        frequency = self.getValue('frequency')
        theta = self.getValue('theta')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            theta_rad = np.deg2rad(theta)
            testimage, _ = skimage.filters.gabor(testimage, frequency=frequency, theta=theta_rad)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

gabor_filter = Gabor_Filter()


###############################################################################
##################   MORPHOLOGICAL / RANK FILTERS   ###########################
###############################################################################


@per_plane
def _maximum_filter_impl(tif, size):
    if tif.ndim == 2:
        return nd_maximum_filter(tif.astype(np.float64), size=size)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = nd_maximum_filter(tif[i].astype(np.float64), size=size)
        return result


class Maximum_Filter(BaseProcess):
    """ maximum_filter(size, keepSourceWindow=False)

    Applies a spatial maximum filter to every frame of your stack.

    Args:
        size (int): Size of the filter kernel (must be odd)

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        size = SliderLabelOdd()
        size.setRange(3, 51)
        size.setValue(3)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'size', 'string': 'Kernel Size', 'object': size})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, size, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _maximum_filter_impl(self.tif, size)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Maximum Filter size=' + str(size)
        return self.end()

    def preview(self):
        size = self.getValue('size')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            testimage = nd_maximum_filter(testimage, size=size)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

maximum_filter = Maximum_Filter()


@per_plane
def _minimum_filter_impl(tif, size):
    if tif.ndim == 2:
        return nd_minimum_filter(tif.astype(np.float64), size=size)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = nd_minimum_filter(tif[i].astype(np.float64), size=size)
        return result


class Minimum_Filter(BaseProcess):
    """ minimum_filter(size, keepSourceWindow=False)

    Applies a spatial minimum filter to every frame of your stack.

    Args:
        size (int): Size of the filter kernel (must be odd)

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        size = SliderLabelOdd()
        size.setRange(3, 51)
        size.setValue(3)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'size', 'string': 'Kernel Size', 'object': size})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, size, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _minimum_filter_impl(self.tif, size)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Minimum Filter size=' + str(size)
        return self.end()

    def preview(self):
        size = self.getValue('size')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            testimage = nd_minimum_filter(testimage, size=size)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

minimum_filter = Minimum_Filter()


@per_plane
def _percentile_filter_impl(tif, percentile, size):
    if tif.ndim == 2:
        return nd_percentile_filter(tif.astype(np.float64), percentile=percentile, size=size)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = nd_percentile_filter(tif[i].astype(np.float64), percentile=percentile, size=size)
        return result


class Percentile_Filter(BaseProcess):
    """ percentile_filter(percentile, size, keepSourceWindow=False)

    Applies a spatial percentile filter to every frame of your stack.

    Args:
        percentile (float): The percentile to compute (0-100)
        size (int): Size of the filter kernel (must be odd)

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        percentile = SliderLabel(2)
        percentile.setRange(0, 100)
        percentile.setValue(50)
        size = SliderLabelOdd()
        size.setRange(3, 51)
        size.setValue(3)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'percentile', 'string': 'Percentile', 'object': percentile})
        self.items.append({'name': 'size', 'string': 'Kernel Size', 'object': size})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, percentile, size, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _percentile_filter_impl(self.tif, percentile, size)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Percentile Filter p={} size={}'.format(percentile, size)
        return self.end()

    def preview(self):
        percentile = self.getValue('percentile')
        size = self.getValue('size')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            testimage = nd_percentile_filter(testimage, percentile=percentile, size=size)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

percentile_filter = Percentile_Filter()


###############################################################################
##################   DENOISING FILTERS   ######################################
###############################################################################


@per_plane
def _tv_denoise_impl(tif, weight):
    if tif.ndim == 2:
        return skimage.restoration.denoise_tv_chambolle(tif.astype(np.float64), weight=weight)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = skimage.restoration.denoise_tv_chambolle(tif[i].astype(np.float64), weight=weight)
        return result


class TV_Denoise(BaseProcess):
    """ tv_denoise(weight, keepSourceWindow=False)

    Applies Total Variation denoising (Chambolle) to every frame of your stack.

    Args:
        weight (float): Denoising weight. Higher values remove more noise but also more detail.

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        weight = SliderLabel(2)
        weight.setRange(0.01, 1.0)
        weight.setValue(0.1)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'weight', 'string': 'Weight', 'object': weight})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, weight, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _tv_denoise_impl(self.tif, weight)
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - TV Denoise weight=' + str(weight)
        return self.end()

    def preview(self):
        weight = self.getValue('weight')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            testimage = skimage.restoration.denoise_tv_chambolle(testimage, weight=weight)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

tv_denoise = TV_Denoise()


###############################################################################
##################   FLASH REMOVER   ##########################################
###############################################################################
from ..utils.BaseProcess import ComboBox


class Flash_Remover(BaseProcess):
    """ flash_remover(method, flash_start, flash_end, auto_detect, window_size, keepSourceWindow=False)

    Detects and removes flash artifacts from image stacks.

    Args:
        method (str): 'Linear Interpolation' or 'Noise Scaling'
        flash_start (int): First frame of the flash artifact
        flash_end (int): Last frame of the flash artifact
        auto_detect (bool): If True, automatically detect flash frames
        window_size (int): Window size for moving average in auto-detection

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def _auto_detect_flash(self, tif, window_size):
        """Detect flash frames using moving average on mean frame intensity."""
        if tif.ndim == 2:
            return 0, 0
        mean_intensity = np.mean(tif.reshape(tif.shape[0], -1), axis=1)
        kernel = np.ones(window_size) / window_size
        if len(mean_intensity) < window_size:
            return 0, 0
        moving_avg = np.convolve(mean_intensity, kernel, mode='same')
        deviation = mean_intensity - moving_avg
        threshold = np.std(deviation) * 3
        flash_frames = np.where(np.abs(deviation) > threshold)[0]
        if len(flash_frames) == 0:
            return 0, 0
        return int(flash_frames[0]), int(flash_frames[-1])

    def _linear_interpolation(self, tif, flash_start, flash_end):
        """Replace flash frames by linearly interpolating between pre/post flash frames."""
        result = tif.copy().astype(np.float64)
        if tif.ndim < 3:
            return result
        mt = tif.shape[0]
        pre_frame = max(0, flash_start - 1)
        post_frame = min(mt - 1, flash_end + 1)
        if pre_frame == flash_start and post_frame == flash_end:
            return result
        n_flash = flash_end - flash_start + 1
        for idx, i in enumerate(range(flash_start, flash_end + 1)):
            alpha = (idx + 1) / (n_flash + 1)
            result[i] = np.interp(
                [alpha],
                [0, 1],
                [0, 1]
            )[0] * result[post_frame] + (1 - alpha) * result[pre_frame]
        return result

    def _noise_scaling(self, tif, flash_start, flash_end, window_size):
        """Scale flash frames by baseline noise ratio."""
        result = tif.copy().astype(np.float64)
        if tif.ndim < 3:
            return result
        mt = tif.shape[0]
        pre_start = max(0, flash_start - window_size)
        pre_end = flash_start
        post_start = flash_end + 1
        post_end = min(mt, flash_end + 1 + window_size)
        if pre_end > pre_start and post_end > post_start:
            baseline = np.concatenate([result[pre_start:pre_end], result[post_start:post_end]], axis=0)
        elif pre_end > pre_start:
            baseline = result[pre_start:pre_end]
        elif post_end > post_start:
            baseline = result[post_start:post_end]
        else:
            return result
        baseline_mean = np.mean(baseline, axis=0)
        baseline_std = np.std(baseline, axis=0)
        baseline_std[baseline_std == 0] = 1  # avoid division by zero
        for i in range(flash_start, flash_end + 1):
            frame_mean = np.mean(result[i])
            frame = result[i] - np.mean(result[i])
            frame_std = np.std(result[i])
            if frame_std > 0:
                result[i] = baseline_mean + (frame / frame_std) * baseline_std
            else:
                result[i] = baseline_mean
        return result

    def gui(self):
        self.gui_reset()
        method = ComboBox()
        method.addItem('Linear Interpolation')
        method.addItem('Noise Scaling')
        flash_start = SliderLabel(0)
        flash_start.setRange(0, 10000)
        flash_start.setValue(0)
        flash_end = SliderLabel(0)
        flash_end.setRange(0, 10000)
        flash_end.setValue(0)
        auto_detect = CheckBox()
        auto_detect.setChecked(True)
        window_size = SliderLabel(0)
        window_size.setRange(10, 500)
        window_size.setValue(50)
        self.items.append({'name': 'method', 'string': 'Method', 'object': method})
        self.items.append({'name': 'flash_start', 'string': 'Flash Start Frame', 'object': flash_start})
        self.items.append({'name': 'flash_end', 'string': 'Flash End Frame', 'object': flash_end})
        self.items.append({'name': 'auto_detect', 'string': 'Auto Detect', 'object': auto_detect})
        self.items.append({'name': 'window_size', 'string': 'Window Size', 'object': window_size})
        super().gui()

    def __call__(self, method='Linear Interpolation', flash_start=0, flash_end=0, auto_detect=True, window_size=50, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.ndim < 3:
            g.alert("Flash Remover requires at least 3-dimensional movies.")
            return
        if auto_detect:
            flash_start, flash_end = self._auto_detect_flash(self.tif, window_size)
            if flash_start == 0 and flash_end == 0:
                g.alert("No flash artifact detected.")
                return
        if flash_start > flash_end:
            g.alert("Flash start must be <= flash end.")
            return
        if method == 'Linear Interpolation':
            self.newtif = self._linear_interpolation(self.tif, flash_start, flash_end)
        elif method == 'Noise Scaling':
            self.newtif = self._noise_scaling(self.tif, flash_start, flash_end, window_size)
        else:
            self.newtif = self.tif
        self.newtif = self.newtif.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Flash Removed ({})'.format(method)
        return self.end()

flash_remover = Flash_Remover()



#from scipy import signal
#data=g.currentTrace.rois[0]['roi'].getTrace()
#wavelet = signal.ricker
#widths = np.arange(1, 200)
#cwtmatr = signal.cwt(data, wavelet, widths)
#import pyqtgraph as pg
#i=pg.image(cwtmatr.T)
#i.view.setAspectLocked(lock=True, ratio=cwtmatr.shape[0]/cwtmatr.shape[1]*20)
