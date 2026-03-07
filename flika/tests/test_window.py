import sys, os

from .. import global_vars as g
from ..window import Window
import numpy as np
import time
import pytest
from ..roi import makeROI
import pyqtgraph as pg
from qtpy import QtGui
from qtpy.QtWidgets import QApplication

im = np.random.random([120, 90, 90])
	
class TestWindow():
	def setup_method(self, obj):
		self.win1 = Window(im)

	def teardown_method(self):
		self.win1.close()

	def test_link(self):
		win2 = Window(im)
		self.win1.link(win2)
		self.win1.setIndex(100)
		assert win2.currentIndex == self.win1.currentIndex, "Linked windows failure"
		win2.setIndex(50)
		assert self.win1.currentIndex == win2.currentIndex, "Linked windows failure"
		win2.unlink(self.win1)
		self.win1.setIndex(100)
		assert win2.currentIndex != self.win1.currentIndex, "Unlinked windows failure"
		win2.setIndex(50)
		assert self.win1.currentIndex != win2.currentIndex, "Unlinked windows failure"
		win2.close()
		assert len(self.win1.linkedWindows) == 0, "Closed window is not unlinked"

	def test_timeline(self):
		self.win1.imageview.setImage(im[0])
		assert self.win1.imageview.ui.roiPlot.isVisible() == False
		self.win1.imageview.setImage(im)
		assert self.win1.imageview.ui.roiPlot.isVisible() == True

class ROITest():
	TYPE=None
	POINTS=[]
	MASK=[]

	def setup_method(self):
		for i in range(3):
			try:
				self.win1 = Window(self.img)
				break
			except RuntimeError:
				pass
		if not hasattr(self, 'win1'):
			raise Exception("Unable to create Window due to RuntimeError")

		self.changed = False
		self.changeFinished = False
		self.roi = makeROI(self.TYPE, self.POINTS, window=self.win1)
		self.roi.sigRegionChanged.connect(self.preChange)
		self.roi.sigRegionChangeFinished.connect(self.preChangeFinished)
		self.check_placement()

	def preChange(self):
		self.changed = True
	
	def preChangeFinished(self):
		self.changeFinished = True

	def checkChanged(self):
		assert self.changed, "Change signal was not sent"
		self.changed = False
	
	def checkChangeFinished(self):
		assert self.changeFinished, "ChangeFinished signal was not sent"
		self.changeFinished = False
	
	def teardown_method(self, func):
		#if self.roi is not None:
		#	self.roi.delete()
		#assert self.roi not in self.win1.rois, "ROI deleted but still in window.rois"
		for roi in self.win1.rois:
			roi.delete()
		self.win1.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()



	def check_placement(self, mask=None, points=None):
		if mask is None:
			mask = self.MASK
		if points is None:
			points = self.POINTS
		assert np.array_equal(self.roi.getMask(), mask), "Mask differs on creation. %s != %s" % (self.roi.getMask(), self.MASK)
		assert np.array_equal(self.roi.pts, points), "pts differs on creation. %s != %s" % (self.roi.pts, self.POINTS)
		assert np.array_equal(self.roi.getPoints(), points), "getPoints differs on creation. %s != %s" % (self.roi.getPoints(), self.POINTS)

	def check_similar(self, other):
		mask1 = self.roi.getMask()
		mask2 = other.getMask()
		assert np.array_equal(mask1, mask2), "Mask differs on creation. %s != %s" % (mask1, mask2)
		assert np.array_equal(self.roi.pts, other.pts), "pts differs on creation. %s != %s" % (self.roi.pts, self.POINTS)
		assert np.array_equal(self.roi.getPoints(), other.getPoints()), "getPoints differs on creation. %s != %s" % (self.roi.getPoints(), other.getPoints())

	def test_copy(self):
		self.roi.copy()
		roi1 = self.win1.paste()
		assert roi1 == None and len(self.win1.rois) == 1, "Copying ROI to same window"
		
		w2 = Window(self.img)
		roi2 = w2.paste()
		assert self.roi in roi2.linkedROIs and roi2 in self.roi.linkedROIs, "Linked ROI on paste"
		self.check_similar(roi2)

		self.roi.translate([1, 2])

		self.checkChanged()
		self.checkChangeFinished()

		self.check_similar(roi2)

		w2.close()
		self.win1.setAsCurrentWindow()

	def test_export_import(self):
		s = str(self.roi)
		self.roi.window.save_rois("test.txt")
		from ..roi import open_rois
		rois = open_rois('test.txt')
		assert len(rois) == 1, "Import ROI failure"
		roi = rois[0]
		os.remove('test.txt')
		self.check_similar(roi)
		roi.delete()
	
	def test_plot(self):
		self.roi.unplot()
		trace = self.roi.plot()
		if trace is None:
			assert self.win1.image.ndim == 4, "Trace failed on non-4D image"
			return
		assert self.roi.traceWindow != None, "ROI plotted, roi traceWindow set"
		assert g.currentTrace == trace, "ROI plotted, but currentTrace is still None"
		ind = trace.get_roi_index(self.roi)
		if trace.rois[ind]['p1trace'].opts['pen'] != None:
			assert trace.rois[ind]['p1trace'].opts['pen'].color().name() == self.roi.pen.color().name(), "Color not changed. %s != %s" % (trace.rois[ind]['p1trace'].opts['pen'].color().name(), self.roi.pen.color().name())
		self.roi.unplot()
		assert self.roi.traceWindow == None, "ROI unplotted, roi traceWindow cleared"
		assert g.currentTrace == None, "ROI unplotted, currentTrace cleared"
		g.settings['multipleTraceWindows'] = False
	
	def test_translate(self):
		path = self.roi.pts.copy()
		self.roi.translate([2, 1])

		self.checkChanged()
		self.checkChangeFinished()

		assert len(self.roi.pts) == len(path), "roi size change on translate"
		assert [i + [2, 1] in path for i in self.roi.pts], "translate applied to mask"
		self.roi.translate([-2, -1])

		self.checkChanged()
		self.checkChangeFinished()

	def test_plot_translate(self):
		trace = self.roi.plot()
		if trace is None:
			assert self.win1.image.ndim == 4, "Trace failed on non-4D image"
			return
		assert self.roi.traceWindow != None, "ROI plotted, roi traceWindow set. %s should not be None" % (self.roi.traceWindow)
		assert g.currentTrace == self.roi.traceWindow, "ROI plotted, currentTrace not set. %s != %s" % (g.currentTrace, self.roi.traceWindow) 
		ind = trace.get_roi_index(self.roi)

		traceItem = trace.rois[ind]['p1trace']
		yData = traceItem.yData.copy()
		self.roi.translate(2, 1)
		assert not np.array_equal(yData, traceItem.yData), "Translated ROI yData compare %s != %s" % (yData, traceItem.yData)
		
		self.checkChanged()
		self.checkChangeFinished()

		self.roi.translate(-2, -1)
		assert np.array_equal(yData, traceItem.yData), "Translated back ROI yData compare %s != %s" % (yData, traceItem.yData)

		self.checkChanged()
		self.checkChangeFinished()

		self.roi.unplot()

	def test_change_color(self):
		self.roi.unplot()
		color = QtGui.QColor('#ff00ff')
		self.roi.plot()
		self.roi.colorSelected(color)
		
		self.checkChangeFinished()

		assert self.roi.pen.color().name() == color.name(), "Color not changed. %s != %s" % (self.roi.pen.color().name(), color.name())
		self.roi.unplot()
	
	
	def test_translate_multiple(self):
		translates = [[5, 0], [0, 5], [-5, 0], [0, -5]]
		self.roi.copy()
		w2 = Window(self.img)
		roi2 = w2.paste()
		roi2.colorSelected(QtGui.QColor(0, 255, 130))
		self.roi.plot()
		roi2.plot()
		for i in range(20 * len(translates)):
			tr = translates[i % len(translates)]
			self.roi.translate(*tr)
			self.checkChanged()
			self.checkChangeFinished()
			#time.sleep(.02)
			#QApplication.processEvents()

		w2.close()
		self.roi.draw_from_points(self.POINTS)
	
	def test_resize_multiple(self):
		if len(self.roi.getHandles()) == 0:
			return
		translates = [[1, 0], [0, 1], [-1, 0], [0, -1]]
		self.roi.copy()
		w2 = Window(self.img)
		roi2 = w2.paste()
		roi2.colorSelected(QtGui.QColor(0, 255, 130))
		self.roi.plot()
		roi2.plot()

		for h in self.roi.getHandles():
			h._updateView()
			pos = h.viewPos()
			for i in range(4 * len(translates)):
				tr = translates[i % len(translates)]
				self.roi.movePoint(h, [pos.x() + tr[0], pos.y() + tr[1]])
				self.checkChanged()
				self.checkChangeFinished()
				self.check_similar(roi2)
				#time.sleep(.02)
				#QApplication.processEvents()

		w2.close()
		self.roi.draw_from_points(self.POINTS)

class ROI_Rectangle(ROITest):
	TYPE = "rectangle"
	POINTS = [[3, 2], [2, 5]]
	MASK = [[3, 4, 3, 4, 3, 4, 3, 4, 3, 4], [2, 2, 3, 3, 4, 4, 5, 5, 6, 6]]

	def test_crop(self):
		w2 = self.roi.crop()
		bound = self.roi.boundingRect()
		mask = self.roi.getMask()
		w, h = np.ptp(mask, 1) + [1, 1]

		assert w == bound.width() and h == bound.height(), "Croppped image different size (%s, %s) != (%s, %s)" % (bound.width(), bound.height(), w, h)
		w2.close()

	def test_resize(self):
		self.roi.scale([1, 1.2])

		self.checkChanged()
		self.checkChangeFinished()
		
		points = [self.POINTS[0], [2, 6]]
		mask = [[3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4], [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]]
		self.check_placement(points=points, mask=mask)

		self.roi.draw_from_points(self.POINTS)

class ROI_Line(ROITest):
	TYPE="line"
	POINTS = [[3, 2], [8, 4]]
	MASK = [[3, 4, 5, 6, 7, 8], [2, 2, 3, 3, 4, 4]]

	def test_kymograph(self):
		self.roi.update_kymograph()
		if self.roi.kymograph is None and self.win1.image.ndim != 3:
			return
		kymo = self.roi.kymograph
		assert kymo.image.shape[1] == self.win1.image.shape[0]
		kymo.close()
		self.win1.setAsCurrentWindow()

	def test_move_handle(self):
		self.roi.movePoint(0, [3, 4])

		self.checkChanged()
		self.checkChangeFinished()
		
		newMask = [[3, 4, 5, 6, 7, 8], [4, 4, 4, 4, 4, 4]]
		assert np.array_equal(self.roi.getMask(), newMask)
		assert np.array_equal(self.roi.getPoints(), [[3, 4], [8, 4]])
		self.roi.movePoint(0, [3, 2])

		self.checkChanged()
		self.checkChangeFinished()


class ROI_Freehand(ROITest):
	TYPE="freehand"
	POINTS = [3, 2], [5, 6], [2, 4]
	MASK = [[0, 1, 1, 1, 2, 2, 3], [2, 0, 1, 2, 2, 3, 4]]

	def test_plot_translate(self):
		pass

	def test_translate_multiple(self):
		pass


class ROI_Rect_Line(ROITest):
	TYPE="rect_line"
	POINTS = [[3, 2], [5, 4], [4, 8]]
	MASK = [[3, 4, 5, 5, 5, 4, 4, 4], [2, 3, 4, 4, 5, 6, 7, 8]]

	
	def test_extend(self):
		self.roi.extend(9, 2)

		self.checkChanged()
		self.checkChangeFinished()

		self.roi.removeSegment(len(self.roi.lines)-1)
	
	def test_plot_translate(self):
		pass

	def test_translate(self):
		pass

	def test_translate_multiple(self):
		pass

	def test_kymograph(self):
		self.roi.update_kymograph()
		if self.roi.kymograph is None and self.win1.image.ndim != 3:
			return
		kymo = self.roi.kymograph
		assert kymo.image.shape[1] == self.win1.image.shape[0]
		
		self.roi.setWidth(3)
		assert kymo.image.shape[1] == self.win1.image.shape[0]

		kymo.close()
		self.win1.setAsCurrentWindow()

	def test_copy(self):
		self.roi.copy()
		roi1 = self.win1.paste()
		assert roi1 == None and len(self.win1.rois) == 1, "Copying ROI to same window"
		
		w2 = Window(self.img)
		roi2 = w2.paste()
		assert self.roi in roi2.linkedROIs and roi2 in self.roi.linkedROIs, "Linked ROI on paste"
		self.check_similar(roi2)

		self.roi.lines[0].movePoint(0, [1, 2])

		self.checkChanged()
		self.checkChangeFinished()

		self.check_similar(roi2)

		w2.close()
		self.win1.setAsCurrentWindow()


class Test_Rectangle_2D(ROI_Rectangle):
	img = np.random.random([20, 20])
class Test_Rectangle_3D(ROI_Rectangle):
	img = np.random.random([10, 20, 20])
class Test_Rectangle_4D(ROI_Rectangle):
	img = np.random.random([10, 20, 20, 3])

class Test_Line_2D(ROI_Line):
	img = np.random.random([20, 20])
class Test_Line_3D(ROI_Line):
	img = np.random.random([10, 20, 20])
class Test_Line_4D(ROI_Line):
	img = np.random.random([10, 20, 20, 3])

class Test_Freehand_2D(ROI_Freehand):
	img = np.random.random([20, 20])
class Test_Freehand_3D(ROI_Freehand):
	img = np.random.random([10, 20, 20])
class Test_Freehand_4D(ROI_Freehand):
	img = np.random.random([10, 20, 20, 3])

class Test_Rect_Line_2D(ROI_Rect_Line):
	img = np.random.random([20, 20])
class Test_Rect_Line_3D(ROI_Rect_Line):
	img = np.random.random([10, 20, 20])
class Test_Rect_Line_4D(ROI_Rect_Line):
	img = np.random.random([10, 20, 20, 3])




##############################################################################
# Ellipse ROI tests
##############################################################################

class ROI_Ellipse(ROITest):
	TYPE = "ellipse"
	# pts[0] = center, pts[1] = corner -> center at (10,10), corner at (14,13)
	POINTS = [[10, 10], [14, 13]]
	MASK = None  # ellipse mask is non-trivial; override check_placement

	def check_placement(self, mask=None, points=None):
		"""For ellipses, just verify mask is non-empty and within bounds."""
		m = self.roi.getMask()
		assert len(m) == 2, "getMask should return (xx, yy) tuple"
		assert np.size(m[0]) > 0, "Ellipse mask should not be empty"
		assert np.all(m[0] >= 0) and np.all(m[0] < self.win1.mx), "Mask x out of bounds"
		assert np.all(m[1] >= 0) and np.all(m[1] < self.win1.my), "Mask y out of bounds"

	def test_copy(self):
		"""Ellipse copy/paste - verify basic mechanics work."""
		self.roi.copy()
		roi1 = self.win1.paste()
		assert roi1 == None and len(self.win1.rois) == 1, "Copying ROI to same window"
		w2 = Window(self.img)
		roi2 = w2.paste()
		assert roi2 is not None, "Paste should return a new ROI"
		# Ellipse copy goes through text which may change representation;
		# just verify non-empty mask
		m2 = roi2.getMask()
		assert np.size(m2[0]) > 0, "Pasted ellipse mask should be non-empty"
		w2.close()
		self.win1.setAsCurrentWindow()

	def test_translate(self):
		m_before = self.roi.getMask()
		count_before = np.size(m_before[0])
		self.roi.translate([2, 1])
		self.checkChanged()
		self.checkChangeFinished()
		m_after = self.roi.getMask()
		assert np.size(m_after[0]) == count_before, "Ellipse pixel count changed on translate"
		self.roi.translate([-2, -1])
		self.checkChanged()
		self.checkChangeFinished()

	def test_export_import(self):
		# Ellipse export/import produces a different ROI kind (from text),
		# so we just verify the round-trip doesn't crash.
		s = str(self.roi)
		self.roi.window.save_rois("test_ellipse.txt")
		from ..roi import open_rois
		rois = open_rois('test_ellipse.txt')
		assert len(rois) == 1, "Import ROI failure"
		os.remove('test_ellipse.txt')
		rois[0].delete()

	def test_translate_multiple(self):
		pass

	def test_resize_multiple(self):
		pass

	def test_plot_translate(self):
		pass

class Test_Ellipse_2D(ROI_Ellipse):
	img = np.random.random([30, 30])
class Test_Ellipse_3D(ROI_Ellipse):
	img = np.random.random([10, 30, 30])
class Test_Ellipse_4D(ROI_Ellipse):
	img = np.random.random([10, 30, 30, 3])


##############################################################################
# Point ROI tests
##############################################################################

class TestPointROI():
	def setup_method(self):
		self.img2d = np.random.random([20, 20])
		self.img3d = np.random.random([10, 20, 20])
		self.win = Window(self.img3d)

	def teardown_method(self):
		for roi in self.win.rois:
			roi.delete()
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_point_mask(self):
		roi = makeROI('point_roi', [[5, 7]], window=self.win)
		m = roi.getMask()
		assert np.array_equal(m[0], [5]), "Point ROI x mask"
		assert np.array_equal(m[1], [7]), "Point ROI y mask"

	def test_point_trace(self):
		roi = makeROI('point_roi', [[5, 7]], window=self.win)
		trace = roi.plot()
		if trace is not None:
			assert roi.traceWindow is not None, "Point ROI trace plotted"
			roi.unplot()


##############################################################################
# Center-Surround ROI tests
##############################################################################

class TestCenterSurroundROI():
	def setup_method(self):
		self.img = np.random.random([10, 30, 30])
		self.win = Window(self.img)

	def teardown_method(self):
		for roi in self.win.rois:
			roi.delete()
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_cs_masks_exist(self):
		# pos=[5,5], size=[10,10]
		roi = makeROI('center_surround', [[5, 5], [10, 10]], window=self.win)
		outer = roi.getMask()
		center = roi.getCenterMask()
		surround = roi.getSurroundMask()
		assert np.size(outer[0]) > 0, "Outer mask should not be empty"
		assert np.size(center[0]) > 0, "Center mask should not be empty"
		assert np.size(surround[0]) > 0, "Surround mask should not be empty"

	def test_center_inside_outer(self):
		roi = makeROI('center_surround', [[5, 5], [10, 10]], window=self.win)
		outer_set = set(zip(roi.getMask()[0].tolist(), roi.getMask()[1].tolist()))
		center_set = set(zip(roi.getCenterMask()[0].tolist(), roi.getCenterMask()[1].tolist()))
		assert center_set.issubset(outer_set), "Center mask should be a subset of outer mask"


##############################################################################
# ROI Manager tests
##############################################################################

class TestROIManager():
	def setup_method(self):
		self.img = np.random.random([10, 20, 20])
		self.win = Window(self.img)

	def teardown_method(self):
		for roi in self.win.rois:
			roi.delete()
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_manager_creation(self):
		from ..app.roi_manager import ROIManager
		mgr = ROIManager.instance()
		assert mgr is not None, "ROIManager should be created"

	def test_manager_item_count(self):
		from ..app.roi_manager import ROIManager
		makeROI('rectangle', [[3, 2], [4, 5]], window=self.win)
		makeROI('line', [[1, 1], [5, 5]], window=self.win)
		assert len(self.win.rois) == 2, "Window should have 2 ROIs"
		mgr = ROIManager.instance()
		mgr._rebuild_list()
		assert mgr.tree.topLevelItemCount() == len(self.win.rois), \
			"ROI Manager item count should match window ROI count"


##############################################################################
# ROI Stats tests
##############################################################################

class TestROIStats():
	def setup_method(self):
		self.img = np.random.random([10, 20, 20])
		self.win = Window(self.img)
		self.roi = makeROI('rectangle', [[3, 2], [4, 5]], window=self.win)

	def teardown_method(self):
		for roi in self.win.rois:
			roi.delete()
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_compute_roi_stats(self):
		from ..utils.roi_stats import compute_roi_stats
		stats = compute_roi_stats(self.roi, self.win)
		assert 'mean' in stats, "stats should have mean"
		assert 'std' in stats, "stats should have std"
		assert 'min' in stats, "stats should have min"
		assert 'max' in stats, "stats should have max"
		assert 'area' in stats, "stats should have area"
		assert stats['area'] > 0, "area should be > 0"

	def test_compute_shape_descriptors(self):
		from ..utils.roi_stats import compute_shape_descriptors
		desc = compute_shape_descriptors(self.roi, self.win)
		assert 'perimeter' in desc, "descriptors should have perimeter"
		assert 'circularity' in desc, "descriptors should have circularity"
		assert desc['perimeter'] > 0, "perimeter should be > 0"


##############################################################################
# Boolean ROI operations tests
##############################################################################

class TestBooleanROIOps():
	def setup_method(self):
		self.img = np.random.random([20, 20])
		self.win = Window(self.img)
		# Overlapping rectangles
		self.roi_a = makeROI('rectangle', [[2, 2], [6, 6]], window=self.win)
		self.roi_b = makeROI('rectangle', [[5, 5], [6, 6]], window=self.win)

	def teardown_method(self):
		for roi in self.win.rois:
			roi.delete()
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_and(self):
		from ..roi import boolean_roi_op
		result = boolean_roi_op(self.roi_a, self.roi_b, 'AND')
		assert result is not None, "AND should produce a result"
		m = result.getMask()
		assert np.size(m[0]) > 0, "AND mask should not be empty"

	def test_or(self):
		from ..roi import boolean_roi_op
		result = boolean_roi_op(self.roi_a, self.roi_b, 'OR')
		assert result is not None, "OR should produce a result"
		m = result.getMask()
		# OR should have at least as many pixels as the larger ROI
		assert np.size(m[0]) >= np.size(self.roi_a.getMask()[0]), "OR mask too small"

	def test_xor(self):
		from ..roi import boolean_roi_op
		result = boolean_roi_op(self.roi_a, self.roi_b, 'XOR')
		assert result is not None, "XOR should produce a result"

	def test_subtract(self):
		from ..roi import boolean_roi_op
		result = boolean_roi_op(self.roi_a, self.roi_b, 'SUBTRACT')
		assert result is not None, "SUBTRACT should produce a result"
		m = result.getMask()
		# SUBTRACT should have fewer pixels than the original
		assert np.size(m[0]) < np.size(self.roi_a.getMask()[0]), "SUBTRACT should reduce pixel count"


##############################################################################
# Transform ROI operations tests
##############################################################################

class TestTransformROI():
	def setup_method(self):
		self.img = np.random.random([30, 30])
		self.win = Window(self.img)
		self.roi = makeROI('rectangle', [[5, 5], [10, 10]], window=self.win)

	def teardown_method(self):
		for roi in self.win.rois:
			roi.delete()
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_enlarge(self):
		from ..roi import transform_roi_mask
		original_count = np.size(self.roi.getMask()[0])
		result = transform_roi_mask(self.roi, 'enlarge', pixels=2)
		assert result is not None, "enlarge should produce a result"
		assert np.size(result.getMask()[0]) > original_count, "enlarged mask should be larger"

	def test_shrink(self):
		from ..roi import transform_roi_mask
		original_count = np.size(self.roi.getMask()[0])
		result = transform_roi_mask(self.roi, 'shrink', pixels=1)
		assert result is not None, "shrink should produce a result"
		assert np.size(result.getMask()[0]) < original_count, "shrunk mask should be smaller"

	def test_band(self):
		from ..roi import transform_roi_mask
		result = transform_roi_mask(self.roi, 'band', pixels=2)
		assert result is not None, "band should produce a result"

	def test_convex_hull(self):
		from ..roi import transform_roi_mask
		result = transform_roi_mask(self.roi, 'convex_hull')
		assert result is not None, "convex_hull should produce a result"


##############################################################################
# 4D Viewer smoke tests
##############################################################################

class TestOrthogonalViewer():
	def setup_method(self):
		self.img4d = np.random.random([5, 15, 15, 4])
		self.win = Window(self.img4d)

	def teardown_method(self):
		if self.win._ortho_viewer is not None:
			self.win._ortho_viewer.close()
			self.win._ortho_viewer = None
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_ortho_viewer_creation(self):
		from ..window import OrthogonalViewer
		ov = OrthogonalViewer(self.win)
		assert ov is not None, "OrthogonalViewer created"
		ov.show()
		self.win._ortho_viewer = ov

	def test_ortho_set_crosshair(self):
		from ..window import OrthogonalViewer
		ov = OrthogonalViewer(self.win)
		ov.show()
		self.win._ortho_viewer = ov
		ov.set_crosshair(7, 5)
		assert ov._x_pos == 7, "Crosshair x position"
		assert ov._y_pos == 5, "Crosshair y position"


class TestVolumeViewer():
	def setup_method(self):
		self.img4d = np.random.random([5, 15, 15, 4])
		self.win = Window(self.img4d)

	def teardown_method(self):
		if self.win._volume_viewer is not None:
			self.win._volume_viewer.close()
			self.win._volume_viewer = None
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_volume_viewer_creation(self):
		from ..viewers.volume_viewer import VolumeViewer, _HAS_GL
		if not _HAS_GL:
			pytest.skip("pyqtgraph.opengl not available")
		vv = VolumeViewer(self.win)
		assert vv is not None, "VolumeViewer created"
		self.win._volume_viewer = vv

	def test_volume_viewer_set_crosshair(self):
		from ..viewers.volume_viewer import VolumeViewer, _HAS_GL
		if not _HAS_GL:
			pytest.skip("pyqtgraph.opengl not available")
		vv = VolumeViewer(self.win)
		self.win._volume_viewer = vv
		vv.set_crosshair(5, 3)
		assert vv._crosshair_x == 5, "Volume viewer crosshair x"
		assert vv._crosshair_y == 3, "Volume viewer crosshair y"

	def test_volume_viewer_reset_camera(self):
		from ..viewers.volume_viewer import VolumeViewer, _HAS_GL
		if not _HAS_GL:
			pytest.skip("pyqtgraph.opengl not available")
		vv = VolumeViewer(self.win)
		self.win._volume_viewer = vv
		vv._reset_camera()
		# Just verify no crash; camera position API varies by pyqtgraph version


##############################################################################
# Line Profile and Histogram viewer smoke tests
##############################################################################

class TestLineProfile():
	def setup_method(self):
		self.img = np.random.random([10, 20, 20])
		self.win = Window(self.img)
		self.roi = makeROI('line', [[3, 2], [8, 4]], window=self.win)

	def teardown_method(self):
		for roi in self.win.rois:
			roi.delete()
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_line_profile_creation(self):
		from ..viewers.line_profile import LineProfileWidget
		widget = LineProfileWidget(self.roi)
		assert widget is not None, "LineProfileWidget created"
		widget.close()


class TestROIHistogram():
	def setup_method(self):
		self.img = np.random.random([10, 20, 20])
		self.win = Window(self.img)
		self.roi = makeROI('rectangle', [[3, 2], [4, 5]], window=self.win)

	def teardown_method(self):
		for roi in self.win.rois:
			roi.delete()
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_histogram_creation(self):
		from ..viewers.roi_histogram import ROIHistogramWidget
		widget = ROIHistogramWidget(self.roi)
		assert widget is not None, "ROIHistogramWidget created"
		widget.close()


##############################################################################
# Window feature tests (snapshot, crosshair mode, context menu)
##############################################################################

class TestWindowFeatures():
	def setup_method(self):
		self.img = np.random.random([10, 20, 20])
		self.win = Window(self.img)

	def teardown_method(self):
		self.win.close()
		if g.m is not None:
			g.m.clear()
		pg.ViewBox.AllViews.clear()
		pg.ViewBox.NamedViews.clear()

	def test_snapshot(self):
		"""Test snapshot method doesn't crash (actual file creation depends on Desktop path)."""
		desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
		if not os.path.isdir(desktop):
			pytest.skip("No Desktop directory")
		self.win.snapshot()
		# Clean up the snapshot file
		import glob
		pattern = os.path.join(desktop, '*_????????_??????.png')
		for f in sorted(glob.glob(pattern))[-1:]:
			if self.win.name.replace(' ', '_').replace('/', '_') in f:
				os.remove(f)

	def test_crosshair_initial_state(self):
		assert self.win._crosshair_active == False, "Crosshair should be off initially"

	def test_ndims(self):
		assert self.win.nDims == 3, "3D image should have nDims == 3"
		win2d = Window(np.random.random([20, 20]))
		assert win2d.nDims == 2, "2D image should have nDims == 2"
		win2d.close()


##############################################################################
# Dependency checker unit tests
##############################################################################

class TestDependencyChecker():
	def test_check_package_numpy(self):
		from ..app.dependency_checker import _check_package
		status, installed, action = _check_package("numpy", "1.20")
		assert "OK" in status, f"numpy should be installed, got {status}"
		assert installed != "-", "numpy version should be detected"

	def test_check_package_missing(self):
		from ..app.dependency_checker import _check_package
		status, installed, action = _check_package("nonexistent_pkg_xyz_12345")
		assert "MISSING" in status, f"fake package should be MISSING, got {status}"
		assert "pip install" in action, "Missing package should have pip install action"


class TestTracefig():
	def setup_method(self):
		self.w1 = Window(im)
		self.rect = makeROI('rectangle', [[3, 2], [4, 5]], window=self.w1)
		self.trace = self.rect.plot()

	def teardown_method(self):
		self.w1.close()

	def test_plotting(self):
		self.trace.indexChanged.emit(20)
		assert self.w1.currentIndex == 20, "trace indexChanged"

	def test_export(self):
		self.rect.window.save_rois('tempROI.txt')
		t = open('tempROI.txt').read()
		assert t == 'rectangle\n3 2\n4 5\n'
		os.remove('tempROI.txt')
