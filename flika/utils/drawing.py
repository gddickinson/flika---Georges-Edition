# -*- coding: utf-8 -*-
"""Image annotation drawing utilities.

Consolidated drawing helpers used by process modules for overlaying
markers (circles, lines, crosses, points) on 2D images.
"""
import numpy as np
from skimage import draw as skdraw


def draw_circle(image, center_y, center_x, radius, value=1.0):
    """Draw a circle perimeter on a 2D image using skimage.draw.

    Parameters
    ----------
    image : 2D ndarray
        Image to draw on (modified in-place).
    center_y, center_x : float
        Center coordinates.
    radius : float
        Circle radius in pixels.
    value : float
        Pixel value for the circle perimeter.
    """
    rr, cc = skdraw.circle_perimeter(int(round(center_y)), int(round(center_x)),
                                     int(round(radius)), shape=image.shape)
    image[rr, cc] = value


def draw_circles(image, centers_yx, radii, value=1.0):
    """Draw multiple circles on a 2D image.

    Parameters
    ----------
    image : 2D ndarray
        Image to draw on (modified in-place).
    centers_yx : array-like (N, 2)
        Each row is (y, x).
    radii : array-like (N,)
        Radius for each circle.
    value : float
        Pixel value for the circle perimeters.
    """
    for (cy, cx), r in zip(centers_yx, radii):
        draw_circle(image, cy, cx, r, value)


def draw_line(image, y0, x0, y1, x1, value=1.0):
    """Draw a line on a 2D image using skimage.draw.line.

    Coordinates are clipped to image bounds.

    Parameters
    ----------
    image : 2D ndarray
        Image to draw on (modified in-place).
    y0, x0 : float
        Start coordinates.
    y1, x1 : float
        End coordinates.
    value : float
        Pixel value for the line.
    """
    rr, cc = skdraw.line(int(round(y0)), int(round(x0)),
                         int(round(y1)), int(round(x1)))
    mask = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
    image[rr[mask], cc[mask]] = value


def draw_crosses(image, centers_yx, size=3, value=1.0):
    """Draw cross markers at specified positions.

    Parameters
    ----------
    image : 2D ndarray
        Image to draw on (modified in-place).
    centers_yx : array-like (N, 2)
        Each row is (y, x).
    size : int
        Arm length of the cross in pixels.
    value : float
        Pixel value for the cross markers.
    """
    h, w = image.shape[:2]
    for y, x in centers_yx:
        y, x = int(round(y)), int(round(x))
        for dy in range(-size, size + 1):
            cy = y + dy
            if 0 <= cy < h and 0 <= x < w:
                image[cy, x] = value
        for dx in range(-size, size + 1):
            cx = x + dx
            if 0 <= y < h and 0 <= cx < w:
                image[y, cx] = value


def draw_points(image, centers_yx, value=1.0):
    """Draw single-pixel points at specified positions.

    Parameters
    ----------
    image : 2D ndarray
        Image to draw on (modified in-place).
    centers_yx : array-like (N, 2)
        Each row is (y, x).
    value : float
        Pixel value for the points.
    """
    h, w = image.shape[:2]
    for y, x in centers_yx:
        y, x = int(round(y)), int(round(x))
        if 0 <= y < h and 0 <= x < w:
            image[y, x] = value
