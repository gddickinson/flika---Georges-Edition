# -*- coding: utf-8 -*-
import os

__all__ = ['image_path']

def image_path(image_name):
    """
    Return the absolute path to an image

    Parameters
    ----------
    image_name : str
       Name of image

    Returns
    -------
    path : str
      Full path to image
    """
    # Try the simple filesystem path first (works in most installations)
    result = os.path.join(os.path.dirname(__file__), image_name)
    if os.path.exists(result):
        return result
    # Fallback for zipped installations
    result_alt = os.path.join(
        os.path.dirname(__file__).replace('site-packages.zip', 'flika'),
        image_name,
    )
    if os.path.exists(result_alt):
        return result_alt
    raise RuntimeError("image does not exist: %s" % image_name)