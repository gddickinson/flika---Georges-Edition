#!/usr/bin/env python
"""Thin compatibility shim -- see pyproject.toml for the canonical build config.

Commands to upload to pypi:
    python -m build
    twine upload dist/*
"""
from setuptools import setup, find_packages
import sys

with open('flika/version.py') as version_file:
    __version__ = '0.0.0'
    exec(version_file.read())  # This sets the __version__ variable.

try:
    with open('README.rst') as readme:
        LONG_DESCRIPTION = readme.read()
        LONG_DESCRIPTION = LONG_DESCRIPTION.replace(
            '.. image:: flika/docs/_static/img/flika_screencapture.gif', '')
except FileNotFoundError:
    LONG_DESCRIPTION = ''

entry_points = """
[console_scripts]
flika = flika.flika:exec_
flika_post_install = flika.flika:post_install
"""

install_requires = [
      'numpy>=1.24',
      'scipy>=1.10',
      'pandas>=1.5',
      'matplotlib>=3.6',
      'pyqtgraph>=0.13',
      'PyQt6',
      'qtpy>=2.3',
      'setuptools>=1.0',
      'scikit-image>=0.20',
      'scikit-learn>=1.2',
      'ipython>=8.0',
      'ipykernel',
      'qtconsole',
      'pyopengl',
      'requests',
      'nd2reader',
      'markdown',
      'packaging',
      'tifffile>=2022.5.4']

extras_require = {
    ':sys_platform == "win32"': ['winshell', 'pypiwin32'],
    'ai': ['anthropic>=0.18'],
    'gpu': ['cupy'],
    'accel': ['numba>=0.57', 'torch>=2.0'],
    'all-formats': ['h5py', 'zarr', 'ome-zarr>=0.9', 'aicsimageio'],
    'lazy': ['dask[array]>=2022.1'],
    'segmentation': ['cellpose', 'stardist', 'csbdeep', 'micro_sam'],
    'model-zoo': ['bioimageio.core>=0.6', 'bioimageio.spec>=0.5'],
    'denoising': ['careamics>=0.1'],
    'spt': ['trackpy>=0.6'],
    'dev': ['pytest', 'pytest-qt', 'flake8', 'mypy'],
}

setup(name='flika',
      version=__version__,
      description='An interactive image processing program for biologists written in Python.',
      long_description=LONG_DESCRIPTION,
      author='Kyle Ellefsen, Brett Settle, George Bhatt',
      author_email='kyleellefsen@gmail.com',
      url='http://flika-org.github.io',
      install_requires=install_requires,
      extras_require=extras_require,
      license='MIT',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3.13',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Image Processing',
          ],
      packages=find_packages(),
      entry_points=entry_points,
      include_package_data=True,
      package_data={'gui': ['*.ui'],
                    'images': ['*.ico', '*.png', '*.txt', '*.tif']})
