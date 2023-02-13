"""
# Helper functions for analysis of FIJI registration
"""
import numpy as np

use_cp = False
if use_cp:
    import cupy as cp
import pandas as pd
import os
from pathlib import Path
import time
import glob
import re

import matplotlib
import matplotlib.image as mpimg
from matplotlib import pylab, mlab, pyplot

plt = pyplot
from IPython.core.pylabtools import figsize, getfigs
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as PILImage
from PIL.TiffTags import TAGS

from struct import *

# from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm

import skimage

# print(skimage.__version__)
from skimage.measure import ransac
from skimage.transform import (
    ProjectiveTransform,
    AffineTransform,
    EuclideanTransform,
    warp,
)

try:
    import skimage.external.tifffile as tiff
except:
    import tifffile as tiff
from scipy.signal import savgol_filter
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

from sklearn.linear_model import (
    LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


import dask
import dask.array as da
from dask.distributed import Client, progress, get_task_stream
from dask.diagnostics import ProgressBar

import cv2

print("Open CV version: ", cv2.__version__)
import mrcfile
import h5py
import npy2bdv
import pickle
import webbrowser
from IPython.display import IFrame

EPS = np.finfo(float).eps

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def read_transformation_matrix_from_xf_file(xf_filename):
    """
    Reads transformation matrix created by FiJi-based workflow from *.xf file. ©G.Shtengel 10/2022 gleb.shtengel@gmail.com

    Parameters:
    xf_filename : str
        Full path to *.xf file containing the transformation matrix data

    Returns:
    transformation_matrix : array
    """
    npdt_recalled = pd.read_csv(xf_filename, sep="  ", header=None)
    tr = npdt_recalled.to_numpy()
    transformation_matrix = np.zeros((len(tr), 3, 3))
    transformation_matrix[:, 0, 0:2] = tr[:, 0:2]
    transformation_matrix[:, 0, 2] = tr[:, 6]
    transformation_matrix[:, 1, 0:2] = tr[:, 2:4]
    transformation_matrix[:, 1, 2] = tr[:, 9]
    transformation_matrix[:, 2, 2] = np.ones((len(tr)))
    return transformation_matrix


def analyze_transformation_matrix(transformation_matrix, xf_filename):
    """
    Analyzes the transformation matrix created by FiJi-based workflow. ©G.Shtengel 10/2022 gleb.shtengel@gmail.com

    Parameters:
    transformation_matrix : array
        Transformation matrix (read by read_transformation_matrix_from_xf_file above).
    xf_filename : str
        Full path to *.xf file containing the transformation matrix data

    Returns:
    tr_matr_cum : array
    Cumulative transformation matrix
    """
    Xshift_orig = transformation_matrix[:, 0, 2]
    Yshift_orig = transformation_matrix[:, 1, 2]
    Xscale_orig = transformation_matrix[:, 0, 0]
    Yscale_orig = transformation_matrix[:, 1, 1]
    tr_matr_cum = transformation_matrix.copy()

    prev_mt = np.eye(3, 3)
    for j, cur_mt in enumerate(
        tqdm(
            transformation_matrix, desc="Calculating Cummilative Transformation Matrix"
        )
    ):
        if any(np.isnan(cur_mt)):
            print(
                "Frame: {:d} has ill-defined transformation matrix, will use identity transformation instead:".format(
                    j
                )
            )
            print(cur_mt)
        else:
            prev_mt = np.matmul(cur_mt, prev_mt)
        tr_matr_cum[j] = prev_mt
    # Now insert identity matrix for the zero frame which does not need to be trasformed
    tr_matr_cum_orig = tr_matr_cum.copy()

    s00_cum_orig = tr_matr_cum[:, 0, 0].copy()
    s11_cum_orig = tr_matr_cum[:, 1, 1].copy()
    s01_cum_orig = tr_matr_cum[:, 0, 1].copy()
    s10_cum_orig = tr_matr_cum[:, 1, 0].copy()

    Xshift_cum_orig = tr_matr_cum_orig[:, 0, 2]
    Yshift_cum_orig = tr_matr_cum_orig[:, 1, 2]

    # print('Recalculating Shifts')
    s00_cum_orig = tr_matr_cum[:, 0, 0]
    s11_cum_orig = tr_matr_cum[:, 1, 1]
    fr = np.arange(0, len(s00_cum_orig), dtype=float)
    s00_slp = (
        -1.0 * (np.sum(fr) - np.dot(s00_cum_orig, fr)) / np.dot(fr, fr)
    )  # find the slope of a linear fit with fiorced first scale=1
    s00_fit = 1.0 + s00_slp * fr
    s00_cum_new = s00_cum_orig + 1.0 - s00_fit
    s11_slp = (
        -1.0 * (np.sum(fr) - np.dot(s11_cum_orig, fr)) / np.dot(fr, fr)
    )  # find the slope of a linear fit with fiorced first scale=1
    s11_fit = 1.0 + s11_slp * fr
    s11_cum_new = s11_cum_orig + 1.0 - s11_fit

    s01_slp = np.dot(s01_cum_orig, fr) / np.dot(
        fr, fr
    )  # find the slope of a linear fit with forced first scale=1
    s01_fit = s01_slp * fr
    s01_cum_new = s01_cum_orig - s01_fit
    s10_slp = np.dot(s10_cum_orig, fr) / np.dot(
        fr, fr
    )  # find the slope of a linear fit with forced first scale=1
    s10_fit = s10_slp * fr
    s10_cum_new = s10_cum_orig - s10_fit

    Xshift_cum = tr_matr_cum[:, 0, 2]
    Yshift_cum = tr_matr_cum[:, 1, 2]

    subtract_linear_fit = True

    # Subtract linear trend from offsets
    if subtract_linear_fit:
        fr = np.arange(0, len(Xshift_cum))
        pX = np.polyfit(fr, Xshift_cum, 1)
        Xfit = np.polyval(pX, fr)
        pY = np.polyfit(fr, Yshift_cum, 1)
        Yfit = np.polyval(pY, fr)
        Xshift_residual = Xshift_cum - Xfit
        Yshift_residual = Yshift_cum - Yfit
    else:
        Xshift_residual = Xshift_cum.copy()
        Yshift_residual = Yshift_cum.copy()

    # define new cum. transformation matrix where the offests may have linear slopes subtracted
    tr_matr_cum_residual = tr_matr_cum.copy()
    tr_matr_cum_residual[:, 0, 2] = Xshift_residual
    tr_matr_cum_residual[:, 1, 2] = Yshift_residual
    tr_matr_cum_residual[:, 0, 0] = s00_cum_new
    tr_matr_cum_residual[:, 1, 1] = s11_cum_new
    tr_matr_cum_residual[:, 0, 1] = s01_cum_new
    tr_matr_cum_residual[:, 1, 0] = s10_cum_new

    fs = 12
    fig5, axs5 = subplots(3, 3, figsize=(18, 12), sharex=True)
    fig5.subplots_adjust(left=0.15, bottom=0.08, right=0.99, top=0.94)

    # plot scales
    axs5[0, 0].plot(Xscale_orig, "r", label="Sxx fr.-to-fr.")
    axs5[0, 0].plot(Yscale_orig, "b", label="Syy fr.-to-fr.")
    axs5[0, 0].set_title("Frame-to-Frame Scale Change", fontsize=fs + 1)
    axs5[1, 0].plot(
        tr_matr_cum_orig[:, 0, 0], "r", linestyle="dotted", label="Sxx cum."
    )
    axs5[1, 0].plot(
        tr_matr_cum_orig[:, 1, 1], "b", linestyle="dotted", label="Syy cum."
    )
    axs5[1, 0].plot(s00_fit, "r", label="Sxx cum. - lin. fit")
    axs5[1, 0].plot(s11_fit, "b", label="Syy cum. - lin. fit")
    axs5[1, 0].set_title("Cumulative Scale", fontsize=fs + 1)
    axs5[2, 0].plot(tr_matr_cum_residual[:, 0, 0], "r", label="Sxx cum. - residual")
    axs5[2, 0].plot(tr_matr_cum_residual[:, 1, 1], "b", label="Syy cum. - residual")
    axs5[2, 0].set_title("Residual Cumulative Scale", fontsize=fs + 1)
    axs5[2, 0].set_xlabel("Frame", fontsize=fs + 1)

    # plot shears
    axs5[0, 1].plot(transformation_matrix[:, 0, 1], "r", label="Sxy fr.-to-fr.")
    axs5[0, 1].plot(transformation_matrix[:, 1, 0], "b", label="Syx fr.-to-fr.")
    axs5[0, 1].set_title("Frame-to-Frame Shear", fontsize=fs + 1)
    axs5[1, 1].plot(
        tr_matr_cum_orig[:, 0, 1], "r", linestyle="dotted", label="Sxy cum."
    )
    axs5[1, 1].plot(
        tr_matr_cum_orig[:, 1, 0], "b", linestyle="dotted", label="Syx cum."
    )
    axs5[1, 1].plot(s01_fit, "r", label="Sxy cum. - lin. fit")
    axs5[1, 1].plot(s10_fit, "b", label="Syx cum. - lin. fit")
    axs5[1, 1].set_title("Cumulative Shear", fontsize=fs + 1)
    axs5[2, 1].plot(tr_matr_cum_residual[:, 0, 1], "r", label="Sxy cum. - residual")
    axs5[2, 1].plot(tr_matr_cum_residual[:, 1, 0], "b", label="Syx cum. - residual")
    axs5[2, 1].set_title("Residual Cumulative Shear", fontsize=fs + 1)
    axs5[2, 1].set_xlabel("Frame", fontsize=fs + 1)

    # plot shifts
    axs5[0, 2].plot(Xshift_orig, "r", label="Tx fr.-to-fr.")
    axs5[0, 2].plot(Yshift_orig, "b", label="Ty fr.-to-fr.")
    axs5[0, 2].set_title("Frame-to-Frame Shift", fontsize=fs + 1)
    axs5[1, 2].plot(Xshift_cum, "r", linestyle="dotted", label="Tx cum.")
    axs5[1, 2].plot(Xfit, "r", label="Tx cum. - lin. fit")
    axs5[1, 2].plot(Yshift_cum, "b", linestyle="dotted", label="Ty cum.")
    axs5[1, 2].plot(Yfit, "b", label="Ty cum. - lin. fit")
    axs5[1, 2].set_title("Cumulative Shift", fontsize=fs + 1)
    axs5[2, 2].plot(tr_matr_cum_residual[:, 0, 2], "r", label="Tx cum. - residual")
    axs5[2, 2].plot(tr_matr_cum_residual[:, 1, 2], "b", label="Ty cum. - residual")
    axs5[2, 2].set_title("Residual Cumulative Shift", fontsize=fs + 1)
    axs5[2, 2].set_xlabel("Frame", fontsize=fs + 1)

    for ax in axs5.ravel():
        ax.grid(True)
        ax.legend()
    fig5.suptitle(xf_filename, fontsize=fs + 2)
    fig5.savefig(xf_filename + "_Transform_Summary.png", dpi=300)
    return tr_matr_cum
