"""
# Single-frame image processing functions
"""
import numpy as np
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

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
use_cp = False
if use_cp:
    import cupy as cp


EPS = np.finfo(float).eps


def _center_and_normalize_points_gs(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.
    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.
    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(D).
    If the points are all identical, the returned values will contain nan.

    Parameters:
    ----------
    points : (N, D) array
        The coordinates of the image points.
    Returns:
    -------
    matrix : (D+1, D+1) array
        The transformation matrix to obtain the new points.
    new_points : (N, D) array
        The transformed image points.
    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.
    """
    n, d = points.shape
    centroid = np.mean(points, axis=0)

    centered = points - centroid
    rms = np.sqrt(np.sum(centered**2) / n)

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if rms == 0:
        return np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan)

    norm_factor = np.sqrt(d) / rms

    part_matrix = norm_factor * np.concatenate(
        (np.eye(d), -centroid[:, np.newaxis]), axis=1
    )
    matrix = np.concatenate(
        (
            part_matrix,
            [
                [
                    0,
                ]
                * d
                + [1]
            ],
        ),
        axis=0,
    )

    points_h = np.row_stack([points.T, np.ones(n)])

    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points


def Single_Image_SNR(img, **kwargs):
    """
    Estimates SNR based on a single image.
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    Calculates SNR of a single image base on auto-correlation analysis after [1].

    Parameters
    ---------
    img : 2D array

    kwargs:
    edge_fraction : float
        fraction of the full autocetrrelation range used to calculate the "mean value" (default is 0.10)
    extrapolate_signal : boolean
        extrapolate to find signal autocorrelationb at 0-point (without noise). Default is True
    disp_res : boolean
        display results (plots) (default is True)
    save_res_png : boolean
        save the analysis output into a PNG file (default is True)
    res_fname : string
        filename for the result image ('SNR_result.png')
    img_label : string
        optional image label
    dpi : int
        dots-per-inch resolution for the output image

    Returns:
        xSNR, ySNR, rSNR : float, float, float
            SNR determined using the method in [1] along X- and Y- directions.
            If there is a direction with slow varying data - that direction provides more accurate SNR estimate
            Y-streaks in typical FIB-SEM data provide slow varying Y-component because streaks
            usually get increasingly worse with increasing Y.
            So for typical FIB-SEM data use ySNR

    [1] J. T. L. Thong et al, Single-image signal-to-noise ratio estimation. Scanning, 328–336 (2001).
    """
    edge_fraction = kwargs.get("edge_fraction", 0.10)
    extrapolate_signal = kwargs.get("extrapolate_signal", True)
    disp_res = kwargs.get("disp_res", True)
    nbins_disp = kwargs.get("nbins_disp", 256)
    thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    save_res_png = kwargs.get("save_res_png", True)
    res_fname = kwargs.get("res_fname", "SNR_results.png")
    img_label = kwargs.get("img_label", "Orig. Image")
    dpi = kwargs.get("dpi", 300)

    # first make image size even
    ysz, xsz = img.shape
    img = img[0 : ysz // 2 * 2, 0 : xsz // 2 * 2]
    ysz, xsz = img.shape

    xy_ratio = xsz / ysz
    data_FT = fftshift(fftn(ifftshift(img - img.mean())))
    data_FC = (np.multiply(data_FT, np.conj(data_FT))) / xsz / ysz
    data_ACR = np.abs(fftshift(fftn(ifftshift(data_FC))))
    data_ACR_peak = data_ACR[ysz // 2, xsz // 2]
    data_ACR_log = np.log(data_ACR)
    data_ACR = data_ACR / data_ACR_peak
    radial_ACR = radial_profile(data_ACR, [xsz // 2, ysz // 2])
    r_ACR = np.concatenate((radial_ACR[::-1], radial_ACR[1:-1]))

    # rsz = xsz
    rsz = len(r_ACR)
    rcr = np.linspace(-rsz // 2, rsz // 2 - 1, rsz)
    xcr = np.linspace(-xsz // 2, xsz // 2 - 1, xsz)
    ycr = np.linspace(-ysz // 2, ysz // 2 - 1, ysz)

    xl = xcr[xsz // 2 - 2 : xsz // 2]
    xacr_left = data_ACR[ysz // 2, (xsz // 2 - 2) : (xsz // 2)]
    xc = xcr[xsz // 2]
    xr = xcr[xsz // 2 + 1 : xsz // 2 + 3]
    xacr_right = data_ACR[ysz // 2, (xsz // 2 + 1) : (xsz // 2 + 3)]
    if extrapolate_signal:
        xNFacl = xacr_left[0] + (xc - xl[0]) / (xl[1] - xl[0]) * (
            xacr_left[1] - xacr_left[0]
        )
        xNFacr = xacr_right[0] + (xc - xr[0]) / (xr[1] - xr[0]) * (
            xacr_right[1] - xacr_right[0]
        )
    else:
        xNFacl = xacr_left[1]
        xNFacr = xacr_right[0]
    x_left = xcr[xsz // 2 - 2 : xsz // 2 + 1]
    xacr_left = np.concatenate((xacr_left, np.array([xNFacl])))
    x_right = xcr[xsz // 2 : xsz // 2 + 3]
    xacr_right = np.concatenate((np.array([xNFacr]), xacr_right))

    yl = ycr[ysz // 2 - 2 : ysz // 2]
    yacr_left = data_ACR[(ysz // 2 - 2) : (ysz // 2), xsz // 2]
    yc = ycr[ysz // 2]
    yr = ycr[ysz // 2 + 1 : ysz // 2 + 3]
    yacr_right = data_ACR[(ysz // 2 + 1) : (ysz // 2 + 3), xsz // 2]
    if extrapolate_signal:
        yNFacl = yacr_left[0] + (yc - yl[0]) / (yl[1] - yl[0]) * (
            yacr_left[1] - yacr_left[0]
        )
        yNFacr = yacr_right[0] + (yc - yr[0]) / (yr[1] - yr[0]) * (
            yacr_right[1] - yacr_right[0]
        )
    else:
        yNFacl = yacr_left[1]
        yNFacr = yacr_right[0]
    y_left = ycr[ysz // 2 - 2 : ysz // 2 + 1]
    yacr_left = np.concatenate((yacr_left, np.array([yNFacl])))
    y_right = ycr[ysz // 2 : ysz // 2 + 3]
    yacr_right = np.concatenate((np.array([yNFacr]), yacr_right))

    rl = rcr[rsz // 2 - 2 : rsz // 2]
    racr_left = r_ACR[(rsz // 2 - 2) : (rsz // 2)]
    rc = rcr[rsz // 2]
    rr = rcr[rsz // 2 + 1 : rsz // 2 + 3]
    racr_right = r_ACR[(rsz // 2 + 1) : (rsz // 2 + 3)]
    if extrapolate_signal:
        rNFacl = racr_left[0] + (rc - rl[0]) / (rl[1] - rl[0]) * (
            racr_left[1] - racr_left[0]
        )
        rNFacr = racr_right[0] + (rc - rr[0]) / (rr[1] - rr[0]) * (
            racr_right[1] - racr_right[0]
        )
    else:
        rNFacl = racr_left[1]
        rNFacr = racr_right[0]
    r_left = rcr[rsz // 2 - 2 : rsz // 2 + 1]
    racr_left = np.concatenate((racr_left, np.array([rNFacl])))
    r_right = rcr[rsz // 2 : rsz // 2 + 3]
    racr_right = np.concatenate((np.array([rNFacr]), racr_right))

    x_acr = data_ACR[ysz // 2, xsz // 2]
    x_noise_free_acr = xacr_right[0]
    xedge = int32(xsz * edge_fraction)
    x_mean_value = np.mean(data_ACR[ysz // 2, 0:xedge])
    xx_mean_value = np.linspace(-xsz // 2, (-xsz // 2 + xedge - 1), xedge)
    yedge = int32(ysz * edge_fraction)
    y_acr = data_ACR[ysz // 2, xsz // 2]
    y_noise_free_acr = yacr_right[0]
    y_mean_value = np.mean(data_ACR[0:yedge, xsz // 2])
    yy_mean_value = np.linspace(-ysz // 2, (-ysz // 2 + yedge - 1), yedge)
    redge = int32(rsz * edge_fraction)
    r_acr = data_ACR[ysz // 2, xsz // 2]
    r_noise_free_acr = racr_right[0]
    r_mean_value = np.mean(r_ACR[0:redge])
    rr_mean_value = np.linspace(-rsz // 2, (-rsz // 2 + redge - 1), redge)

    xSNR = (x_noise_free_acr - x_mean_value) / (x_acr - x_noise_free_acr)
    ySNR = (y_noise_free_acr - y_mean_value) / (y_acr - y_noise_free_acr)
    rSNR = (r_noise_free_acr - r_mean_value) / (r_acr - r_noise_free_acr)
    if disp_res:
        fs = 12

        if xy_ratio < 2.5:
            fig, axs = subplots(1, 4, figsize=(20, 5))
        else:
            fig = plt.figure(figsize=(20, 5))
            ax0 = fig.add_subplot(2, 2, 1)
            ax1 = fig.add_subplot(2, 2, 3)
            ax2 = fig.add_subplot(1, 4, 3)
            ax3 = fig.add_subplot(1, 4, 4)
            axs = [ax0, ax1, ax2, ax3]
        fig.subplots_adjust(
            left=0.03, bottom=0.06, right=0.99, top=0.92, wspace=0.25, hspace=0.10
        )

        range_disp = get_min_max_thresholds(
            img,
            thr_min=thresholds_disp[0],
            thr_max=thresholds_disp[1],
            nbins=nbins_disp,
            disp_res=False,
        )
        axs[0].imshow(img, cmap="Greys", vmin=range_disp[0], vmax=range_disp[1])
        axs[0].grid(True)
        axs[0].set_title(img_label)
        if save_res_png:
            axs[0].text(
                0, 1.1 + (xy_ratio - 1.0) / 20.0, res_fname, transform=axs[0].transAxes
            )
        axs[1].imshow(
            data_ACR_log, extent=[-xsz // 2 - 1, xsz // 2, -ysz // 2 - 1, ysz // 2]
        )
        axs[1].grid(True)
        axs[1].set_title("Autocorrelation (log scale)")

        axs[2].plot(xcr, data_ACR[ysz // 2, :], "r", linewidth=0.5, label="X")
        axs[2].plot(ycr, data_ACR[:, xsz // 2], "b", linewidth=0.5, label="Y")
        axs[2].plot(rcr, r_ACR, "g", linewidth=0.5, label="R")
        axs[2].plot(
            xx_mean_value,
            xx_mean_value * 0 + x_mean_value,
            "r--",
            linewidth=2.0,
            label="<X>={:.5f}".format(x_mean_value),
        )
        axs[2].plot(
            yy_mean_value,
            yy_mean_value * 0 + y_mean_value,
            "b--",
            linewidth=2.0,
            label="<Y>={:.5f}".format(y_mean_value),
        )
        axs[2].plot(
            rr_mean_value,
            rr_mean_value * 0 + r_mean_value,
            "g--",
            linewidth=2.0,
            label="<R>={:.5f}".format(r_mean_value),
        )
        axs[2].grid(True)
        axs[2].legend()
        axs[2].set_title("Normalized autocorr. cross-sections")
        axs[3].plot(xcr, data_ACR[ysz // 2, :], "rx", label="X data")
        axs[3].plot(ycr, data_ACR[:, xsz // 2], "bd", label="Y data")
        axs[3].plot(rcr, r_ACR, "g+", ms=10, label="R data")
        axs[3].plot(
            xcr[xsz // 2],
            data_ACR[ysz // 2, xsz // 2],
            "md",
            label="Peak: {:.4f}, {:.4f}".format(
                xcr[xsz // 2], data_ACR[ysz // 2, xsz // 2]
            ),
        )
        axs[3].plot(x_left, xacr_left, "r")
        axs[3].plot(
            x_right,
            xacr_right,
            "r",
            label="X extrap.: {:.4f}, {:.4f}".format(x_right[0], xacr_right[0]),
        )
        axs[3].plot(y_left, yacr_left, "b")
        axs[3].plot(
            y_right,
            yacr_right,
            "b",
            label="Y extrap.: {:.4f}, {:.4f}".format(y_right[0], yacr_right[0]),
        )
        axs[3].plot(r_left, racr_left, "g")
        axs[3].plot(
            r_right,
            racr_right,
            "g",
            label="R extrap.: {:.4f}, {:.4f}".format(r_right[0], racr_right[0]),
        )
        axs[3].text(
            0.03,
            0.92,
            "xSNR = {:.2f}".format(xSNR),
            color="r",
            transform=axs[3].transAxes,
            fontsize=fs,
        )
        axs[3].text(
            0.03,
            0.86,
            "ySNR = {:.2f}".format(ySNR),
            color="b",
            transform=axs[3].transAxes,
            fontsize=fs,
        )
        axs[3].text(
            0.03,
            0.80,
            "rSNR = {:.2f}".format(rSNR),
            color="g",
            transform=axs[3].transAxes,
            fontsize=fs,
        )
        axs[3].grid(True)
        axs[3].legend()
        axs[3].set_xlim(-5, 5)
        axs[3].set_title("Normalized autocorr. cross-sections")

        if save_res_png:
            # print('X:   ACR peak = {:.4f}, Noise-Free ACR Peak = {:.4f}, Squared Mean = {:.4f}'.format(x_acr, x_noise_free_acr, x_mean_value ))
            # print('xSNR = {:.2f}'.format(xSNR))
            # print('Y:   ACR peak = {:.4f}, Noise-Free ACR Peak = {:.4f}, Squared Mean = {:.4f}'.format(y_acr, y_noise_free_acr, y_mean_value ))
            # print('ySNR = {:.4f}'.format(ySNR))
            # print('R:   ACR peak = {:.4f}, Noise-Free ACR Peak = {:.4f}, Squared Mean = {:.4f}'.format(r_acr, r_noise_free_acr, r_mean_value ))
            # print('rSNR = {:.4f}'.format(rSNR))
            fig.savefig(res_fname, dpi=dpi)
            print("Saved the results into the file: ", res_fname)

    return xSNR, ySNR, rSNR


def Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs):
    """
    Analyses the noise statistics in the selected ROI's of the EM data
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

    Performs following:
    1.  Smooth the image by 2D convolution with a given kernel.
    2.  Determine "Noise" as difference between the original raw and smoothed data.
    3.  Build a histogram of Smoothed Image.
    4.  For each histogram bin of the Smoothed Image (Step 3), calculate the mean value and variance for the same pixels in the original image.
    5.  Plot the dependence of the noise variance vs. image intensity.
    6.  One of the parameters is a DarkCount. If it is not explicitly defined as input parameter, it will be set to 0.
    7.  The equation is determined for a line that passes through the points Intensity=DarkCount and Noise Variance = 0 and is a best fit for
        the [Mean Intensity, Noise Variance] points determined for each ROI (Step 1 above).
    8.  The data is plotted. Following values of SNR are defined from the slope of the line in Step 7:
        a.  PSNR (Peak SNR) = Intensity /sqrt(Noise Variance) at the intensity at the histogram peak determined in the Step 3.
        b.  MSNR (Mean SNR) = Mean Intensity /sqrt(Noise Variance)
        c.  DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance), where Max and Min Intensity are determined by
            corresponding cumulative threshold parameters, and Noise Variance is taken at the intensity in the middle of the range
            (Min Intensity + Max Intensity)/2.0.

    Parameters
    ----------
    img : 2D array
        original image
    Noise_ROIs : list of lists: [[left, right, top, bottom]]
        list of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the noise.
    Hist_ROI : list [left, right, top, bottom]
        coordinates (indices) of the boundaries of the image subset to evaluate the real data histogram.

    kwargs:
    DarkCount : float
        the value of the Intensity Data at 0.
    kernel : 2D float array
        a kernel to perform 2D smoothing convolution.
    nbins_disp : int
        (default 256) number of histogram bins for building the PDF and CDF to determine the data range for data display.
    thresholds_disp : list [thr_min_disp, thr_max_disp]
        (default [1e-3, 1e-3]) CDF threshold for determining the min and max data values for display.
    nbins_analysis : int
        (default 256) number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5.
    thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
        (default [2e-2, 2e-2]) CDF threshold for building the data histogram in Step 5.
    nbins_analysis : int
         (default 256) number of histogram bins for building the data histogram in Step 5.
    disp_res : boolean
        (default is False) - to plot/ display the results
    save_res_png : boolean
        save the analysis output into a PNG file (default is True)
    res_fname : string
        filename for the sesult image ('SNR_result.png')
    img_label : string
        optional image label
    Notes : string
        optional additional notes
    dpi : int

    Returns:
    mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR
        mean_vals and var_vals are the Mean Intensity and Noise Variance values for the Noise_ROIs (Step 1)
        NF_slope is the slope of the linear fit curve (Step 4)
        PSNR and DSNR are Peak and Dynamic SNR's (Step 6)
    """
    st = 1.0 / np.sqrt(2.0)
    def_kernel = np.array([[st, 1.0, st], [1.0, 1.0, 1.0], [st, 1.0, st]]).astype(float)
    def_kernel = def_kernel / def_kernel.sum()
    kernel = kwargs.get("kernel", def_kernel)
    DarkCount = kwargs.get("DarkCount", 0)
    nbins_disp = kwargs.get("nbins_disp", 256)
    thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    nbins_analysis = kwargs.get("nbins_analysis", 100)
    thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
    disp_res = kwargs.get("disp_res", True)
    save_res_png = kwargs.get("save_res_png", True)
    res_fname = kwargs.get("res_fname", "Noise_Analysis_ROIs.png")
    img_label = kwargs.get("img_label", "")
    Notes = kwargs.get("Notes", "")
    dpi = kwargs.get("dpi", 300)

    fs = 11
    img_filtered = convolve2d(img, kernel, mode="same")
    range_disp = get_min_max_thresholds(
        img_filtered,
        thr_min=thresholds_disp[0],
        thr_max=thresholds_disp[1],
        nbins=nbins_disp,
        disp_res=False,
    )

    xi, xa, yi, ya = Hist_ROI
    img_hist = img[yi:ya, xi:xa]
    img_hist_filtered = img_filtered[yi:ya, xi:xa]

    range_analysis = get_min_max_thresholds(
        img_hist_filtered,
        thr_min=thresholds_analysis[0],
        thr_max=thresholds_analysis[1],
        nbins=nbins_analysis,
        disp_res=False,
    )
    if disp_res:
        print(
            "The EM data range for noise analysis: {:.1f} - {:.1f},  DarkCount={:.1f}".format(
                range_analysis[0], range_analysis[1], DarkCount
            )
        )
    bins_analysis = np.linspace(range_analysis[0], range_analysis[1], nbins_analysis)

    xy_ratio = img.shape[1] / img.shape[0]
    xsz = 15
    ysz = xsz * 3.5 / xy_ratio

    n_ROIs = len(Noise_ROIs) + 1
    mean_vals = np.zeros(n_ROIs)
    var_vals = np.zeros(n_ROIs)
    mean_vals[0] = DarkCount

    if disp_res:
        fig = plt.figure(figsize=(xsz, ysz))
        axs0 = fig.add_subplot(3, 1, 1)
        axs1 = fig.add_subplot(3, 1, 2)
        axs2 = fig.add_subplot(3, 3, 7)
        axs3 = fig.add_subplot(3, 3, 8)
        axs4 = fig.add_subplot(3, 3, 9)
        fig.subplots_adjust(
            left=0.01, bottom=0.06, right=0.99, top=0.95, wspace=0.25, hspace=0.10
        )

        axs0.text(
            0.01,
            1.13,
            res_fname + ",   " + Notes,
            transform=axs0.transAxes,
            fontsize=fs - 3,
        )
        axs0.imshow(img, cmap="Greys", vmin=range_disp[0], vmax=range_disp[1])
        axs0.axis(False)
        axs0.set_title("Original Image: " + img_label, color="r", fontsize=fs + 1)
        Hist_patch = patches.Rectangle(
            (xi, yi),
            abs(xa - xi) - 2,
            abs(ya - yi) - 2,
            linewidth=1.0,
            edgecolor="white",
            facecolor="none",
        )
        axs1.add_patch(Hist_patch)

        axs2.imshow(
            img_hist_filtered, cmap="Greys", vmin=range_disp[0], vmax=range_disp[1]
        )
        axs2.axis(False)
        axs2.set_title("Smoothed ROI", fontsize=fs + 1)

    if disp_res:
        hist, bins, hist_patches = axs3.hist(
            img_hist_filtered.ravel(), range=range_disp, bins=nbins_disp
        )
    else:
        hist, bins = np.histogram(
            img_hist_filtered.ravel(), range=range_disp, bins=nbins_disp
        )

    bin_centers = np.array(bins[1:] - (bins[1] - bins[0]) / 2.0)
    hist_center_ind = np.argwhere(
        (bin_centers > range_analysis[0]) & (bin_centers < range_analysis[1])
    )
    hist_smooth = savgol_filter(np.array(hist), (nbins_disp // 10) * 2 + 1, 7)
    I_peak = bin_centers[hist_smooth.argmax()]
    I_mean = np.mean(img)
    C_peak = hist_smooth.max()

    if disp_res:
        axs3.plot(
            bin_centers[hist_center_ind],
            hist_smooth[hist_center_ind],
            color="grey",
            linestyle="dashed",
            linewidth=2,
        )
        Ipeak_lbl = "$I_{peak}$" + "={:.1f}".format(I_peak)
        axs3.plot(I_peak, C_peak, "rd", label=Ipeak_lbl)
        axs3.set_title("Histogram of the Smoothed ROI", fontsize=fs + 1)
        axs3.grid(True)
        axs3.set_xlabel("Smoothed ROI Image Intensity", fontsize=fs + 1)
        for hist_patch in np.array(hist_patches)[bin_centers < range_analysis[0]]:
            hist_patch.set_facecolor("lime")
        for hist_patch in np.array(hist_patches)[bin_centers > range_analysis[1]]:
            hist_patch.set_facecolor("red")
        ylim3 = array(axs3.get_ylim())
        I_min, I_max = range_analysis
        axs3.plot(
            [I_min, I_min],
            [ylim3[0] - 1000, ylim3[1]],
            color="lime",
            linestyle="dashed",
            label="$I_{min}$" + "={:.1f}".format(I_min),
        )
        axs3.plot(
            [I_max, I_max],
            [ylim3[0] - 1000, ylim3[1]],
            color="red",
            linestyle="dashed",
            label="$I_{max}$" + "={:.1f}".format(I_max),
        )
        axs3.set_ylim(ylim3)
        axs3.legend(loc="upper right", fontsize=fs + 1)
        axs1.imshow(img_filtered, cmap="Greys", vmin=range_disp[0], vmax=range_disp[1])
        axs1.axis(False)
        axs1.set_title("Smoothed Image")
        axs4.plot(DarkCount, 0, "d", color="black", label="Dark Count")

    for j, ROI in enumerate(tqdm(Noise_ROIs, desc="analyzing ROIs")):
        xi, xa, yi, ya = ROI
        img_ROI = img[yi:ya, xi:xa]
        img_ROI_filtered = img_filtered[yi:ya, xi:xa]

        imdiff = img_ROI - img_ROI_filtered
        x = np.mean(img_ROI)
        y = np.var(imdiff)
        mean_vals[j + 1] = x
        var_vals[j + 1] = y

        if disp_res:
            patch_col = get_cmap("gist_rainbow_r")((j) / (n_ROIs))
            rect_patch = patches.Rectangle(
                (xi, yi),
                abs(xa - xi) - 2,
                abs(ya - yi) - 2,
                linewidth=0.5,
                edgecolor=patch_col,
                facecolor="none",
            )
            axs0.add_patch(rect_patch)
            axs4.plot(x, y, "d", color=patch_col)  # , label='patch {:d}'.format(j))

    NF_slope = np.mean(var_vals[1:] / (mean_vals[1:] - DarkCount))
    mean_val_fit = np.array([mean_vals.min(), mean_vals.max()])
    var_val_fit = (mean_val_fit - DarkCount) * NF_slope
    VAR = (I_peak - DarkCount) * NF_slope
    VAR_at_mean = ((I_max + I_min) / 2.0 - DarkCount) * NF_slope
    PSNR = (I_peak - DarkCount) / np.sqrt(VAR)
    MSNR = (I_mean - DarkCount) / np.sqrt((I_mean - DarkCount) * NF_slope)
    DSNR = (I_max - I_min) / np.sqrt(VAR_at_mean)

    if disp_res:
        axs4.grid(True)
        axs4.set_title("Noise Distribution", fontsize=fs + 1)
        axs4.set_xlabel("ROI Image Intensity Mean", fontsize=fs + 1)
        axs4.set_ylabel("ROI Image Intensity Variance", fontsize=fs + 1)
        axs4.plot(
            mean_val_fit,
            var_val_fit,
            color="orange",
            label="Fit:  y = (x {:.1f}) * {:.2f}".format(DarkCount, NF_slope),
        )
        axs4.legend(loc="upper left", fontsize=fs + 2)
        ylim4 = array(axs4.get_ylim())
        V_min = (I_min - DarkCount) * NF_slope
        V_max = (I_max - DarkCount) * NF_slope
        V_peak = (I_peak - DarkCount) * NF_slope
        axs4.plot(
            [I_min, I_min],
            [ylim4[0], V_min],
            color="lime",
            linestyle="dashed",
            label="$I_{min}$" + "={:.1f}".format(I_min),
        )
        axs4.plot(
            [I_max, I_max],
            [ylim4[0], V_max],
            color="red",
            linestyle="dashed",
            label="$I_{max}$" + "={:.1f}".format(I_max),
        )
        axs4.plot(
            [I_peak, I_peak],
            [ylim4[0], V_peak],
            color="black",
            linestyle="dashed",
            label="$I_{peak}$" + "={:.1f}".format(I_peak),
        )
        axs4.set_ylim(ylim4)
        txt1 = "Peak Intensity:  {:.1f}".format(I_peak)
        axs4.text(0.05, 0.65, txt1, transform=axs4.transAxes, fontsize=fs + 1)
        txt2 = "Variance={:.1f}, STD={:.1f}".format(VAR, np.sqrt(VAR))
        axs4.text(0.05, 0.55, txt2, transform=axs4.transAxes, fontsize=fs + 1)
        txt3 = "PSNR = {:.2f}".format(PSNR)
        axs4.text(0.05, 0.45, txt3, transform=axs4.transAxes, fontsize=fs + 1)
        txt3 = "DSNR = {:.2f}".format(DSNR)
        axs4.text(0.05, 0.35, txt3, transform=axs4.transAxes, fontsize=fs + 1)
        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print("results saved into the file: " + res_fname)

    return mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR


def Single_Image_Noise_Statistics(img, **kwargs):
    """
    Analyses the noise statistics of the EM data image.
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

    Performs following:
    1. Smooth the image by 2D convolution with a given kernel.
    2. Determine "Noise" as difference between the original raw and smoothed data.
    3. Build a histogram of Smoothed Image.
    4. For each histogram bin of the Smoothed Image (Step 3), calculate the mean value and variance for the same pixels in the original image.
    5. Plot the dependence of the noise variance vs. image intensity.
    6. One of the parameters is a DarkCount. If it is not explicitly defined as input parameter,
        it will be set to 0
    7. The equation is determined for a line that passes through the point:
            Intensity=DarkCount and Noise Variance = 0
            and is a best fit for the [Mean Intensity, Noise Variance] points
            determined for each ROI (Step 1 above).
    8. The data is plotted. Two values of SNR are defined from the slope of the line in Step 4:
        PSNR (Peak SNR) = Intensity /sqrt(Noise Variance) at the intensity
            at the histogram peak determined in the Step 5.
        MSNR (Mean SNR) = Mean Intensity /sqrt(Noise Variance)
        DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance),
            where Max and Min Intensity are determined by corresponding cummulative
            threshold parameters, and Noise Variance is taken at the intensity
            in the middle of the range (Min Intensity + Max Intensity)/2.0

    Parameters
    ----------
        img : 2d array

        kwargs:
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        DarkCount : float
            the value of the Intensity Data at 0.
        kernel : 2D float array
            a kernel to perfrom 2D smoothing convolution.
        nbins_disp : int
            (default 256) number of histogram bins for building the PDF and CDF to determine the data range for data display.
        thresholds_disp : list [thr_min_disp, thr_max_disp]
            (default [1e-3, 1e-3]) CDF threshold for determining the min and max data values for display.
        nbins_analysis : int
            (default 256) number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5.
        thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
            (default [2e-2, 2e-2]) CDF threshold for building the data histogram in Step 5.
        nbins_analysis : int
             (default 256) number of histogram bins for building the data histogram in Step 5.
        disp_res : boolean
            (default is False) - to plot/ display the results
        save_res_png : boolean
            save the analysis output into a PNG file (default is True)
        res_fname : string
            filename for the result image ('Noise_Analysis.png')
        img_label : string
            optional image label
        Notes : string
            optional additional notes
        dpi : int

    Returns:
    mean_vals, var_vals, I0, PSNR, DSNR, popt, result
        mean_vals and var_vals are the Mean Intensity and Noise Variance values for Step5, I0 is zero intercept (should be close to DarkCount)
        PSNR and DSNR are Peak and Dynamic SNR's (Step 8)
    """
    st = 1.0 / np.sqrt(2.0)
    def_kernel = np.array([[st, 1.0, st], [1.0, 1.0, 1.0], [st, 1.0, st]]).astype(float)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    def_kernel = def_kernel / def_kernel.sum()
    kernel = kwargs.get("kernel", def_kernel)
    DarkCount = kwargs.get("DarkCount", 0)
    nbins_disp = kwargs.get("nbins_disp", 256)
    thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    nbins_analysis = kwargs.get("nbins_analysis", 100)
    thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
    disp_res = kwargs.get("disp_res", True)
    save_res_png = kwargs.get("save_res_png", True)
    res_fname = kwargs.get("res_fname", "Noise_Analysis.png")
    image_name = kwargs.get("image_name", "")
    Notes = kwargs.get("Notes", "")
    dpi = kwargs.get("dpi", 300)

    xi = 0
    yi = 0
    ysz, xsz = img.shape
    xa = xi + xsz
    ya = yi + ysz

    xi_eval = xi + evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = xa
    yi_eval = yi + evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ya

    img = img[yi_eval:ya_eval, xi_eval:xa_eval]
    img_filtered = convolve2d(img, kernel, mode="same")[1:-1, 1:-1]
    img = img[1:-1, 1:-1]

    range_disp = get_min_max_thresholds(
        img_filtered,
        thr_min=thresholds_disp[0],
        thr_max=thresholds_disp[1],
        nbins=nbins_disp,
        disp_res=disp_res,
    )
    if disp_res:
        print(
            "The EM data range for display:            {:.1f} - {:.1f}".format(
                range_disp[0], range_disp[1]
            )
        )
    range_analysis = get_min_max_thresholds(
        img_filtered,
        thr_min=thresholds_analysis[0],
        thr_max=thresholds_analysis[1],
        nbins=nbins_analysis,
        disp_res=disp_res,
    )
    if disp_res:
        print(
            "The EM data range for noise analysis:     {:.1f} - {:.1f}".format(
                range_analysis[0], range_analysis[1]
            )
        )
    bins_analysis = np.linspace(range_analysis[0], range_analysis[1], nbins_analysis)

    imdiff = img - img_filtered
    range_imdiff = get_min_max_thresholds(
        imdiff,
        thr_min=thresholds_disp[0],
        thr_max=thresholds_disp[1],
        nbins=nbins_disp,
        disp_res=disp_res,
    )

    xy_ratio = img.shape[1] / img.shape[0]
    xsz = 15
    ysz = xsz / 1.5 * xy_ratio
    if disp_res:
        fs = 11
        fig, axss = subplots(
            2, 3, figsize=(xsz, ysz), gridspec_kw={"height_ratios": [1, 1]}
        )
        fig.subplots_adjust(
            left=0.07, bottom=0.06, right=0.99, top=0.92, wspace=0.15, hspace=0.10
        )
        axs = axss.ravel()
        axs[0].text(
            -0.1,
            1.1,
            res_fname + ",       " + Notes,
            transform=axs[0].transAxes,
            fontsize=fs,
        )

        axs[0].imshow(img, cmap="Greys", vmin=range_disp[0], vmax=range_disp[1])
        axs[0].axis(False)
        axs[0].set_title("Original Image: " + image_name, color="r", fontsize=fs + 1)

        axs[1].imshow(
            img_filtered, cmap="Greys", vmin=range_disp[0], vmax=range_disp[1]
        )
        axs[1].axis(False)
        axs[1].set_title("Smoothed Image")
        Low_mask = img * 0.0 + 255.0
        High_mask = Low_mask.copy()
        Low_mask[img_filtered > range_analysis[0]] = np.nan
        axs[1].imshow(Low_mask, cmap="brg_r")
        High_mask[img_filtered < range_analysis[1]] = np.nan
        axs[1].imshow(High_mask, cmap="gist_rainbow")

        axs[2].imshow(imdiff, cmap="Greys", vmin=range_imdiff[0], vmax=range_imdiff[1])
        axs[2].axis(False)
        axs[2].set_title("Image Difference", fontsize=fs + 1)

    if disp_res:
        hist, bins, patches = axs[4].hist(
            img_filtered.ravel(), range=range_disp, bins=nbins_disp
        )
    else:
        hist, bins = np.histogram(
            img_hist_filtered.ravel(), range=range_disp, bins=nbins_disp
        )
    bin_centers = np.array(bins[1:] - (bins[1] - bins[0]) / 2.0)
    hist_center_ind = np.argwhere(
        (bin_centers > range_analysis[0]) & (bin_centers < range_analysis[1])
    )
    hist_smooth = savgol_filter(np.array(hist), (nbins_disp // 10) * 2 + 1, 7)
    I_peak = bin_centers[hist_smooth.argmax()]
    C_peak = hist_smooth.max()
    Ipeak_lbl = "$I_{peak}$" + "={:.1f}".format(I_peak)

    if disp_res:
        axs[4].plot(
            bin_centers[hist_center_ind],
            hist_smooth[hist_center_ind],
            color="grey",
            linestyle="dashed",
            linewidth=2,
        )
        axs[4].plot(I_peak, C_peak, "rd", label=Ipeak_lbl)
        axs[4].legend(loc="upper left", fontsize=fs + 1)
        axs[4].set_title("Histogram of the Smoothed Image", fontsize=fs + 1)
        axs[4].grid(True)
        axs[4].set_xlabel("Image Intensity", fontsize=fs + 1)
        for patch in np.array(patches)[bin_centers < range_analysis[0]]:
            patch.set_facecolor("lime")
        for patch in np.array(patches)[bin_centers > range_analysis[1]]:
            patch.set_facecolor("red")
        ylim4 = array(axs[4].get_ylim())
        axs[4].plot(
            [range_analysis[0], range_analysis[0]],
            [ylim4[0] - 1000, ylim4[1]],
            color="lime",
            linestyle="dashed",
            label="Ilow",
        )
        axs[4].plot(
            [range_analysis[1], range_analysis[1]],
            [ylim4[0] - 1000, ylim4[1]],
            color="red",
            linestyle="dashed",
            label="Ihigh",
        )
        axs[4].set_ylim(ylim4)
        txt1 = "Smoothing Kernel"
        axs[4].text(
            0.69,
            0.955,
            txt1,
            transform=axs[4].transAxes,
            backgroundcolor="white",
            fontsize=fs - 1,
        )
        txt2 = "{:.3f}  {:.3f}  {:.3f}".format(kernel[0, 0], kernel[0, 1], kernel[0, 2])
        axs[4].text(
            0.69,
            0.910,
            txt2,
            transform=axs[4].transAxes,
            backgroundcolor="white",
            fontsize=fs - 2,
        )
        txt3 = "{:.3f}  {:.3f}  {:.3f}".format(kernel[1, 0], kernel[1, 1], kernel[1, 2])
        axs[4].text(
            0.69,
            0.865,
            txt3,
            transform=axs[4].transAxes,
            backgroundcolor="white",
            fontsize=fs - 2,
        )
        txt3 = "{:.3f}  {:.3f}  {:.3f}".format(kernel[2, 0], kernel[2, 1], kernel[2, 2])
        axs[4].text(
            0.69,
            0.820,
            txt3,
            transform=axs[4].transAxes,
            backgroundcolor="white",
            fontsize=fs - 2,
        )

    if disp_res:
        hist, bins, patches = axs[5].hist(imdiff.ravel(), bins=nbins_disp)
        axs[5].grid(True)
        axs[5].set_title("Histogram of the Difference Map", fontsize=fs + 1)
        axs[5].set_xlabel("Image Difference", fontsize=fs + 1)
    else:
        hist, bins = np.histogram(imdiff.ravel(), bins=nbins_disp)

    ind_new = np.digitize(img_filtered, bins_analysis)
    result = np.array(
        [
            (np.mean(img_filtered[ind_new == j]), np.var(imdiff[ind_new == j]))
            for j in range(1, nbins_analysis)
        ]
    )
    non_nan_ind = np.argwhere(np.invert(np.isnan(result[:, 0])))
    mean_vals = np.squeeze(result[non_nan_ind, 0])
    var_vals = np.squeeze(result[non_nan_ind, 1])
    try:
        popt = np.polyfit(mean_vals, var_vals, 1)
        if disp_res:
            print("popt: ", popt)

        I_array = np.array((range_analysis[0], range_analysis[1], I_peak))
        if disp_res:
            print("I_array: ", I_array)
        Var_array = np.polyval(popt, I_array)
        Var_peak = Var_array[2]
    except:
        if disp_res:
            print("np.polyfit could not converge")
        popt = np.array([np.var(imdiff) / np.mean(img_filtered - DarkCount), 0])
        I_array = np.array((range_analysis[0], range_analysis[1], I_peak))
        Var_peak = np.var(imdiff)
    var_fit = np.polyval(popt, mean_vals)
    I0 = -popt[1] / popt[0]
    Slope_header = np.mean(var_vals / (mean_vals - DarkCount))
    var_fit_header = (mean_vals - DarkCount) * Slope_header
    if disp_res:
        axs[3].plot(mean_vals, var_vals, "r.", label="data")
        axs[3].plot(
            mean_vals,
            var_fit,
            "b",
            label="linear fit: {:.1f}*x + {:.1f}".format(popt[0], popt[1]),
        )
        axs[3].plot(
            mean_vals, var_fit_header, "magenta", label="linear fit with header offset"
        )
        axs[3].grid(True)
        axs[3].set_title("Noise Distribution", fontsize=fs + 1)
        axs[3].set_xlabel("Image Intensity Mean", fontsize=fs + 1)
        axs[3].set_ylabel("Image Intensity Variance", fontsize=fs + 1)
        ylim3 = array(axs[3].get_ylim())
        lbl_low = "$I_{low}$" + ", thr={:.1e}".format(thresholds_analysis[0])
        lbl_high = "$I_{high}$" + ", thr={:.1e}".format(thresholds_analysis[1])
        axs[3].plot(
            [range_analysis[0], range_analysis[0]],
            [ylim3[0] - 1000, ylim3[1]],
            color="lime",
            linestyle="dashed",
            label=lbl_low,
        )
        axs[3].plot(
            [range_analysis[1], range_analysis[1]],
            [ylim3[0] - 1000, ylim3[1]],
            color="red",
            linestyle="dashed",
            label=lbl_high,
        )
        axs[3].legend(loc="upper center", fontsize=fs + 1)
        axs[3].set_ylim(ylim3)

    PSNR = (I_peak - I0) / np.sqrt(Var_peak)
    PSNR_header = (I_peak - DarkCount) / np.sqrt(Var_peak)
    DSNR = (range_analysis[1] - range_analysis[0]) / np.sqrt(Var_peak)
    if disp_res:
        print("Var at peak: {:.1f}".format(Var_peak))
        print(
            "PSNR={:.2f}, PSNR_header={:.2f}, DSNR={:.2f}".format(
                PSNR, PSNR_header, DSNR
            )
        )

        txt1 = "Histogram Peak:  " + Ipeak_lbl
        axs[3].text(0.25, 0.27, txt1, transform=axs[3].transAxes, fontsize=fs + 1)
        txt2 = (
            "DSNR = "
            + "$(I_{high}$"
            + "$ - I_{low})$"
            + "/"
            + "$σ_{peak}$"
            + " = {:.2f}".format(DSNR)
        )
        axs[3].text(0.25, 0.22, txt2, transform=axs[3].transAxes, fontsize=fs + 1)

        txt3 = "Zero Intercept:    " + "$I_{0}$" + "={:.1f}".format(I0)
        axs[3].text(
            0.25, 0.17, txt3, transform=axs[3].transAxes, color="blue", fontsize=fs + 1
        )
        txt4 = (
            "PSNR = "
            + "$(I_{peak}$"
            + "$ - I_{0})$"
            + "/"
            + "$σ_{peak}$"
            + " = {:.2f}".format(PSNR)
        )
        axs[3].text(
            0.25, 0.12, txt4, transform=axs[3].transAxes, color="blue", fontsize=fs + 1
        )

        txt5 = "Header Zero Int.:    " + "$I_{0}$" + "={:.1f}".format(DarkCount)
        axs[3].text(
            0.25,
            0.07,
            txt5,
            transform=axs[3].transAxes,
            color="magenta",
            fontsize=fs + 1,
        )
        txt6 = (
            "PSNR = "
            + "$(I_{peak}$"
            + "$ - I_{0})$"
            + "/"
            + "$σ_{peak}$"
            + " = {:.2f}".format(PSNR_header)
        )
        axs[3].text(
            0.25,
            0.02,
            txt6,
            transform=axs[3].transAxes,
            color="magenta",
            fontsize=fs + 1,
        )

        if save_res_png:
            fig.savefig(res_fname, dpi=300)
            print("results saved into the file: " + res_fname)
    return mean_vals, var_vals, I0, PSNR, DSNR, popt, result


def Perform_2D_fit(img, estimator, **kwargs):
    """
    Bin the image and then perform 2D polynomial (currently only 2D parabolic) fit on the binned image.
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

    Parameters
    ----------
    img : 2D array
        original image
    estimator : RANSACRegressor(),
                LinearRegression(),
                TheilSenRegressor(),
                HuberRegressor()
    kwargs:
    image_name : str
        Image name (for display purposes)
    bins : int
        binsize for image binning. If not provided, bins=10
    Analysis_ROIs : list of lists: [[left, right, top, bottom]]
        list of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the parabolic fit.
    calc_corr : boolean
        If True - the full image correction is calculated
    ignore_Y  : boolean
        If True - the parabolic fit to only X is perfromed
    disp_res : boolean
        (default is False) - to plot/ display the results
    save_res_png : boolean
        save the analysis output into a PNG file (default is False)
    res_fname : string
        filename for the result image ('Image_Flattening.png')
    Xsect : int
        X - coordinate for Y-crossection
    Ysect : int
        Y - coordinate for X-crossection
    dpi : int

    Returns:
    intercept, coefs, mse, img_correction_array
    """
    image_name = kwargs.get("image_name", "RawImageA")
    ysz, xsz = img.shape
    calc_corr = kwargs.get("calc_corr", False)
    ignore_Y = kwargs.get("ignore_Y", False)
    Xsect = kwargs.get("Xsect", xsz // 2)
    Ysect = kwargs.get("Ysect", ysz // 2)
    disp_res = kwargs.get("disp_res", True)
    bins = kwargs.get("bins", 10)  # bins = 10
    Analysis_ROIs = kwargs.get("Analysis_ROIs", [])
    save_res_png = kwargs.get("save_res_png", False)
    res_fname = kwargs.get("res_fname", "Image_Flattening.png")
    dpi = kwargs.get("dpi", 300)

    img_binned = (
        img[0 : ysz // bins * bins, 0 : xsz // bins * bins]
        .astype(float)
        .reshape(ysz // bins, bins, xsz // bins, bins)
        .sum(3)
        .sum(1)
        / bins
        / bins
    )
    if len(Analysis_ROIs) > 0:
        Analysis_ROIs_binned = [
            [ind // bins for ind in Analysis_ROI] for Analysis_ROI in Analysis_ROIs
        ]
    else:
        Analysis_ROIs_binned = []
    vmin, vmax = get_min_max_thresholds(img_binned, disp_res=False)
    yszb, xszb = img_binned.shape
    yb, xb = np.indices(img_binned.shape)
    if len(Analysis_ROIs_binned) > 0:
        img_1D_list = []
        xb_1d_list = []
        yb_1d_list = []
        for Analysis_ROI_binned in Analysis_ROIs_binned:
            # Analysis_ROI : list of [left, right, top, bottom]
            img_1D_list = (
                img_1D_list
                + img_binned[
                    Analysis_ROI_binned[2] : Analysis_ROI_binned[3],
                    Analysis_ROI_binned[0] : Analysis_ROI_binned[1],
                ]
                .ravel()
                .tolist()
            )
            xb_1d_list = (
                xb_1d_list
                + xb[
                    Analysis_ROI_binned[2] : Analysis_ROI_binned[3],
                    Analysis_ROI_binned[0] : Analysis_ROI_binned[1],
                ]
                .ravel()
                .tolist()
            )
            yb_1d_list = (
                yb_1d_list
                + yb[
                    Analysis_ROI_binned[2] : Analysis_ROI_binned[3],
                    Analysis_ROI_binned[0] : Analysis_ROI_binned[1],
                ]
                .ravel()
                .tolist()
            )
        img_1D = np.array(img_1D_list)
        xb_1d = np.array(xb_1d_list)
        yb_1d = np.array(yb_1d_list)
    else:
        img_1D = img_binned.ravel()
        xb_1d = xb.ravel()
        yb_1d = yb.ravel()

    img_binned_1D = img_binned.ravel()
    X_binned = np.vstack((xb.ravel(), yb.ravel())).T
    X = np.vstack((xb_1d, yb_1d)).T

    ysz, xsz = img.shape
    yf, xf = np.indices(img.shape)
    xf_1d = xf.ravel() / bins
    yf_1d = yf.ravel() / bins
    Xf = np.vstack((xf_1d, yf_1d)).T

    model = make_pipeline(PolynomialFeatures(2), estimator)

    if ignore_Y:
        ymean = np.mean(yb_1d)
        yb_1d_flat = yb_1d * 0.0 + ymean
        X_yflat = np.vstack((xb_1d, yb_1d_flat)).T
        model.fit(X_yflat, img_1D)

    else:
        model.fit(X, img_1D)

    model = make_pipeline(PolynomialFeatures(2), estimator)
    model.fit(X, img_1D)
    if hasattr(model[-1], "estimator_"):
        if ignore_Y:
            model[-1].estimator_.coef_[0] = (
                model[-1].estimator_.coef_[0]
                + model[-1].estimator_.coef_[2] * ymean
                + model[-1].estimator_.coef_[5] * ymean * ymean
            )
            model[-1].estimator_.coef_[1] = (
                model[-1].estimator_.coef_[1] + model[-1].estimator_.coef_[4] * ymean
            )
            model[-1].estimator_.coef_[2] = 0.0
            model[-1].estimator_.coef_[4] = 0.0
            model[-1].estimator_.coef_[5] = 0.0
        coefs = model[-1].estimator_.coef_
        intercept = model[-1].estimator_.intercept_
    else:
        if ignore_Y:
            model[-1].coef_[0] = (
                model[-1].coef_[0]
                + model[-1].coef_[2] * ymean
                + model[-1].coef_[5] * ymean * ymean
            )
            model[-1].coef_[1] = model[-1].coef_[1] + model[-1].coef_[4] * ymean
            model[-1].coef_[2] = 0.0
            model[-1].coef_[4] = 0.0
            model[-1].coef_[5] = 0.0
        coefs = model[-1].coef_
        intercept = model[-1].intercept_
    img_fit_1d = model.predict(X)
    scr = model.score(X, img_1D)
    mse = mean_squared_error(img_fit_1d, img_1D)
    img_fit = model.predict(X_binned).reshape(yszb, xszb)
    if calc_corr:
        img_correction_array = np.mean(img_fit_1d) / model.predict(Xf).reshape(ysz, xsz)
    else:
        img_correction_array = img * 0.0

    if disp_res:
        print("Estimator coefficients ( 1  x  y  x^2  x*y  y^2): ", coefs)
        print("Estimator intercept: ", intercept)

        fig, axs = subplots(2, 2, figsize=(12, 8))
        axs[0, 0].imshow(img_binned, cmap="Greys", vmin=vmin, vmax=vmax)
        axs[0, 0].grid(True)
        axs[0, 0].plot([Xsect // bins, Xsect // bins], [0, yszb], "lime", linewidth=0.5)
        axs[0, 0].plot([0, xszb], [Ysect // bins, Ysect // bins], "cyan", linewidth=0.5)
        if len(Analysis_ROIs_binned) > 0:
            col_ROIs = "yellow"
            axs[0, 0].text(
                0.3,
                0.9,
                "with Analysis ROIs",
                color=col_ROIs,
                transform=axs[0, 0].transAxes,
            )
            for Analysis_ROI_binned in Analysis_ROIs_binned:
                # Analysis_ROI : list of [left, right, top, bottom]
                xi, xa, yi, ya = Analysis_ROI_binned
                ROI_patch = patches.Rectangle(
                    (xi, yi),
                    abs(xa - xi) - 2,
                    abs(ya - yi) - 2,
                    linewidth=0.75,
                    edgecolor=col_ROIs,
                    facecolor="none",
                )
                axs[0, 0].add_patch(ROI_patch)

        axs[0, 0].set_xlim((0, xszb))
        axs[0, 0].set_ylim((yszb, 0))
        axs[0, 0].set_title("{:d}-x Binned Raw:".format(bins) + image_name)

        axs[0, 1].imshow(img_fit, cmap="Greys", vmin=vmin, vmax=vmax)
        axs[0, 1].grid(True)
        axs[0, 1].plot([Xsect // bins, Xsect // bins], [0, yszb], "lime", linewidth=0.5)
        axs[0, 1].plot([0, xszb], [Ysect // bins, Ysect // bins], "cyan", linewidth=0.5)
        axs[0, 1].set_xlim((0, xszb))
        axs[0, 1].set_ylim((yszb, 0))
        axs[0, 1].set_title("{:d}-x Binned Fit: ".format(bins) + image_name)

        axs[1, 0].plot(img[Ysect, :], "b", label=image_name, linewidth=0.5)
        axs[1, 0].plot(
            xb[0, :] * bins,
            img_binned[Ysect // bins, :],
            "cyan",
            label="Binned " + image_name,
        )
        axs[1, 0].plot(
            xb[0, :] * bins,
            img_fit[Ysect // bins, :],
            "yellow",
            linewidth=4,
            linestyle="--",
            label="Fit: " + image_name,
        )
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_xlabel("X-coordinate")

        axs[1, 1].plot(img[:, Xsect], "g", label=image_name, linewidth=0.5)
        axs[1, 1].plot(
            yb[:, 0] * bins,
            img_binned[:, Xsect // bins],
            "lime",
            label="Binned " + image_name,
        )
        axs[1, 1].plot(
            yb[:, 0] * bins,
            img_fit[:, Xsect // bins],
            "yellow",
            linewidth=4,
            linestyle="--",
            label="Fit: " + image_name,
        )
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_xlabel("Y-coordinate")
        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print("results saved into the file: " + res_fname)
    return intercept, coefs, mse, img_correction_array
