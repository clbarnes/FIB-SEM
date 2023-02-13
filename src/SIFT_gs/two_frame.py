"""
# Two-frame image processing functions
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


def mutual_information_2d(x, y, sigma=1, bin=256, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    mi: float
        the computed similarity measure
    """
    bins = (bin, bin)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    # ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
    #                             output=jh)
    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = (
            (np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
            / np.sum(jh * np.log(jh))
        ) - 1
    else:
        mi = np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2))
    return mi


def mutual_information_2d_cp(x, y, sigma=1, bin=256, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram using CUPY package.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    mi: float
        the computed similarity measure
    """
    bins = (bin, bin)

    jhf = cp.histogram2d(x, y, bins=bins)

    # smooth the jh with a gaussian filter of given sigma
    # ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
    #                             output=jh)
    # compute marginal histograms
    jh = jhf[0] + EPS
    sh = cp.sum(jh)
    jh = jh / sh
    s1 = cp.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = cp.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = (
            (cp.sum(s1 * cp.log(s1)) + cp.sum(s2 * cp.log(s2)))
            / cp.sum(jh * cp.log(jh))
        ) - 1
    else:
        mi = cp.sum(jh * cp.log(jh)) - np.sum(s1 * cp.log(s1)) - cp.sum(s2 * cp.log(s2))
    return mi


def Two_Image_NCC_SNR(img1, img2, **kwargs):
    """
     Estimates normalized cross-correlation and SNR of two images.
     ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

     Calculates SNR from cross-correlation of two images after [1, 2, 3].

     Parameters
     ---------
     img1 : 2D array
     img2 : 2D array

     kwargs:
     zero_mean: boolean
         if True, cross-correlation is zero-mean

     Returns:
         NCC, SNR : float, float
             NCC - normalized cross-correlation coefficient
             SNR - Signal-to-Noise ratio based on NCC

    [1] J. Frank, L. AI-Ali, Signal-to-noise ratio of electron micrographs obtained by cross correlation. Nature 256, 4 (1975).
    [2] J. Frank, in: Computer Processing of Electron Microscopic Images. Ed. P.W. Hawkes (Springer, Berlin, 1980).
    [3] M. Radermacher, T. Ruiz, On cross-correlations, averages and noise in electron microscopy. Acta Crystallogr. Sect. F Struct. Biol. Commun. 75, 12–18 (2019).

    """
    zero_mean = kwargs.get("zero_mean", True)

    if img1.shape == img2.shape:
        ysz, xsz = img1.shape
        if zero_mean:
            img1 = img1 - img1.mean()
            img2 = img2 - img2.mean()
        xy = np.sum((img1.ravel() * img2.ravel())) / (xsz * ysz)
        xx = np.sum((img1.ravel() * img1.ravel())) / (xsz * ysz)
        yy = np.sum((img2.ravel() * img2.ravel())) / (xsz * ysz)
        NCC = xy / np.sqrt(xx * yy)
        SNR = NCC / (1 - NCC)

    else:
        print("img1 and img2 shapes must be equal")
        NCC = 0.0
        SNR = 0.0

    return NCC, SNR


def Two_Image_FSC(img1, img2, **kwargs):
    """
    Perform Fourier Shell Correlation to determine the image resolution, after [1]. ©G.Shtengel, 10/2019. gleb.shtengel@gmail.com
    FSC is determined from radially averaged foirier cross-correlation (with optional selection of range of angles for radial averaging).

    Parameters
    ---------
    img1 : 2D array
    img2 : 2D array

    kwargs:
    SNRt : float
        SNR threshold for determining the resolution bandwidth
    astart : float
        Start angle for radial averaging. Default is 0
    astop : float
        Stop angle for radial averaging. Default is 90
    symm : int
        Symmetry factor (how many times Start and stop angle intervalks are repeated within 360 deg). Default is 4.
    disp_res : boolean
        display results (plots) (default is False)
    ax : axis object (matplotlip)
        to export the plot
    save_res_png : boolean
        save results into PNG file (default is False)
    res_fname : string
        filename for the result image ('SNR_result.png')
    img_labels : [string, string]
        optional image labels
    dpi : int
        dots-per-inch resolution for the output image
    pixel : float
        optional pixel size in nm. If not provided, will be ignored.
        if provided, second axis will be added on top with inverse pixels
    xrange : [float, float]
        range of x axis in FSC plot in inverse pixels
        if not provided [0, 0.5] range will be used

    Returns FSC_sp_frequencies, FSC_data, x2, T, FSC_bw
        FSC_sp_frequencies : float array
            Spatial Frequency (/Nyquist) - for FSC plot
        FSC_data: float array
        x2 : float array
            Spatial Frequency (/Nyquist) - for threshold line plot
        T : float array
            threshold line plot
        FSC_bw : float
            the value of FSC determined as an intersection of smoothed data threshold

    [1]. M. van Heela, and M. Schatzb, "Fourier shell correlation threshold criteria," Journal of Structural Biology 151, 250-262 (2005)
    """
    SNRt = kwargs.get("SNRt", 0.1)
    astart = kwargs.get("astart", 0.0)
    astop = kwargs.get("astop", 90.0)
    symm = kwargs.get("symm", 4)
    disp_res = kwargs.get("disp_res", False)
    ax = kwargs.get("ax", "")
    save_res_png = kwargs.get("save_res_png", False)
    res_fname = kwargs.get("res_fname", "FSC_results.png")
    img_labels = kwargs.get("img_labels", ["Image 1", "Image 2"])
    dpi = kwargs.get("dpi", 300)
    pixel = kwargs.get("pixel", 0.0)
    xrange = kwargs.get("xrange", [0, 0.5])

    # Check whether the inputs dimensions match and the images are square
    if disp_res:
        if np.shape(img1) != np.shape(img2):
            print("input images must have the same dimensions")
        if np.shape(img1)[0] != np.shape(img1)[1]:
            print("input images must be squares")
    I1 = fftshift(
        fftn(ifftshift(img1))
    )  # I1 and I2 store the FFT of the images to be used in the calcuation for the FSC
    I2 = fftshift(fftn(ifftshift(img2)))

    C_imre = np.multiply(I1, np.conj(I2))
    C12_ar = abs(np.multiply((I1 + I2), np.conj(I1 + I2)))
    y0, x0 = argmax2d(C12_ar)
    C1 = radial_profile_select_angles(
        abs(np.multiply(I1, np.conj(I1))), [x0, y0], astart, astop, symm
    )
    C2 = radial_profile_select_angles(
        abs(np.multiply(I2, np.conj(I2))), [x0, y0], astart, astop, symm
    )
    C = radial_profile_select_angles(
        np.real(C_imre), [x0, y0], astart, astop, symm
    ) + 1j * radial_profile_select_angles(
        np.imag(C_imre), [x0, y0], astart, astop, symm
    )

    FSC_data = abs(C) / np.sqrt(abs(np.multiply(C1, C2)))
    """
    T is the SNR threshold calculated accoring to the input SNRt, if nothing is given
    a default value of 0.1 is used.

    x2 contains the normalized spatial frequencies
    """
    r = np.arange(1 + np.shape(img1)[0])
    n = 2 * np.pi * r
    n[0] = 1
    eps = np.finfo(float).eps
    t1 = np.divide(np.ones(np.shape(n)), n + eps)
    t2 = SNRt + 2 * np.sqrt(SNRt) * t1 + np.divide(np.ones(np.shape(n)), np.sqrt(n))
    t3 = SNRt + 2 * np.sqrt(SNRt) * t1 + 1
    T = np.divide(t2, t3)
    # FSC_sp_frequencies = np.arange(np.shape(C)[0])/(np.shape(img1)[0]/sqrt(2.0))
    # x2 = r/(np.shape(img1)[0]/sqrt(2.0))
    FSC_sp_frequencies = np.arange(np.shape(C)[0]) / (np.shape(img1)[0])
    x2 = r / (np.shape(img1)[0])
    FSC_data_smooth = smooth(FSC_data, 20)
    FSC_bw = find_BW(FSC_sp_frequencies, FSC_data_smooth, SNRt)
    """
    If the disp_res input is set to True, an output plot is generated.
    """
    if disp_res:
        if ax == "":
            fig = plt.figure(figsize=(8, 12))
            axs0 = fig.add_subplot(3, 2, 1)
            axs1 = fig.add_subplot(3, 2, 2)
            axs2 = fig.add_subplot(3, 2, 3)
            axs3 = fig.add_subplot(3, 2, 4)
            ax = fig.add_subplot(3, 1, 3)
            fig.subplots_adjust(
                left=0.01, bottom=0.06, right=0.99, top=0.975, wspace=0.25, hspace=0.10
            )
            vmin1, vmax1 = get_min_max_thresholds(img1, disp_res=False)
            vmin2, vmax2 = get_min_max_thresholds(img2, disp_res=False)
            axs0.imshow(img1, cmap="Greys", vmin=vmin1, vmax=vmax1)
            axs1.imshow(img2, cmap="Greys", vmin=vmin2, vmax=vmax2)
            x = np.linspace(0, 1.41, 500)
            axs2.set_xlim(-1, 1)
            axs2.set_ylim(-1, 1)
            axs2.imshow(np.log(abs(I1)), extent=[-1, 1, -1, 1], cmap="Greys_r")
            axs3.set_xlim(-1, 1)
            axs3.set_ylim(-1, 1)
            axs3.imshow(np.log(abs(I2)), extent=[-1, 1, -1, 1], cmap="Greys_r")
            for i in np.arange(symm):
                ai = np.radians(astart + 360.0 / symm * i)
                aa = np.radians(astop + 360.0 / symm * i)
                axs2.plot(x * np.cos(ai), x * np.sin(ai), color="orange", linewidth=0.5)
                axs3.plot(x * np.cos(ai), x * np.sin(ai), color="orange", linewidth=0.5)
                axs2.plot(x * np.cos(aa), x * np.sin(aa), color="orange", linewidth=0.5)
                axs3.plot(x * np.cos(aa), x * np.sin(aa), color="orange", linewidth=0.5)
            ttls = img_labels.copy()
            ttls.append("FFT of " + img_labels[0])
            ttls.append("FFT of " + img_labels[1])
            for axi, ttl in zip([axs0, axs1, axs2, axs3], ttls):
                axi.grid(False)
                axi.axis(False)
                axi.set_title(ttl)

    if disp_res or ax != "":
        ax.plot(FSC_sp_frequencies, FSC_data, label="FSC data", color="r")
        ax.plot(
            FSC_sp_frequencies, FSC_data_smooth, label="FSC data smoothed", color="b"
        )
        ax.plot(
            x2,
            x2 * 0.0 + SNRt,
            "--",
            label="Threshold SNR = {:.3f}".format(SNRt),
            color="m",
        )
        if pixel > 1e-6:
            label = "FSC BW = {:.3f} inv.pix., or {:.2f} nm".format(
                FSC_bw, pixel / FSC_bw
            )
        else:
            label = "FSC BW = {:.3f}".format(FSC_bw)
        ax.plot(
            np.array((FSC_bw, FSC_bw)),
            np.array((0.0, 1.0)),
            "--",
            label=label,
            color="g",
        )
        ax.set_xlim(xrange)
        ax.legend()
        ax.set_xlabel("Spatial Frequency (inverse pixels)")
        ax.set_ylabel("FSC Magnitude")
        ax.grid(True)
        if pixel > 1e-6:

            def forward(x):
                return x / pixel

            def inverse(x):
                return x * pixel

            secax = ax.secondary_xaxis("top", functions=(forward, inverse))
            secax.set_xlabel("Spatial Frequency ($nm^{-1}$)")

    if disp_res:
        ax.set_position([0.1, 0.05, 0.85, 0.28])
        print("FSC BW = {:.5f}".format(FSC_bw))
        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print("Saved the results into the file: ", res_fname)
    return (FSC_sp_frequencies, FSC_data, x2, T, FSC_bw)


def Two_Image_Analysis(params):
    """
    Analyzes the registration  quality between two frames (for DASK registration analysis)

    Parameterss:
    params : list of params
        params = [frame1_filename, frame2_filename, eval_bounds, eval_metrics]
        eval_bounds = [xi,  xa, yi, ya]
        eval_metrics = ['NSAD', 'NCC', 'NMI', 'FSC']

    Returns
        results : list of results
    """
    frame1_filename, frame2_filename, eval_bounds, eval_metrics = params
    xi_eval, xa_eval, yi_eval, ya_eval = eval_bounds

    I1 = tiff.imread(frame1_filename)
    I1c = I1[yi_eval:ya_eval, xi_eval:xa_eval]
    I2 = tiff.imread(frame2_filename)
    I2c = I2[yi_eval:ya_eval, xi_eval:xa_eval]
    fr_mean = abs(I1c / 2.0 + I2c / 2.0)
    dy, dx = shape(I2c)

    results = []
    for metric in eval_metrics:
        if metric == "NSAD":
            results.append(mean(abs(I1c - I2c)) / (np.mean(fr_mean) - np.amin(fr_mean)))
        if metric == "NCC":
            results.append(Two_Image_NCC_SNR(I1c, I2c)[0])
        if metric == "NMI":
            results.append(
                mutual_information_2d(
                    I1c.ravel(), I2c.ravel(), sigma=1.0, bin=2048, normalized=True
                )
            )
        if metric == "FSC":
            # SNRt is SNR threshold for determining the resolution bandwidth
            # force square images for FSC
            if dx != dy:
                d = min((dx // 2, dy // 2))
                results.append(
                    Two_Image_FSC(
                        I1c[dy // 2 - d : dy // 2 + d, dx // 2 - d : dx // 2 + d],
                        I2c[dy // 2 - d : dy // 2 + d, dx // 2 - d : dx // 2 + d],
                        SNRt=0.143,
                        disp_res=False,
                    )[4]
                )
            else:
                results.append(Two_Image_FSC(I1c, I2c, SNRt=0.143, disp_res=False)[4])

    return results
