"""
# General help functions
"""
import numpy as np
from pylab import *

from struct import *

# from tqdm import tqdm_notebook as tqdm

from scipy.signal import savgol_filter

import cv2

print("Open CV version: ", cv2.__version__)

use_cp = False
if use_cp:
    import cupy as cp


def get_spread(data, window=501, porder=3):
    """
    Calculates spread - standard deviation of the (signal - Sav-Gol smoothed signal).
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com

    Parameters:
    data : 1D array
    window : int
        aperture (number of points) for Sav-Gol filter)
    porder : int
        polynomial order for Sav-Gol filter

    Returns:
        data_spread : float

    """

    try:
        # sm_data = savgol_filter(data.astype(double), window, porder)
        sm_data = savgol_filter(data.astype(double), window, porder, mode="mirror")
        data_spread = np.std(data - sm_data)
    except:
        print("spread error")
        data_spread = 0.0
    return data_spread


def get_min_max_thresholds(image, **kwargs):
    """
    Determines the data range (min and max) with given fractional thresholds for cumulative distribution.
    ©G.Shtengel 11/2022 gleb.shtengel@gmail.com

    Calculates the histogram of pixel intensities of the image with number of bins determined by parameter nbins (default = 256)
    and normalizes it to get the probability distribution function (PDF), from which a cumulative distribution function (CDF) is calculated.
    Then given the thr_min, thr_max parameters, the minimum and maximum values of the data range are found
    by determining the intensities at which CDF= thr_min and (1- thr_max), respectively

    Parameters:
    ----------
    image : 2D array
        Image to be analyzed

    kwargs:
     ----------
    thr_min : float
        lower CDF threshold for determining the minimum data value. Default is 1.0e-3
    thr_max : float
        upper CDF threshold for determining the maximum data value. Default is 1.0e-3
    nbins : int
        number of histogram bins for building the PDF and CDF
    log  : bolean
        If True, the histogram will have log scale. Default is false
    disp_res : bolean
        If True display the results. Default is True.
    save_res : boolean
        If True the image will be saved. Default is False.
    dpi : int
        Default is 300
    save_filename : string
        the name of the image to perform this operations (defaulut is 'min_max_thresholds.png').

    Returns
    (dmin, dmax) : float array
    """
    thr_min = kwargs.get("thr_min", 1.0e-3)
    thr_max = kwargs.get("thr_max", 1.0e-3)
    nbins = kwargs.get("nbins", 256)
    disp_res = kwargs.get("disp_res", True)
    log = kwargs.get("log", False)
    save_res = kwargs.get("save_res", False)
    dpi = kwargs.get("dpi", 300)
    save_filename = kwargs.get("save_filename", "min_max_thresholds.png")

    if disp_res:
        fsz = 11
        fig, axs = subplots(2, 1, figsize=(6, 8))
        hist, bins, patches = axs[0].hist(image.ravel(), bins=nbins, log=log)
    else:
        hist, bins = np.histogram(image.ravel(), bins=nbins)
    pdf = hist / np.prod(image.shape)
    cdf = np.cumsum(pdf)
    data_max = bins[argmin(abs(cdf - (1.0 - thr_max)))]
    data_min = bins[argmin(abs(cdf - thr_min))]

    if disp_res:
        xCDF = bins[0:-1] + (bins[1] - bins[0]) / 2.0
        xthr = [xCDF[0], xCDF[-1]]
        ythr_min = [thr_min, thr_min]
        ythr_max = [1 - thr_max, 1 - thr_max]
        axs[1].plot(xCDF, cdf, label="CDF")
        axs[1].plot(xthr, ythr_min, "r", label="thr_min={:.5f}".format(thr_min))
        axs[1].plot(
            xthr, ythr_max, "g", label="1.0 - thr_max = {:.5f}".format(1 - thr_max)
        )
        axs[1].set_xlabel("Intensity Level", fontsize=fsz)
        axs[0].set_ylabel("PDF", fontsize=fsz)
        axs[1].set_ylabel("CDF", fontsize=fsz)
        xi = data_min - (np.abs(data_max - data_min) / 2)
        xa = data_max + (np.abs(data_max - data_min) / 2)
        rys = [[0, np.max(hist)], [0, 1]]
        for ax, ry in zip(axs, rys):
            ax.plot(
                [data_min, data_min],
                ry,
                "r",
                linestyle="--",
                label="data_min={:.1f}".format(data_min),
            )
            ax.plot(
                [data_max, data_max],
                ry,
                "g",
                linestyle="--",
                label="data_max={:.1f}".format(data_max),
            )
            ax.set_xlim(xi, xa)
            ax.grid(True)
        axs[1].legend(loc="center", fontsize=fsz)
        axs[1].set_title(
            "Data Min and max with thr_min={:.0e},  thr_max={:.0e}".format(
                thr_min, thr_max
            ),
            fontsize=fsz,
        )
        if save_res:
            fig.savefig(save_filename, dpi=dpi)
    return np.array((data_min, data_max))


def argmax2d(X):
    return np.unravel_index(X.argmax(), X.shape)


def find_BW(fr, FSC, SNRt):
    npts = np.shape(FSC)[0] * 0.75
    j = 15
    while (j < npts - 1) and FSC[j] > SNRt:
        j = j + 1
    BW = fr[j - 1] + (fr[j] - fr[j - 1]) * (SNRt - FSC[j - 1]) / (FSC[j] - FSC[j - 1])
    return BW


def radial_profile(data, center):
    """
    Calculates radially average profile of the 2D array (used for FRC and auto-correlation)
    ©G.Shtengel 08/2020 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array

    center : list of two ints
        [xcenter, ycenter]

    Returns
        radialprofile : float array
            limited to x-size//2 of the input data
    """
    ysz, xsz = data.shape
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = (np.array(tbin) / nr)[0 : (xsz // 2 + 1)]
    return radialprofile


def radial_profile_select_angles(data, center, astart=89, astop=91, symm=4):
    """
    Calculates radially average profile of the 2D array (used for FRC) within a select range of angles.
    ©G.Shtengel 08/2020 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array
    center : list of two ints
        [xcenter, ycenter]

    astart : float
        Start angle for radial averaging. Default is 0
    astop : float
        Stop angle for radial averaging. Default is 90
    symm : int
        Symmetry factor (how many times Start and stop angle intervalks are repeated within 360 deg). Default is 4.


    Returns
        radialprofile : float array
            limited to x-size//2 of the input data
    """
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r_ang = (np.angle(x - center[0] + 1j * (y - center[1]), deg=True)).ravel()
    r = r.astype(np.int)

    if symm > 0:
        ind = np.squeeze(np.array(np.where((r_ang > astart) & (r_ang < astart))))
        for i in np.arange(symm):
            ai = astart + 360.0 / symm * i - 180.0
            aa = astop + 360.0 / symm * i - 180.0
            ind = np.concatenate(
                (ind, np.squeeze((np.array(np.where((r_ang > ai) & (r_ang < aa)))))),
                axis=0,
            )
    else:
        ind = np.squeeze(np.where((r_ang > astart) & (r_ang < astop)))

    rr = np.ravel(r)[ind]
    dd = np.ravel(data)[ind]

    tbin = np.bincount(rr, dd)
    nr = np.bincount(rr)
    radialprofile = tbin / nr
    return radialprofile


def smooth(x, window_len=11, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]

    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[(window_len // 2 - 1) : -(window_len // 2)]
