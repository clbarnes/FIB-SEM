"""
# Helper functions for results presentation
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


def read_kwargs_xlsx(file_xlsx, kwargs_sheet_name, **kwargs):
    """
    Reads (SIFT processing) kwargs from XLSX file and returns them as dictionary. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com

    Parameters:
    file_xlsx : str
        Full path to XLSX file containing a worksheet with SIFt parameters saves as two columns: (name, value)
    kwargs_sheet_name : str
        Name of the worksheet containing SIFT parameters
    """
    disp_res_local = kwargs.get("disp_res", False)

    kwargs_dict_initial = {}
    try:
        stack_info = pd.read_excel(
            file_xlsx, header=None, sheet_name=kwargs_sheet_name
        ).T  # read from transposed
        if len(stack_info.keys()) > 0:
            if len(stack_info.keys()) > 0:
                for key in stack_info.keys():
                    kwargs_dict_initial[stack_info[key][0]] = stack_info[key][1]
            else:
                kwargs_dict_initial["Stack Filename"] = stack_info.index[1]
    except:
        if disp_res_local:
            print("No stack info record present, using defaults")
    kwargs_dict = {}
    for key in kwargs_dict_initial:
        if "TransformType" in key:
            exec(
                'kwargs_dict["TransformType"] = '
                + kwargs_dict_initial[key].split(".")[-1].split("'")[0]
            )
        elif "targ_vector" in key:
            exec(
                'kwargs_dict["targ_vector"] = np.array('
                + kwargs_dict_initial[key].replace(" ", ",")
                + ")"
            )
        elif "l2_matrix" in key:
            exec(
                'kwargs_dict["l2_matrix"] = np.array('
                + kwargs_dict_initial[key].replace(" ", ",")
                + ")"
            )
        elif "fit_params" in key:
            exec('kwargs_dict["fit_params"] = ' + kwargs_dict_initial[key])
        elif "subtract_linear_fit" in key:
            exec(
                'kwargs_dict["subtract_linear_fit"] = np.array('
                + kwargs_dict_initial[key]
                + ")"
            )
        elif "Stack Filename" in key:
            exec('kwargs_dict["Stack Filename"] = str(kwargs_dict_initial[key])')
        else:
            try:
                exec(
                    'kwargs_dict["' + str(key) + '"] = ' + str(kwargs_dict_initial[key])
                )
            except:
                exec(
                    'kwargs_dict["'
                    + str(key)
                    + '"] = "'
                    + kwargs_dict_initial[key].replace("\\", "/").replace("\n", ",")
                    + '"'
                )
    if "dump_filename" in kwargs.keys():
        kwargs_dict["dump_filename"] = kwargs["dump_filename"]
    # correct for pandas mixed read failures
    try:
        if kwargs_dict["mrc_mode"]:
            kwargs_dict["mrc_mode"] = 1
    except:
        pass
    try:
        if kwargs_dict["int_order"]:
            kwargs_dict["int_order"] = 1
    except:
        pass
    try:
        if kwargs_dict["flipY"] == 1:
            kwargs_dict["flipY"] = True
        else:
            kwargs_dict["flipY"] = False
    except:
        pass
    try:
        if kwargs_dict["BFMatcher"] == 1:
            kwargs_dict["BFMatcher"] = True
        else:
            kwargs_dict["BFMatcher"] = False
    except:
        pass
    try:
        if kwargs_dict["invert_data"] == 1:
            kwargs_dict["invert_data"] = True
        else:
            kwargs_dict["invert_data"] = False
    except:
        pass
    try:
        if kwargs_dict["sliding_evaluation_box"] == 1:
            kwargs_dict["sliding_evaluation_box"] = True
        else:
            kwargs_dict["sliding_evaluation_box"] = False
    except:
        pass

    return kwargs_dict


def generate_report_mill_rate_xlsx(Mill_Rate_Data_xlsx, **kwargs):
    """
    Generate Report Plot for mill rate evaluation from XLSX spreadsheet file. ©G.Shtengel 12/2022 gleb.shtengel@gmail.com

    Parameters:
    Mill_Rate_Data_xlsx : str
        Path to the xlsx workbook containing the Working Distance (WD), Milling Y Voltage (MV), and FOV center shifts data.

    kwargs:
    Mill_Volt_Rate_um_per_V : float
        Milling Voltage to Z conversion (µm/V). Default is 31.235258870176065.

    """
    disp_res = kwargs.get("disp_res", False)
    if disp_res:
        print("Loading kwarg Data")
    saved_kwargs = read_kwargs_xlsx(Mill_Rate_Data_xlsx, "kwargs Info", **kwargs)
    data_dir = saved_kwargs.get("data_dir", "")
    Sample_ID = saved_kwargs.get("Sample_ID", "")
    Saved_Mill_Volt_Rate_um_per_V = saved_kwargs.get(
        "Mill_Volt_Rate_um_per_V", 31.235258870176065
    )
    Mill_Volt_Rate_um_per_V = kwargs.get(
        "Mill_Volt_Rate_um_per_V", Saved_Mill_Volt_Rate_um_per_V
    )

    if disp_res:
        print("Loading Working Distance and Milling Y Voltage Data")
    try:
        int_results = pd.read_excel(Mill_Rate_Data_xlsx, sheet_name="FIBSEM Data")
    except:
        int_results = pd.read_excel(Mill_Rate_Data_xlsx, sheet_name="Milling Rate Data")
    fr = int_results["Frame"]
    WD = int_results["Working Distance (mm)"]
    MillingYVoltage = int_results["Milling Y Voltage (V)"]

    if disp_res:
        print("Generating Plot")
    fs = 12
    Mill_Volt_Rate_um_per_V = 31.235258870176065

    fig, axs = subplots(2, 1, figsize=(6, 7), sharex=True)
    fig.subplots_adjust(
        left=0.12, bottom=0.06, right=0.99, top=0.96, wspace=0.05, hspace=0.05
    )
    axs[0].plot(fr, WD, label="WD, Exp. Data", color="blue")
    axs[0].grid(True)
    axs[0].set_ylabel("Working Distance (mm)")
    # axs[0].set_xlim(xi, xa)
    WD_fit_coef = np.polyfit(fr, WD, 1)
    WD_fit = np.polyval(WD_fit_coef, fr)
    axs[0].plot(
        fr,
        WD_fit,
        label="Fit, slope = {:.2f} nm/line".format(WD_fit_coef[0] * 1.0e6),
        color="red",
    )
    axs[0].legend(fontsize=12)

    axs[1].plot(fr, MillingYVoltage, label="Mill. Y Volt. Exp. Data", color="green")
    axs[1].grid(True)
    axs[1].set_ylabel("Milling Y Voltage (V)")
    MV_fit_coef = np.polyfit(fr, MillingYVoltage, 1)
    MV_fit = np.polyval(MV_fit_coef, fr)
    axs[1].plot(
        fr,
        MV_fit,
        label="Fit, slope = {:.3f} nm/line".format(
            MV_fit_coef[0] * Mill_Volt_Rate_um_per_V * -1.0e3
        ),
        color="orange",
    )
    axs[1].legend(fontsize=12)
    axs[1].text(
        0.02,
        0.05,
        "Milling Voltage to Z conversion: {:.4f} µm/V".format(Mill_Volt_Rate_um_per_V),
        transform=axs[1].transAxes,
        fontsize=12,
    )
    axs[1].set_xlabel("Frame")
    ldm = 70
    data_dir_short = data_dir if len(data_dir) < ldm else "... " + data_dir[-ldm:]
    try:
        axs[0].text(
            -0.15,
            1.05,
            Sample_ID + "    " + data_dir_short,
            fontsize=fs - 2,
            transform=axs[0].transAxes,
        )
    except:
        axs[0].text(
            -0.15, 1.05, data_dir_short, fontsize=fs - 2, transform=axs[0].transAxes
        )
    fig.savefig(
        os.path.join(data_dir, Mill_Rate_Data_xlsx.replace(".xlsx", "_Mill_Rate.png")),
        dpi=300,
    )


def generate_report_FOV_center_shift_xlsx(Mill_Rate_Data_xlsx, **kwargs):
    """
    Generate Report Plot for FOV center shift from XLSX spreadsheet file. ©G.Shtengel 12/2022 gleb.shtengel@gmail.com

    Parameters:
    Mill_Rate_Data_xlsx : str
        Path to the xlsx workbook containing the Working Distance (WD), Milling Y Voltage (MV), and FOV center shifts data.

    kwargs:
    Mill_Volt_Rate_um_per_V : float
        Milling Voltage to Z conversion (µm/V). Defaul is 31.235258870176065.

    Returns: trend_x, trend_y
        Smoothed FOV shifts
    """
    disp_res = kwargs.get("disp_res", False)
    if disp_res:
        print("Loading kwarg Data")
    saved_kwargs = read_kwargs_xlsx(Mill_Rate_Data_xlsx, "kwargs Info", **kwargs)
    data_dir = saved_kwargs.get("data_dir", "")
    Sample_ID = saved_kwargs.get("Sample_ID", "")

    if disp_res:
        print("Loading FOV Center Location Data")
    try:
        int_results = pd.read_excel(Mill_Rate_Data_xlsx, sheet_name="FIBSEM Data")
    except:
        int_results = pd.read_excel(Mill_Rate_Data_xlsx, sheet_name="Milling Rate Data")
    fr = int_results["Frame"]
    center_x = int_results["FOV X Center (Pix)"]
    center_y = int_results["FOV Y Center (Pix)"]
    apert = np.min((51, len(fr) - 1))
    trend_x = savgol_filter(center_x * 1.0, apert, 1) - center_x[0]
    trend_y = savgol_filter(center_y * 1.0, apert, 1) - center_y[0]

    if disp_res:
        print("Generating Plot")
    fs = 12

    fig, axs = subplots(2, 1, figsize=(6, 7), sharex=True)
    fig.subplots_adjust(
        left=0.12, bottom=0.06, right=0.99, top=0.96, wspace=0.05, hspace=0.05
    )
    axs[0].plot(fr, center_x, label="FOV X center, Data", color="red")
    axs[0].plot(fr, center_y, label="FOV Y center, Data", color="blue")
    axs[0].grid(True)
    axs[0].set_ylabel("FOV Center (Pix)")
    # axs[0].set_xlim(xi, xa)
    axs[0].legend(fontsize=12)

    axs[1].plot(fr, trend_x, label="FOV X center shift, smoothed", color="red")
    axs[1].plot(fr, trend_y, label="FOV Y center shift, smoothed", color="blue")
    axs[1].grid(True)
    axs[1].set_ylabel("FOV Center Shift (Pix)")
    axs[1].legend(fontsize=12)
    axs[1].set_xlabel("Frame")
    ldm = 70
    data_dir_short = data_dir if len(data_dir) < ldm else "... " + data_dir[-ldm:]
    try:
        axs[0].text(
            -0.15,
            1.05,
            Sample_ID + "    " + data_dir_short,
            fontsize=fs - 2,
            transform=axs[0].transAxes,
        )
    except:
        axs[0].text(
            -0.15, 1.05, data_dir_short, fontsize=fs - 2, transform=axs[0].transAxes
        )
    fig.savefig(
        os.path.join(
            data_dir, Mill_Rate_Data_xlsx.replace(".xlsx", "_FOV_XYcenter.png")
        ),
        dpi=300,
    )
    return


def generate_report_data_minmax_xlsx(minmax_xlsx_file, **kwargs):
    """
    Generate Report Plot for data Min-Max from XLSX spreadsheet file. ©G.Shtengel 10/2022 gleb.shtengel@gmail.com

    Parameters:
    minmax_xlsx_file : str
        Path to the xlsx workbook containing Min-Max data
    """
    disp_res = kwargs.get("disp_res", False)
    if disp_res:
        print("Loading kwarg Data")
    saved_kwargs = read_kwargs_xlsx(minmax_xlsx_file, "kwargs Info", **kwargs)
    data_dir = saved_kwargs.get("data_dir", "")
    fnm_reg = saved_kwargs.get("fnm_reg", "Registration_file.mrc")
    Sample_ID = saved_kwargs.get("Sample_ID", "")
    threshold_min = saved_kwargs.get("threshold_min", 0.0)
    threshold_max = saved_kwargs.get("threshold_min", 0.0)
    fit_params_saved = saved_kwargs.get("fit_params", ["SG", 101, 3])
    fit_params = kwargs.get("fit_params", fit_params_saved)
    preserve_scales = saved_kwargs.get(
        "preserve_scales", True
    )  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below

    if disp_res:
        print("Loading MinMax Data")
    try:
        int_results = pd.read_excel(minmax_xlsx_file, sheet_name="FIBSEM Data")
    except:
        int_results = pd.read_excel(minmax_xlsx_file, sheet_name="MinMax Data")
    frames = int_results["Frame"]
    frame_min = int_results["Min"]
    frame_max = int_results["Max"]
    data_min_glob = np.min(frame_min)
    data_max_glob = np.max(frame_max)
    """
    sliding_min = int_results['Sliding Min']
    sliding_max = int_results['Sliding Max']
    """
    sliding_min = savgol_filter(
        frame_min.astype(double), min([fit_params[1], fit_params[1]]), fit_params[2]
    )
    sliding_max = savgol_filter(
        frame_max.astype(double), min([fit_params[1], fit_params[1]]), fit_params[2]
    )

    if disp_res:
        print("Generating Plot")
    fs = 12
    fig0, ax0 = subplots(1, 1, figsize=(6, 4))
    fig0.subplots_adjust(left=0.14, bottom=0.11, right=0.99, top=0.94)
    ax0.plot(frame_min, "b", linewidth=1, label="Frame Minima")
    ax0.plot(sliding_min, "b", linewidth=2, linestyle="dotted", label="Sliding Minima")
    ax0.plot(frame_max, "r", linewidth=1, label="Frame Maxima")
    ax0.plot(sliding_max, "r", linewidth=2, linestyle="dotted", label="Sliding Maxima")
    ax0.legend()
    ax0.grid(True)
    ax0.set_xlabel("Frame")
    ax0.set_ylabel("Minima and Maxima Values")
    dxn = (data_max_glob - data_min_glob) * 0.1
    ax0.set_ylim((data_min_glob - dxn, data_max_glob + dxn))
    # if needed, display the data in a narrower range
    # ax0.set_ylim((-4500, -1500))
    xminmax = [0, len(frame_min)]
    y_min = [data_min_glob, data_min_glob]
    y_max = [data_max_glob, data_max_glob]
    ax0.plot(xminmax, y_min, "b", linestyle="--")
    ax0.plot(xminmax, y_max, "r", linestyle="--")
    ax0.text(
        len(frame_min) / 20.0,
        data_min_glob - dxn / 1.75,
        "data_min_glob={:.1f}".format(data_min_glob),
        fontsize=fs - 2,
        c="b",
    )
    ax0.text(
        len(frame_min) / 20.0,
        data_max_glob + dxn / 2.25,
        "data_max_glob={:.1f}".format(data_max_glob),
        fontsize=fs - 2,
        c="r",
    )
    ax0.text(
        len(frame_min) / 20.0,
        data_min_glob + dxn * 4.5,
        "threshold_min={:.1e}".format(threshold_min),
        fontsize=fs - 2,
        c="b",
    )
    ax0.text(
        len(frame_min) / 20.0,
        data_min_glob + dxn * 5.5,
        "threshold_max={:.1e}".format(threshold_max),
        fontsize=fs - 2,
        c="r",
    )
    ldm = 70
    data_dir_short = data_dir if len(data_dir) < ldm else "... " + data_dir[-ldm:]

    try:
        ax0.text(
            -0.15,
            1.05,
            Sample_ID + "    " + data_dir_short,
            fontsize=fs - 2,
            transform=axs[0].transAxes,
        )
    except:
        ax0.text(-0.15, 1.05, data_dir_short, fontsize=fs - 2, transform=ax0.transAxes)
    """
    try:
        fig0.suptitle(Sample_ID + '    ' +  data_dir_short, fontsize = fs-2)
    except:
        fig0.suptitle(data_dir_short, fontsize = fs-2)
    """
    fig0.savefig(
        os.path.join(data_dir, minmax_xlsx_file.replace(".xlsx", "_Min_Max.png")),
        dpi=300,
    )


def generate_report_transf_matrix_from_xlsx(transf_matrix_xlsx_file, **kwargs):
    """
    Generate Report Plot for Transformation Matrix from XLSX spreadsheet file. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com

    Parameters:
    transf_matrix_xlsx_file : str
        Path to the xlsx workbook containing Transformation Matrix data

    """
    disp_res = kwargs.get("disp_res", False)
    if disp_res:
        print("Loading kwarg Data")
    saved_kwargs = read_kwargs_xlsx(transf_matrix_xlsx_file, "kwargs Info", **kwargs)
    data_dir = saved_kwargs.get("data_dir", "")
    fnm_reg = saved_kwargs.get("fnm_reg", "Registration_file.mrc")
    TransformType = saved_kwargs.get("TransformType", RegularizedAffineTransform)
    Sample_ID = saved_kwargs.get("Sample_ID", "")
    SIFT_nfeatures = saved_kwargs.get("SIFT_nfeatures", 0)
    SIFT_nOctaveLayers = saved_kwargs.get("SIFT_nOctaveLayers", 3)
    SIFT_contrastThreshold = saved_kwargs.get("SIFT_contrastThreshold", 0.025)
    SIFT_edgeThreshold = saved_kwargs.get("SIFT_edgeThreshold", 10)
    SIFT_sigma = saved_kwargs.get("SIFT_sigma", 1.6)
    l2_param_default = 1e-5  # regularization strength (shrinkage parameter)
    l2_matrix_default = (
        np.eye(6) * l2_param_default
    )  # initially set equal shrinkage on all coefficients
    l2_matrix_default[2, 2] = 0  # turn OFF the regularization on shifts
    l2_matrix_default[5, 5] = 0  # turn OFF the regularization on shifts
    l2_matrix = saved_kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = saved_kwargs.get(
        "targ_vector", np.array([1, 0, 0, 0, 1, 0])
    )  # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = saved_kwargs.get("solver", "RANSAC")
    drmax = saved_kwargs.get("drmax", 2.0)
    max_iter = saved_kwargs.get("max_iter", 1000)
    BFMatcher = saved_kwargs.get(
        "BFMatcher", False
    )  # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = saved_kwargs.get(
        "save_matches", True
    )  # If True, matches will be saved into individual files
    kp_max_num = saved_kwargs.get("kp_max_num", -1)
    save_res_png = saved_kwargs.get("save_res_png", True)

    preserve_scales = saved_kwargs.get(
        "preserve_scales", True
    )  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params = saved_kwargs.get(
        "fit_params", False
    )  # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
    # window size 701, polynomial order 3
    subtract_linear_fit = saved_kwargs.get(
        "subtract_linear_fit", [True, True]
    )  # The linear slopes along X- and Y- directions (respectively) will be subtracted from the cumulative shifts.
    subtract_FOVtrend_from_fit = saved_kwargs.get(
        "subtract_FOVtrend_from_fit", [True, True]
    )
    pad_edges = saved_kwargs.get("pad_edges", True)

    if disp_res:
        print("Loading Original Transformation Data")
    orig_transf_matrix = pd.read_excel(
        transf_matrix_xlsx_file, sheet_name="Orig. Transformation Matrix"
    )
    transformation_matrix = np.vstack(
        (
            orig_transf_matrix["T00 (Sxx)"],
            orig_transf_matrix["T01 (Sxy)"],
            orig_transf_matrix["T02 (Tx)"],
            orig_transf_matrix["T10 (Syx)"],
            orig_transf_matrix["T11 (Syy)"],
            orig_transf_matrix["T12 (Ty)"],
            orig_transf_matrix["T20 (0.0)"],
            orig_transf_matrix["T21 (0.0)"],
            orig_transf_matrix["T22 (1.0)"],
        )
    ).T.reshape((len(orig_transf_matrix["T00 (Sxx)"]), 3, 3))

    if disp_res:
        print("Loading Cumulative Transformation Data")
    cum_transf_matrix = pd.read_excel(
        transf_matrix_xlsx_file, sheet_name="Cum. Transformation Matrix"
    )
    tr_matr_cum = np.vstack(
        (
            cum_transf_matrix["T00 (Sxx)"],
            cum_transf_matrix["T01 (Sxy)"],
            cum_transf_matrix["T02 (Tx)"],
            cum_transf_matrix["T10 (Syx)"],
            cum_transf_matrix["T11 (Syy)"],
            cum_transf_matrix["T12 (Ty)"],
            cum_transf_matrix["T20 (0.0)"],
            cum_transf_matrix["T21 (0.0)"],
            cum_transf_matrix["T22 (1.0)"],
        )
    ).T.reshape((len(cum_transf_matrix["T00 (Sxx)"]), 3, 3))

    if disp_res:
        print("Loading Intermediate Data")
    int_results = pd.read_excel(
        transf_matrix_xlsx_file, sheet_name="Intermediate Results"
    )
    s00_cum_orig = int_results["s00_cum_orig"]
    s11_cum_orig = int_results["s11_cum_orig"]
    s00_fit = int_results["s00_fit"]
    s11_fit = int_results["s11_fit"]
    s01_cum_orig = int_results["s01_cum_orig"]
    s10_cum_orig = int_results["s10_cum_orig"]
    s01_fit = int_results["s01_fit"]
    s10_fit = int_results["s10_fit"]
    Xshift_cum_orig = int_results["Xshift_cum_orig"]
    Yshift_cum_orig = int_results["Yshift_cum_orig"]
    Xshift_cum = int_results["Xshift_cum"]
    Yshift_cum = int_results["Yshift_cum"]
    Xfit = int_results["Xfit"]
    Yfit = int_results["Yfit"]

    if disp_res:
        print("Loading Statistics")
    stat_results = pd.read_excel(transf_matrix_xlsx_file, sheet_name="Reg. Stat. Info")
    npts = stat_results["Npts"]
    error_abs_mean = stat_results["Mean Abs Error"]

    fs = 14
    lwf = 2
    lwl = 1
    fig5, axs5 = subplots(4, 3, figsize=(18, 16), sharex=True)
    fig5.subplots_adjust(left=0.07, bottom=0.03, right=0.99, top=0.95)
    # display the info
    axs5[0, 0].axis(False)
    axs5[0, 0].text(-0.1, 0.9, Sample_ID, fontsize=fs + 4)
    # axs5[0,0].text(-0.1, 0.73, 'Global Data Range:  Min={:.2f}, Max={:.2f}'.format(data_min_glob, data_max_glob), transform=axs5[0,0].transAxes, fontsize = fs)

    if TransformType == RegularizedAffineTransform:
        tstr = ["{:d}".format(x) for x in targ_vector]
        otext = (
            "Reg.Aff.Transf., λ= {:.1e}, t=[".format(l2_matrix[0, 0])
            + " ".join(tstr)
            + "], w/"
            + solver
        )
    else:
        otext = TransformType.__name__ + " with " + solver + " solver"
    axs5[0, 0].text(-0.1, 0.80, otext, transform=axs5[0, 0].transAxes, fontsize=fs)

    SIFT1text = "SIFT: nFeatures = {:d}, nOctaveLayers = {:d}, ".format(
        SIFT_nfeatures, SIFT_nOctaveLayers
    )
    axs5[0, 0].text(-0.1, 0.65, SIFT1text, transform=axs5[0, 0].transAxes, fontsize=fs)

    SIFT2text = "SIFT: contrThr = {:.3f}, edgeThr = {:.2f}, σ= {:.2f}".format(
        SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma
    )
    axs5[0, 0].text(-0.1, 0.50, SIFT2text, transform=axs5[0, 0].transAxes, fontsize=fs)

    sbtrfit = (
        ("ON, " if subtract_linear_fit[0] else "OFF, ")
        + ("ON" if subtract_linear_fit[1] else "OFF")
        + ("(ON, " if subtract_FOVtrend_from_fit[0] else "(OFF, ")
        + ("ON)" if subtract_FOVtrend_from_fit[1] else "OFF)")
    )
    axs5[0, 0].text(
        -0.1,
        0.35,
        "drmax={:.1f}, Max # of KeyPts={:d}, Max # of Iter.={:d}".format(
            drmax, kp_max_num, max_iter
        ),
        transform=axs5[0, 0].transAxes,
        fontsize=fs,
    )
    padedges = "ON" if pad_edges else "OFF"
    if preserve_scales:
        fit_method = fit_params[0]
        if fit_method == "LF":
            fit_str = ", Meth: Linear Fit"
            fm_string = "linear"
        else:
            if fit_method == "SG":
                fit_str = ", Meth: Sav.-Gol., " + str(fit_params[1:])
                fm_string = "Sav.-Gol."
            else:
                fit_str = ", Meth: Pol.Fit, ord.={:d}".format(fit_params[1])
                fm_string = "polyn."
        preserve_scales_string = "Pres. Scls: ON" + fit_str
    else:
        preserve_scales_string = "Preserve Scales: OFF"
    axs5[0, 0].text(
        -0.1, 0.20, preserve_scales_string, transform=axs5[0, 0].transAxes, fontsize=fs
    )
    axs5[0, 0].text(
        -0.1,
        0.05,
        "Subtract Shift Fit: " + sbtrfit + ", Pad Edges: " + padedges,
        transform=axs5[0, 0].transAxes,
        fontsize=fs,
    )
    # plot number of keypoints
    axs5[0, 1].plot(npts, "g", linewidth=lwl, label="# of key-points per frame")
    axs5[0, 1].set_title("# of key-points per frame")
    axs5[0, 1].text(
        0.03,
        0.2,
        "Mean # of kpts= {:.0f}   Median # of kpts= {:.0f}".format(
            np.mean(npts), np.median(npts)
        ),
        transform=axs5[0, 1].transAxes,
        fontsize=fs - 1,
    )
    # plot Standard deviations
    axs5[0, 2].plot(
        error_abs_mean,
        "magenta",
        linewidth=lwl,
        label="Mean Abs Error over keyponts per frame",
    )
    axs5[0, 2].set_title("Mean Abs Error keyponts per frame")
    axs5[0, 2].text(
        0.03,
        0.2,
        "Mean Abs Error= {:.3f}   Median Abs Error= {:.3f}".format(
            np.mean(error_abs_mean), np.median(error_abs_mean)
        ),
        transform=axs5[0, 2].transAxes,
        fontsize=fs - 1,
    )

    # plot scales terms
    axs5[1, 0].plot(
        transformation_matrix[:, 0, 0], "r", linewidth=lwl, label="Sxx frame-to-frame"
    )
    axs5[1, 0].plot(
        transformation_matrix[:, 1, 1], "b", linewidth=lwl, label="Syy frame-to-frame"
    )
    axs5[1, 0].set_title("Frame-to-Frame Scale Change", fontsize=fs)
    axs5[2, 0].plot(
        s00_cum_orig, "r", linewidth=lwl, linestyle="dotted", label="Sxx cum."
    )
    axs5[2, 0].plot(
        s11_cum_orig, "b", linewidth=lwl, linestyle="dotted", label="Syy cum."
    )
    if preserve_scales:
        axs5[2, 0].plot(
            s00_fit,
            "orange",
            linewidth=lwf,
            linestyle="dashed",
            label="Sxx cum. - " + fm_string + " fit",
        )
        axs5[2, 0].plot(
            s11_fit,
            "cyan",
            linewidth=lwf,
            linestyle="dashed",
            label="Syy cum. - " + fm_string + " fit",
        )
    axs5[2, 0].set_title("Cumulative Scale", fontsize=fs)
    yi10, ya10 = axs5[1, 0].get_ylim()
    dy0 = (ya10 - yi10) / 2.0
    yi20, ya20 = axs5[2, 0].get_ylim()
    if (ya20 - yi20) < 0.01 * dy0:
        axs5[2, 0].set_ylim((yi20 - dy0, ya20 + dy0))
    axs5[3, 0].plot(
        tr_matr_cum[:, 0, 0], "r", linewidth=lwl, label="Sxx cum. - residual"
    )
    axs5[3, 0].plot(
        tr_matr_cum[:, 1, 1], "b", linewidth=lwl, label="Syy cum. - residual"
    )
    axs5[3, 0].set_title("Residual Cumulative Scale", fontsize=fs)
    axs5[3, 0].set_xlabel("Frame", fontsize=fs + 1)
    yi30, ya30 = axs5[3, 0].get_ylim()
    if (ya30 - yi30) < 0.01 * dy0:
        axs5[3, 0].set_ylim((yi30 - dy0, ya30 + dy0))

    # plot shear terms
    axs5[1, 1].plot(
        transformation_matrix[:, 0, 1], "r", linewidth=lwl, label="Sxy frame-to-frame"
    )
    axs5[1, 1].plot(
        transformation_matrix[:, 1, 0], "b", linewidth=lwl, label="Syx frame-to-frame"
    )
    axs5[1, 1].set_title("Frame-to-Frame Shear Change", fontsize=fs)
    axs5[2, 1].plot(
        s01_cum_orig, "r", linewidth=lwl, linestyle="dotted", label="Sxy cum."
    )
    axs5[2, 1].plot(
        s10_cum_orig, "b", linewidth=lwl, linestyle="dotted", label="Syx cum."
    )
    if preserve_scales:
        axs5[2, 1].plot(
            s01_fit,
            "orange",
            linewidth=lwf,
            linestyle="dashed",
            label="Sxy cum. - " + fm_string + " fit",
        )
        axs5[2, 1].plot(
            s10_fit,
            "cyan",
            linewidth=lwf,
            linestyle="dashed",
            label="Syx cum. - " + fm_string + " fit",
        )
    axs5[2, 1].set_title("Cumulative Shear", fontsize=fs)
    yi11, ya11 = axs5[1, 1].get_ylim()
    dy1 = (ya11 - yi11) / 2.0
    yi21, ya21 = axs5[2, 1].get_ylim()
    if (ya21 - yi21) < 0.01 * dy1:
        axs5[2, 1].set_ylim((yi21 - dy1, ya21 + dy1))
    axs5[3, 1].plot(
        tr_matr_cum[:, 0, 1], "r", linewidth=lwl, label="Sxy cum. - residual"
    )
    axs5[3, 1].plot(
        tr_matr_cum[:, 1, 0], "b", linewidth=lwl, label="Syx cum. - residual"
    )
    axs5[3, 1].set_title("Residual Cumulative Shear", fontsize=fs)
    axs5[3, 1].set_xlabel("Frame", fontsize=fs + 1)
    yi31, ya31 = axs5[3, 1].get_ylim()
    if (ya31 - yi21) < 0.01 * dy1:
        axs5[3, 1].set_ylim((yi31 - dy1, ya31 + dy1))

    # plot shifts
    axs5[1, 2].plot(
        transformation_matrix[:, 0, 2], "r", linewidth=lwl, label="Tx fr.-to-fr."
    )
    axs5[1, 2].plot(
        transformation_matrix[:, 1, 2], "b", linewidth=lwl, label="Ty fr.-to-fr."
    )
    axs5[1, 2].set_title("Frame-to-Frame Shift", fontsize=fs)
    if preserve_scales:
        axs5[2, 2].plot(Xshift_cum_orig, "r", linewidth=lwl, label="Tx cum. - orig.")
        axs5[2, 2].plot(Yshift_cum_orig, "b", linewidth=lwl, label="Ty cum. - orig.")
        axs5[2, 2].plot(
            Xshift_cum,
            "r",
            linewidth=lwl,
            linestyle="dotted",
            label="Tx cum. - pres. scales",
        )
        axs5[2, 2].plot(
            Yshift_cum,
            "b",
            linewidth=lwl,
            linestyle="dotted",
            label="Ty cum. - pres. scales",
        )
    else:
        axs5[2, 2].plot(
            Xshift_cum, "r", linewidth=lwl, linestyle="dotted", label="Tx cum."
        )
        axs5[2, 2].plot(
            Yshift_cum, "b", linewidth=lwl, linestyle="dotted", label="Ty cum."
        )
    if subtract_linear_fit[0]:
        axs5[2, 2].plot(
            Xfit,
            "orange",
            linewidth=lwf,
            linestyle="dashed",
            label="Tx cum. - lin. fit",
        )
    if subtract_linear_fit[1]:
        axs5[2, 2].plot(
            Yfit, "cyan", linewidth=lwf, linestyle="dashed", label="Ty cum. - lin. fit"
        )
    axs5[2, 2].set_title("Cumulative Shift", fontsize=fs)
    axs5[3, 2].plot(
        tr_matr_cum[:, 0, 2], "r", linewidth=lwl, label="Tx cum. - residual"
    )
    axs5[3, 2].plot(
        tr_matr_cum[:, 1, 2], "b", linewidth=lwl, label="Ty cum. - residual"
    )
    axs5[3, 2].set_title("Residual Cumulative Shift", fontsize=fs)
    axs5[3, 2].set_xlabel("Frame", fontsize=fs + 1)

    for ax in axs5.ravel()[1:]:
        ax.grid(True)
        ax.legend(fontsize=fs - 1)
    fig5.suptitle(transf_matrix_xlsx_file, fontsize=fs)
    if save_res_png:
        fig5.savefig(transf_matrix_xlsx_file.replace(".xlsx", ".png"), dpi=300)


def generate_report_transf_matrix_details(transf_matrix_bin_file, *kwarrgs):
    """
    Generate Report Plot for Transformation Matrix from binary dump file. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com
    The binary dump file should contain list with these parameters (in this order):
        [saved_kwargs, npts, error_abs_mean, transformation_matrix,
        s00_cum_orig, s11_cum_orig, s00_fit, s11_fit, tr_matr_cum, s01_cum_orig, s10_cum_orig, s01_fit, s10_fit,
        Xshift_cum_orig, Yshift_cum_orig, Xshift_cum, Yshift_cum, Xfit, Yfit]

    Parameters:
    transf_matrix_bin_file : str
        Path to the binary dump file

    """
    with open(transf_matrix_bin_file, "rb") as f:
        [
            saved_kwargs,
            npts,
            error_abs_mean,
            transformation_matrix,
            s00_cum_orig,
            s11_cum_orig,
            s00_fit,
            s11_fit,
            tr_matr_cum,
            s01_cum_orig,
            s10_cum_orig,
            s01_fit,
            s10_fit,
            Xshift_cum_orig,
            Yshift_cum_orig,
            Xshift_cum,
            Yshift_cum,
            Xfit,
            Yfit,
        ] = pickle.load(f)

    data_dir = saved_kwargs.get("data_dir", "")
    fnm_reg = saved_kwargs.get("fnm_reg", "Registration_file.mrc")
    TransformType = saved_kwargs.get("TransformType", RegularizedAffineTransform)
    Sample_ID = saved_kwargs.get("Sample_ID", "")
    SIFT_nfeatures = saved_kwargs.get("SIFT_nfeatures", 0)
    SIFT_nOctaveLayers = saved_kwargs.get("SIFT_nOctaveLayers", 3)
    SIFT_contrastThreshold = saved_kwargs.get("SIFT_contrastThreshold", 0.025)
    SIFT_edgeThreshold = saved_kwargs.get("SIFT_edgeThreshold", 10)
    SIFT_sigma = saved_kwargs.get("SIFT_sigma", 1.6)
    l2_param_default = 1e-5  # regularization strength (shrinkage parameter)
    l2_matrix_default = (
        np.eye(6) * l2_param_default
    )  # initially set equal shrinkage on all coefficients
    l2_matrix_default[2, 2] = 0  # turn OFF the regularization on shifts
    l2_matrix_default[5, 5] = 0  # turn OFF the regularization on shifts
    l2_matrix = saved_kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = saved_kwargs.get(
        "targ_vector", np.array([1, 0, 0, 0, 1, 0])
    )  # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = saved_kwargs.get("solver", "RANSAC")
    drmax = saved_kwargs.get("drmax", 2.0)
    max_iter = saved_kwargs.get("max_iter", 1000)
    BFMatcher = saved_kwargs.get(
        "BFMatcher", False
    )  # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = saved_kwargs.get(
        "save_matches", True
    )  # If True, matches will be saved into individual files
    kp_max_num = saved_kwargs.get("kp_max_num", -1)
    save_res_png = saved_kwargs.get("save_res_png", True)

    preserve_scales = saved_kwargs.get(
        "preserve_scales", True
    )  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params = saved_kwargs.get(
        "fit_params", False
    )  # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
    # window size 701, polynomial order 3
    subtract_linear_fit = saved_kwargs.get(
        "subtract_linear_fit", [True, True]
    )  # The linear slopes along X- and Y- directions (respectively) will be subtracted from the cumulative shifts.
    subtract_FOVtrend_from_fit = saved_kwargs.get(
        "subtract_FOVtrend_from_fit", [True, True]
    )
    # print("subtract_linear_fit:", subtract_linear_fit)
    pad_edges = saved_kwargs.get("pad_edges", True)

    fs = 14
    lwf = 2
    lwl = 1
    fig5, axs5 = subplots(4, 3, figsize=(18, 16), sharex=True)
    fig5.subplots_adjust(left=0.07, bottom=0.03, right=0.99, top=0.95)
    # display the info
    axs5[0, 0].axis(False)
    axs5[0, 0].text(-0.1, 0.9, Sample_ID, fontsize=fs + 4)
    # axs5[0,0].text(-0.1, 0.73, 'Global Data Range:  Min={:.2f}, Max={:.2f}'.format(data_min_glob, data_max_glob), transform=axs5[0,0].transAxes, fontsize = fs)

    if TransformType == RegularizedAffineTransform:
        tstr = ["{:d}".format(x) for x in targ_vector]
        otext = (
            "Reg.Aff.Transf., λ= {:.1e}, t=[".format(l2_matrix[0, 0])
            + " ".join(tstr)
            + "], w/"
            + solver
        )
    else:
        otext = TransformType.__name__ + " with " + solver + " solver"
    axs5[0, 0].text(-0.1, 0.80, otext, transform=axs5[0, 0].transAxes, fontsize=fs)

    SIFT1text = "SIFT: nFeatures = {:d}, nOctaveLayers = {:d}, ".format(
        SIFT_nfeatures, SIFT_nOctaveLayers
    )
    axs5[0, 0].text(-0.1, 0.65, SIFT1text, transform=axs5[0, 0].transAxes, fontsize=fs)

    SIFT2text = "SIFT: contrThr = {:.3f}, edgeThr = {:.2f}, σ= {:.2f}".format(
        SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma
    )
    axs5[0, 0].text(-0.1, 0.50, SIFT2text, transform=axs5[0, 0].transAxes, fontsize=fs)

    sbtrfit = ("ON, " if subtract_linear_fit[0] else "OFF, ") + (
        "ON" if subtract_linear_fit[1] else "OFF"
    )
    axs5[0, 0].text(
        -0.1,
        0.35,
        "drmax={:.1f}, Max # of KeyPts={:d}, Max # of Iter.={:d}".format(
            drmax, kp_max_num, max_iter
        ),
        transform=axs5[0, 0].transAxes,
        fontsize=fs,
    )
    padedges = "ON" if pad_edges else "OFF"
    if preserve_scales:
        fit_method = fit_params[0]
        if fit_method == "LF":
            fit_str = ", Meth: Linear Fit"
            fm_string = "linear"
        else:
            if fit_method == "SG":
                fit_str = ", Meth: Sav.-Gol., " + str(fit_params[1:])
                fm_string = "Sav.-Gol."
            else:
                fit_str = ", Meth: Pol.Fit, ord.={:d}".format(fit_params[1])
                fm_string = "polyn."
        preserve_scales_string = "Pres. Scls: ON" + fit_str
    else:
        preserve_scales_string = "Preserve Scales: OFF"
    axs5[0, 0].text(
        -0.1, 0.20, preserve_scales_string, transform=axs5[0, 0].transAxes, fontsize=fs
    )
    axs5[0, 0].text(
        -0.1,
        0.05,
        "Subtract Shift Fit: " + sbtrfit + ", Pad Edges: " + padedges,
        transform=axs5[0, 0].transAxes,
        fontsize=fs,
    )
    # plot number of keypoints
    axs5[0, 1].plot(npts, "g", linewidth=lwl, label="# of key-points per frame")
    axs5[0, 1].set_title("# of key-points per frame")
    axs5[0, 1].text(
        0.03,
        0.2,
        "Mean # of kpts= {:.0f}   Median # of kpts= {:.0f}".format(
            np.mean(npts), np.median(npts)
        ),
        transform=axs5[0, 1].transAxes,
        fontsize=fs - 1,
    )
    # plot Standard deviations
    axs5[0, 2].plot(
        error_abs_mean,
        "magenta",
        linewidth=lwl,
        label="Mean Abs Error over keyponts per frame",
    )
    axs5[0, 2].set_title("Mean Abs Error keyponts per frame")
    axs5[0, 2].text(
        0.03,
        0.2,
        "Mean Abs Error= {:.3f}   Median Abs Error= {:.3f}".format(
            np.mean(error_abs_mean), np.median(error_abs_mean)
        ),
        transform=axs5[0, 2].transAxes,
        fontsize=fs - 1,
    )

    # plot scales terms
    axs5[1, 0].plot(
        transformation_matrix[:, 0, 0], "r", linewidth=lwl, label="Sxx frame-to-frame"
    )
    axs5[1, 0].plot(
        transformation_matrix[:, 1, 1], "b", linewidth=lwl, label="Syy frame-to-frame"
    )
    axs5[1, 0].set_title("Frame-to-Frame Scale Change", fontsize=fs)
    axs5[2, 0].plot(
        s00_cum_orig, "r", linewidth=lwl, linestyle="dotted", label="Sxx cum."
    )
    axs5[2, 0].plot(
        s11_cum_orig, "b", linewidth=lwl, linestyle="dotted", label="Syy cum."
    )
    if preserve_scales:
        axs5[2, 0].plot(
            s00_fit,
            "orange",
            linewidth=lwf,
            linestyle="dashed",
            label="Sxx cum. - " + fm_string + " fit",
        )
        axs5[2, 0].plot(
            s11_fit,
            "cyan",
            linewidth=lwf,
            linestyle="dashed",
            label="Syy cum. - " + fm_string + " fit",
        )
    axs5[2, 0].set_title("Cumulative Scale", fontsize=fs)
    yi10, ya10 = axs5[1, 0].get_ylim()
    dy0 = (ya10 - yi10) / 2.0
    yi20, ya20 = axs5[2, 0].get_ylim()
    if (ya20 - yi20) < 0.01 * dy0:
        axs5[2, 0].set_ylim((yi20 - dy0, ya20 + dy0))
    axs5[3, 0].plot(
        tr_matr_cum[:, 0, 0], "r", linewidth=lwl, label="Sxx cum. - residual"
    )
    axs5[3, 0].plot(
        tr_matr_cum[:, 1, 1], "b", linewidth=lwl, label="Syy cum. - residual"
    )
    axs5[3, 0].set_title("Residual Cumulative Scale", fontsize=fs)
    axs5[3, 0].set_xlabel("Frame", fontsize=fs + 1)
    yi30, ya30 = axs5[3, 0].get_ylim()
    if (ya30 - yi30) < 0.01 * dy0:
        axs5[3, 0].set_ylim((yi30 - dy0, ya30 + dy0))

    # plot shear terms
    axs5[1, 1].plot(
        transformation_matrix[:, 0, 1], "r", linewidth=lwl, label="Sxy frame-to-frame"
    )
    axs5[1, 1].plot(
        transformation_matrix[:, 1, 0], "b", linewidth=lwl, label="Syx frame-to-frame"
    )
    axs5[1, 1].set_title("Frame-to-Frame Shear Change", fontsize=fs)
    axs5[2, 1].plot(
        s01_cum_orig, "r", linewidth=lwl, linestyle="dotted", label="Sxy cum."
    )
    axs5[2, 1].plot(
        s10_cum_orig, "b", linewidth=lwl, linestyle="dotted", label="Syx cum."
    )
    if preserve_scales:
        axs5[2, 1].plot(
            s01_fit,
            "orange",
            linewidth=lwf,
            linestyle="dashed",
            label="Sxy cum. - " + fm_string + " fit",
        )
        axs5[2, 1].plot(
            s10_fit,
            "cyan",
            linewidth=lwf,
            linestyle="dashed",
            label="Syx cum. - " + fm_string + " fit",
        )
    axs5[2, 1].set_title("Cumulative Shear", fontsize=fs)
    yi11, ya11 = axs5[1, 1].get_ylim()
    dy1 = (ya11 - yi11) / 2.0
    yi21, ya21 = axs5[2, 1].get_ylim()
    if (ya21 - yi21) < 0.01 * dy1:
        axs5[2, 1].set_ylim((yi21 - dy1, ya21 + dy1))
    axs5[3, 1].plot(
        tr_matr_cum[:, 0, 1], "r", linewidth=lwl, label="Sxy cum. - residual"
    )
    axs5[3, 1].plot(
        tr_matr_cum[:, 1, 0], "b", linewidth=lwl, label="Syx cum. - residual"
    )
    axs5[3, 1].set_title("Residual Cumulative Shear", fontsize=fs)
    axs5[3, 1].set_xlabel("Frame", fontsize=fs + 1)
    yi31, ya31 = axs5[3, 1].get_ylim()
    if (ya31 - yi21) < 0.01 * dy1:
        axs5[3, 1].set_ylim((yi31 - dy1, ya31 + dy1))

    # plot shifts
    axs5[1, 2].plot(
        transformation_matrix[:, 0, 2], "r", linewidth=lwl, label="Tx fr.-to-fr."
    )
    axs5[1, 2].plot(
        transformation_matrix[:, 1, 2], "b", linewidth=lwl, label="Ty fr.-to-fr."
    )
    axs5[1, 2].set_title("Frame-to-Frame Shift", fontsize=fs)
    if preserve_scales:
        axs5[2, 2].plot(Xshift_cum_orig, "r", linewidth=lwl, label="Tx cum. - orig.")
        axs5[2, 2].plot(Yshift_cum_orig, "b", linewidth=lwl, label="Ty cum. - orig.")
        axs5[2, 2].plot(
            Xshift_cum,
            "r",
            linewidth=lwl,
            linestyle="dotted",
            label="Tx cum. - pres. scales",
        )
        axs5[2, 2].plot(
            Yshift_cum,
            "b",
            linewidth=lwl,
            linestyle="dotted",
            label="Ty cum. - pres. scales",
        )
    else:
        axs5[2, 2].plot(
            Xshift_cum, "r", linewidth=lwl, linestyle="dotted", label="Tx cum."
        )
        axs5[2, 2].plot(
            Yshift_cum, "b", linewidth=lwl, linestyle="dotted", label="Ty cum."
        )
    if subtract_linear_fit[0]:
        axs5[2, 2].plot(
            Xfit,
            "orange",
            linewidth=lwf,
            linestyle="dashed",
            label="Tx cum. - lin. fit",
        )
    if subtract_linear_fit[1]:
        axs5[2, 2].plot(
            Yfit, "cyan", linewidth=lwf, linestyle="dashed", label="Ty cum. - lin. fit"
        )
    axs5[2, 2].set_title("Cumulative Shift", fontsize=fs)
    axs5[3, 2].plot(
        tr_matr_cum[:, 0, 2], "r", linewidth=lwl, label="Tx cum. - residual"
    )
    axs5[3, 2].plot(
        tr_matr_cum[:, 1, 2], "b", linewidth=lwl, label="Ty cum. - residual"
    )
    axs5[3, 2].set_title("Residual Cumulative Shift", fontsize=fs)
    axs5[3, 2].set_xlabel("Frame", fontsize=fs + 1)

    for ax in axs5.ravel()[1:]:
        ax.grid(True)
        ax.legend(fontsize=fs - 1)
    fn = os.path.join(data_dir, fnm_reg)
    fig5.suptitle(fn, fontsize=fs)
    if save_res_png:
        fig5.savefig(fn.replace(".mrc", "_Transform_Summary.png"), dpi=300)


def generate_report_from_xls_registration_summary(file_xlsx, **kwargs):
    """
    Generate Report Plot for FIB-SEM data set registration from xlxs workbook file. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com
    XLS file should have pages (sheets):
        - 'Registration Quality Statistics' - containing columns with the the evaluation box data and registration quality metrics data:
            'Frame', 'xi_eval', 'xa_eval', 'yi_eval', 'ya_eval', 'Npts', 'Mean Abs Error', 'Image NSAD', 'Image NCC', 'Image MI'
        - 'Stack Info' - containing the fields:
            'Stack Filename' and 'data_dir'
        - 'SIFT kwargs' (optional) - containg the kwargs with SIFT registration parameters.

    Parameters:
    xlsx_fname : str
        full path to the XLSX workbook file

    kwargs
    ---------
    sample_frame_files : list
        List of paths to sample frame images
    png_file : str
        filename to save the results. Default is file_xlsx with extension '.xlsx' replaced with '.png'
    invert_data : bolean
        If True, the representative data frames will use inverse LUT.
    dump_filename : str
        Filename of a binary dump of the FIBSEM_dataset object.

    """
    xlsx_name = os.path.basename(os.path.abspath(file_xlsx))
    base_dir = os.path.dirname(os.path.abspath(file_xlsx))
    sample_frame_mask = xlsx_name.replace(
        "_RegistrationQuality.xlsx", "_sample_image_frame*.*"
    )
    unsorted_sample_frame_files = glob.glob(os.path.join(base_dir, sample_frame_mask))
    try:
        unsorter_frames = [
            int(x.split("frame")[1].split(".png")[0])
            for x in unsorted_sample_frame_files
        ]
        sorted_inds = argsort(unsorter_frames)
        existing_sample_frame_files = [
            unsorted_sample_frame_files[i] for i in sorted_inds
        ]
    except:
        existing_sample_frame_files = unsorted_sample_frame_files
    sample_frame_files = kwargs.get("sample_frame_files", existing_sample_frame_files)
    png_file_default = file_xlsx.replace(".xlsx", ".png")
    png_file = kwargs.get("png_file", png_file_default)
    dump_filename = kwargs.get("dump_filename", "")

    Regisration_data = pd.read_excel(
        file_xlsx, sheet_name="Registration Quality Statistics"
    )
    # columns=['Frame', 'xi_eval', 'xa_eval', 'yi_eval', 'ya_eval', 'Npts', 'Mean Abs Error', 'Image NSAD', 'Image NCC', 'Image MI']
    frames = Regisration_data["Frame"]
    xi_evals = Regisration_data["xi_eval"]
    xa_evals = Regisration_data["xa_eval"]
    yi_evals = Regisration_data["yi_eval"]
    ya_evals = Regisration_data["ya_eval"]

    """
    image_nsad = Regisration_data['NSAD']
    image_ncc = Regisration_data['NCC']
    image_nmi = Regisration_data['NMI']
    nsads = [np.mean(image_nsad), np.median(image_nsad), np.std(image_nsad)]
    nccs = [np.mean(image_ncc), np.median(image_ncc), np.std(image_ncc)]
    nmis = [np.mean(image_nmi), np.median(image_nmi), np.std(image_nmi)]
    """
    eval_metrics = Regisration_data.columns[5:]
    num_metrics = len(eval_metrics)

    num_frames = len(frames)

    stack_info_dict = read_kwargs_xlsx(file_xlsx, "Stack Info", **kwargs)
    if "dump_filename" in stack_info_dict.keys():
        dump_filename = kwargs.get("dump_filename", stack_info_dict["dump_filename"])
    else:
        dump_filename = kwargs.get("dump_filename", "")
    try:
        if np.isnan(dump_filename):
            dump_filename = ""
    except:
        pass
    stack_info_dict["dump_filename"] = dump_filename

    try:
        invert_data = kwargs.get("invert_data", stack_info_dict["invert_data"])
    except:
        invert_data = kwargs.get("invert_data", False)

    default_stack_name = file_xlsx.replace("_RegistrationQuality.xlsx", ".mrc")
    stack_filename = os.path.normpath(
        stack_info_dict.get("Stack Filename", default_stack_name)
    )
    data_dir = stack_info_dict.get("data_dir", "")
    ftype = stack_info_dict.get("ftype", 0)

    heights = [0.8] * 3 + [1.5] * num_metrics
    gs_kw = dict(height_ratios=heights)
    fig, axs = subplots(
        (num_metrics + 3), 1, figsize=(6, 2 * (num_metrics + 2)), gridspec_kw=gs_kw
    )
    fig.subplots_adjust(
        left=0.14, bottom=0.04, right=0.99, top=0.98, wspace=0.18, hspace=0.04
    )
    for ax in axs[0:3]:
        ax.axis("off")

    fs = 12
    lwl = 1

    if len(sample_frame_files) > 0:
        sample_frame_images_available = True
        for jf, ax in enumerate(axs[0:3]):
            try:
                ax.imshow(mpimg.imread(sample_frame_files[jf]))
                ax.axis(False)
            except:
                pass
    else:
        sample_frame_images_available = False
        sample_data_available = True
        if stack_exists:
            print("Will use sample images from the registered stack")
            use_raw_data = False
            if Path(stack_filename).suffix == ".mrc":
                mrc_obj = mrcfile.mmap(stack_filename, mode="r")
                header = mrc_obj.header
                mrc_mode = header.mode
                """
                mode 0 -> uint8
                mode 1 -> int16
                mode 2 -> float32
                mode 4 -> complex64
                mode 6 -> uint16
                """
                if mrc_mode == 0:
                    dt_mrc = uint8
                if mrc_mode == 1:
                    dt_mrc = int16
                if mrc_mode == 2:
                    dt_mrc = float32
                if mrc_mode == 4:
                    dt_mrc = complex64
                if mrc_mode == 6:
                    dt_mrc = uint16
        else:
            print("Will use sample images from the raw data")
            if os.path.exists(dump_filename):
                print("Trying to recall the data from ", dump_filename)
            try:
                print("Looking for the raw data in the directory", data_dir)
                if ftype == 0:
                    fls = sorted(glob.glob(os.path.join(data_dir, "*.dat")))
                    if len(fls) < 1:
                        fls = sorted(glob.glob(os.path.join(data_dir, "*/*.dat")))
                if ftype == 1:
                    fls = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
                    if len(fls) < 1:
                        fls = sorted(glob.glob(os.path.join(data_dir, "*/*.tif")))
                num_frames = len(fls)
                stack_info_dict["disp_res"] = False
                raw_dataset = FIBSEM_dataset(
                    fls,
                    recall_parameters=os.path.exists(dump_filename),
                    **stack_info_dict
                )
                XResolution = raw_dataset.XResolution
                YResolution = raw_dataset.YResolution
                if pad_edges and perfrom_transformation:
                    # shape = [test_frame.YResolution, test_frame.XResolution]
                    shape = [YResolution, XResolution]
                    xmn, xmx, ymn, ymx = determine_pad_offsets(
                        shape, raw_dataset.tr_matr_cum_residual
                    )
                    padx = int(xmx - xmn)
                    pady = int(ymx - ymn)
                    xi = int(np.max([xmx, 0]))
                    yi = int(np.max([ymx, 0]))
                    # The initial transformation matrices are calculated with no padding.Padding is done prior to transformation
                    # so that the transformed images are not clipped.
                    # Such padding means shift (by xi and yi values). Therefore the new transformation matrix
                    # for padded frames will be (Shift Matrix)x(Transformation Matrix)x(Inverse Shift Matrix)
                    # those are calculated below base on the amount of padding calculated above
                    shift_matrix = np.array(
                        [[1.0, 0.0, xi], [0.0, 1.0, yi], [0.0, 0.0, 1.0]]
                    )
                    inv_shift_matrix = np.linalg.inv(shift_matrix)
                else:
                    padx = 0
                    pady = 0
                    xi = 0
                    yi = 0
                    shift_matrix = np.eye(3, 3)
                    inv_shift_matrix = np.eye(3, 3)
                xsz = XResolution + padx
                xa = xi + XResolution
                ysz = YResolution + pady
                ya = yi + YResolution
                use_raw_data = True
            except:
                sample_data_available = False
                use_raw_data = False
        if sample_data_available:
            print("Sample data is available")
        else:
            print("Sample data is NOT available")

        if num_frames // 10 * 9 > 0:
            ev_ind2 = num_frames // 10 * 9
        else:
            ev_ind2 = num_frames - 1
        eval_inds = [num_frames // 10, num_frames // 2, ev_ind2]
        # print(eval_inds)

        for j, eval_ind in enumerate(eval_inds):
            ax = axs[j]
            if sample_data_available:
                if stack_exists:
                    if Path(stack_filename).suffix == ".mrc":
                        frame_img = (
                            mrc_obj.data[frames[eval_ind], :, :].astype(dt_mrc)
                        ).astype(float)
                    if Path(stack_filename).suffix == ".tif":
                        frame_img = tiff.imread(stack_filename, key=eval_ind)
                else:
                    dtp = float
                    chunk_frames = np.arange(
                        eval_ind, min(eval_ind + zbin_factor, len(fls) - 2)
                    )
                    frame_filenames = np.array(raw_dataset.fls)[chunk_frames]
                    tr_matrices = np.array(raw_dataset.tr_matr_cum_residual)[
                        chunk_frames
                    ]
                    frame_img = transform_chunk_of_frames(
                        frame_filenames,
                        xsz,
                        ysz,
                        ftype,
                        flatten_image,
                        image_correction_file,
                        perfrom_transformation,
                        tr_matrices,
                        shift_matrix,
                        inv_shift_matrix,
                        xi,
                        xa,
                        yi,
                        ya,
                        ImgB_fraction=0.0,
                        invert_data=False,
                        int_order=1,
                        flipY=raw_dataset.flipY,
                    )
                # print(eval_ind, np.shape(frame_img), yi_evals[eval_ind], ya_evals[eval_ind], xi_evals[eval_ind], xa_evals[eval_ind])
                if use_raw_data:
                    eval_ind = eval_ind // zbin_factor
                dmin, dmax = get_min_max_thresholds(
                    frame_img[
                        yi_evals[eval_ind] : ya_evals[eval_ind],
                        xi_evals[eval_ind] : xa_evals[eval_ind],
                    ]
                )
                if invert_data:
                    ax.imshow(frame_img, cmap="Greys_r", vmin=dmin, vmax=dmax)
                else:
                    ax.imshow(frame_img, cmap="Greys", vmin=dmin, vmax=dmax)

                ax.text(
                    0.03,
                    1.01,
                    "Frame={:d},  NSAD={:.3f},  NCC={:.3f},  NMI={:.3f}".format(
                        frames[eval_ind],
                        image_nsad[eval_ind],
                        image_ncc[eval_ind],
                        image_nmi[eval_ind],
                    ),
                    color="red",
                    transform=ax.transAxes,
                )
                rect_patch = patches.Rectangle(
                    (xi_evals[eval_ind], yi_evals[eval_ind]),
                    abs(xa_evals[eval_ind] - xi_evals[eval_ind]) - 2,
                    abs(ya_evals[eval_ind] - yi_evals[eval_ind]) - 2,
                    linewidth=1.0,
                    edgecolor="yellow",
                    facecolor="none",
                )
                ax.add_patch(rect_patch)
            ax.axis("off")

        if stack_exists:
            if Path(stack_filename).suffix == ".mrc":
                mrc_obj.close()

    axes_names = {
        "NSAD": "Norm. Sum of Abs. Diff",
        "NCC": "Norm. Cross-Corr.",
        "NMI": "Norm. Mutual Inf.",
        "FSC": "FSC BW (inv pix)",
    }
    colors = ["red", "blue", "green", "magenta", "lime"]

    for j, metric in enumerate(eval_metrics):
        metric_data = Regisration_data[metric]
        nmis = []
        axs[j + 3].plot(
            frames, Regisration_data[metric], linewidth=lwl, color=colors[j]
        )
        try:
            axs[j + 3].set_ylabel(axes_names[metric], fontsize=fs - 2)
        except:
            axs[j + 3].set_ylabel(metric, fontsize=fs - 2)
        axs[j + 3].text(
            0.02,
            0.04,
            (
                metric
                + " mean = {:.3f}   "
                + metric
                + " median = {:.3f}  "
                + metric
                + " STD = {:.3f}"
            ).format(np.mean(metric_data), np.median(metric_data), np.std(metric_data)),
            transform=axs[j + 3].transAxes,
            fontsize=fs - 4,
        )

    axs[-1].set_xlabel("Binned Frame #")
    for ax in axs[2:]:
        ax.grid(True)

    axs[0].text(-0.15, 2.7, stack_filename, transform=axs[3].transAxes)
    fig.savefig(png_file, dpi=300)


def plot_registrtion_quality_xlsx(data_files, labels, **kwargs):
    """
    Read and plot together multiple registration quality summaries.
    ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

    Parameters:
    data_files : array of str
        Filenames (full paths) of the registration summaries (*.xlsx files)
    labels : array of str
        Labels (for each registration)

    kwargs:
    frame_inds : array or list of int
        Array or list oif frame indecis to use to azalyze the data.
    save_res_png : boolean
        If True, the PNG's of summary plots as well as summary Excel notebook are saved
    save_filename : str
        Filename (full path) to save the results (default is data_dir +'Regstration_Summary.png')
    nsad_bounds : list of floats
        Bounds for NSAD plot (default is determined by get_min_max_thresholds with thresholds of 1e-4)
    ncc_bounds : list of floats
        Bounds for NCC plot (default is determined by get_min_max_thresholds with thresholds of 1e-4)
    nmi_bounds : list of floats
        Bounds for NMI plot (default is determined by get_min_max_thresholds with thresholds of 1e-4)
    colors : array or list of colors
        Optional colors for each plot/file. If not provided, will be auto-generated.
    linewidths : array of float
        linewidths for individual files. If not provided, all linewidts are set to 0.5

    Returns
    xlsx_fname : str
        Filename of the summary Excel notebook
    """
    save_res_png = kwargs.get("save_res_png", True)
    linewidths = kwargs.get("linewidths", np.ones(len(data_files)) * 0.5)
    data_dir = os.path.split(data_files[0])[0]
    default_save_filename = os.path.join(data_dir, "Regstration_Summary.png")
    save_filename = kwargs.get("save_filename", default_save_filename)
    nsad_bounds = kwargs.get("nsad_bounds", [0.0, 0.0])
    ncc_bounds = kwargs.get("ncc_bounds", [0.0, 0.0])
    nmi_bounds = kwargs.get("nmi_bounds", [0.0, 0.0])
    frame_inds = kwargs.get("frame_inds", [])

    nfls = len(data_files)
    reg_datas = []
    for data_file in data_files:
        # fl = os.path.join(data_dir, df)
        # data = pd.read_csv(fl)
        data = pd.read_excel(data_file, sheet_name="Registration Quality Statistics")
        reg_datas.append(data)

    lw0 = 0.5
    lw1 = 1

    fs = 12
    fs2 = 10
    fig1, axs1 = subplots(3, 1, figsize=(7, 11), sharex=True)
    fig1.subplots_adjust(
        left=0.1, bottom=0.05, right=0.99, top=0.96, wspace=0.2, hspace=0.1
    )

    ax_nsad = axs1[0]
    ax_ncc = axs1[1]
    ax_nmi = axs1[2]
    ax_nsad.set_ylabel("Normalized Sum of Abs. Differences", fontsize=fs)
    ax_ncc.set_ylabel("Normalized Cross-Correlation", fontsize=fs)
    ax_nmi.set_ylabel("Normalized Mutual Information", fontsize=fs)
    ax_nmi.set_xlabel("Frame", fontsize=fs)

    spreads = []
    my_cols = [get_cmap("gist_rainbow_r")((nfls - j) / (nfls)) for j in np.arange(nfls)]
    my_cols[0] = "grey"
    my_cols[-1] = "red"
    my_cols = kwargs.get("colors", my_cols)

    means = []
    image_nsads = []
    image_nccs = []
    image_snrs = []
    image_nmis = []
    frame_inds_glob = []
    for j, reg_data in enumerate(
        tqdm(reg_datas, desc="generating the registration quality summary plots")
    ):
        # my_col = get_cmap("gist_rainbow_r")((nfls-j)/(nfls))
        # my_cols.append(my_col)
        my_col = my_cols[j]
        pf = labels[j]
        lw0 = linewidths[j]
        if len(frame_inds) > 0:
            try:
                image_nsad = np.array(reg_data["Image NSAD"])[frame_inds]
                image_ncc = np.array(reg_data["Image NCC"])[frame_inds]
                image_nmi = np.array(reg_data["Image MI"])[frame_inds]
            except:
                image_nsad = np.array(reg_data["NSAD"])[frame_inds]
                image_ncc = np.array(reg_data["NCC"])[frame_inds]
                image_nmi = np.array(reg_data["NMI"])[frame_inds]
            frame_inds_loc = frame_inds.copy()
        else:
            try:
                image_nsad = np.array(reg_data["Image NSAD"])
                image_ncc = np.array(reg_data["Image NCC"])
                image_nmi = np.array(reg_data["Image MI"])
            except:
                image_nsad = np.array(reg_data["NSAD"])
                image_ncc = np.array(reg_data["NCC"])
                image_nmi = np.array(reg_data["NMI"])
            frame_inds_loc = np.arange(len(image_ncc))
        fr_i = min(frame_inds_loc) - (max(frame_inds_loc) - min(frame_inds_loc)) * 0.05
        fr_a = max(frame_inds_loc) + (max(frame_inds_loc) - min(frame_inds_loc)) * 0.05
        image_nsads.append(image_nsad)
        image_nccs.append(image_ncc)
        image_snr = image_ncc / (1.0 - image_ncc)
        image_snrs.append(image_snr)
        image_nmis.append(image_nmi)
        frame_inds_glob.append(frame_inds_loc)

        eval_metrics = [image_nsad, image_ncc, image_snr, image_nmi]
        spreads.append([get_spread(metr) for metr in eval_metrics])
        means.append([np.mean(metr) for metr in eval_metrics])

        ax_nsad.plot(frame_inds_loc, image_nsad, c=my_col, linewidth=lw0)
        ax_nsad.plot(image_nsad[0], c=my_col, linewidth=lw1, label=pf)
        ax_ncc.plot(frame_inds_loc, image_ncc, c=my_col, linewidth=lw0)
        ax_ncc.plot(image_ncc[0], c=my_col, linewidth=lw1, label=pf)
        ax_nmi.plot(frame_inds_loc, image_nmi, c=my_col, linewidth=lw0)
        ax_nmi.plot(image_nmi[0], c=my_col, linewidth=lw1, label=pf)

    for ax in axs1.ravel():
        ax.grid(True)
        ax.legend(fontsize=fs2)

    if nsad_bounds[0] == nsad_bounds[1]:
        nsad_min, nsad_max = get_min_max_thresholds(
            np.concatenate(image_nsads),
            thr_min=1e-4,
            thr_max=1e-4,
            nbins=256,
            disp_res=False,
        )
    else:
        nsad_min, nsad_max = nsad_bounds
    ax_nsad.set_ylim(nsad_min, nsad_max)

    if ncc_bounds[0] == ncc_bounds[1]:
        ncc_min, ncc_max = get_min_max_thresholds(
            np.concatenate(image_nccs),
            thr_min=1e-4,
            thr_max=1e-4,
            nbins=256,
            disp_res=False,
        )
    else:
        ncc_min, ncc_max = ncc_bounds
    ax_ncc.set_ylim(ncc_min, ncc_max)

    if nmi_bounds[0] == nmi_bounds[1]:
        nmi_min, nmi_max = get_min_max_thresholds(
            np.concatenate(image_nmis),
            thr_min=1e-4,
            thr_max=1e-4,
            nbins=256,
            disp_res=False,
        )
    else:
        nmi_min, nmi_max = nmi_bounds
    ax_nmi.set_ylim(nmi_min, nmi_max)
    ax_nmi.set_xlim(fr_i, fr_a)

    ax_nsad.text(-0.05, 1.05, data_dir, transform=ax_nsad.transAxes, fontsize=10)
    if save_res_png:
        fig1.savefig(save_filename, dpi=300)

    # Generate the Cell Text
    cell_text = []
    fig2_data = []
    limits = []
    rows = labels
    fst = 9

    for j, (mean, spread) in enumerate(zip(means, spreads)):
        cell_text.append(
            [
                "{:.4f}".format(mean[0]),
                "{:.4f}".format(spread[0]),
                "{:.4f}".format(mean[1]),
                "{:.4f}".format(spread[1]),
                "{:.4f}".format(mean[2]),
                "{:.4f}".format(mean[3]),
                "{:.4f}".format(spread[3]),
            ]
        )
        fig2_data.append(
            [mean[0], spread[0], mean[1], spread[1], mean[2], mean[3], spread[3]]
        )

    # Generate the table
    fig2, ax = subplots(1, 1, figsize=(9.5, 1.3))
    fig2.subplots_adjust(
        left=0.32, bottom=0.01, right=0.98, top=0.86, wspace=0.05, hspace=0.05
    )
    ax.axis(False)
    ax.text(-0.30, 1.07, "SIFT Registration Comparisons:  " + data_dir, fontsize=fst)
    llw1 = 0.3
    clw = [llw1, llw1]

    columns = [
        "NSAD Mean",
        "NSAD Spread",
        "NCC Mean",
        "NCC Spread",
        "Mean SNR",
        "NMI Mean",
        "NMI Spread",
    ]

    n_cols = len(columns)
    n_rows = len(rows)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0, 0.995, 1.0],
        fontsize=16,
    )
    tbl.auto_set_column_width(col=3)

    table_props = tbl.properties()
    try:
        table_cells = table_props["child_artists"]
    except:
        table_cells = table_props["children"]

    tbl.auto_set_font_size(False)
    for j, cell in enumerate(table_cells[0 : n_cols * n_rows]):
        cell.get_text().set_color(my_cols[j // n_cols])
        cell.get_text().set_fontsize(fst)
    for j, cell in enumerate(table_cells[n_cols * (n_rows + 1) :]):
        cell.get_text().set_color(my_cols[j])
    for cell in table_cells[n_cols * n_rows :]:
        #    cell.get_text().set_fontweight('bold')
        cell.get_text().set_fontsize(fst)
    save_filename2 = save_filename.replace(".png", "_table.png")
    if save_res_png:
        fig2.savefig(save_filename2, dpi=300)

    ysize_fig = 4
    ysize_tbl = 0.25 * nfls
    fst3 = 8
    fig3, axs3 = subplots(
        2,
        1,
        figsize=(7, ysize_fig + ysize_tbl),
        gridspec_kw={"height_ratios": [ysize_tbl, ysize_fig]},
    )
    fig3.subplots_adjust(
        left=0.10, bottom=0.10, right=0.98, top=0.96, wspace=0.05, hspace=0.05
    )

    for j, reg_data in enumerate(reg_datas):
        my_col = my_cols[j]
        pf = labels[j]
        lw0 = linewidths[j]
        if len(frame_inds) > 0:
            try:
                image_ncc = np.array(reg_data["Image NCC"])[frame_inds]
            except:
                image_ncc = np.array(reg_data["NCC"])[frame_inds]
            frame_inds_loc = frame_inds.copy()
        else:
            try:
                image_ncc = np.array(reg_data["Image NCC"])
            except:
                image_ncc = np.array(reg_data["NCC"])
            frame_inds_loc = np.arange(len(image_ncc))

        axs3[1].plot(frame_inds_loc, image_ncc, c=my_col, linewidth=lw0)
        axs3[1].plot(image_ncc[0], c=my_col, linewidth=lw1, label=pf)
    axs3[1].grid(True)
    axs3[1].legend(fontsize=fs2)
    axs3[1].set_ylabel("Normalized Cross-Correlation", fontsize=fs)
    axs3[1].set_xlabel("Frame", fontsize=fs)
    axs3[1].set_ylim(ncc_min, ncc_max)
    axs3[1].set_xlim(fr_i, fr_a)
    axs3[0].axis(False)
    axs3[0].text(
        -0.1, 1.07, "SIFT Registration Comparisons:  " + data_dir, fontsize=fst3
    )
    llw1 = 0.3
    clw = [llw1, llw1]

    columns2 = ["NCC Mean", "NCC Spread", "SNR Mean", "SNR Spread"]
    cell2_text = []
    fig3_data = []
    limits = []
    rows = labels

    for j, (mean, spread) in enumerate(zip(means, spreads)):
        cell2_text.append(
            [
                "{:.4f}".format(mean[1]),
                "{:.4f}".format(spread[1]),
                "{:.4f}".format(mean[2]),
                "{:.4f}".format(spread[2]),
            ]
        )
    n_cols = len(columns2)
    n_rows = len(rows)

    tbl2 = axs3[0].table(
        cellText=cell2_text,
        rowLabels=rows,
        colLabels=columns2,
        cellLoc="center",
        colLoc="center",
        bbox=[0.38, 0, 0.55, 1.0],  # (left, bottom, width, height)
        fontsize=16,
    )
    rl = max([len(pf) for pf in labels])
    # tbl2.auto_set_column_width(col=[rl+5, len(columns2[0]), len(columns2[1]), len(columns2[2]), len(columns2[3])])
    tbl2.auto_set_column_width(col=list(range(len(columns2) + 1)))
    tbl2.auto_set_font_size(False)

    table2_props = tbl2.properties()
    try:
        table2_cells = table2_props["child_artists"]
    except:
        table2_cells = table2_props["children"]

    tbl.auto_set_font_size(False)
    for j, cell in enumerate(table2_cells[0 : n_cols * n_rows]):
        cell.get_text().set_color(my_cols[j // n_cols])
        cell.get_text().set_fontsize(fst3)
    for j, cell in enumerate(table2_cells[n_cols * (n_rows + 1) :]):
        cell.get_text().set_color(my_cols[j])
    for cell in table2_cells[n_cols * n_rows :]:
        #    cell.get_text().set_fontweight('bold')
        cell.get_text().set_fontsize(fst3)
    save_filename3 = save_filename.replace(".png", "_fig_and_table.png")
    if save_res_png:
        fig3.savefig(save_filename3, dpi=300)

    if save_res_png:
        # Generate a single multi-page CSV file
        xlsx_fname = save_filename.replace(".png", ".xlsx")
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(xlsx_fname, engine="xlsxwriter")
        fig2_df = pd.DataFrame(fig2_data, columns=columns)
        fig2_df.insert(0, "", labels)
        fig2_df.insert(1, "Path", data_files)
        fig2_df.to_excel(writer, index=None, sheet_name="Summary")
        for reg_data, label in zip(
            tqdm(reg_datas, desc="saving the data into xlsx file"), labels
        ):
            data_fn = label[0:31]
            reg_data.to_excel(writer, sheet_name=data_fn)
        writer.save()
    return xlsx_fname
