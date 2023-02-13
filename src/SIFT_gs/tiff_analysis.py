"""
# TIF stack analysis functions
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


def show_eval_box_tif_stack(tif_filename, **kwargs):
    """
    Read tif stack and display the eval box for each frame from the list.
    ©G.Shtengel, 08/2022. gleb.shtengel@gmail.com

    Parameters
    ---------
    tif_filename : str
        File name (full path) of the tif stack to be analyzed

    kwargs:
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    ax : matplotlib ax artist
        if provided, the data is exported to external ax object.
    frame_inds : array
        List of frame indices to display the evaluation box. If not provided, three frames will be used:
        [nz//10,  nz//2, nz//10*9] where nz is number of frames in tif stack
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    sliding_evaluation_box : boolean
        if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above
    invert_data : Boolean
    """
    Sample_ID = kwargs.get("Sample_ID", "")
    save_res_png = kwargs.get("save_res_png", True)
    save_filename = kwargs.get("save_filename", tif_filename)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    invert_data = kwargs.get("invert_data", False)
    ax = kwargs.get("ax", "")
    plot_internal = ax == ""

    with tiff.TiffFile(tif_filename) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
    try:
        shape = eval(tif_tags["ImageDescription"])
        nz, ny, nx = shape["shape"]
    except:
        try:
            shape = eval(tif_tags["image_description"])
            nz, ny, nx = shape["shape"]
        except:
            fr0 = tiff.imread(tif_filename, key=0)
            ny, nx = np.shape(fr0)
            nz = eval(tif_tags["nimages"])

    frame_inds = kwargs.get("frame_inds", [nz // 10, nz // 2, nz // 10 * 9])

    xi_eval = evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = nx
    yi_eval = evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ny

    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2] - start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0] - start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0

    for fr_ind in frame_inds:
        # eval_frame = (tif.data[fr_ind, :, :].astype(dt)).astype(float)
        eval_frame = tiff.imread(tif_filename, key=fr_ind).astype(float)

        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval * fr_ind // nz
            yi_eval = start_evaluation_box[0] + dy_eval * fr_ind // nz
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny

        if plot_internal:
            fig, ax = subplots(1, 1, figsize=(10.0, 11.0 * ny / nx))
        dmin, dmax = get_min_max_thresholds(
            eval_frame[yi_eval:ya_eval, xi_eval:xa_eval], disp_res=False
        )
        if invert_data:
            ax.imshow(eval_frame, cmap="Greys_r", vmin=dmin, vmax=dmax)
        else:
            ax.imshow(eval_frame, cmap="Greys", vmin=dmin, vmax=dmax)
        ax.grid(True, color="cyan")
        ax.set_title(Sample_ID + " " + tif_filename + ",  frame={:d}".format(fr_ind))
        rect_patch = patches.Rectangle(
            (xi_eval, yi_eval),
            abs(xa_eval - xi_eval) - 2,
            abs(ya_eval - yi_eval) - 2,
            linewidth=1.0,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(rect_patch)
        if save_res_png and plot_internal:
            fname = os.path.splitext(save_filename)[
                0
            ] + "_frame_{:d}_evaluation_box.png".format(fr_ind)
            fig.savefig(fname, dpi=300)


def evaluate_registration_two_frames_tif(params_tif):
    """
    Helper function used by DASK routine. Analyzes registration between two frames.
    ©G.Shtengel, 08/2022. gleb.shtengel@gmail.com

    Parameters:
    params_tif : list of tif_filename, fr, evals
    tif_filename  : string
        full path to tif filename
    fr : int
        Index of the SECOND frame
    evals :  list of image bounds to be used for evaluation exi_eval, xa_eval, yi_eval, ya_eval, save_frame_png, filename_frame_png


    Returns:
    image_nsad, image_ncc, image_mi   : float, float, float

    """
    (
        tif_filename,
        fr,
        invert_data,
        evals,
        save_frame_png,
        filename_frame_png,
    ) = params_tif
    xi_eval, xa_eval, yi_eval, ya_eval = evals

    frame0 = tiff.imread(tif_filename, key=int(fr - 1)).astype(float)
    frame1 = tiff.imread(tif_filename, key=int(fr)).astype(float)

    if invert_data:
        prev_frame = -1.0 * frame0[yi_eval:ya_eval, xi_eval:xa_eval]
        curr_frame = -1.0 * frame1[yi_eval:ya_eval, xi_eval:xa_eval]
    else:
        prev_frame = frame0[yi_eval:ya_eval, xi_eval:xa_eval]
        curr_frame = frame1[yi_eval:ya_eval, xi_eval:xa_eval]
    fr_mean = np.abs(curr_frame / 2.0 + prev_frame / 2.0)
    # image_nsad =  np.mean(np.abs(curr_frame-prev_frame))/(np.mean(fr_mean)-np.amin(fr_mean))
    image_nsad = np.mean(np.abs(curr_frame - prev_frame)) / (
        np.mean(fr_mean) - np.amin(fr_mean)
    )
    image_ncc = Two_Image_NCC_SNR(curr_frame, prev_frame)[0]
    image_mi = mutual_information_2d(
        prev_frame.ravel(), curr_frame.ravel(), sigma=1.0, bin=2048, normalized=True
    )
    if save_frame_png:
        yshape, xshape = frame0.shape
        fig, ax = subplots(1, 1, figsize=(3.0 * xshape / yshape, 3))
        fig.subplots_adjust(left=0.0, bottom=0.00, right=1.0, top=1.0)
        dmin, dmax = get_min_max_thresholds(frame0[yi_eval:ya_eval, xi_eval:xa_eval])
        if invert_data:
            ax.imshow(frame0, cmap="Greys_r", vmin=dmin, vmax=dmax)
        else:
            ax.imshow(frame0, cmap="Greys", vmin=dmin, vmax=dmax)
        ax.text(
            0.06,
            0.95,
            "Frame={:d},  NSAD={:.3f},  NCC={:.3f},  NMI={:.3f}".format(
                fr, image_nsad, image_ncc, image_mi
            ),
            color="red",
            transform=ax.transAxes,
            fontsize=12,
        )
        rect_patch = patches.Rectangle(
            (xi_eval, yi_eval),
            abs(xa_eval - xi_eval) - 2,
            abs(ya_eval - yi_eval) - 2,
            linewidth=1.0,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(rect_patch)
        ax.axis("off")
        fig.savefig(
            filename_frame_png,
            dpi=300,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )  # save the figure to file
        plt.close(fig)

    return image_nsad, image_ncc, image_mi


def analyze_tif_stack_registration(tif_filename, DASK_client, **kwargs):
    """
    Read MRC stack and analyze registration - calculate NSAD, NCC, and MI.
    ©G.Shtengel, 08/2022. gleb.shtengel@gmail.com

    Parameters
    ---------
    tif_filename : str
        File name (full path) of the mrc stack to be analyzed
    DASK client (needs to be initialized and running by this time)

    kwargs:
    use_DASK : boolean
        use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
    DASK_client_retries : int (default to 0)
        Number of allowed automatic retries if a task fails
    frame_inds : array
        Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames
    invert_data : boolean
        If True, the data will be inverted
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    sliding_evaluation_box : boolean
        if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    save_filename : str
        Path to the filename to save the results. If empty, tif_filename+'_RegistrationQuality.csv' will be used

    Returns reg_summary : PD data frame, registration_summary_xlsx : path to summary XLSX workbook
    """
    Sample_ID = kwargs.get("Sample_ID", "")
    use_DASK = kwargs.get("use_DASK", False)
    DASK_client_retries = kwargs.get("DASK_client_retries", 0)
    invert_data = kwargs.get("invert_data", False)
    save_res_png = kwargs.get("save_res_png", True)
    save_filename = kwargs.get("save_filename", tif_filename)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    registration_summary_xlsx = save_filename.replace(
        ".mrc", "_RegistrationQuality.xlsx"
    )

    if sliding_evaluation_box:
        print("Will use sliding (linearly) evaluation box")
        print("   Starting with box:  ", start_evaluation_box)
        print("   Finishing with box: ", stop_evaluation_box)
    else:
        print("Will use fixed evaluation box: ", evaluation_box)

    with tiff.TiffFile(tif_filename) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
    # print(tif_tags)
    try:
        shape = eval(tif_tags["ImageDescription"])
        nz, ny, nx = shape["shape"]
    except:
        try:
            shape = eval(tif_tags["image_description"])
            nz, ny, nx = shape["shape"]
        except:
            fr0 = tiff.imread(tif_filename, key=0)
            ny, nx = np.shape(fr0)
            nz = eval(tif_tags["nimages"])
    header_dict = {"nx": nx, "ny": ny, "nz": nz}

    xi_eval = evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = nx
    yi_eval = evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ny
    evals = [xi_eval, xa_eval, yi_eval, ya_eval]

    frame_inds_default = np.arange(nz - 1) + 1
    frame_inds = np.array(kwargs.get("frame_inds", frame_inds_default))
    nf = frame_inds[-1] - frame_inds[0] + 1
    if frame_inds[0] == 0:
        frame_inds = frame_inds + 1
    sample_frame_inds = [
        frame_inds[nf // 10],
        frame_inds[nf // 2],
        frame_inds[nf // 10 * 9],
    ]
    print("Will analyze regstrations in {:d} frames".format(len(frame_inds)))
    print("Will save the data into " + registration_summary_xlsx)
    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2] - start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0] - start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0

    params_tif_mult = []
    xi_evals = np.zeros(nf, dtype=int16)
    xa_evals = np.zeros(nf, dtype=int16)
    yi_evals = np.zeros(nf, dtype=int16)
    ya_evals = np.zeros(nf, dtype=int16)
    for j, fr in enumerate(frame_inds):
        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval * (fr - frame_inds[0]) // nf
            yi_eval = start_evaluation_box[0] + dy_eval * (fr - frame_inds[0]) // nf
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny
            evals = [xi_eval, xa_eval, yi_eval, ya_eval]
        xi_evals[j] = xi_eval
        xa_evals[j] = xa_eval
        yi_evals[j] = yi_eval
        ya_evals[j] = ya_eval
        if fr in sample_frame_inds:
            save_frame_png = save_sample_frames_png
            filename_frame_png = os.path.splitext(save_filename)[
                0
            ] + "_sample_image_frame{:d}.png".format(fr)
        else:
            save_frame_png = False
            filename_frame_png = (
                os.path.splitext(save_filename)[0] + "_sample_image_frame.png"
            )
        params_tif_mult.append(
            [tif_filename, fr, invert_data, evals, save_frame_png, filename_frame_png]
        )

    if use_DASK:
        print("Using DASK distributed")
        futures = DASK_client.map(
            evaluate_registration_two_frames_tif,
            params_tif_mult,
            retries=DASK_client_retries,
        )
        dask_results = DASK_client.gather(futures)
        image_nsad = np.array([res[0] for res in dask_results])
        image_ncc = np.array([res[1] for res in dask_results])
        image_mi = np.array([res[2] for res in dask_results])
    else:
        print("Using Local Computation")
        image_nsad = np.zeros((nf), dtype=float)
        image_ncc = np.zeros((nf), dtype=float)
        image_mi = np.zeros((nf), dtype=float)
        results = []
        for params_tif_mult_pair in tqdm(
            params_tif_mult, desc="Evaluating frame registration: "
        ):
            print(params_tif_mult_pair)
            [tif_filename, fr, invert_data, evals] = params_tif_mult_pair
            print(fr)
            results.append(evaluate_registration_two_frames_tif(params_tif_mult_pair))
        image_nsad = np.array([res[0] for res in results])
        image_ncc = np.array([res[1] for res in results])
        image_mi = np.array([res[2] for res in results])

    nsads = [np.mean(image_nsad), np.median(image_nsad), np.std(image_nsad)]
    # image_ncc = image_ncc[1:-1]
    nccs = [np.mean(image_ncc), np.median(image_ncc), np.std(image_ncc)]
    nmis = [np.mean(image_mi), np.median(image_mi), np.std(image_mi)]

    print(
        "Saving the Registration Quality Statistics into the file: ",
        registration_summary_xlsx,
    )
    xlsx_writer = pd.ExcelWriter(registration_summary_xlsx, engine="xlsxwriter")
    columns = [
        "Frame",
        "xi_eval",
        "xa_eval",
        "yi_eval",
        "ya_eval",
        "Image NSAD",
        "Image NCC",
        "Image MI",
    ]
    reg_summary = pd.DataFrame(
        np.vstack(
            (
                frame_inds,
                xi_evals,
                xa_evals,
                yi_evals,
                ya_evals,
                image_nsad,
                image_ncc,
                image_mi,
            )
        ).T,
        columns=columns,
        index=None,
    )
    reg_summary.to_excel(
        xlsx_writer, index=None, sheet_name="Registration Quality Statistics"
    )
    Stack_info = pd.DataFrame(
        [
            {
                "Stack Filename": tif_filename,
                "Sample_ID": Sample_ID,
                "invert_data": invert_data,
            }
        ]
    ).T  # prepare to be save in transposed format
    header_info = pd.DataFrame([header_dict]).T
    Stack_info = Stack_info.append(header_info)
    Stack_info.to_excel(xlsx_writer, header=False, sheet_name="Stack Info")
    xlsx_writer.save()

    return reg_summary, registration_summary_xlsx
