"""
# MRC stack analysis functions
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


def evaluate_registration_two_frames(params_mrc):
    """
    Helper function used by DASK routine. Analyzes registration between two frames.
    ©G.Shtengel, 10/2020. gleb.shtengel@gmail.com

    Parameters:
    params_mrc : list of mrc_filename, fr, evals, save_frame_png, filename_frame_png
    mrc_filename  : string
        full path to mrc filename
    fr : int
        Index of the SECOND frame
    evals :  list of image bounds to be used for evaluation exi_eval, xa_eval, yi_eval, ya_eval


    Returns:
    image_nsad, image_ncc, image_mi   : float, float, float

    """
    (
        mrc_filename,
        fr,
        invert_data,
        evals,
        save_frame_png,
        filename_frame_png,
    ) = params_mrc
    mrc_obj = mrcfile.mmap(mrc_filename, mode="r")
    header = mrc_obj.header
    """
    mode 0 -> uint8
    mode 1 -> int16
    mode 2 -> float32
    mode 4 -> complex64
    mode 6 -> uint16
    """
    mrc_mode = mrc_obj.header.mode
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

    xi_eval, xa_eval, yi_eval, ya_eval = evals
    if invert_data:
        prev_frame = -1.0 * (
            (
                (mrc_obj.data[fr - 1, yi_eval:ya_eval, xi_eval:xa_eval]).astype(dt_mrc)
            ).astype(float)
        )
        curr_frame = -1.0 * (
            (
                (mrc_obj.data[fr, yi_eval:ya_eval, xi_eval:xa_eval]).astype(dt_mrc)
            ).astype(float)
        )
    else:
        prev_frame = (
            mrc_obj.data[fr - 1, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt_mrc)
        ).astype(float)
        curr_frame = (
            mrc_obj.data[fr, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt_mrc)
        ).astype(float)
    fr_mean = np.abs(curr_frame / 2.0 + prev_frame / 2.0)
    image_nsad = np.mean(np.abs(curr_frame - prev_frame)) / (
        np.mean(fr_mean) - np.amin(fr_mean)
    )
    # image_nsad =  np.mean(np.abs(curr_frame-prev_frame))/(np.mean(fr_mean))
    image_ncc = Two_Image_NCC_SNR(curr_frame, prev_frame)[0]
    image_mi = mutual_information_2d(
        prev_frame.ravel(), curr_frame.ravel(), sigma=1.0, bin=2048, normalized=True
    )

    if save_frame_png:
        fr_img = (mrc_obj.data[fr, :, :].astype(dt_mrc)).astype(float)
        yshape, xshape = fr_img.shape
        fig, ax = subplots(1, 1, figsize=(3.0 * xshape / yshape, 3))
        fig.subplots_adjust(left=0.0, bottom=0.00, right=1.0, top=1.0)
        dmin, dmax = get_min_max_thresholds(fr_img[yi_eval:ya_eval, xi_eval:xa_eval])
        if invert_data:
            ax.imshow(fr_img, cmap="Greys_r", vmin=dmin, vmax=dmax)
        else:
            ax.imshow(fr_img, cmap="Greys", vmin=dmin, vmax=dmax)
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

    mrc_obj.close()
    return image_nsad, image_ncc, image_mi


def analyze_mrc_stack_registration(mrc_filename, DASK_client, **kwargs):
    """
    Read MRC stack and analyze registration - calculate NSAD, NCC, and MI.
    ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

    Parameters
    ---------
    mrc_filename : str
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
        Path to the filename to save the results. If empty, mrc_filename+'_RegistrationQuality.csv' will be used
    save_sample_frames_png : bolean
        If True, sample frames with superimposed eval box and registration analysis data will be saved into png files. Default is True

    Returns reg_summary : PD data frame, registration_summary_xlsx : path to summary XLSX workbook
    """
    Sample_ID = kwargs.get("Sample_ID", "")
    use_DASK = kwargs.get("use_DASK", False)
    DASK_client_retries = kwargs.get("DASK_client_retries", 0)
    invert_data = kwargs.get("invert_data", False)
    save_res_png = kwargs.get("save_res_png", True)
    save_filename = kwargs.get("save_filename", mrc_filename)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    registration_summary_xlsx = save_filename.replace(
        ".mrc", "_RegistrationQuality.xlsx"
    )
    save_sample_frames_png = kwargs.get("save_sample_frames_png", True)

    if sliding_evaluation_box:
        print("Will use sliding (linearly) evaluation box")
        print("   Starting with box:  ", start_evaluation_box)
        print("   Finishing with box: ", stop_evaluation_box)
    else:
        print("Will use fixed evaluation box: ", evaluation_box)

    mrc_obj = mrcfile.mmap(mrc_filename, mode="r")
    header = mrc_obj.header
    mrc_mode = header.mode
    nx, ny, nz = int32(header["nx"]), int32(header["ny"]), int32(header["nz"])
    header_dict = {}
    for record in header.dtype.names:  # create dictionary from the header data
        if ("extra" not in record) and ("label" not in record):
            header_dict[record] = header[record]
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
    print("mrc_mode={:d} ".format(mrc_mode), ", dt_mrc=", dt_mrc)

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

    params_mrc_mult = []
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
        params_mrc_mult.append(
            [mrc_filename, fr, invert_data, evals, save_frame_png, filename_frame_png]
        )

    if use_DASK:
        mrc_obj.close()
        print("Using DASK distributed")
        futures = DASK_client.map(
            evaluate_registration_two_frames,
            params_mrc_mult,
            retries=DASK_client_retries,
        )
        dask_results = DASK_client.gather(futures)
        image_nsad = np.array([res[0] for res in dask_results])
        image_ncc = np.array([res[1] for res in dask_results])
        image_mi = np.array([res[2] for res in dask_results])
    else:
        print("Using Local Computation")
        image_nsad = np.zeros(nf, dtype=float)
        image_ncc = np.zeros(nf, dtype=float)
        image_mi = np.zeros(nf, dtype=float)
        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval * frame_inds[0] // nf
            yi_eval = start_evaluation_box[0] + dy_eval * frame_inds[0] // nf
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny
        if invert_data:
            prev_frame = -1.0 * (
                (
                    mrc_obj.data[
                        frame_inds[0] - 1, yi_eval:ya_eval, xi_eval:xa_eval
                    ].astype(dt_mrc)
                ).astype(float)
            )
        else:
            prev_frame = (
                mrc_obj.data[
                    frame_inds[0] - 1, yi_eval:ya_eval, xi_eval:xa_eval
                ].astype(dt_mrc)
            ).astype(float)
        for j, frame_ind in enumerate(
            tqdm(frame_inds, desc="Evaluating frame registration: ")
        ):
            if sliding_evaluation_box:
                xi_eval = start_evaluation_box[2] + dx_eval * j // nf
                yi_eval = start_evaluation_box[0] + dy_eval * j // nf
                if start_evaluation_box[3] > 0:
                    xa_eval = xi_eval + start_evaluation_box[3]
                else:
                    xa_eval = nx
                if start_evaluation_box[1] > 0:
                    ya_eval = yi_eval + start_evaluation_box[1]
                else:
                    ya_eval = ny

            if invert_data:
                curr_frame = -1.0 * (
                    (
                        mrc_obj.data[
                            frame_ind, yi_eval:ya_eval, xi_eval:xa_eval
                        ].astype(dt_mrc)
                    ).astype(float)
                )
            else:
                curr_frame = (
                    mrc_obj.data[frame_ind, yi_eval:ya_eval, xi_eval:xa_eval].astype(
                        dt_mrc
                    )
                ).astype(float)
            if use_cp:
                curr_frame_cp = cp.array(curr_frame)
                prev_frame_cp = cp.array(prev_frame)
                fr_mean = cp.abs(curr_frame_cp / 2.0 + prev_frame_cp / 2.0)
            else:
                fr_mean = abs(curr_frame / 2.0 + prev_frame / 2.0)

            image_ncc[j - 1] = Two_Image_NCC_SNR(curr_frame, prev_frame)[0]
            if use_cp:
                image_nsad[j - 1] = cp.asnumpy(
                    cp.mean(cp.abs(curr_frame_cp - prev_frame_cp))
                    / (cp.mean(fr_mean) - cp.amin(fr_mean))
                )
                image_mi[j - 1] = cp.asnumpy(
                    mutual_information_2d_cp(
                        prev_frame_cp.ravel(),
                        curr_frame_cp.ravel(),
                        sigma=1.0,
                        bin=2048,
                        normalized=True,
                    )
                )
            else:
                image_nsad[j - 1] = mean(abs(curr_frame - prev_frame)) / (
                    np.mean(fr_mean) - np.amin(fr_mean)
                )
                image_mi[j - 1] = mutual_information_2d(
                    prev_frame.ravel(),
                    curr_frame.ravel(),
                    sigma=1.0,
                    bin=2048,
                    normalized=True,
                )
            prev_frame = curr_frame.copy()
            del curr_frame_cp, prev_frame_cp
            if (frame_ind in sample_frame_inds) and save_sample_frames_png:
                filename_frame_png = os.path.splitext(save_filename)[
                    0
                ] + "_sample_image_frame{:d}.png".format(j)
                fr_img = (mrc_obj.data[frame_ind, :, :].astype(dt_mrc)).astype(float)
                yshape, xshape = fr_img.shape
                fig, ax = subplots(1, 1, figsize=(3.0 * xshape / yshape, 3))
                fig.subplots_adjust(left=0.0, bottom=0.00, right=1.0, top=1.0)
                dmin, dmax = get_min_max_thresholds(
                    fr_img[yi_eval:ya_eval, xi_eval:xa_eval]
                )
                if invert_data:
                    ax.imshow(fr_img, cmap="Greys_r", vmin=dmin, vmax=dmax)
                else:
                    ax.imshow(fr_img, cmap="Greys", vmin=dmin, vmax=dmax)
                ax.text(
                    0.06,
                    0.95,
                    "Frame={:d},  NSAD={:.3f},  NCC={:.3f},  NMI={:.3f}".format(
                        frame_ind, image_nsad[j - 1], image_ncc[j - 1], image_mi[j - 1]
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

        mrc_obj.close()

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
                "Stack Filename": mrc_filename,
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


def show_eval_box_mrc_stack(mrc_filename, **kwargs):
    """
    Read MRC stack and display the eval box for each frame from the list.
    ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

    Parameters
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed

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
        [nz//10,  nz//2, nz//10*9] where nz is number of frames in mrc stack
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
    save_filename = kwargs.get("save_filename", mrc_filename)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    invert_data = kwargs.get("invert_data", False)
    ax = kwargs.get("ax", "")
    plot_internal = ax == ""

    mrc = mrcfile.mmap(mrc_filename, mode="r")
    header = mrc.header
    nx, ny, nz = int32(header["nx"]), int32(header["ny"]), int32(header["nz"])
    """
        mode 0 -> uint8
        mode 1 -> int16
        mode 2 -> float32
        mode 4 -> complex64
        mode 6 -> uint16
    """
    mrc_mode = header.mode
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
        eval_frame = (mrc.data[fr_ind, :, :].astype(dt_mrc)).astype(float)

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
        ax.set_title(Sample_ID + " " + mrc_filename + ",  frame={:d}".format(fr_ind))
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

    mrc.close()


def bin_crop_mrc_stack(mrc_filename, **kwargs):
    """
    Bins and crops a 3D mrc stack along X-, Y-, or Z-directions and saves it into MRC or HDF5 format. ©G.Shtengel 08/2022 gleb.shtengel@gmail.com

    Parameters:
        mrc_filename : str
            name (full path) of the mrc file to be binned
    **kwargs:
        fnm_types : list of strings.
            File type(s) for output data. Options are: ['h5', 'mrc'].
            Defauls is ['mrc']. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        zbin_factor : int
            binning factor in z-direction
        xbin_factor : int
            binning factor in x-direction
        ybin_factor : int
            binning factor in y-direction
        mode  : str
            Binning mode. Default is 'mean', other option is 'sum'
        frmax : int
            Maximum frame to bin. If not present, the entire file is binned
        binned_copped_filename : str
            name (full path) of the mrc file to save the results into. If not present, the new file name is constructed from the original by adding "_zbinXX" at the end.
        xi : int
            left edge of the crop
        xa : int
            right edge of the crop
        yi : int
            top edge of the crop
        ya : int
            bottom edge of the crop
        fri : int
            start frame
        fra : int
            stop frame
    Returns:
        fnms_saved : list of str
            Names of the new (binned and cropped) data files.
    """
    fnm_types = kwargs.get("fnm_types", ["mrc"])
    xbin_factor = kwargs.get("xbin_factor", 1)  # binning factor in in x-direction
    ybin_factor = kwargs.get("ybin_factor", 1)  # binning factor in in y-direction
    zbin_factor = kwargs.get("zbin_factor", 1)  # binning factor in in z-direction

    mode = kwargs.get(
        "mode", "mean"
    )  # binning mode. Default is 'mean', other option is 'sum'
    mrc_obj = mrcfile.mmap(mrc_filename, mode="r", permissive=True)
    header = mrc_obj.header
    """
        mode 0 -> uint8
        mode 1 -> int16
        mode 2 -> float32
        mode 4 -> complex64
        mode 6 -> uint16
    """
    mrc_mode = mrc_obj.header.mode
    voxel_size_angstr = mrc_obj.voxel_size
    voxel_size_angstr_new = voxel_size_angstr.copy()
    voxel_size_angstr_new.x = voxel_size_angstr.x * xbin_factor
    voxel_size_angstr_new.y = voxel_size_angstr.y * ybin_factor
    voxel_size_angstr_new.z = voxel_size_angstr.z * zbin_factor
    voxel_size_new = voxel_size_angstr.copy()
    voxel_size_new.x = voxel_size_angstr_new.x / 10.0
    voxel_size_new.y = voxel_size_angstr_new.y / 10.0
    voxel_size_new.z = voxel_size_angstr_new.z / 10.0
    nx, ny, nz = int32(header["nx"]), int32(header["ny"]), int32(header["nz"])
    frmax = kwargs.get("frmax", nz)
    xi = kwargs.get("xi", 0)
    xa = kwargs.get("xa", nx)
    yi = kwargs.get("yi", 0)
    ya = kwargs.get("ya", ny)
    fri = kwargs.get("fri", 0)
    fra = kwargs.get("fra", nz)
    nx_binned = (xa - xi) // xbin_factor
    ny_binned = (ya - yi) // ybin_factor
    xa = xi + nx_binned * xbin_factor
    ya = yi + ny_binned * ybin_factor
    binned_copped_filename_default = (
        os.path.splitext(mrc_filename)[0] + "_binned_croped.mrc"
    )
    binned_copped_filename = kwargs.get(
        "binned_copped_filename", binned_copped_filename_default
    )
    binned_mrc_filename = os.path.splitext(binned_copped_filename)[0] + ".mrc"
    dt = type(mrc_obj.data[0, 0, 0])
    print("Source mrc_mode: {:d}, source data type:".format(mrc_mode), dt)
    print(
        "Source Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}".format(
            voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z
        )
    )
    if mode == "sum":
        mrc_mode = 1
        dt = int16
    print("Result mrc_mode: {:d}, source data type:".format(mrc_mode), dt)
    st_frames = np.arange(fri, fra, zbin_factor)
    print(
        "New Data Set Shape:  {:d} x {:d} x {:d}".format(
            nx_binned, ny_binned, len(st_frames)
        )
    )

    fnms_saved = []
    if "mrc" in fnm_types:
        fnms_saved.append(binned_mrc_filename)
        mrc_new = mrcfile.new_mmap(
            binned_mrc_filename,
            shape=(len(st_frames), ny_binned, nx_binned),
            mrc_mode=mrc_mode,
            overwrite=True,
        )
        mrc_new.voxel_size = voxel_size_angstr_new
        # mrc_new.header.cella = voxel_size_angstr_new
        print(
            "Result Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}".format(
                voxel_size_angstr_new.x,
                voxel_size_angstr_new.y,
                voxel_size_angstr_new.z,
            )
        )
        desc = "Saving the data stack into MRC file"

    if "h5" in fnm_types:
        binned_h5_filename = os.path.splitext(binned_mrc_filename)[0] + ".h5"
        try:
            os.remove(binned_h5_filename)
        except:
            pass
        fnms_saved.append(binned_h5_filename)
        bdv_writer = npy2bdv.BdvWriter(
            binned_h5_filename, nchannels=1, blockdim=((1, 256, 256),)
        )
        bdv_writer.append_view(
            stack=None,
            virtual_stack_dim=(len(st_frames), ny_binned, nx_binned),
            time=0,
            channel=0,
            voxel_size_xyz=(voxel_size_new.x, voxel_size_new.y, voxel_size_new.z),
            voxel_units="nm",
        )
        if "mrc" in fnm_types:
            desc = "Saving the data stack into MRC and H5 files"
        else:
            desc = "Saving the data stack into H5 file"

    for j, st_frame in enumerate(tqdm(st_frames, desc=desc)):
        # need to fix this
        if mode == "mean":
            zbinnd_fr = np.mean(
                mrc_obj.data[
                    st_frame : min(st_frame + zbin_factor, nz - 1), yi:ya, xi:xa
                ],
                axis=0,
            )
        else:
            zbinnd_fr = np.sum(
                mrc_obj.data[
                    st_frame : min(st_frame + zbin_factor, nz - 1), yi:ya, xi:xa
                ],
                axis=0,
            )
        if (xbin_factor > 1) or (ybin_factor > 1):
            if mode == "mean":
                zbinnd_fr = np.mean(
                    np.mean(
                        zbinnd_fr.reshape(
                            ny_binned, ybin_factor, nx_binned, xbin_factor
                        ),
                        axis=3,
                    ),
                    axis=1,
                )
            else:
                zbinnd_fr = np.sum(
                    np.sum(
                        zbinnd_fr.reshape(
                            ny_binned, ybin_factor, nx_binned, xbin_factor
                        ),
                        axis=3,
                    ),
                    axis=1,
                )
        if "mrc" in fnm_types:
            mrc_new.data[j, :, :] = zbinnd_fr.astype(dt)
        if "h5" in fnm_types:
            bdv_writer.append_plane(plane=zbinnd_fr, z=j, time=0, channel=0)

    if "mrc" in fnm_types:
        mrc_new.close()

    if "h5" in fnm_types:
        bdv_writer.write_xml()
        bdv_writer.close()

    mrc_obj.close()

    return fnms_saved


def bin_crop_frames(bin_crop_parameters):
    """
    Help function used by bin_crop_mrc_stack_DASK
    """
    import logging

    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.ERROR)
    (
        mrc_filename,
        save_filename,
        dtp,
        start_frame,
        stop_frame,
        xbin_factor,
        ybin_factor,
        zbin_factor,
        mode,
        xi,
        xa,
        yi,
        ya,
    ) = bin_crop_parameters
    mrc_obj = mrcfile.mmap(mrc_filename, mode="r", permissive=True)
    if mode == "mean":
        zbinnd_fr = np.mean(mrc_obj.data[start_frame:stop_frame, yi:ya, xi:xa], axis=0)
    else:
        zbinnd_fr = np.sum(mrc_obj.data[start_frame:stop_frame, yi:ya, xi:xa], axis=0)
    if (xbin_factor > 1) or (ybin_factor > 1):
        if mode == "mean":
            zbinnd_fr = np.mean(
                np.mean(
                    zbinnd_fr.reshape(ny_binned, ybin_factor, nx_binned, xbin_factor),
                    axis=3,
                ),
                axis=1,
            )
        else:
            zbinnd_fr = np.sum(
                np.sum(
                    zbinnd_fr.reshape(ny_binned, ybin_factor, nx_binned, xbin_factor),
                    axis=3,
                ),
                axis=1,
            )
    tiff.imsave(save_filename, zbinnd_fr.astype(dtp))
    mrc_obj.close()
    return save_filename


def bin_crop_mrc_stack_DASK(DASK_client, mrc_filename, **kwargs):
    """
    Bins a 3D mrc stack along Z-direction (optional binning in X-Y plane as well) and crops it along X- and Y- directions. ©G.Shtengel 08/2022 gleb.shtengel@gmail.com

    Parameters:
        DASK_client
        mrc_filename : str
            name (full path) of the mrc file to be binned
    **kwargs:
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        DASK_client_retries : int (default to 0)
            Number of allowed automatic retries if a task fails
        fnm_types : list of strings.
            File type(s) for output data. Options are: ['h5', 'mrc'].
            Defauls is ['mrc']. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        zbin_factor : int
            binning factor in z-direction
        xbin_factor : int
            binning factor in x-direction
        ybin_factor : int
            binning factor in y-direction
        mode  : str
            Binning mode. Default is 'mean', other option is 'sum'
        frmax : int
            Maximum frame to bin. If not present, the entire file is binned
        binned_copped_filename : str
            name (full path) of the mrc file to save the results into. If not present, the new file name is constructed from the original by adding "_zbinXX" at the end.
        xi : int
            left edge of the crop
        xa : int
            right edge of the crop
        yi : int
            top edge of the crop
        ya : int
            bottom edge of the crop
        fri : int
            start frame
        fra : int
            stop frame
        disp_res : bolean
            Display messages and intermediate results
    Returns:
        fnms_saved : list of str
            Names of the new (binned and cropped) data files.
    """
    use_DASK = kwargs.get("use_DASK", False)
    DASK_client_retries = kwargs.get("DASK_client_retries", 3)
    fnm_types = kwargs.get("fnm_types", ["mrc"])
    xbin_factor = kwargs.get("xbin_factor", 1)  # binning factor in in x-direction
    ybin_factor = kwargs.get("ybin_factor", 1)  # binning factor in in y-direction
    zbin_factor = kwargs.get("zbin_factor", 1)  # binning factor in in z-direction
    disp_res = kwargs.get("disp_res", True)

    mode = kwargs.get(
        "mode", "mean"
    )  # binning mode. Default is 'mean', other option is 'sum'
    mrc_obj = mrcfile.mmap(mrc_filename, mode="r", permissive=True)
    header = mrc_obj.header
    """
        mode 0 -> uint8
        mode 1 -> int16
        mode 2 -> float32
        mode 4 -> complex64
        mode 6 -> uint16
    """
    mrc_mode = mrc_obj.header.mode
    # voxel_size_angstr = mrc_obj.header.cella
    voxel_size_angstr = mrc_obj.voxel_size
    voxel_size_angstr_new = voxel_size_angstr.copy()
    voxel_size_angstr_new.x = voxel_size_angstr.x * xbin_factor
    voxel_size_angstr_new.y = voxel_size_angstr.y * ybin_factor
    voxel_size_angstr_new.z = voxel_size_angstr.z * zbin_factor
    voxel_size_new = voxel_size_angstr.copy()
    voxel_size_new.x = voxel_size_angstr_new.x / 10.0
    voxel_size_new.y = voxel_size_angstr_new.y / 10.0
    voxel_size_new.z = voxel_size_angstr_new.z / 10.0
    nx, ny, nz = int32(header["nx"]), int32(header["ny"]), int32(header["nz"])
    frmax = kwargs.get("frmax", nz)
    xi = kwargs.get("xi", 0)
    xa = kwargs.get("xa", nx)
    yi = kwargs.get("yi", 0)
    ya = kwargs.get("ya", ny)
    fri = kwargs.get("fri", 0)
    fra = kwargs.get("fra", nz)
    nx_binned = (xa - xi) // xbin_factor
    ny_binned = (ya - yi) // ybin_factor
    xa = xi + nx_binned * xbin_factor
    ya = yi + ny_binned * ybin_factor
    binned_copped_filename_default = (
        os.path.splitext(mrc_filename)[0] + "_binned_croped.mrc"
    )
    binned_copped_filename = kwargs.get(
        "binned_copped_filename", binned_copped_filename_default
    )
    binned_mrc_filename = os.path.splitext(binned_copped_filename)[0] + ".mrc"
    dtp = type(mrc_obj.data[0, 0, 0])
    print("Source mrc_mode: {:d}, source data type:".format(mrc_mode), dtp)
    print(
        "Source Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}".format(
            voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z
        )
    )
    if mode == "sum":
        mrc_mode = 1
        dtp = int16
    print("Result mrc_mode: {:d}, source data type:".format(mrc_mode), dtp)

    st_frames = np.arange(fri, fra, zbin_factor)
    print(
        "New Data Set Shape:  {:d} x {:d} x {:d}".format(
            nx_binned, ny_binned, len(st_frames)
        )
    )

    bin_crop_parameters_dataset = []
    for j, st_frame in enumerate(
        tqdm(st_frames, desc="Setting up DASK parameter sets", display=disp_res)
    ):
        save_filename = os.path.join(
            os.path.split(mrc_filename)[0], "Binned_Cropped_Frame_{:d}.tif".format(j)
        )
        start_frame = st_frame
        stop_frame = min(st_frame + zbin_factor, nz - 1)
        bin_crop_parameters_dataset.append(
            [
                mrc_filename,
                save_filename,
                dtp,
                start_frame,
                stop_frame,
                xbin_factor,
                ybin_factor,
                zbin_factor,
                mode,
                xi,
                xa,
                yi,
                ya,
            ]
        )

    print("Binning / Cropping and Saving Intermediate Frames")
    if use_DASK:
        if disp_res:
            print("Starting DASK jobs")
        futures_bin_crop = DASK_client.map(
            bin_crop_frames, bin_crop_parameters_dataset, retries=DASK_client_retries
        )
        binned_cropped_filenames = np.array(DASK_client.gather(futures_bin_crop))
        if disp_res:
            print("Finished DASK jobs")
    else:  # if DASK is not used - perform local computations
        if disp_res:
            print("Will perform local computations")
        binned_cropped_filenames = []
        for bin_crop_parameters in tqdm(
            bin_crop_parameters_dataset,
            desc="Transforming and saving frame chunks",
            display=disp_res,
        ):
            binned_cropped_filenames.append(bin_crop_frames(bin_crop_parameters))

    print("Creating Dask Array Stack")
    # now build dask array of the transformed dataset
    # read the first file to get the shape and dtype (ASSUMING THAT ALL FILES SHARE THE SAME SHAPE/TYPE)
    frame0 = tiff.imread(binned_cropped_filenames[0])
    lazy_imread = dask.delayed(tiff.imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in binned_cropped_filenames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=frame0.shape, dtype=frame0.dtype)
        for delayed_reader in lazy_arrays
    ]
    # Stack infividual frames into one large dask.array
    FIBSEMstack = da.stack(dask_arrays, axis=0)
    nz, ny, nx = FIBSEMstack.shape

    print("Saving Intermediate Frames into Final Stacks")
    save_kwargs = {
        "fnm_types": fnm_types,
        "fnm_reg": binned_mrc_filename,
        "voxel_size": voxel_size_new,
        "dtp": dtp,
        "disp_res": True,
    }
    fnms_saved = save_data_stack(FIBSEMstack, **save_kwargs)

    for fnm in tqdm(
        binned_cropped_filenames, desc="Removing Intermediate Files: ", display=disp_res
    ):
        try:
            os.remove(os.path.join(fnm))
        except:
            pass

    return fnms_saved
