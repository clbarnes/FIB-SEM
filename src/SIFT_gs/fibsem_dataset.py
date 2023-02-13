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


class FIBSEM_dataset:
    """
    A class representing a FIB-SEM data set
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    Contains the info/settings on the FIB-SEM dataset and the procedures that can be performed on it.

    Attributes
    ----------
    fls : array of str
        filenames for the individual data frames in the set
    data_dir : str
        data directory (path)
    Sample_ID : str
            Sample ID
    ftype : int
        file type (0 - Shan Xu's .dat, 1 - tif)
    PixelSize : float
        pixel size in nm. This is inherited from FIBSEM_frame object. Default is 8.0
    voxel_size : rec.array(( float,  float,  float), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        voxel size in nm. Default is isotropic (PixelSize, PixelSize, PixelSize)
    Scaling : 2D array of floats
        scaling parameters allowing to convert I16 data into actual electron counts
    fnm_reg : str
        filename for the final registed dataset
    use_DASK : boolean
        use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
    threshold_min : float
        CDF threshold for determining the minimum data value
    threshold_max : float
        CDF threshold for determining the maximum data value
    nbins : int
        number of histogram bins for building the PDF and CDF
    sliding_minmax : boolean
        if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
        if False - same data_min_glob and data_max_glob will be used for all files
    TransformType : object reference
        Transformation model used by SIFT for determining the transformation matrix from Key-Point pairs.
        Choose from the following options:
            ShiftTransform - only x-shift and y-shift
            XScaleShiftTransform  -  x-scale, x-shift, y-shift
            ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
            AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
            RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
    l2_matrix : 2D float array
        matrix of regularization (shrinkage) parameters
    targ_vector = 1D float array
        target vector for regularization
    solver : str
        Solver used for SIFT ('RANSAC' or 'LinReg')
    drmax : float
        In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
        In the case of 'LinReg' - outlier threshold for iterative regression
    max_iter : int
        Max number of iterations in the iterative procedure above (RANSAC or LinReg)
    BFMatcher : boolean
        If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches : boolean
        If True, matches will be saved into individual files
    kp_max_num : int
        Max number of key-points to be matched.
        Key-points in every frame are indexed (in descending order) by the strength of the response.
        Only kp_max_num is kept for further processing.
        Set this value to -1 if you want to keep ALL keypoints (may take forever to process!)
    SIFT_nfeatures : int
        SIFT libary default is 0. The number of best features to retain.
        The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
    SIFT_nOctaveLayers : int
        SIFT libary default  is 3. The number of layers in each octave.
        3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
    SIFT_contrastThreshold : double
        SIFT libary default  is 0.04. The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
        The larger the threshold, the less features are produced by the detector.
        The contrast threshold will be divided by nOctaveLayers when the filtering is applied.
        When nOctaveLayers is set to default and if you want to use the value used in
        D. Lowe paper (0.03), set this argument to 0.09.
    SIFT_edgeThreshold : double
        SIFT libary default  is 10. The threshold used to filter out edge-like features.
        Note that the its meaning is different from the contrastThreshold,
        i.e. the larger the edgeThreshold, the less features are filtered out
        (more features are retained).
    SIFT_sigma : double
        SIFT library default is 1.6.  The sigma of the Gaussian applied to the input image at the octave #0.
        If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    dtp : Data Type
        Python data type for saving. Deafult is int16, the other option currently is uint8.
    zbin_factor : int
        binning factor in z-direction (milling direction). Data will be binned when saving the final result. Default is 1.
    flipY : boolean
        If True, the data will be flipped along Y-axis. Default is False.
    preserve_scales : boolean
        If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
    fit_params : list
        Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
        Other options are:
            ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
            ['PF', 2]  - use polynomial fit (in this case of order 2)
    int_order : int
        The order of interpolation (when transforming the data).
            The order has to be in the range 0-5:
                0: Nearest-neighbor
                1: Bi-linear (default)
                2: Bi-quadratic
                3: Bi-cubic
                4: Bi-quartic
                5: Bi-quintic
    subtract_linear_fit : [boolean, boolean]
        List of two Boolean values for two directions: X- and Y-.
        If True, the linear slopes along X- and Y- directions (respectively)
        will be subtracted from the cumulative shifts.
        This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
    pad_edges : boolean
        If True, the data will be padded before transformation to avoid clipping.
    ImgB_fraction : float
            fractional ratio of Image B to be used for constructing the fuksed image:
            ImageFused = ImageA * (1.0-ImgB_fraction) + ImageB * ImgB_fraction
    evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.

    Methods
    -------
    SIFT_evaluation(eval_fls = [], **kwargs):
        Evaluate SIFT settings and perfromance of few test frames (eval_fls).

    convert_raw_data_to_tif_files(DASK_client = '', **kwargs):
        Convert binary ".dat" files into ".tif" files

    evaluate_FIBSEM_statistics(self, DASK_client, **kwargs):
        Evaluates parameters of FIBSEM data set (data Min/Max, Working Distance, Milling Y Voltage, FOV center positions).

    extract_keypoints(DASK_client, **kwargs):
        Extract Key-Points and Descriptors

    determine_transformations(DASK_client, **kwargs):
        Determine transformation matrices for sequential frame pairs

    process_transformation_matrix(**kwargs):
        Calculate cumulative transformation matrix

    save_parameters(**kwargs):
        Save transformation attributes and parameters (including transformation matrices)

    check_for_nomatch_frames(thr_npt, **kwargs):
        Check for frames with low number of Key-Point matches,m exclude them and re-calculate the cumulative transformation matrix

    transform_and_save(DASK_client, **kwargs):
        Transform the frames using the cumulative transformation matrix and save the data set into .mrc file

    show_eval_box(**kwargs):
        Show the box used for evaluating the registration quality

    estimate_SNRs(**kwargs):
        Estimate SNRs in Image A and Image B based on single-image SNR calculation.

    evaluate_ImgB_fractions(ImgB_fractions, frame_inds, **kwargs):
        Calculate NCC and SNR vs Image B fraction over a set of frames.
    """

    def __init__(self, fls, **kwargs):
        """
        Initializes an instance of  FIBSEM_dataset object. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com

        Parameters
        ----------
        fls : array of str
            filenames for the individual data frames in the set
        data_dir : str
            data directory (path)

        kwargs
        ---------
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        DASK_client_retries : int (default to 0)
            Number of allowed automatic retries if a task fails
        Sample_ID : str
                Sample ID
        PixelSize : float
            pixel size in nm. Default is 8.0
        Scaling : 2D array of floats
            scaling parameters allowing to convert I16 data into actual electron counts
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        sliding_minmax : boolean
            if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
            if False - same data_min_glob and data_max_glob will be used for all files
        TransformType : object reference
            Transformation model used by SIFT for determining the transformation matrix from Key-Point pairs.
            Choose from the following options:
                ShiftTransform - only x-shift and y-shift
                XScaleShiftTransform  -  x-scale, x-shift, y-shift
                ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D float array
            matrix of regularization (shrinkage) parameters
        targ_vector = 1D float array
            target vector for regularization
        solver : str
            Solver used for SIFT ('RANSAC' or 'LinReg')
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        kp_max_num : int
            Max number of key-points to be matched.
            Key-points in every frame are indexed (in descending order) by the strength of the response.
            Only kp_max_num is kept for further processing.
            Set this value to -1 if you want to keep ALL keypoints (may take forever to process!)
        SIFT_nfeatures : int
            SIFT libary default is 0. The number of best features to retain.
            The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        SIFT_nOctaveLayers : int
            SIFT libary default  is 3. The number of layers in each octave.
            3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        SIFT_contrastThreshold : double
            SIFT libary default  is 0.04. The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
            The larger the threshold, the less features are produced by the detector.
            The contrast threshold will be divided by nOctaveLayers when the filtering is applied.
            When nOctaveLayers is set to default and if you want to use the value used in
            D. Lowe paper (0.03), set this argument to 0.09.
        SIFT_edgeThreshold : double
            SIFT libary default  is 10. The threshold used to filter out edge-like features.
            Note that the its meaning is different from the contrastThreshold,
            i.e. the larger the edgeThreshold, the less features are filtered out
            (more features are retained).
        SIFT_sigma : double
            SIFT library default is 1.6.  The sigma of the Gaussian applied to the input image at the octave #0.
            If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        dtp : Data Type
            Python data type for saving. Deafult is int16, the other option currently is uint8.
        zbin_factor : int
            binning factor in z-direction (milling direction). Data will be binned when saving the final result. Default is 1.
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        int_order : int
            The order of interpolation (when transforming the data).
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        disp_res : boolean
            If False, the intermediate printouts will be suppressed
        """

        disp_res = kwargs.get("disp_res", True)
        self.fls = fls
        self.fnms = [os.path.splitext(fl)[0] + "_kpdes.bin" for fl in fls]
        self.nfrs = len(fls)
        if disp_res:
            print("Total Number of frames: ", self.nfrs)
        self.data_dir = kwargs.get("data_dir", os.getcwd())
        self.ftype = kwargs.get(
            "ftype", 0
        )  # ftype=0 - Shan Xu's binary format  ftype=1 - tif files
        mid_frame = FIBSEM_frame(fls[self.nfrs // 2], ftype=self.ftype)
        self.XResolution = kwargs.get("XResolution", mid_frame.XResolution)
        self.YResolution = kwargs.get("YResolution", mid_frame.YResolution)
        self.Scaling = kwargs.get("Scaling", mid_frame.Scaling)
        if hasattr(mid_frame, "PixelSize"):
            self.PixelSize = kwargs.get("PixelSize", mid_frame.PixelSize)
        else:
            self.PixelSize = kwargs.get("PixelSize", 8.0)
        self.voxel_size = np.rec.array(
            (self.PixelSize, self.PixelSize, self.PixelSize),
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
        )
        if hasattr(self, "YResolution"):
            YResolution_default = self.YResolution
        else:
            YResolution_default = FIBSEM_frame(self.fls[len(self.fls) // 2]).YResolution
        YResolution = kwargs.get("YResolution", YResolution_default)

        test_frame = FIBSEM_frame(fls[0], ftype=self.ftype)
        self.DetA = test_frame.DetA
        self.DetB = test_frame.DetB
        self.ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)
        if self.DetB == "None":
            ImgB_fraction = 0.0
        self.Sample_ID = kwargs.get("Sample_ID", "")
        self.EightBit = kwargs.get("EightBit", 1)
        self.use_DASK = kwargs.get("use_DASK", True)
        self.DASK_client_retries = kwargs.get("DASK_client_retries", 0)
        self.threshold_min = kwargs.get("threshold_min", 1e-3)
        self.threshold_max = kwargs.get("threshold_max", 1e-3)
        self.nbins = kwargs.get("nbins", 256)
        self.sliding_minmax = kwargs.get("sliding_minmax", True)
        self.TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
        self.tr_matr_cum_residual = [
            np.eye(3, 3) for i in np.arange(self.nfrs)
        ]  # placeholder - identity transformation matrix
        l2_param_default = 1e-5  # regularization strength (shrinkage parameter)
        l2_matrix_default = (
            np.eye(6) * l2_param_default
        )  # initially set equal shrinkage on all coefficients
        l2_matrix_default[2, 2] = 0  # turn OFF the regularization on shifts
        l2_matrix_default[5, 5] = 0  # turn OFF the regularization on shifts
        self.l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
        self.targ_vector = kwargs.get(
            "targ_vector", np.array([1, 0, 0, 0, 1, 0])
        )  # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
        self.solver = kwargs.get("solver", "RANSAC")
        self.drmax = kwargs.get("drmax", 2.0)
        self.max_iter = kwargs.get("max_iter", 1000)
        self.BFMatcher = kwargs.get(
            "BFMatcher", False
        )  # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        self.save_matches = kwargs.get(
            "save_matches", True
        )  # If True, matches will be saved into individual files
        self.kp_max_num = kwargs.get("kp_max_num", -1)
        self.SIFT_nfeatures = kwargs.get("SIFT_nfeatures", 0)
        self.SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", 3)
        self.SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", 0.04)
        self.SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", 10)
        self.SIFT_sigma = kwargs.get("SIFT_sigma", 1.6)
        self.save_res_png = kwargs.get("save_res_png", True)
        self.zbin_factor = kwargs.get(
            "zbin_factor", 1
        )  # binning factor in z-direction (milling direction). Data will be binned when saving the final result. Default is 1.
        self.eval_metrics = kwargs.get("eval_metrics", ["NSAD", "NCC", "NMI", "FSC"])
        self.fnm_types = kwargs.get("fnm_types", ["mrc"])
        self.flipY = kwargs.get(
            "flipY", False
        )  # If True, the registered data will be flipped along Y axis
        self.preserve_scales = kwargs.get(
            "preserve_scales", True
        )  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
        self.fit_params = kwargs.get(
            "fit_params", False
        )  # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
        # window size 701, polynomial order 3

        self.int_order = kwargs.get(
            "int_order", False
        )  #     The order of interpolation. The order has to be in the range 0-5:
        #    - 0: Nearest-neighbor
        #    - 1: Bi-linear (default)
        #    - 2: Bi-quadratic
        #    - 3: Bi-cubic
        #    - 4: Bi-quartic
        #    - 5: Bi-quintic
        self.subtract_linear_fit = kwargs.get(
            "subtract_linear_fit", [True, True]
        )  # If True, the linear slope will be subtracted from the cumulative shifts.
        self.subtract_FOVtrend_from_fit = kwargs.get(
            "subtract_FOVtrend_from_fit", [True, True]
        )
        self.FOVtrend_x = np.zeros(len(fls))
        self.FOVtrend_y = np.zeros(len(fls))
        self.pad_edges = kwargs.get("pad_edges", True)
        build_fnm_reg, build_dtp = build_filename(fls[0], **kwargs)
        self.fnm_reg = kwargs.get("fnm_reg", build_fnm_reg)
        self.dtp = kwargs.get("dtp", build_dtp)
        if disp_res:
            print("Registered data will be saved into: ", self.fnm_reg)

        kwargs.update(
            {"data_dir": self.data_dir, "fnm_reg": self.fnm_reg, "dtp": self.dtp}
        )

        if kwargs.get("recall_parameters", False):
            dump_filename = kwargs.get("dump_filename", "")
            try:
                dump_data = pickle.load(open(dump_filename, "rb"))
                dump_loaded = True
            except Exception as ex1:
                dump_loaded = False
                if disp_res:
                    print("Failed to open Parameter dump filename: ", dump_filename)
                    print(ex1.message)

            if dump_loaded:
                try:
                    for key in tqdm(
                        dump_data, desc="Recalling the data set parameters"
                    ):
                        setattr(self, key, dump_data[key])
                except Exception as ex2:
                    if disp_res:
                        print("Parameter dump filename: ", dump_filename)
                        print("Failed to restore the object parameters")
                        print(ex2.message)

    def SIFT_evaluation(self, eval_fls=[], **kwargs):
        """
        Evaluate SIFT settings and perfromance of few test frames (eval_fls). ©G.Shtengel 10/2021 gleb.shtengel@gmail.com

        Parameters:
        eval_fls : array of str
            filenames for the data frames to be used for SIFT evaluation

        kwargs
        ---------
        data_dir : str
            data directory (path)
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        fnm_reg : str
            filename for the final registed dataset
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for key-point extraction
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        TransformType : object reference
            Transformation model used by SIFT for determining the transformation matrix from Key-Point pairs.
            Choose from the following options:
                ShiftTransform - only x-shift and y-shift
                XScaleShiftTransform  -  x-scale, x-shift, y-shift
                ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D float array
            matrix of regularization (shrinkage) parameters
        targ_vector = 1D float array
            target vector for regularization
        solver : str
            Solver used for SIFT ('RANSAC' or 'LinReg')
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        kp_max_num : int
            Max number of key-points to be matched.
            Key-points in every frame are indexed (in descending order) by the strength of the response.
            Only kp_max_num is kept for further processing.
            Set this value to -1 if you want to keep ALL keypoints (may take forever to process!)
        SIFT_nfeatures : int
            SIFT libary default is 0. The number of best features to retain.
            The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        SIFT_nOctaveLayers : int
            SIFT libary default  is 3. The number of layers in each octave.
            3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        SIFT_contrastThreshold : double
            SIFT libary default  is 0.04. The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
            The larger the threshold, the less features are produced by the detector.
            The contrast threshold will be divided by nOctaveLayers when the filtering is applied.
            When nOctaveLayers is set to default and if you want to use the value used in
            D. Lowe paper (0.03), set this argument to 0.09.
        SIFT_edgeThreshold : double
            SIFT libary default  is 10. The threshold used to filter out edge-like features.
            Note that the its meaning is different from the contrastThreshold,
            i.e. the larger the edgeThreshold, the less features are filtered out
            (more features are retained).
        SIFT_sigma : double
            SIFT library default is 1.6.  The sigma of the Gaussian applied to the input image at the octave #0.
            If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check

        Returns:
        dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts
        """
        if len(eval_fls) == 0:
            eval_fls = [self.fls[self.nfrs // 2], self.fls[self.nfrs // 2 + 1]]
        data_dir = kwargs.get("data_dir", self.data_dir)
        ftype = kwargs.get("ftype", self.ftype)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        threshold_min = kwargs.get("threshold_min", self.threshold_min)
        threshold_max = kwargs.get("threshold_max", self.threshold_max)
        nbins = kwargs.get("nbins", self.nbins)
        TransformType = kwargs.get("TransformType", self.TransformType)
        l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
        targ_vector = kwargs.get("targ_vector", self.targ_vector)
        solver = kwargs.get("solver", self.solver)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)
        SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
        SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
        SIFT_contrastThreshold = kwargs.get(
            "SIFT_contrastThreshold", self.SIFT_contrastThreshold
        )
        SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
        SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
        Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        save_matches = kwargs.get("save_matches", self.save_matches)
        save_res_png = kwargs.get("save_res_png", self.save_res_png)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])

        SIFT_evaluation_kwargs = {
            "ftype": ftype,
            "Sample_ID": Sample_ID,
            "data_dir": data_dir,
            "fnm_reg": fnm_reg,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "nbins": nbins,
            "evaluation_box": evaluation_box,
            "TransformType": TransformType,
            "l2_matrix": l2_matrix,
            "targ_vector": targ_vector,
            "solver": solver,
            "drmax": drmax,
            "max_iter": max_iter,
            "kp_max_num": kp_max_num,
            "SIFT_Transform": TransformType,
            "SIFT_nfeatures": SIFT_nfeatures,
            "SIFT_nOctaveLayers": SIFT_nOctaveLayers,
            "SIFT_contrastThreshold": SIFT_contrastThreshold,
            "SIFT_edgeThreshold": SIFT_edgeThreshold,
            "SIFT_sigma": SIFT_sigma,
            "Lowe_Ratio_Threshold": Lowe_Ratio_Threshold,
            "BFMatcher": BFMatcher,
            "save_matches": save_matches,
            "save_res_png": save_res_png,
        }

        (
            dmin,
            dmax,
            comp_time,
            transform_matrix,
            n_matches,
            iteration,
            kpts,
        ) = SIFT_evaluation_dataset(eval_fls, **SIFT_evaluation_kwargs)
        src_pts_filtered, dst_pts_filtered = kpts
        print(
            "Transformation Matrix determined using "
            + TransformType.__name__
            + " using "
            + solver
            + " solver"
        )
        print(transform_matrix)
        print(
            "{:d} keypoint matches were detected with {:.1f} pixel outlier threshold".format(
                n_matches, drmax
            )
        )
        print("Number of iterations: {:d}".format(iteration))
        return dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts

    def convert_raw_data_to_tif_files(self, DASK_client="", **kwargs):
        """
        Convert binary ".dat" files into ".tif" files.

        Parameters:
        DASK_client : instance of the DASK client object

        kwargs
        ---------
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        DASK_client_retries : int (default to 0)
            Number of allowed automatic retries if a task fails
        """
        if hasattr(self, "use_DASK"):
            use_DASK = kwargs.get("use_DASK", self.use_DASK)
        else:
            use_DASK = kwargs.get("use_DASK", False)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get(
                "DASK_client_retries", self.DASK_client_retries
            )
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 0)
        if self.ftype == 0:
            print('Step 2a: Creating "*InLens.tif" files using DASK distributed')
            t00 = time.time()
            if use_DASK:
                try:
                    futures = DASK_client.map(
                        save_inlens_data, self.fls, retries=DASK_client_retries
                    )
                    fls_new = np.array(DASK_client.gather(futures))
                except:
                    fls_new = []
                    for fl in tqdm(
                        self.fls, desc="Converting .dat data files into .tif format"
                    ):
                        fls_new.append(save_inlens_data(fl))
            else:
                fls_new = []
                for fl in tqdm(
                    self.fls, desc="Converting .dat data files into .tif format"
                ):
                    fls_new.append(save_inlens_data(fl))

            t01 = time.time()
            print("Step 2a: Elapsed time: {:.2f} seconds".format(t01 - t00))
            print(
                "Step 2a: Quick check if all files were converted: ",
                np.array_equal(self.fls, fls_new),
            )
        else:
            print("Step 2a: data is already in TIF format")

    def evaluate_FIBSEM_statistics(self, DASK_client, **kwargs):
        """
        Evaluates parameters of FIBSEM data set (Min/Max, Working Distance (WD), Milling Y Voltage (MV), FOV center positions).

        Parameters:
        use_DASK : boolean
            perform remote DASK computations
        DASK_client_retries : int (default to 0)
            Number of allowed automatic retries if a task fails

        kwargs:
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        frame_inds : array
            Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames
        data_dir : str
            data directory (path)  for saving the data
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        sliding_minmax : boolean
            if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
            if False - same data_min_glob and data_max_glob will be used for all files
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        Mill_Volt_Rate_um_per_V : float
            Milling Voltage to Z conversion (µm/V). Defaul is 31.235258870176065.
        FIBSEM_Data_xlsx : str
            Filepath of the Excell file for the FIBSEM data set data to be saved (Data Min/Max, Working Distance, Milling Y Voltage, FOV center positions)
        disp_res : bolean
            If True (default), intermediate messages and results will be displayed.

        Returns:
        list of 9 parameters: FIBSEM_Data_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding, mill_rate_WD, mill_rate_MV, center_x, center_y
            FIBSEM_Data_xlsx : str
                path to Excel file with the FIBSEM data
            data_min_glob : float
                min data value for I8 conversion (open CV SIFT requires I8)
            data_man_glob : float
                max data value for I8 conversion (open CV SIFT requires I8)
            data_min_sliding : float array
                min data values (one per file) for I8 conversion
            data_max_sliding : float array
                max data values (one per file) for I8 conversion

            mill_rate_WD : float array
                Milling rate calculated based on Working Distance (WD)
            mill_rate_MV : float array
                Milling rate calculated based on Milling Y Voltage (MV)
            center_x : float array
                FOV Center X-coordinate extrated from the header data
            center_y : float array
                FOV Center Y-coordinate extrated from the header data
        """
        if hasattr(self, "use_DASK"):
            use_DASK = kwargs.get("use_DASK", self.use_DASK)
        else:
            use_DASK = kwargs.get("use_DASK", False)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get(
                "DASK_client_retries", self.DASK_client_retries
            )
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 0)
        ftype = kwargs.get("ftype", self.ftype)
        frame_inds = kwargs.get("frame_inds", np.arange(len(self.fls)))
        data_dir = self.data_dir
        threshold_min = kwargs.get("threshold_min", self.threshold_min)
        threshold_max = kwargs.get("threshold_max", self.threshold_max)
        nbins = kwargs.get("nbins", self.nbins)
        sliding_minmax = kwargs.get("sliding_minmax", self.sliding_minmax)
        fit_params = kwargs.get("fit_params", self.fit_params)

        if hasattr(self, "Mill_Volt_Rate_um_per_V"):
            Mill_Volt_Rate_um_per_V = kwargs.get(
                "Mill_Volt_Rate_um_per_V", self.Mill_Volt_Rate_um_per_V
            )
        else:
            Mill_Volt_Rate_um_per_V = kwargs.get(
                "Mill_Volt_Rate_um_per_V", 31.235258870176065
            )
        FIBSEM_Data_xlsx = kwargs.get("FIBSEM_Data_xlsx", "FIBSEM_Data_xlsx.xlsx")
        disp_res = kwargs.get("disp_res", True)

        local_kwargs = {
            "use_DASK": use_DASK,
            "DASK_client_retries": DASK_client_retries,
            "ftype": ftype,
            "frame_inds": frame_inds,
            "data_dir": data_dir,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "nbins": nbins,
            "sliding_minmax": sliding_minmax,
            "fit_params": fit_params,
            "Mill_Volt_Rate_um_per_V": Mill_Volt_Rate_um_per_V,
            "FIBSEM_Data_xlsx": FIBSEM_Data_xlsx,
            "disp_res": disp_res,
        }

        if disp_res:
            print(
                "Evaluating the parameters of FIBSEM data set (data Min/Max, Working Distance, Milling Y Voltage, FOV center positions)"
            )
        self.FIBSEM_Data = evaluate_FIBSEM_frames_dataset(
            self.fls, DASK_client, **local_kwargs
        )
        self.data_minmax = self.FIBSEM_Data[0:5]
        WD = self.FIBSEM_Data[5]
        MillingYVoltage = self.FIBSEM_Data[6]

        apert = np.min((51, len(self.FIBSEM_Data[7]) - 1))
        self.FOVtrend_x = (
            savgol_filter(self.FIBSEM_Data[7] * 1.0, apert, 1) - self.FIBSEM_Data[7][0]
        )
        self.FOVtrend_y = (
            savgol_filter(self.FIBSEM_Data[8] * 1.0, apert, 1) - self.FIBSEM_Data[8][0]
        )

        WD_fit_coef = np.polyfit(frame_inds, WD, 1)
        rate_WD = WD_fit_coef[0] * 1.0e6

        MV_fit_coef = np.polyfit(frame_inds, MillingYVoltage, 1)
        rate_MV = MV_fit_coef[0] * Mill_Volt_Rate_um_per_V * -1.0e3

        Z_pixel_size_WD = rate_WD
        Z_pixel_size_MV = rate_MV

        if ftype == 0:
            if disp_res:
                if self.zbin_factor > 1:
                    print(
                        "Z pixel (after {:d}-x Z-binning) = {:.2f} nm - based on WD data".format(
                            self.zbin_factor, Z_pixel_size_WD * self.zbin_factor
                        )
                    )
                    print(
                        "Z pixel (after {:d}-x Z-binning) = {:.2f} nm - based on Milling Voltage data".format(
                            self.zbin_factor, Z_pixel_size_MV * self.zbin_factor
                        )
                    )
                else:
                    print(
                        "Z pixel = {:.2f} nm  - based on WD data".format(
                            Z_pixel_size_WD
                        )
                    )
                    print(
                        "Z pixel = {:.2f} nm  - based on Milling Voltage data".format(
                            Z_pixel_size_MV
                        )
                    )

            self.voxel_size = np.rec.array(
                (self.PixelSize, self.PixelSize, Z_pixel_size_WD),
                dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
            )
        else:
            if disp_res:
                print(
                    "No milling rate data is available, isotropic voxel size is set to {:.2f} nm".format(
                        self.PixelSize
                    )
                )
            self.voxel_size = np.rec.array(
                (self.PixelSize, self.PixelSize, Z_pixel_size_WD),
                dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
            )

        return self.FIBSEM_Data

    def extract_keypoints(self, DASK_client, **kwargs):
        """
        Extract Key-Points and Descriptors

        Parameters:
        DASK_client : instance of the DASK client object

        kwargs
        ---------
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        DASK_client_retries : int (default to 0)
            Number of allowed automatic retries if a task fails
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        EightBit : int
            0 - 16-bit data, 1: 8-bit data
        fnm_reg : str
            filename for the final registed dataset
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        sliding_minmax : boolean
            if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
            if False - same data_min_glob and data_max_glob will be used for all files
        data_minmax : list of 5 parameters
            minmax_xlsx : str
                path to Excel file with Min/Max data
            data_min_glob : float
                min data value for I8 conversion (open CV SIFT requires I8)
            data_min_sliding : float array
                min data values (one per file) for I8 conversion
            data_max_sliding : float array
                max data values (one per file) for I8 conversion
            data_minmax_glob : 2D float array
                min and max data values without sliding averaging
        kp_max_num : int
            Max number of key-points to be matched.
            Key-points in every frame are indexed (in descending order) by the strength of the response.
            Only kp_max_num is kept for further processing.
            Set this value to -1 if you want to keep ALL keypoints (may take forever to process!)

        Returns:
        fnms : array of str
            filenames for binary files kontaining Key-Point and Descriptors for each frame
        """
        if len(self.fls) == 0:
            print("Data set not defined, perform initialization first")
            fnms = []
        else:
            if hasattr(self, "use_DASK"):
                use_DASK = kwargs.get("use_DASK", self.use_DASK)
            else:
                use_DASK = kwargs.get("use_DASK", False)
            if hasattr(self, "DASK_client_retries"):
                DASK_client_retries = kwargs.get(
                    "DASK_client_retries", self.DASK_client_retries
                )
            else:
                DASK_client_retries = kwargs.get("DASK_client_retries", 0)
            ftype = kwargs.get("ftype", self.ftype)
            data_dir = self.data_dir
            fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
            threshold_min = kwargs.get("threshold_min", self.threshold_min)
            threshold_max = kwargs.get("threshold_max", self.threshold_max)
            nbins = kwargs.get("nbins", self.nbins)
            sliding_minmax = kwargs.get("sliding_minmax", self.sliding_minmax)
            data_minmax = kwargs.get("data_minmax", self.data_minmax)
            kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)

            SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
            SIFT_nOctaveLayers = kwargs.get(
                "SIFT_nOctaveLayers", self.SIFT_nOctaveLayers
            )
            SIFT_contrastThreshold = kwargs.get(
                "SIFT_contrastThreshold", self.SIFT_contrastThreshold
            )
            SIFT_edgeThreshold = kwargs.get(
                "SIFT_edgeThreshold", self.SIFT_edgeThreshold
            )
            SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)

            (
                minmax_xlsx,
                data_min_glob,
                data_max_glob,
                data_min_sliding,
                data_max_sliding,
            ) = data_minmax
            kpt_kwargs = {
                "ftype": ftype,
                "threshold_min": threshold_min,
                "threshold_max": threshold_max,
                "nbins": nbins,
                "kp_max_num": kp_max_num,
                "SIFT_nfeatures": SIFT_nfeatures,
                "SIFT_nOctaveLayers": SIFT_nOctaveLayers,
                "SIFT_contrastThreshold": SIFT_contrastThreshold,
                "SIFT_edgeThreshold": SIFT_edgeThreshold,
                "SIFT_sigma": SIFT_sigma,
            }

            if sliding_minmax:
                params_s3 = [
                    [dts3[0], dts3[1], dts3[2], kpt_kwargs]
                    for dts3 in zip(self.fls, data_min_sliding, data_max_sliding)
                ]
            else:
                params_s3 = [
                    [fl, data_min_glob, data_max_glob, kpt_kwargs] for fl in self.fls
                ]
            if use_DASK:
                print("Using DASK distributed")
                futures_s3 = DASK_client.map(
                    extract_keypoints_descr_files,
                    params_s3,
                    retries=DASK_client_retries,
                )
                fnms = DASK_client.gather(futures_s3)
            else:
                print("Using Local Computation")
                fnms = []
                for j, param_s3 in enumerate(
                    tqdm(params_s3, desc="Extracting Key Points and Descriptors: ")
                ):
                    fnms.append(extract_keypoints_descr_files(param_s3))

            self.fnms = fnms
        return fnms

    def determine_transformations(self, DASK_client, **kwargs):
        """
        Determine transformation matrices for sequential frame pairs

        Parameters:
        DASK_client : instance of the DASK client object

        kwargs
        ---------
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        DASK_client_retries : int (default to 0)
            Number of allowed automatic retries if a task fails
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        TransformType : object reference
                Transformation model used by SIFT for determining the transformation matrix from Key-Point pairs.
                Choose from the following options:
                    ShiftTransform - only x-shift and y-shift
                    XScaleShiftTransform  -  x-scale, x-shift, y-shift
                    ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                    AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                    RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D float array
            matrix of regularization (shrinkage) parameters
        targ_vector = 1D float array
            target vector for regularization
        solver : str
            Solver used for SIFT ('RANSAC' or 'LinReg')
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        Lowe_Ratio_Threshold : float
            threshold for Lowe's Ratio Test
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check


        Returns:
        results_s4 : array of lists containing the reults:
            results_s4 = [transformation_matrix, fnm_matches, npt, error_abs_mean]
            transformation_matrix : 2D float array
                transformation matrix for each sequential frame pair
            fnm_matches : str
                filename containing the matches used to determin the transformation for the par of frames
            npts : int
                number of matches
            error_abs_mean : float
                mean abs error of registration for all matched Key-Points
        """
        if len(self.fnms) == 0:
            print("No data on individual key-point data files, peform key-point search")
            results_s4 = []
        else:
            if hasattr(self, "use_DASK"):
                use_DASK = kwargs.get("use_DASK", self.use_DASK)
            else:
                use_DASK = kwargs.get("use_DASK", False)
            if hasattr(self, "DASK_client_retries"):
                DASK_client_retries = kwargs.get(
                    "DASK_client_retries", self.DASK_client_retries
                )
            else:
                DASK_client_retries = kwargs.get("DASK_client_retries", 0)
            ftype = kwargs.get("ftype", self.ftype)
            TransformType = kwargs.get("TransformType", self.TransformType)
            l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
            targ_vector = kwargs.get("targ_vector", self.targ_vector)
            solver = kwargs.get("solver", self.solver)
            drmax = kwargs.get("drmax", self.drmax)
            max_iter = kwargs.get("max_iter", self.max_iter)
            kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)
            Lowe_Ratio_Threshold = kwargs.get(
                "Lowe_Ratio_Threshold", 0.7
            )  # threshold for Lowe's Ratio Test
            BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
            save_matches = kwargs.get("save_matches", self.save_matches)
            save_res_png = kwargs.get("save_res_png", self.save_res_png)
            dt_kwargs = {
                "ftype": ftype,
                "TransformType": TransformType,
                "l2_matrix": l2_matrix,
                "targ_vector": targ_vector,
                "solver": solver,
                "drmax": drmax,
                "max_iter": max_iter,
                "BFMatcher": BFMatcher,
                "save_matches": save_matches,
                "kp_max_num": kp_max_num,
                "Lowe_Ratio_Threshold": Lowe_Ratio_Threshold,
            }

            params_s4 = []
            for j, fnm in enumerate(self.fnms[:-1]):
                fname1 = self.fnms[j]
                fname2 = self.fnms[j + 1]
                params_s4.append([fname1, fname2, dt_kwargs])
            if use_DASK:
                print("Using DASK distributed")
                futures4 = DASK_client.map(
                    determine_transformations_files,
                    params_s4,
                    retries=DASK_client_retries,
                )
                # determine_transformations_files returns (transform_matrix, fnm_matches, kpts, iteration)
                results_s4 = DASK_client.gather(futures4)
            else:
                print("Using Local Computation")
                results_s4 = []
                for param_s4 in tqdm(
                    params_s4, desc="Extracting Transformation Parameters: "
                ):
                    results_s4.append(determine_transformations_files(param_s4))
            # determine_transformations_files returns (transform_matrix, fnm_matches, kpts, errors, iteration)
            self.transformation_matrix = np.nan_to_num(
                np.array([result[0] for result in results_s4])
            )
            self.fnms_matches = [result[1] for result in results_s4]
            self.error_abs_mean = np.nan_to_num(
                np.array([result[3] for result in results_s4])
            )
            self.npts = np.nan_to_num(
                np.array([len(result[2][0]) for result in results_s4])
            )
            print("Mean Number of Keypoints :", np.mean(self.npts).astype(np.int16))
        return results_s4

    def process_transformation_matrix(self, **kwargs):
        """
        Calculate cumulative transformation matrix

        kwargs
        ---------
        data_dir : str
            data directory (path)
        fnm_reg : str
            filename for the final registed dataset
        Sample_ID : str
            Sample ID
        TransformType : object reference
                Transformation model used by SIFT for determining the transformation matrix from Key-Point pairs.
                Choose from the following options:
                    ShiftTransform - only x-shift and y-shift
                    XScaleShiftTransform  -  x-scale, x-shift, y-shift
                    ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                    AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                    RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D float array
            matrix of regularization (shrinkage) parameters
        targ_vector = 1D float array
            target vector for regularization
        solver : str
            Solver used for SIFT ('RANSAC' or 'LinReg')
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.

        Returns:
        tr_matr_cum_residual, tr_matr_cum_xlsx_file : list of 2D arrays of float and the filename of the XLSX file with the transf matrix results
            Cumulative transformation matrices
        """
        if len(self.transformation_matrix) == 0:
            print(
                "No data on individual key-point matches, peform key-point search / matching first"
            )
            self.tr_matr_cum_residual = []
        else:
            data_dir = kwargs.get("data_dir", self.data_dir)
            fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
            TransformType = kwargs.get("TransformType", self.TransformType)
            SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
            SIFT_nOctaveLayers = kwargs.get(
                "SIFT_nOctaveLayers", self.SIFT_nOctaveLayers
            )
            SIFT_contrastThreshold = kwargs.get(
                "SIFT_contrastThreshold", self.SIFT_contrastThreshold
            )
            SIFT_edgeThreshold = kwargs.get(
                "SIFT_edgeThreshold", self.SIFT_edgeThreshold
            )
            SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
            Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
            l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
            targ_vector = kwargs.get("targ_vector", self.targ_vector)
            solver = kwargs.get("solver", self.solver)
            drmax = kwargs.get("drmax", self.drmax)
            max_iter = kwargs.get("max_iter", self.max_iter)
            BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
            save_matches = kwargs.get("save_matches", self.save_matches)
            kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)
            save_res_png = kwargs.get("save_res_png", self.save_res_png)
            preserve_scales = kwargs.get("preserve_scales", self.preserve_scales)
            fit_params = kwargs.get("fit_params", self.fit_params)
            subtract_linear_fit = kwargs.get(
                "subtract_linear_fit", self.subtract_linear_fit
            )
            subtract_FOVtrend_from_fit = kwargs.get(
                "subtract_FOVtrend_from_fit", self.subtract_FOVtrend_from_fit
            )
            pad_edges = kwargs.get("pad_edges", self.pad_edges)

            TM_kwargs = {
                "fnm_reg": fnm_reg,
                "data_dir": data_dir,
                "TransformType": TransformType,
                "SIFT_nfeatures": SIFT_nfeatures,
                "SIFT_nOctaveLayers": SIFT_nOctaveLayers,
                "SIFT_contrastThreshold": SIFT_contrastThreshold,
                "SIFT_edgeThreshold": SIFT_edgeThreshold,
                "SIFT_sigma": SIFT_sigma,
                "Sample_ID": Sample_ID,
                "l2_matrix": l2_matrix,
                "targ_vector": targ_vector,
                "solver": solver,
                "drmax": drmax,
                "max_iter": max_iter,
                "BFMatcher": BFMatcher,
                "save_matches": save_matches,
                "kp_max_num": kp_max_num,
                "save_res_png ": save_res_png,
                "preserve_scales": preserve_scales,
                "fit_params": fit_params,
                "subtract_linear_fit": subtract_linear_fit,
                "subtract_FOVtrend_from_fit": subtract_FOVtrend_from_fit,
                "pad_edges": pad_edges,
            }
            (
                self.tr_matr_cum_residual,
                self.transf_matrix_xlsx_file,
            ) = process_transf_matrix(
                self.transformation_matrix,
                self.FOVtrend_x,
                self.FOVtrend_y,
                self.fnms_matches,
                self.npts,
                self.error_abs_mean,
                **TM_kwargs
            )
        return self.tr_matr_cum_residual, self.transf_matrix_xlsx_file

    def save_parameters(self, **kwargs):
        """
        Save transformation attributes and parameters (including transformation matrices).

        kwargs:
        -------
        dump_filename : string
            String containing the name of the binary dump for saving all attributes of the current istance of the FIBSEM_dataset object.


        Returns:
        dump_filename : string
        """
        default_dump_filename = os.path.join(
            self.data_dir, self.fnm_reg.replace(".mrc", "_params.bin")
        )
        dump_filename = kwargs.get("dump_filename", default_dump_filename)

        pickle.dump(self.__dict__, open(dump_filename, "wb"))

        npts_fnm = dump_filename.replace("_params.bin", "_Npts_Errs_data.csv")
        Tr_matrix_xls_fnm = dump_filename.replace(
            "_params.bin", "_Transform_Matrix_data.csv"
        )

        # Save the keypoint statistics into a CSV file
        columns = ["Npts", "Mean Abs Error"]
        npdt = pd.DataFrame(
            np.vstack((self.npts, self.error_abs_mean)).T, columns=columns, index=None
        )
        npdt.to_csv(npts_fnm, index=None)

        # Save the X-Y shift data and keypoint statistics into a CSV file
        columns = [
            "T00 (Sxx)",
            "T01 (Sxy)",
            "T02 (Tx)",
            "T10 (Syx)",
            "T11 (Syy)",
            "T12 (Ty)",
            "T20 (0.0)",
            "T21 (0.0)",
            "T22 (1.0)",
        ]
        tr_mx_dt = pd.DataFrame(
            self.transformation_matrix.reshape((len(self.transformation_matrix), 9)),
            columns=columns,
            index=None,
        )
        tr_mx_dt.to_csv(Tr_matrix_xls_fnm, index=None)
        return dump_filename

    def check_for_nomatch_frames(self, thr_npt, **kwargs):
        """
        Calculate cumulative transformation matrix

        Parameters:
        -----------
        thr_npt : int
            minimum number of matches. If the pair has less than this - it is reported as "suspicious" and is excluded.

        kwargs
        ---------
        data_dir : str
            data directory (path)
        fnm_reg : str
            filename for the final registed dataset
        Sample_ID : str
            Sample ID
        TransformType : object reference
                Transformation model used by SIFT for determining the transformation matrix from Key-Point pairs.
                Choose from the following options:
                    ShiftTransform - only x-shift and y-shift
                    XScaleShiftTransform  -  x-scale, x-shift, y-shift
                    ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                    AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                    RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D float array
            matrix of regularization (shrinkage) parameters
        targ_vector = 1D float array
            target vector for regularization
        solver : str
            Solver used for SIFT ('RANSAC' or 'LinReg')
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.

        Returns:
        tr_matr_cum_residual : list of 2D arrays of float
            Cumulative transformation matrices
        """
        self.thr_npt = thr_npt
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        TransformType = kwargs.get("TransformType", self.TransformType)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
        targ_vector = kwargs.get("targ_vector", self.targ_vector)
        solver = kwargs.get("solver", self.solver)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        save_matches = kwargs.get("save_matches", self.save_matches)
        kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)
        save_res_png = kwargs.get("save_res_png", self.save_res_png)
        preserve_scales = kwargs.get("preserve_scales", self.preserve_scales)
        fit_params = kwargs.get("fit_params", self.fit_params)
        subtract_linear_fit = kwargs.get(
            "subtract_linear_fit", self.subtract_linear_fit
        )
        subtract_FOVtrend_from_fit = kwargs.get(
            "subtract_FOVtrend_from_fit", self.subtract_FOVtrend_from_fit
        )
        pad_edges = kwargs.get("pad_edges", self.pad_edges)

        res_nomatch_check = check_for_nomatch_frames_dataset(
            self.fls,
            self.fnms,
            self.fnms_matches,
            self.transformation_matrix,
            self.error_abs_mean,
            self.npts,
            thr_npt,
            data_dir=self.data_dir,
            fnm_reg=self.fnm_reg,
        )
        (
            frames_to_remove,
            self.fls,
            self.fnms,
            self.fnms_matches,
            self.error_abs_mean,
            self.npts,
            self.transformation_matrix,
        ) = res_nomatch_check

        if len(frames_to_remove) > 0:
            TM_kwargs = {
                "fnm_reg": fnm_reg,
                "data_dir": data_dir,
                "TransformType": TransformType,
                "Sample_ID": Sample_ID,
                "l2_matrix": l2_matrix,
                "targ_vector": targ_vector,
                "solver": solver,
                "drmax": drmax,
                "max_iter": max_iter,
                "BFMatcher": BFMatcher,
                "save_matches": save_matches,
                "kp_max_num": kp_max_num,
                "save_res_png ": save_res_png,
                "preserve_scales": preserve_scales,
                "fit_params": fit_params,
                "subtract_linear_fit": subtract_linear_fit,
                "subtract_FOVtrend_from_fit": subtract_FOVtrend_from_fit,
                "pad_edges": pad_edges,
            }
            (
                self.tr_matr_cum_residual,
                self.transf_matrix_xlsx_file,
            ) = process_transf_matrix(
                self.transformation_matrix,
                self.FOVtrend_x,
                self.FOVtrend_y,
                self.fnms_matches,
                self.npts,
                self.error_abs_mean,
                **TM_kwargs
            )
        return self.tr_matr_cum_residual, self.transf_matrix_xlsx_file

    def transform_and_save(
        self,
        DASK_client,
        save_transformed_dataset=True,
        save_registration_summary=True,
        frame_inds=np.array((-1)),
        **kwargs
    ):
        """
        Transform the frames using the cumulative transformation matrix and save the data set into .mrc and/or .h5 file

        Parameters
        DASK_client : instance of the DASK client object
        save_transformed_dataset : boolean
            If true, the transformed data set will be saved into MRC file
        save_registration_summary : bolean
            If True, the registration analysis data will be saved into XLSX file
        frame_inds : int array (or list)
            Array of frame indecis. If not set or set to np.array((-1)), all frames will be transformed

        kwargs
        ---------
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        DASK_client_retries : int (default to 0)
            Number of allowed automatic retries if a task fails
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        data_dir : str
            data directory (path)
        fnm_types : list of strings
            File type(s) for output data. Options are: ['h5', 'mrc'].
            Defauls is 'mrc'. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        fnm_reg : str
            filename for the final registed dataset
        ImgB_fraction : float
            fractional ratio of Image B to be used for constructing the fuksed image:
            ImageFused = ImageA * (1.0-ImgB_fraction) + ImageB * ImgB_fraction
        add_offset : boolean
            If True - the Dark Oount offset will be added before saving to make values positive (set True if saving into BigDataViewer HDF5 - it uses UI16 data format)
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        perfrom_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed.
        invert_data : boolean
            If True - the data is inverted.
        flatten_image : bolean
            perform image flattening
        image_correction_file : str
            full path to a binary filename that contains source name (image_correction_source) and correction array (img_correction_array)
        flipY : boolean
            If True, the data will be flipped along Y-axis. Default is False.
        zbin_factor : int
            binning factor along Z-axis
        eval_metrics : list of str
            list of evaluation metrics to use. default is ['NSAD', 'NCC', 'NMI', 'FSC']
        fnm_types : list of strings
            File type(s) for output data. Options are: ['h5', 'mrc'].
            Defauls is 'mrc'. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
        start_evaluation_box : list of 4 int
            see above
        stop_evaluation_box : list of 4 int
            see above
        save_sample_frames_png : bolean
            If True, sample frames with superimposed eval box and registration analysis data will be saved into png files
        dtp  : dtype
            Python data type for saving. Deafult is int16, the other option currently is uint8.
        disp_res : bolean
            If True (default), intermediate messages and results will be displayed.

        Returns:
        reg_summary, reg_summary_xlsx
            reg_summary : pandas DataFrame
            reg_summary = pd.DataFrame(np.vstack((npts, error_abs_mean, image_nsad, image_ncc, image_mi)
            reg_summary_xlsx : name of the XLSX workbook containing the data
        """
        if (frame_inds == np.array((-1))).all():
            frame_inds = np.arange(len(self.fls))

        if hasattr(self, "use_DASK"):
            use_DASK = kwargs.get("use_DASK", self.use_DASK)
        else:
            use_DASK = kwargs.get("use_DASK", False)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get(
                "DASK_client_retries", self.DASK_client_retries
            )
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 0)
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        if hasattr(self, "XResolution"):
            XResolution_default = self.XResolution
        else:
            XResolution_default = FIBSEM_frame(self.fls[len(self.fls) // 2]).XResolution
        XResolution = kwargs.get("XResolution", XResolution_default)
        if hasattr(self, "YResolution"):
            YResolution_default = self.YResolution
        else:
            YResolution_default = FIBSEM_frame(self.fls[len(self.fls) // 2]).YResolution
        YResolution = kwargs.get("YResolution", YResolution_default)

        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        if hasattr(self, "fnm_types"):
            fnm_types = kwargs.get("fnm_types", self.fnm_types)
        else:
            fnm_types = kwargs.get("fnm_types", ["mrc"])
        ImgB_fraction = kwargs.get("ImgB_fraction", self.ImgB_fraction)
        if self.DetB == "None":
            ImgB_fraction = 0.0
        if hasattr(self, "add_offset"):
            add_offset = kwargs.get("add_offset", self.add_offset)
        else:
            add_offset = kwargs.get("add_offset", False)
        save_sample_frames_png = kwargs.get("save_sample_frames_png", True)
        pad_edges = kwargs.get("pad_edges", self.pad_edges)
        save_res_png = kwargs.get("save_res_png", self.save_res_png)
        if hasattr(self, "eval_metrics"):
            eval_metrics = kwargs.get("eval_metrics", self.eval_metrics)
        else:
            eval_metrics = kwargs.get("eval_metrics", ["NSAD", "NCC", "NMI", "FSC"])
        if hasattr(self, "zbin_factor"):
            zbin_factor = kwargs.get(
                "zbin_factor", self.zbin_factor
            )  # binning factor in z-direction (milling direction). Data will be binned when saving the final result. Default is 1.
        else:
            zbin_factor = kwargs.get("zbin_factor", 1)
        if hasattr(self, "voxel_size"):
            voxel_size = kwargs.get("voxel_size", self.voxel_size)
        else:
            voxel_size_default = np.rec.array(
                (self.PixelSize, self.PixelSize, self.PixelSize),
                dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
            )
            voxel_size = kwargs.get("voxel_size", voxel_size_default)
        voxel_size_zbinned = voxel_size.copy()
        voxel_size_zbinned.z = voxel_size.z * zbin_factor
        if hasattr(self, "flipY"):
            flipY = kwargs.get("flipY", self.flipY)
        else:
            flipY = kwargs.get("flipY", False)
        if hasattr(self, "dump_filename"):
            dump_filename = kwargs.get("dump_filename", self.dump_filename)
        else:
            dump_filename = kwargs.get("dump_filename", "")
        int_order = kwargs.get("int_order", self.int_order)
        preserve_scales = kwargs.get("preserve_scales", self.preserve_scales)
        if hasattr(self, "flatten_image"):
            flatten_image = kwargs.get("flatten_image", self.flatten_image)
        else:
            flatten_image = kwargs.get("flatten_image", False)
        if hasattr(self, "image_correction_file"):
            image_correction_file = kwargs.get(
                "image_correction_file", self.image_correction_file
            )
        else:
            image_correction_file = kwargs.get("image_correction_file", "")
        perfrom_transformation = kwargs.get("perfrom_transformation", True) and hasattr(
            self, "tr_matr_cum_residual"
        )
        if hasattr(self, "invert_data"):
            invert_data = kwargs.get("invert_data", self.invert_data)
        else:
            invert_data = kwargs.get("invert_data", False)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        disp_res = kwargs.get("disp_res", True)
        dtp = kwargs.get(
            "dtp", int16
        )  # Python data type for saving. Deafult is int16, the other option currently is uint8.

        save_kwargs = {
            "fnm_types": fnm_types,
            "fnm_reg": fnm_reg,
            "use_DASK": use_DASK,
            "DASK_client_retries": DASK_client_retries,
            "ftype": ftype,
            "XResolution": XResolution,
            "YResolution": YResolution,
            "data_dir": data_dir,
            "voxel_size": voxel_size_zbinned,
            "pad_edges": pad_edges,
            "ImgB_fraction": ImgB_fraction,
            "save_res_png ": save_res_png,
            "dump_filename": dump_filename,
            "dtp": dtp,
            "zbin_factor": zbin_factor,
            "eval_metrics": eval_metrics,
            "flipY": flipY,
            "int_order": int_order,
            "preserve_scales": preserve_scales,
            "flatten_image": flatten_image,
            "image_correction_file": image_correction_file,
            "perfrom_transformation": perfrom_transformation,
            "invert_data": invert_data,
            "evaluation_box": evaluation_box,
            "sliding_evaluation_box": sliding_evaluation_box,
            "start_evaluation_box": start_evaluation_box,
            "stop_evaluation_box": stop_evaluation_box,
            "save_sample_frames_png": save_sample_frames_png,
            "save_registration_summary": save_registration_summary,
            "disp_res": disp_res,
        }

        # first, transform, bin and save frame chunks into individual tif files
        if disp_res:
            print("Transforming and Saving Intermediate Registered Frames")
        registered_filenames = transform_and_save_frames(
            DASK_client, frame_inds, self.fls, self.tr_matr_cum_residual, **save_kwargs
        )

        frame0 = tiff.imread(registered_filenames[0])
        ny, nx = np.shape(frame0)
        if disp_res:
            print("Analyzing Registration Quality")
        if pad_edges and perfrom_transformation:
            xmn, xmx, ymn, ymx = determine_pad_offsets(
                [ny, nx], self.tr_matr_cum_residual
            )
            padx = int(xmx - xmn)
            pady = int(ymx - ymn)
            xi = int(np.max([xmx, 0]))
            yi = int(np.max([ymx, 0]))
        else:
            padx = 0
            pady = 0
            xi = 0
            yi = 0
        if sliding_evaluation_box:
            dx_eval = stop_evaluation_box[2] - start_evaluation_box[2]
            dy_eval = stop_evaluation_box[0] - start_evaluation_box[0]
        else:
            dx_eval = 0
            dy_eval = 0

        eval_bounds = []
        for j, registered_filename in enumerate(
            tqdm(
                registered_filenames,
                desc="Setting up evaluation bounds",
                display=disp_res,
            )
        ):
            if sliding_evaluation_box:
                xi_eval = xi + start_evaluation_box[2] + dx_eval * j // nz
                yi_eval = yi + start_evaluation_box[0] + dy_eval * j // nz
                if start_evaluation_box[3] > 0:
                    xa_eval = xi_eval + start_evaluation_box[3]
                else:
                    xa_eval = nx
                if start_evaluation_box[1] > 0:
                    ya_eval = yi_eval + start_evaluation_box[1]
                else:
                    ya_eval = ny
            else:
                xi_eval = xi + evaluation_box[2]
                if evaluation_box[3] > 0:
                    xa_eval = xi_eval + evaluation_box[3]
                else:
                    xa_eval = nx
                yi_eval = yi + evaluation_box[0]
                if evaluation_box[1] > 0:
                    ya_eval = yi_eval + evaluation_box[1]
                else:
                    ya_eval = ny
            eval_bounds.append([xi_eval, xa_eval, yi_eval, ya_eval])

        save_kwargs["eval_bounds"] = eval_bounds

        reg_summary, reg_summary_xlsx = analyze_registration_frames(
            DASK_client, registered_filenames, **save_kwargs
        )

        if save_transformed_dataset:
            if add_offset:
                offset = (
                    self.Scaling[1, 0] * (1.0 - ImgB_fraction)
                    + self.Scaling[1, 1] * ImgB_fraction
                )
            if disp_res:
                print("Creating Dask Array Stack")
            # now build dask array of the transformed dataset
            # read the first file to get the shape and dtype (ASSUMING THAT ALL FILES SHARE THE SAME SHAPE/TYPE)
            lazy_imread = dask.delayed(tiff.imread)  # lazy reader
            lazy_arrays = [lazy_imread(fn) for fn in registered_filenames]
            dask_arrays = [
                da.from_delayed(delayed_reader, shape=frame0.shape, dtype=frame0.dtype)
                for delayed_reader in lazy_arrays
            ]
            # Stack infividual frames into one large dask.array
            if add_offset:
                FIBSEMstack = da.stack(dask_arrays, axis=0) - offset
            else:
                FIBSEMstack = da.stack(dask_arrays, axis=0)
            # nz, ny, nx = FIBSEMstack.shape
            fnms_saved = save_data_stack(FIBSEMstack, **save_kwargs)
        else:
            if disp_res:
                print("Registered data set is NOT saved into a file")

        # Remove Intermediate Registered Frame Files
        for registered_filename in tqdm(
            registered_filenames,
            desc="Removing Intermediate Registered Frame Files: ",
            display=disp_res,
        ):
            try:
                os.remove(registered_filename)
            except:
                pass

        return reg_summary, reg_summary_xlsx

    def show_eval_box(self, **kwargs):
        """
        Show the box used for evaluating the registration quality

        kwargs
        ---------
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
        start_evaluation_box : list of 4 int
            see above
        stop_evaluation_box : list of 4 int
            see above
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        data_dir : str
            data directory (path)
        fnm_reg : str
            filename for the final registed dataset
        Sample_ID : str
            Sample ID
        int_order : int
            The order of interpolation (when transforming the data).
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        perfrom_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed
        invert_data : boolean
            If True - the data is inverted
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        flipY : boolean
            If True, the data will be flipped along Y-axis. Default is False.
        frame_inds : array or list of int
            Array or list oif frame indecis to use to display the evaluation box.
            Default are [nfrs//10, nfrs//2, nfrs//10*9]
        """
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        int_order = kwargs.get("int_order", self.int_order)
        perfrom_transformation = kwargs.get("perfrom_transformation", True) and hasattr(
            self, "tr_matr_cum_residual"
        )
        invert_data = kwargs.get("invert_data", False)
        flipY = kwargs.get("flipY", False)
        pad_edges = kwargs.get("pad_edges", self.pad_edges)
        save_res_png = kwargs.get("save_res_png", self.save_res_png)
        fls = self.fls
        nfrs = len(fls)
        default_indecis = [nfrs // 10, nfrs // 2, nfrs // 10 * 9]
        frame_inds = kwargs.get("frame_inds", default_indecis)

        for j in frame_inds:
            frame = FIBSEM_frame(fls[j], ftype=ftype)
            if pad_edges and perfrom_transformation:
                shape = [frame.YResolution, frame.XResolution]
                xmn, xmx, ymn, ymx = determine_pad_offsets(
                    shape, self.tr_matr_cum_residual
                )
                padx = np.int16(xmx - xmn)
                pady = np.int16(ymx - ymn)
                xi = np.int16(np.max([xmx, 0]))
                yi = np.int16(np.max([ymx, 0]))
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

            xsz = frame.XResolution + padx
            xa = xi + frame.XResolution
            ysz = frame.YResolution + pady
            ya = yi + frame.YResolution

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

            if sliding_evaluation_box:
                dx_eval = stop_evaluation_box[2] - start_evaluation_box[2]
                dy_eval = stop_evaluation_box[0] - start_evaluation_box[0]
            else:
                dx_eval = 0
                dy_eval = 0

            frame_img = np.zeros((ysz, xsz))

            if invert_data:
                frame_img[yi:ya, xi:xa] = np.negative(frame.RawImageA.astype(float))
                """
                if frame.EightBit==0:
                    frame_img[yi:ya, xi:xa] = np.negative(frame.RawImageA)
                else:
                    frame_img[yi:ya, xi:xa]  =  uint8(255) - frame.RawImageA
                """
            else:
                frame_img[yi:ya, xi:xa] = frame.RawImageA.astype(float)

            if perfrom_transformation:
                transf = ProjectiveTransform(
                    matrix=shift_matrix
                    @ (self.tr_matr_cum_residual[j] @ inv_shift_matrix)
                )
                frame_img_reg = warp(
                    frame_img, transf, order=int_order, preserve_range=True
                )
            else:
                frame_img_reg = frame_img.copy()

            if flipY:
                frame_img_reg = np.flip(frame_img_reg, axis=0)

            if sliding_evaluation_box:
                xi_eval = start_evaluation_box[2] + dx_eval * j // nfrs
                yi_eval = start_evaluation_box[0] + dy_eval * j // nfrs
                if start_evaluation_box[3] > 0:
                    xa_eval = xi_eval + start_evaluation_box[3]
                else:
                    xa_eval = xsz
                if start_evaluation_box[1] > 0:
                    ya_eval = yi_eval + start_evaluation_box[1]
                else:
                    ya_eval = ysz

            vmin, vmax = get_min_max_thresholds(
                frame_img_reg[yi_eval:ya_eval, xi_eval:xa_eval], disp_res=False
            )
            fig, ax = subplots(1, 1, figsize=(10.0, 11.0 * ysz / xsz))
            ax.imshow(frame_img_reg, cmap="Greys", vmin=vmin, vmax=vmax)
            ax.grid(True, color="cyan")
            ax.set_title(fls[j])
            rect_patch = patches.Rectangle(
                (xi_eval, yi_eval),
                abs(xa_eval - xi_eval) - 2,
                abs(ya_eval - yi_eval) - 2,
                linewidth=2.0,
                edgecolor="yellow",
                facecolor="none",
            )
            ax.add_patch(rect_patch)
            if save_res_png:
                fig.savefig(
                    os.path.splitext(fls[j])[0] + "_evaluation_box.png", dpi=300
                )

    def estimate_SNRs(self, **kwargs):
        """
        Estimate SNRs in Image A and Image B based on single-image SNR calculation.

        kwargs
        ---------
        frame_inds : list of int
            List oif frame indecis to use to display the evaluation box.
            Default are [nfrs//10, nfrs//2, nfrs//10*9]
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
        start_evaluation_box : list of 4 int
            see above
        stop_evaluation_box : list of 4 int
            see above
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        data_dir : str
            data directory (path)
        Sample_ID : str
            Sample ID
        ImgB_fraction : float
            Optional fractional weight of Image B to use for constructing the fused image: FusedImage = ImageA*(1.0-ImgB_fraction) + ImageB*ImgB_fraction
            If not provided, the value determined from rSNR ratios will be used.
        invert_data : boolean
            If True - the data is inverted
        perfrom_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        extrapolate_signal : boolean
            extrapolate to find signal autocorrelationb at 0-point (without noise). Default is True
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        flipY : boolean
            If True, the data will be flipped along Y-axis. Default is False.

        """
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        int_order = kwargs.get("int_order", self.int_order)
        invert_data = kwargs.get("invert_data", False)
        save_res_png = kwargs.get("save_res_png", False)
        ImgB_fraction = kwargs.get("ImgB_fraction", 0.00)
        flipY = kwargs.get("flipY", False)
        pad_edges = kwargs.get("pad_edges", self.pad_edges)
        perfrom_transformation = kwargs.get("perfrom_transformation", True) and hasattr(
            self, "tr_matr_cum_residual"
        )
        extrapolate_signal = kwargs.get("extrapolate_signal", True)

        fls = self.fls
        nfrs = len(fls)
        default_indecis = [nfrs // 10, nfrs // 2, nfrs // 10 * 9]
        frame_inds = kwargs.get("frame_inds", default_indecis)

        test_frame = FIBSEM_frame(fls[0], ftype=ftype)

        xi = 0
        yi = 0
        xsz = test_frame.XResolution
        xa = xi + xsz
        ysz = test_frame.YResolution
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

        frame_img = np.zeros((ysz, xsz))
        xSNRAs = []
        ySNRAs = []
        rSNRAs = []
        xSNRBs = []
        ySNRBs = []
        rSNRBs = []

        for j in tqdm(frame_inds, desc="Analyzing Auto-Correlation SNRs "):
            frame = FIBSEM_frame(fls[j], ftype=ftype)
            if pad_edges and perfrom_transformation:
                shape = [frame.YResolution, frame.XResolution]
                xmn, xmx, ymn, ymx = determine_pad_offsets(
                    shape, self.tr_matr_cum_residual
                )
                padx = np.int16(xmx - xmn)
                pady = np.int16(ymx - ymn)
                xi = np.int16(np.max([xmx, 0]))
                yi = np.int16(np.max([ymx, 0]))
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

            xsz = frame.XResolution + padx
            xa = xi + frame.XResolution
            ysz = frame.YResolution + pady
            ya = yi + frame.YResolution

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

            if sliding_evaluation_box:
                dx_eval = stop_evaluation_box[2] - start_evaluation_box[2]
                dy_eval = stop_evaluation_box[0] - start_evaluation_box[0]
            else:
                dx_eval = 0
                dy_eval = 0

            frame_imgA = np.zeros((ysz, xsz))
            if self.DetB != "None":
                frame_imgB = np.zeros((ysz, xsz))

            if invert_data:
                frame_imgA[yi:ya, xi:xa] = np.negative(frame.RawImageA.astype(float))
                if self.DetB != "None":
                    frame_imgB[yi:ya, xi:xa] = np.negative(
                        frame.RawImageB.astype(float)
                    )
            else:
                frame_imgA[yi:ya, xi:xa] = frame.RawImageA.astype(float)
                if self.DetB != "None":
                    frame_imgB[yi:ya, xi:xa] = frame.RawImageB.astype(float)

            if perfrom_transformation:
                transf = ProjectiveTransform(
                    matrix=shift_matrix
                    @ (self.tr_matr_cum_residual[j] @ inv_shift_matrix)
                )
                frame_imgA_reg = warp(
                    frame_imgA, transf, order=int_order, preserve_range=True
                )
                if self.DetB != "None":
                    frame_imgB_reg = warp(
                        frame_imgB, transf, order=int_order, preserve_range=True
                    )
            else:
                frame_imgA_reg = frame_imgA.copy()
                if self.DetB != "None":
                    frame_imgB_reg = frame_imgB.copy()

            if flipY:
                frame_imgA_reg = np.flip(frame_imgA_reg, axis=0)
                if self.DetB != "None":
                    frame_imgB_reg = np.flip(frame_imgB_reg, axis=0)

            if sliding_evaluation_box:
                xi_eval = start_evaluation_box[2] + dx_eval * j // nfrs
                yi_eval = start_evaluation_box[0] + dy_eval * j // nfrs
                if start_evaluation_box[3] > 0:
                    xa_eval = xi_eval + start_evaluation_box[3]
                else:
                    xa_eval = xsz
                if start_evaluation_box[1] > 0:
                    ya_eval = yi_eval + start_evaluation_box[1]
                else:
                    ya_eval = ysz

            """
            if invert_data:
                if test_frame.EightBit==0:
                    frame_imgA = np.negative(FIBSEM_frame(fls[j], ftype=ftype).RawImageA)
                    if self.DetB != 'None':
                        frame_imgB = np.negative(FIBSEM_frame(fls[j], ftype=ftype).RawImageB)
                else:
                    frame_imgA  =  uint8(255) - FIBSEM_frame(fls[j], ftype=ftype).RawImageA
                    if self.DetB != 'None':
                        frame_imgB  =  uint8(255) - FIBSEM_frame(fls[j], ftype=ftype).RawImageB

            else:
                frame_imgA  = FIBSEM_frame(fls[j], ftype=ftype).RawImageA
                if self.DetB != 'None':
                    frame_imgB  = FIBSEM_frame(fls[j], ftype=ftype).RawImageB

            if flipY:
                frame_imgA = np.flip(frame_imgA, axis=0)
                frame_imgB = np.flip(frame_imgB, axis=0)
            """
            frame_imgA_eval = frame_imgA_reg[yi_eval:ya_eval, xi_eval:xa_eval]
            SNR_png = os.path.splitext(os.path.split(fls[j])[1])[0] + ".png"
            SNR_png_fname = os.path.join(data_dir, SNR_png)
            ImageA_xSNR, ImageA_ySNR, ImageA_rSNR = Single_Image_SNR(
                frame_imgA_eval,
                extrapolate_signal=extrapolate_signal,
                save_res_png=save_res_png,
                res_fname=SNR_png_fname.replace(".png", "_ImgA_SNR.png"),
                img_label="Image A, frame={:d}".format(j),
            )
            xSNRAs.append(ImageA_xSNR)
            ySNRAs.append(ImageA_ySNR)
            rSNRAs.append(ImageA_rSNR)
            if self.DetB != "None":
                frame_imgB_eval = frame_imgB_reg[yi_eval:ya_eval, xi_eval:xa_eval]
                ImageB_xSNR, ImageB_ySNR, ImageB_rSNR = Single_Image_SNR(
                    frame_imgB_eval,
                    extrapolate_signal=extrapolate_signal,
                    save_res_png=save_res_png,
                    res_fname=SNR_png_fname.replace(".png", "_ImgB_SNR.png"),
                    img_label="Image B, frame={:d}".format(j),
                )
                xSNRBs.append(ImageB_xSNR)
                ySNRBs.append(ImageB_ySNR)
                rSNRBs.append(ImageB_rSNR)

        fig, ax = subplots(1, 1, figsize=(6, 4))
        ax.plot(frame_inds, xSNRAs, "r+", label="Image A x-SNR")
        ax.plot(frame_inds, ySNRAs, "b+", label="Image A y-SNR")
        ax.plot(frame_inds, rSNRAs, "g+", label="Image A r-SNR")
        if self.DetB != "None":
            ax.plot(frame_inds, xSNRBs, "rx", linestyle="dotted", label="Image B x-SNR")
            ax.plot(frame_inds, ySNRBs, "bx", linestyle="dotted", label="Image B y-SNR")
            ax.plot(frame_inds, rSNRBs, "gx", linestyle="dotted", label="Image B r-SNR")
            ImgB_fraction_xSNR = np.mean(
                np.array(xSNRBs) / (np.array(xSNRAs) + np.array(xSNRBs))
            )
            ImgB_fraction_ySNR = np.mean(
                np.array(ySNRBs) / (np.array(ySNRAs) + np.array(ySNRBs))
            )
            ImgB_fraction_rSNR = np.mean(
                np.array(rSNRBs) / (np.array(rSNRAs) + np.array(rSNRBs))
            )
            if ImgB_fraction < 1e-9:
                ImgB_fraction = ImgB_fraction_rSNR
            ax.text(
                0.1,
                0.5,
                "ImgB fraction (x-SNR) = {:.4f}".format(ImgB_fraction_xSNR),
                color="r",
                transform=ax.transAxes,
            )
            ax.text(
                0.1,
                0.42,
                "ImgB fraction (y-SNR) = {:.4f}".format(ImgB_fraction_ySNR),
                color="b",
                transform=ax.transAxes,
            )
            ax.text(
                0.1,
                0.34,
                "ImgB fraction (r-SNR) = {:.4f}".format(ImgB_fraction_rSNR),
                color="g",
                transform=ax.transAxes,
            )

            xSNRFs = []
            ySNRFs = []
            rSNRFs = []
            for j in tqdm(
                frame_inds, desc="Re-analyzing Auto-Correlation SNRs for fused image"
            ):
                frame = FIBSEM_frame(fls[j], ftype=ftype)
                if pad_edges and perfrom_transformation:
                    shape = [frame.YResolution, frame.XResolution]
                    xmn, xmx, ymn, ymx = determine_pad_offsets(
                        shape, self.tr_matr_cum_residual
                    )
                    padx = np.int16(xmx - xmn)
                    pady = np.int16(ymx - ymn)
                    xi = np.int16(np.max([xmx, 0]))
                    yi = np.int16(np.max([ymx, 0]))
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

                xsz = frame.XResolution + padx
                xa = xi + frame.XResolution
                ysz = frame.YResolution + pady
                ya = yi + frame.YResolution

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

                if sliding_evaluation_box:
                    dx_eval = stop_evaluation_box[2] - start_evaluation_box[2]
                    dy_eval = stop_evaluation_box[0] - start_evaluation_box[0]
                else:
                    dx_eval = 0
                    dy_eval = 0

                frame_imgA = np.zeros((ysz, xsz))
                if self.DetB != "None":
                    frame_imgB = np.zeros((ysz, xsz))

                if invert_data:
                    frame_imgA[yi:ya, xi:xa] = np.negative(
                        frame.RawImageA.astype(float)
                    )
                    if self.DetB != "None":
                        frame_imgB[yi:ya, xi:xa] = np.negative(
                            frame.RawImageB.astype(float)
                        )
                else:
                    frame_imgA[yi:ya, xi:xa] = frame.RawImageA.astype(float)
                    if self.DetB != "None":
                        frame_imgB[yi:ya, xi:xa] = frame.RawImageB.astype(float)

                if perfrom_transformation:
                    transf = ProjectiveTransform(
                        matrix=shift_matrix
                        @ (self.tr_matr_cum_residual[j] @ inv_shift_matrix)
                    )
                    frame_imgA_reg = warp(
                        frame_imgA, transf, order=int_order, preserve_range=True
                    )
                    if self.DetB != "None":
                        frame_imgB_reg = warp(
                            frame_imgB, transf, order=int_order, preserve_range=True
                        )
                else:
                    frame_imgA_reg = frame_imgA.copy()
                    if self.DetB != "None":
                        frame_imgB_reg = frame_imgB.copy()

                if flipY:
                    frame_imgA_reg = np.flip(frame_imgA_reg, axis=0)
                    if self.DetB != "None":
                        frame_imgB_reg = np.flip(frame_imgB_reg, axis=0)

                if sliding_evaluation_box:
                    xi_eval = start_evaluation_box[2] + dx_eval * j // nfrs
                    yi_eval = start_evaluation_box[0] + dy_eval * j // nfrs
                    if start_evaluation_box[3] > 0:
                        xa_eval = xi_eval + start_evaluation_box[3]
                    else:
                        xa_eval = xsz
                    if start_evaluation_box[1] > 0:
                        ya_eval = yi_eval + start_evaluation_box[1]
                    else:
                        ya_eval = ysz

                frame_imgA_eval = frame_imgA_reg[yi_eval:ya_eval, xi_eval:xa_eval]
                frame_imgB_eval = frame_imgB_reg[yi_eval:ya_eval, xi_eval:xa_eval]

                frame_imgF_eval = (
                    frame_imgA_eval * (1.0 - ImgB_fraction)
                    + frame_imgB_eval * ImgB_fraction
                )
                ImageF_xSNR, ImageF_ySNR, ImageF_rSNR = Single_Image_SNR(
                    frame_imgF_eval,
                    extrapolate_signal=extrapolate_signal,
                    save_res_png=save_res_png,
                    res_fname=SNR_png_fname.replace(
                        ".png", "_ImgB_fr{:.3f}_SNR.png".format(ImgB_fraction)
                    ),
                    img_label="Fused, ImB_fr={:.4f}, frame={:d}".format(
                        ImgB_fraction, j
                    ),
                )
                xSNRFs.append(ImageF_xSNR)
                ySNRFs.append(ImageF_ySNR)
                rSNRFs.append(ImageF_rSNR)

            ax.plot(
                frame_inds, xSNRFs, "rd", linestyle="dashed", label="Fused Image x-SNR"
            )
            ax.plot(
                frame_inds, ySNRFs, "bd", linestyle="dashed", label="Fused Image y-SNR"
            )
            ax.plot(
                frame_inds, rSNRFs, "gd", linestyle="dashed", label="Fused Image r-SNR"
            )

        else:
            ImgB_fraction_xSNR = 0.0
            ImgB_fraction_ySNR = 0.0
            ImgB_fraction_rSNR = 0.0
        ax.grid(True)
        ax.legend()
        ax.set_title(Sample_ID + "  " + data_dir, fontsize=8)
        ax.set_xlabel("Frame")
        ax.set_ylabel("SNR")
        if save_res_png:
            fig_filename = os.path.join(
                data_dir, os.path.splitext(fnm_reg)[0] + "SNR_evaluation_mult_frame.png"
            )
            fig.savefig(fig_filename, dpi=300)

        return ImgB_fraction_xSNR, ImgB_fraction_ySNR, ImgB_fraction_rSNR

    def evaluate_ImgB_fractions(self, ImgB_fractions, frame_inds, **kwargs):
        """
        Calculate NCC and SNR vs Image B fraction over a set of frames.

        ImgB_fractions : list
            List of fractions to estimate the NCC and SNR
        frame_inds : int array
            array of frame indices to perform NCC / SNR evaluation

        kwargs
        ---------
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        extrapolate_signal : boolean
            extrapolate to find signal autocorrelationb at 0-point (without noise). Default is True
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        data_dir : str
            data directory (path)
        Sample_ID : str
            Sample ID

        invert_data : boolean
            If True - the data is inverted
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.


        Returns
        SNRimpr_max_position, SNRimpr_max, ImgB_fractions, SNRs
        """
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        if hasattr(self, "invert_data"):
            invert_data = kwargs.get("invert_data", self.invert_data)
        else:
            invert_data = kwargs.get("invert_data", False)
        if hasattr(self, "flipY"):
            flipY = kwargs.get("flipY", self.flipY)
        else:
            flipY = kwargs.get("flipY", flipY)
        save_res_png = kwargs.get("save_res_png", self.save_res_png)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        save_sample_frames_png = kwargs.get("save_sample_frames_png", False)
        extrapolate_signal = kwargs.get("extrapolate_signal", True)

        nbr = len(ImgB_fractions)
        kwargs["zbin_factor"] = 1

        test_frame = FIBSEM_frame(self.fls[frame_inds[0]])

        xi = 0
        yi = 0
        xsz = test_frame.XResolution
        xa = xi + xsz
        ysz = test_frame.YResolution
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

        br_results = []
        xSNRFs = []
        ySNRFs = []
        rSNRFs = []

        for ImgB_fraction in tqdm(ImgB_fractions, desc="Evaluating Img B fractions"):
            kwargs["ImgB_fraction"] = ImgB_fraction
            kwargs["disp_res"] = False
            kwargs["evaluation_box"] = evaluation_box
            kwargs["flipY"] = flipY
            kwargs["invert_data"] = invert_data
            DASK_client = ""
            kwargs["disp_res"] = False
            br_res, br_res_xlsx = self.transform_and_save(
                DASK_client,
                save_transformed_dataset=False,
                save_registration_summary=False,
                frame_inds=frame_inds,
                use_DASK=False,
                save_sample_frames_png=False,
                eval_metrics=["NCC"],
                **kwargs
            )
            br_results.append(br_res)

            if invert_data:
                if test_frame.EightBit == 0:
                    frame_imgA = np.negative(test_frame.RawImageA)
                    if self.DetB != "None":
                        frame_imgB = np.negative(test_frame.RawImageB)
                else:
                    frame_imgA = uint8(255) - test_frame.RawImageA
                    if self.DetB != "None":
                        frame_imgB = uint8(255) - test_frame.RawImageB
            else:
                frame_imgA = test_frame.RawImageA
                if self.DetB != "None":
                    frame_imgB = test_frame.RawImageB
            if flipY:
                frame_imgA = np.flip(frame_imgA, axis=0)
                frame_imgB = np.flip(frame_imgB, axis=0)

            frame_imgA_eval = frame_imgA[yi_eval:ya_eval, xi_eval:xa_eval]
            frame_imgB_eval = frame_imgB[yi_eval:ya_eval, xi_eval:xa_eval]

            frame_imgF_eval = (
                frame_imgA_eval * (1.0 - ImgB_fraction)
                + frame_imgB_eval * ImgB_fraction
            )
            ImageF_xSNR, ImageF_ySNR, ImageF_rSNR = Single_Image_SNR(
                frame_imgF_eval,
                extrapolate_signal=extrapolate_signal,
                disp_res=False,
                save_res_png=False,
                res_fname="",
                img_label="",
            )
            xSNRFs.append(ImageF_xSNR)
            ySNRFs.append(ImageF_ySNR)
            rSNRFs.append(ImageF_rSNR)

        fig, axs = subplots(4, 1, figsize=(6, 11))
        fig.subplots_adjust(
            left=0.12, bottom=0.06, right=0.99, top=0.96, wspace=0.25, hspace=0.24
        )
        try:
            ncc0 = (br_results[0])["NCC"]
        except:
            ncc0 = (br_results[0])["Image NCC"]
        SNR0 = ncc0 / (1 - ncc0)
        SNRimpr_cc = []
        SNRs = []

        for j, (ImgB_fraction, br_result) in enumerate(zip(ImgB_fractions, br_results)):
            my_col = get_cmap("gist_rainbow_r")((nbr - j) / (nbr - 1))
            try:
                ncc = br_result["NCC"]
            except:
                ncc = br_result["Image NCC"]
            SNR = ncc / (1.0 - ncc)
            frames_local = br_result["Frame"]
            axs[0].plot(
                frames_local,
                SNR,
                color=my_col,
                label="ImgB fraction = {:.2f}".format(ImgB_fraction),
            )
            axs[1].plot(
                frames_local,
                SNR / SNR0,
                color=my_col,
                label="ImgB fraction = {:.2f}".format(ImgB_fraction),
            )
            SNRimpr_cc.append(np.mean(SNR / SNR0))
            SNRs.append(np.mean(SNR))

        SNRimpr_ac = np.array(rSNRFs) / rSNRFs[0]

        SNRimpr_cc_max = np.max(SNRimpr_cc)
        SNRimpr_cc_max_ind = np.argmax(SNRimpr_cc)
        ImgB_fraction_max = ImgB_fractions[SNRimpr_cc_max_ind]
        xi = max(0, (SNRimpr_cc_max_ind - 3))
        xa = min((SNRimpr_cc_max_ind + 3), len(ImgB_fractions))
        ImgB_fr_range = ImgB_fractions[xi:xa]
        SNRimpr_cc_range = SNRimpr_cc[xi:xa]
        popt = np.polyfit(ImgB_fr_range, SNRimpr_cc_range, 2)
        SNRimpr_cc_fit_max_pos = -0.5 * popt[1] / popt[0]
        ImgB_fr_fit_cc = np.linspace(ImgB_fr_range[0], ImgB_fr_range[-1], 21)
        SNRimpr_cc_fit = np.polyval(popt, ImgB_fr_fit_cc)
        if (
            popt[0] < 0
            and SNRimpr_cc_fit_max_pos > ImgB_fractions[0]
            and SNRimpr_cc_fit_max_pos < ImgB_fractions[-1]
        ):
            SNRimpr_cc_max_position = SNRimpr_cc_fit_max_pos
            SNRimpr_cc_max = np.polyval(popt, SNRimpr_cc_max_position)
        else:
            SNRimpr_cc_max_position = ImgB_fraction_max

        SNRimpr_ac_max = np.max(SNRimpr_ac)
        SNRimpr_ac_max_ind = np.argmax(SNRimpr_ac)
        ImgB_fraction_max = ImgB_fractions[SNRimpr_ac_max_ind]
        xi = max(0, (SNRimpr_ac_max_ind - 3))
        xa = min((SNRimpr_ac_max_ind + 3), len(ImgB_fractions))
        ImgB_fr_range = ImgB_fractions[xi:xa]
        SNRimpr_ac_range = SNRimpr_ac[xi:xa]
        popt = np.polyfit(ImgB_fr_range, SNRimpr_ac_range, 2)
        SNRimpr_ac_fit_max_pos = -0.5 * popt[1] / popt[0]
        ImgB_fr_fit_ac = np.linspace(ImgB_fr_range[0], ImgB_fr_range[-1], 21)
        SNRimpr_ac_fit = np.polyval(popt, ImgB_fr_fit_ac)
        if (
            popt[0] < 0
            and SNRimpr_ac_fit_max_pos > ImgB_fractions[0]
            and SNRimpr_ac_fit_max_pos < ImgB_fractions[-1]
        ):
            SNRimpr_ac_max_position = SNRimpr_ac_fit_max_pos
            SNRimpr_ac_max = np.polyval(popt, SNRimpr_ac_max_position)
        else:
            SNRimpr_ac_max_position = ImgB_fraction_max

        fs = 10
        axs[0].grid(True)
        axs[0].set_ylabel("Frame-to-Frame SNR", fontsize=fs)
        axs[0].set_xlabel("Frame", fontsize=fs)
        axs[0].legend(fontsize=fs - 1)
        axs[0].set_title(Sample_ID + "  " + data_dir, fontsize=fs)
        axs[1].grid(True)
        axs[1].set_ylabel("Frame-to-Frame SNR Improvement", fontsize=fs)
        axs[1].set_xlabel("Frame", fontsize=fs)

        axs[2].plot(ImgB_fractions, rSNRFs, "rd", label="Data (auto-correlation)")
        axs[2].grid(True)
        axs[2].set_ylabel("Auto-Corr SNR", fontsize=fs)

        axs[3].plot(ImgB_fractions, SNRimpr_cc, "cs", label="Data (cross-corr.)")
        axs[3].plot(ImgB_fr_fit_cc, SNRimpr_cc_fit, "b", label="Fit (cross-corr.)")
        axs[3].plot(
            SNRimpr_cc_max_position,
            SNRimpr_cc_max,
            "bx",
            markersize=10,
            label="Max SNR Impr. (cc)",
        )
        axs[3].text(
            0.4,
            0.35,
            "Max CC SNR Improvement={:.3f}".format(SNRimpr_cc_max),
            transform=axs[3].transAxes,
            fontsize=fs,
        )
        axs[3].text(
            0.4,
            0.25,
            "@ Img B Fraction ={:.3f}".format(SNRimpr_cc_max_position),
            transform=axs[3].transAxes,
            fontsize=fs,
        )
        axs[3].plot(ImgB_fractions, SNRimpr_ac, "rd", label="Data (auto-corr.)")
        axs[3].plot(ImgB_fr_fit_ac, SNRimpr_ac_fit, "magenta", label="Fit (auto-corr.)")
        axs[3].plot(
            SNRimpr_ac_max_position,
            SNRimpr_ac_max,
            "mx",
            markersize=10,
            label="Max SNR Impr. (ac)",
        )
        axs[3].text(
            0.4,
            0.15,
            "Max AC SNR Improvement={:.3f}".format(SNRimpr_ac_max),
            transform=axs[3].transAxes,
            fontsize=fs,
        )
        axs[3].text(
            0.4,
            0.05,
            "@ Img B Fraction ={:.3f}".format(SNRimpr_ac_max_position),
            transform=axs[3].transAxes,
            fontsize=fs,
        )

        axs[3].legend(fontsize=fs - 2, loc="upper left")
        axs[3].grid(True)
        axs[3].set_ylabel("Mean SNR improvement", fontsize=fs)
        axs[3].set_xlabel("Image B fraction", fontsize=fs)

        if save_res_png:
            fname_image = os.path.join(
                data_dir,
                os.path.splitext(fnm_reg)[0] + "_SNR_vs_ImgB_ratio_evaluation.png",
            )
            fig.savefig(fname_image, dpi=300)

        return SNRimpr_cc_max_position, SNRimpr_cc_max, ImgB_fractions, SNRs, rSNRFs
