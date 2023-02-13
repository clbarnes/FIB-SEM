"""
# FIBSEM image reader
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


class FIBSEM_frame:
    """
    A class representing single FIB-SEM data frame.
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    Contains the info/settings on a single FIB-SEM data frame and the procedures that can be performed on it.

    Attributes (only some more important are listed here)
    ----------
    fname : str
        filename of the individual data frame
    header : str
        1024 bytes - header
    FileVersion : int
        file version number
    ChanNum : int
        Number of channels
    EightBit : int
        8-bit data switch: 0 for 16-bit data, 1 for 8-bit data
    Scaling : 2D array of floats
        scaling parameters allowing to convert I16 data into actual electron counts
    Sample_ID : str
        Sample_ID
    Notes : str
        Experiment notes
    DetA : str
        Detector A name
    DetB : str
        Detector B name ('None' if there is no Detector B)
    XResolution : int
        number of pixels - frame size in horizontal direction
    YResolution : int
        number of pixels - frame size in vertical direction
    PixelSize : float
        pixel size in nm. Default is 8.0

    Methods
    -------
    print_header()
        Prints a formatted content of the file header

    display_images()
        Display auto-scaled detector images without saving the figure into the file.

    save_images_jpeg(**kwargs)
        Display auto-scaled detector images and save the figure into JPEG file (s).

    save_images_tif(images_to_save = 'Both')
        Save the detector images into TIF file (s).

    get_image_min_max(image_name = 'ImageA', thr_min = 1.0e-4, thr_max = 1.0e-3, nbins=256, disp_res = False)
        Calculates the data range of the EM data.

    RawImageA_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
        Convert the Image A into 8-bit array

    RawImageB_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
            Convert the Image B into 8-bit array

    save_snapshot(display = True, dpi=300, thr_min = 1.0e-3, thr_max = 1.0e-3, nbins=256, **kwargs):
        Builds an image that contains both the Detector A and Detector B (if present) images as well as a table with important FIB-SEM parameters.

    analyze_noise_ROIs(**kwargs):
        Analyses the noise statistics in the selected ROI's of the EM data. (Calls Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs):)

    analyze_noise_statistics(**kwargs):
        Analyses the noise statistics of the EM data image. (Calls Single_Image_Noise_Statistics(img, **kwargs):)

    analyze_SNR_autocorr(**kwargs):
        Estimates SNR using auto-correlation analysis of a single image. (Calls Single_Image_SNR(img, **kwargs):)

    show_eval_box(**kwargs):
        Show the box used for evaluating the noise

    determine_field_fattening_parameters(**kwargs):
        Perfrom 2D parabolic fit (calls Perform_2D_fit(Img, estimator, **kwargs)) and determine the field-flattening parameters

    flatten_image(**kwargs):
        Flatten the image
    """

    def __init__(self, fname, **kwargs):
        self.fname = fname
        self.ftype = kwargs.get(
            "ftype", 0
        )  # ftype=0 - Shan Xu's binary format  ftype=1 - tif files
        self.use_dask_arrays = kwargs.get("use_dask_arrays", False)
        if self.ftype == 1:
            self.RawImageA = tiff.imread(fname)

        # for tif files
        if self.ftype == 1:
            self.FileVersion = -1
            self.DetA = "Detector A"  # Name of detector A
            self.DetB = "None"  # Name of detector B
            try:
                with PILImage.open(self.fname) as img:
                    tif_header = {TAGS[key]: img.tag[key] for key in img.tag_v2}
                    self.header = tif_header
                try:
                    if tif_header["BitsPerSample"][0] == 8:
                        self.EightBit = 1
                    else:
                        self.EightBit = 0
                except:
                    self.EightBit = int(type(self.RawImageA[0, 0]) == np.uint8)
            except:
                self.header = ""
                self.EightBit = int(type(self.RawImageA[0, 0]) == np.uint8)

            self.PixelSize = kwargs.get("PixelSize", 8.0)
            self.Sample_ID = kwargs.get("Sample_ID", "")
            self.YResolution, self.XResolution = self.RawImageA.shape
            self.Scaling = np.array([[1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]]).T

        # for Shan Xu's data files
        if self.ftype == 0:
            fid = open(self.fname, "rb")
            fid.seek(0, 0)
            self.header = fid.read(1024)  # Read in self.header
            self.FileMagicNum = unpack(">L", self.header[0:4])[
                0
            ]  # Read in magic number, should be 3555587570
            self.FileVersion = unpack(">h", self.header[4:6])[
                0
            ]  # Read in file version number
            self.FileType = unpack(">h", self.header[6:8])[
                0
            ]  # Read in file type, 1 is Zeiss Neon detectors
            self.SWdate = (unpack("10s", self.header[8:18])[0]).decode(
                "utf-8"
            )  # Read in SW date
            self.TimeStep = unpack(">d", self.header[24:32])[
                0
            ]  # Read in AI sampling time (including oversampling) in seconds
            self.ChanNum = unpack("b", self.header[32:33])[
                0
            ]  # Read in number of channels
            self.EightBit = unpack("b", self.header[33:34])[
                0
            ]  # Read in 8-bit data switch

            if self.FileVersion == 1:
                # Read in self.AI channel self.Scaling factors, (col#: self.AI#), (row#: offset, gain, 2nd order, 3rd order)
                self.ScalingS = unpack(
                    ">" + str(4 * self.ChanNum) + "d",
                    self.header[36 : (36 + self.ChanNum * 32)],
                )
            elif (
                self.FileVersion == 2
                or self.FileVersion == 3
                or self.FileVersion == 4
                or self.FileVersion == 5
                or self.FileVersion == 6
            ):
                self.ScalingS = unpack(
                    ">" + str(4 * self.ChanNum) + "f",
                    self.header[36 : (36 + self.ChanNum * 16)],
                )
            else:
                self.ScalingS = unpack(">8f", self.header[36:68])
                self.Scaling = transpose(np.asarray(self.ScalingS).reshape(2, 4))

            if self.FileVersion > 8:
                self.RestartFlag = unpack("b", self.header[68:69])[
                    0
                ]  # Read in restart flag
                self.StageMove = unpack("b", self.header[69:70])[
                    0
                ]  # Read in stage move flag
                self.FirstPixelX = unpack(">l", self.header[70:74])[
                    0
                ]  # Read in first pixel X coordinate (center = 0)
                self.FirstPixelY = unpack(">l", self.header[74:78])[
                    0
                ]  # Read in first pixel Y coordinate (center = 0)

            self.XResolution = unpack(">L", self.header[100:104])[
                0
            ]  # Read X resolution
            self.YResolution = unpack(">L", self.header[104:108])[
                0
            ]  # Read Y resolution

            if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3:
                self.Oversampling = unpack(">B", self.header[108:109])[
                    0
                ]  # self.AI oversampling
                self.AIDelay = unpack(">h", self.header[109:111])[
                    0
                ]  # self.AI delay (# of samples)
            else:
                self.Oversampling = unpack(">H", self.header[108:110])[0]

            self.ZeissScanSpeed = unpack("b", self.header[111:112])[
                0
            ]  # Scan speed (Zeiss #)

            if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3:
                self.ScanRate = unpack(">d", self.header[112:120])[
                    0
                ]  # Actual AO (scanning) rate
                self.FramelineRampdownRatio = unpack(">d", self.header[120:128])[
                    0
                ]  # Frameline rampdown ratio
                self.Xmin = unpack(">d", self.header[128:136])[
                    0
                ]  # X coil minimum voltage
                self.Xmax = unpack(">d", self.header[136:144])[
                    0
                ]  # X coil maximum voltage
                self.Detmin = -10  # Detector minimum voltage
                self.Detmax = 10  # Detector maximum voltage
            else:
                self.ScanRate = unpack(">f", self.header[112:116])[
                    0
                ]  # Actual AO (scanning) rate
                self.FramelineRampdownRatio = unpack(">f", self.header[116:120])[
                    0
                ]  # Frameline rampdown ratio
                self.Xmin = unpack(">f", self.header[120:124])[
                    0
                ]  # X coil minimum voltage
                self.Xmax = unpack(">f", self.header[124:128])[
                    0
                ]  # X coil maximum voltage
                self.Detmin = unpack(">f", self.header[128:132])[
                    0
                ]  # Detector minimum voltage
                self.Detmax = unpack(">f", self.header[132:136])[
                    0
                ]  # Detector maximum voltage
                self.DecimatingFactor = unpack(">H", self.header[136:138])[
                    0
                ]  # Decimating factor

            self.AI1 = unpack("b", self.header[151:152])[0]  # self.AI Ch1
            self.AI2 = unpack("b", self.header[152:153])[0]  # self.AI Ch2
            self.AI3 = unpack("b", self.header[153:154])[0]  # self.AI Ch3
            self.AI4 = unpack("b", self.header[154:155])[0]  # self.AI Ch4

            self.Notes = (unpack("200s", self.header[180:380])[0]).decode(
                "utf-8"
            )  # Read in notes

            if self.FileVersion > 8:
                self.Sample_ID = (
                    (unpack("25s", self.header[155:180])[0])
                    .decode("utf-8")
                    .strip("\x00")
                )  # Read in Sample ID
            else:
                self.Sample_ID = self.Notes.split(",")[0].strip("\x00")

            if self.FileVersion == 1 or self.FileVersion == 2:
                self.DetA = (unpack("10s", self.header[380:390])[0]).decode(
                    "utf-8"
                )  # Name of detector A
                self.DetB = (unpack("18s", self.header[390:408])[0]).decode(
                    "utf-8"
                )  # Name of detector B
                self.DetC = (unpack("20s", self.header[700:720])[0]).decode(
                    "utf-8"
                )  # Name of detector C
                self.DetD = (unpack("20s", self.header[720:740])[0]).decode(
                    "utf-8"
                )  # Name of detector D
                self.Mag = unpack(">d", self.header[408:416])[0]  # Magnification
                self.PixelSize = unpack(">d", self.header[416:424])[
                    0
                ]  # Pixel size in nm
                self.WD = unpack(">d", self.header[424:432])[
                    0
                ]  # Working distance in mm
                self.EHT = unpack(">d", self.header[432:440])[0]  # EHT in kV
                self.SEMApr = unpack("b", self.header[440:441])[
                    0
                ]  # SEM aperture number
                self.HighCurrent = unpack("b", self.header[441:442])[
                    0
                ]  # high current mode (1=on, 0=off)
                self.SEMCurr = unpack(">d", self.header[448:456])[
                    0
                ]  # SEM probe current in A
                self.SEMRot = unpack(">d", self.header[456:464])[
                    0
                ]  # SEM scan roation in degree
                self.ChamVac = unpack(">d", self.header[464:472])[0]  # Chamber vacuum
                self.GunVac = unpack(">d", self.header[472:480])[0]  # E-gun vacuum
                self.SEMStiX = unpack(">d", self.header[480:488])[0]  # SEM stigmation X
                self.SEMStiY = unpack(">d", self.header[488:496])[0]  # SEM stigmation Y
                self.SEMAlnX = unpack(">d", self.header[496:504])[
                    0
                ]  # SEM aperture alignment X
                self.SEMAlnY = unpack(">d", self.header[504:512])[
                    0
                ]  # SEM aperture alignment Y
                self.StageX = unpack(">d", self.header[512:520])[
                    0
                ]  # Stage position X in mm
                self.StageY = unpack(">d", self.header[520:528])[
                    0
                ]  # Stage position Y in mm
                self.StageZ = unpack(">d", self.header[528:536])[
                    0
                ]  # Stage position Z in mm
                self.StageT = unpack(">d", self.header[536:544])[
                    0
                ]  # Stage position T in degree
                self.StageR = unpack(">d", self.header[544:552])[
                    0
                ]  # Stage position R in degree
                self.StageM = unpack(">d", self.header[552:560])[
                    0
                ]  # Stage position M in mm
                self.BrightnessA = unpack(">d", self.header[560:568])[
                    0
                ]  # Detector A brightness (%)
                self.ContrastA = unpack(">d", self.header[568:576])[
                    0
                ]  # Detector A contrast (%)
                self.BrightnessB = unpack(">d", self.header[576:584])[
                    0
                ]  # Detector B brightness (%)
                self.ContrastB = unpack(">d", self.header[584:592])[
                    0
                ]  # Detector B contrast (%)
                self.Mode = unpack("b", self.header[600:601])[
                    0
                ]  # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
                self.FIBFocus = unpack(">d", self.header[608:616])[0]  # FIB focus in kV
                self.FIBProb = unpack("b", self.header[616:617])[0]  # FIB probe number
                self.FIBCurr = unpack(">d", self.header[624:632])[
                    0
                ]  # FIB emission current
                self.FIBRot = unpack(">d", self.header[632:640])[0]  # FIB scan rotation
                self.FIBAlnX = unpack(">d", self.header[640:648])[
                    0
                ]  # FIB aperture alignment X
                self.FIBAlnY = unpack(">d", self.header[648:656])[
                    0
                ]  # FIB aperture alignment Y
                self.FIBStiX = unpack(">d", self.header[656:664])[0]  # FIB stigmation X
                self.FIBStiY = unpack(">d", self.header[664:672])[0]  # FIB stigmation Y
                self.FIBShiftX = unpack(">d", self.header[672:680])[
                    0
                ]  # FIB beam shift X in micron
                self.FIBShiftY = unpack(">d", self.header[680:688])[
                    0
                ]  # FIB beam shift Y in micron
            else:
                self.DetA = (unpack("10s", self.header[380:390])[0]).decode(
                    "utf-8"
                )  # Name of detector A
                self.DetB = (unpack("18s", self.header[390:408])[0]).decode(
                    "utf-8"
                )  # Name of detector B
                self.DetC = (unpack("20s", self.header[410:430])[0]).decode(
                    "utf-8"
                )  # Name of detector C
                self.DetD = (unpack("20s", self.header[430:450])[0]).decode(
                    "utf-8"
                )  # Name of detector D
                self.Mag = unpack(">f", self.header[460:464])[0]  # Magnification
                self.PixelSize = unpack(">f", self.header[464:468])[
                    0
                ]  # Pixel size in nm
                self.WD = unpack(">f", self.header[468:472])[
                    0
                ]  # Working distance in mm
                self.EHT = unpack(">f", self.header[472:476])[0]  # EHT in kV
                self.SEMApr = unpack("b", self.header[480:481])[
                    0
                ]  # SEM aperture number
                self.HighCurrent = unpack("b", self.header[481:482])[
                    0
                ]  # high current mode (1=on, 0=off)
                self.SEMCurr = unpack(">f", self.header[490:494])[
                    0
                ]  # SEM probe current in A
                self.SEMRot = unpack(">f", self.header[494:498])[
                    0
                ]  # SEM scan roation in degree
                self.ChamVac = unpack(">f", self.header[498:502])[0]  # Chamber vacuum
                self.GunVac = unpack(">f", self.header[502:506])[0]  # E-gun vacuum
                self.SEMShiftX = unpack(">f", self.header[510:514])[
                    0
                ]  # SEM beam shift X
                self.SEMShiftY = unpack(">f", self.header[514:518])[
                    0
                ]  # SEM beam shift Y
                self.SEMStiX = unpack(">f", self.header[518:522])[0]  # SEM stigmation X
                self.SEMStiY = unpack(">f", self.header[522:526])[0]  # SEM stigmation Y
                self.SEMAlnX = unpack(">f", self.header[526:530])[
                    0
                ]  # SEM aperture alignment X
                self.SEMAlnY = unpack(">f", self.header[530:534])[
                    0
                ]  # SEM aperture alignment Y
                self.StageX = unpack(">f", self.header[534:538])[
                    0
                ]  # Stage position X in mm
                self.StageY = unpack(">f", self.header[538:542])[
                    0
                ]  # Stage position Y in mm
                self.StageZ = unpack(">f", self.header[542:546])[
                    0
                ]  # Stage position Z in mm
                self.StageT = unpack(">f", self.header[546:550])[
                    0
                ]  # Stage position T in degree
                self.StageR = unpack(">f", self.header[550:554])[
                    0
                ]  # Stage position R in degree
                self.StageM = unpack(">f", self.header[554:558])[
                    0
                ]  # Stage position M in mm
                self.BrightnessA = unpack(">f", self.header[560:564])[
                    0
                ]  # Detector A brightness (%)
                self.ContrastA = unpack(">f", self.header[564:568])[
                    0
                ]  # Detector A contrast (%)
                self.BrightnessB = unpack(">f", self.header[568:572])[
                    0
                ]  # Detector B brightness (%)
                self.ContrastB = unpack(">f", self.header[572:576])[
                    0
                ]  # Detector B contrast (%)
                self.Mode = unpack("b", self.header[600:601])[
                    0
                ]  # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
                self.FIBFocus = unpack(">f", self.header[604:608])[0]  # FIB focus in kV
                self.FIBProb = unpack("b", self.header[608:609])[0]  # FIB probe number
                self.FIBCurr = unpack(">f", self.header[620:624])[
                    0
                ]  # FIB emission current
                self.FIBRot = unpack(">f", self.header[624:628])[0]  # FIB scan rotation
                self.FIBAlnX = unpack(">f", self.header[628:632])[
                    0
                ]  # FIB aperture alignment X
                self.FIBAlnY = unpack(">f", self.header[632:636])[
                    0
                ]  # FIB aperture alignment Y
                self.FIBStiX = unpack(">f", self.header[636:640])[0]  # FIB stigmation X
                self.FIBStiY = unpack(">f", self.header[640:644])[0]  # FIB stigmation Y
                self.FIBShiftX = unpack(">f", self.header[644:648])[
                    0
                ]  # FIB beam shift X in micron
                self.FIBShiftY = unpack(">f", self.header[648:652])[
                    0
                ]  # FIB beam shift Y in micron

            if self.FileVersion > 4:
                self.MillingXResolution = unpack(">L", self.header[652:656])[
                    0
                ]  # FIB milling X resolution
                self.MillingYResolution = unpack(">L", self.header[656:660])[
                    0
                ]  # FIB milling Y resolution
                self.MillingXSize = unpack(">f", self.header[660:664])[
                    0
                ]  # FIB milling X size (um)
                self.MillingYSize = unpack(">f", self.header[664:668])[
                    0
                ]  # FIB milling Y size (um)
                self.MillingULAng = unpack(">f", self.header[668:672])[
                    0
                ]  # FIB milling upper left inner angle (deg)
                self.MillingURAng = unpack(">f", self.header[672:676])[
                    0
                ]  # FIB milling upper right inner angle (deg)
                self.MillingLineTime = unpack(">f", self.header[676:680])[
                    0
                ]  # FIB line milling time (s)
                self.FIBFOV = unpack(">f", self.header[680:684])[0]  # FIB FOV (um)
                self.MillingLinesPerImage = unpack(">H", self.header[684:686])[
                    0
                ]  # FIB milling lines per image
                self.MillingPIDOn = unpack(">b", self.header[686:687])[
                    0
                ]  # FIB milling PID on
                self.MillingPIDMeasured = unpack(">b", self.header[689:690])[
                    0
                ]  # FIB milling PID measured (0:specimen, 1:beamdump)
                self.MillingPIDTarget = unpack(">f", self.header[690:694])[
                    0
                ]  # FIB milling PID target
                self.MillingPIDTargetSlope = unpack(">f", self.header[694:698])[
                    0
                ]  # FIB milling PID target slope
                self.MillingPIDP = unpack(">f", self.header[698:702])[
                    0
                ]  # FIB milling PID P
                self.MillingPIDI = unpack(">f", self.header[702:706])[
                    0
                ]  # FIB milling PID I
                self.MillingPIDD = unpack(">f", self.header[706:710])[
                    0
                ]  # FIB milling PID D
                self.MachineID = (unpack("30s", self.header[800:830])[0]).decode(
                    "utf-8"
                )  # Machine ID
                self.SEMSpecimenI = unpack(">f", self.header[672:676])[
                    0
                ]  # SEM specimen current (nA)

            if self.FileVersion > 5:
                self.Temperature = unpack(">f", self.header[850:854])[
                    0
                ]  # Temperature (F)
                self.FaradayCupI = unpack(">f", self.header[854:858])[
                    0
                ]  # Faraday cup current (nA)
                self.FIBSpecimenI = unpack(">f", self.header[858:862])[
                    0
                ]  # FIB specimen current (nA)
                self.BeamDump1I = unpack(">f", self.header[862:866])[
                    0
                ]  # Beam dump 1 current (nA)
                self.SEMSpecimenI = unpack(">f", self.header[866:870])[
                    0
                ]  # SEM specimen current (nA)
                self.MillingYVoltage = unpack(">f", self.header[870:874])[
                    0
                ]  # Milling Y voltage (V)
                self.FocusIndex = unpack(">f", self.header[874:878])[0]  # Focus index
                self.FIBSliceNum = unpack(">L", self.header[878:882])[0]  # FIB slice #

            if self.FileVersion > 7:
                self.BeamDump2I = unpack(">f", self.header[882:886])[
                    0
                ]  # Beam dump 2 current (nA)
                self.MillingI = unpack(">f", self.header[886:890])[
                    0
                ]  # Milling current (nA)

            self.FileLength = unpack(">q", self.header[1000:1008])[
                0
            ]  # Read in file length in bytes

            #                Finish self.header read
            #
            #                Read raw data
            #                fid.seek(1024, 0)
            #                n_elements = self.ChanNum * self.XResolution * self.YResolution
            #                print(n_elements, self.ChanNum, self.XResolution, self.YResolution)
            #                if self.EightBit==1:
            #                    raw_data = fid.read(n_elements) # Read in data
            #                    Raw = unpack('>'+str(n_elements)+'B',raw_data)
            #                else:
            #                    #raw_data = fid.read(2*n_elements) # Read in data
            #                    #Raw = unpack('>'+str(n_elements)+'h',raw_data)
            #                fid.close
            #                finish reading raw data

            n_elements = self.ChanNum * self.XResolution * self.YResolution
            fid.seek(1024, 0)
            if self.EightBit == 1:
                dt = np.dtype(np.uint8)
                dt = dt.newbyteorder(">")
                if self.use_dask_arrays:
                    Raw = da.from_array(np.frombuffer(fid.read(n_elements), dtype=dt))
                else:
                    Raw = np.frombuffer(fid.read(n_elements), dtype=dt)
            else:
                dt = np.dtype(np.int16)
                dt = dt.newbyteorder(">")
                if self.use_dask_arrays:
                    Raw = da.from_array(
                        np.frombuffer(fid.read(2 * n_elements), dtype=dt)
                    )
                else:
                    Raw = np.frombuffer(fid.read(2 * n_elements), dtype=dt)
            fid.close
            # finish reading raw data

            Raw = np.array(Raw).reshape(
                self.YResolution, self.XResolution, self.ChanNum
            )
            # print(shape(Raw), type(Raw), type(Raw[0,0]))

            # data = np.asarray(datab).reshape(self.YResolution,self.XResolution,ChanNum)
            if self.EightBit == 1:
                if self.AI1 == 1:
                    self.RawImageA = Raw[:, :, 0]
                    self.ImageA = (
                        Raw[:, :, 0].astype(float32)
                        * self.ScanRate
                        / self.Scaling[0, 0]
                        / self.Scaling[2, 0]
                        / self.Scaling[3, 0]
                        + self.Scaling[1, 0]
                    ).astype(int32)
                    if self.AI2 == 1:
                        self.RawImageB = Raw[:, :, 1]
                        self.ImageB = (
                            Raw[:, :, 1].astype(float32)
                            * self.ScanRate
                            / self.Scaling[0, 1]
                            / self.Scaling[2, 1]
                            / self.Scaling[3, 1]
                            + self.Scaling[1, 1]
                        ).astype(int32)
                elif self.AI2 == 1:
                    self.RawImageB = Raw[:, :, 0]
                    self.ImageB = (
                        Raw[:, :, 0].astype(float32)
                        * self.ScanRate
                        / self.Scaling[0, 0]
                        / self.Scaling[2, 0]
                        / self.Scaling[3, 0]
                        + self.Scaling[1, 0]
                    ).astype(int32)
            else:
                if (
                    self.FileVersion == 1
                    or self.FileVersion == 2
                    or self.FileVersion == 3
                    or self.FileVersion == 4
                    or self.FileVersion == 5
                    or self.FileVersion == 6
                ):
                    if self.AI1 == 1:
                        self.RawImageA = Raw[:, :, 0]
                        self.ImageA = (
                            self.Scaling[0, 0] + self.RawImageA * self.Scaling[1, 0]
                        )  # Converts raw I16 data to voltage based on self.Scaling factors
                        if self.AI2 == 1:
                            self.RawImageB = Raw[:, :, 1]
                            self.ImageB = (
                                self.Scaling[0, 1] + self.RawImageB * self.Scaling[1, 1]
                            )
                            if self.AI3 == 1:
                                self.RawImageC = (Raw[:, :, 2]).reshape(
                                    self.YResolution, self.XResolution
                                )
                                self.ImageC = (
                                    self.Scaling[0, 2]
                                    + self.RawImageC * self.Scaling[1, 2]
                                )
                                if self.AI4 == 1:
                                    self.RawImageD = (Raw[:, :, 3]).reshape(
                                        self.YResolution, self.XResolution
                                    )
                                    self.ImageD = (
                                        self.Scaling[0, 3]
                                        + self.RawImageD * self.Scaling[1, 3]
                                    )
                            elif self.AI4 == 1:
                                self.RawImageD = (Raw[:, :, 2]).reshape(
                                    self.YResolution, self.XResolution
                                )
                                self.ImageD = (
                                    self.Scaling[0, 2]
                                    + self.RawImageD * self.Scaling[1, 2]
                                )
                        elif self.AI3 == 1:
                            self.RawImageC = Raw[:, :, 1]
                            self.ImageC = (
                                self.Scaling[0, 1] + self.RawImageC * self.Scaling[1, 1]
                            )
                            if self.AI4 == 1:
                                self.RawImageD = (Raw[:, :, 2]).reshape(
                                    self.YResolution, self.XResolution
                                )
                                self.ImageD = (
                                    self.Scaling[0, 2]
                                    + self.RawImageD * self.Scaling[1, 2]
                                )
                        elif self.AI4 == 1:
                            self.RawImageD = Raw[:, :, 1]
                            self.ImageD = (
                                self.Scaling[0, 1] + self.RawImageD * self.Scaling[1, 1]
                            )
                    elif self.AI2 == 1:
                        self.RawImageB = Raw[:, :, 0]
                        self.ImageB = (
                            self.Scaling[0, 0] + self.RawImageB * self.Scaling[1, 0]
                        )
                        if self.AI3 == 1:
                            self.RawImageC = Raw[:, :, 1]
                            self.ImageC = (
                                self.Scaling[0, 1] + self.RawImageC * self.Scaling[1, 1]
                            )
                            if self.AI4 == 1:
                                self.RawImageD = (Raw[:, :, 2]).reshape(
                                    self.YResolution, self.XResolution
                                )
                                self.ImageD = (
                                    self.Scaling[0, 2]
                                    + self.RawImageD * self.Scaling[1, 2]
                                )
                        elif self.AI4 == 1:
                            self.RawImageD = Raw[:, :, 1]
                            self.ImageD = (
                                self.Scaling[0, 1] + self.RawImageD * self.Scaling[1, 1]
                            )
                    elif self.AI3 == 1:
                        self.RawImageC = Raw[:, :, 0]
                        self.ImageC = (
                            self.Scaling[0, 0] + self.RawImageC * self.Scaling[1, 0]
                        )
                        if self.AI4 == 1:
                            self.RawImageD = Raw[:, :, 1]
                            self.ImageD = (
                                self.Scaling[0, 1] + self.RawImageD * self.Scaling[1, 1]
                            )
                    elif self.AI4 == 1:
                        self.RawImageD = Raw[:, :, 0]
                        self.ImageD = (
                            self.Scaling[0, 0] + self.RawImageD * self.Scaling[1, 0]
                        )

                elif self.FileVersion == 7:
                    if self.AI1 == 1:
                        self.RawImageA = Raw[:, :, 0]
                        self.ImageA = (
                            self.RawImageA - self.Scaling[1, 0]
                        ) * self.Scaling[2, 0]
                        if self.AI2 == 1:
                            self.RawImageB = Raw[:, :, 1]
                            self.ImageB = (
                                self.RawImageB - self.Scaling[1, 1]
                            ) * self.Scaling[2, 1]
                    elif self.AI2 == 1:
                        self.RawImageB = Raw[:, :, 0]
                        self.ImageB = (
                            self.RawImageB - self.Scaling[1, 1]
                        ) * self.Scaling[2, 1]

                elif self.FileVersion == 8 or self.FileVersion == 9:
                    self.ElectronFactor1 = 0.1
                    # 16-bit intensity is 10x electron counts
                    self.Scaling[3, 0] = self.ElectronFactor1
                    self.ElectronFactor2 = 0.1
                    # 16-bit intensity is 10x electron counts
                    self.Scaling[3, 1] = self.ElectronFactor2
                    if self.AI1 == 1:
                        self.RawImageA = Raw[:, :, 0]
                        self.ImageA = (
                            (self.RawImageA - self.Scaling[1, 0])
                            * self.Scaling[2, 0]
                            / self.ScanRate
                            * self.Scaling[0, 0]
                            / self.ElectronFactor1
                        )
                        # Converts raw I16 data to voltage based on self.Scaling factors
                        if self.AI2 == 1:
                            self.RawImageB = Raw[:, :, 1]
                            self.ImageB = (
                                (self.RawImageB - self.Scaling[1, 1])
                                * self.Scaling[2, 1]
                                / self.ScanRate
                                * self.Scaling[0, 1]
                                / self.ElectronFactor2
                            )
                    elif self.AI2 == 1:
                        self.RawImageB = Raw[:, :, 0]
                        self.ImageB = (
                            (self.RawImageB - self.Scaling[1, 1])
                            * self.Scaling[2, 1]
                            / self.ScanRate
                            * self.Scaling[0, 1]
                            / self.ElectronFactor2
                        )

    def print_header(self):
        """
        Prints a formatted content of the file header

        """
        if self.FileVersion == -1:
            print("Sample_ID=", self.Sample_ID)
            print("DetA=", self.DetA)
            print("DetB=", self.DetB)
            print("EightBit=", self.EightBit)
            print("XResolution=", self.XResolution)
            print("YResolution=", self.YResolution)
            print("PixelSize=", self.PixelSize)
        else:
            print("FileMagicNum=", self.FileMagicNum)
            print("FileVersion=", self.FileVersion)
            print("FileType=", self.FileType)
            print("SWdate=", self.SWdate)
            print("TimeStep=", self.TimeStep)
            print("ChanNum=", self.ChanNum)
            print("EightBit=", self.EightBit)
            print("Scaling=", self.Scaling)
            if self.FileVersion > 8:
                print("RestartFlag=", self.RestartFlag)
                print("StageMove=", self.StageMove)
                print("FirstPixelX=", self.FirstPixelX)
                print("FirstPixelY=", self.FirstPixelY)
            print("XResolution=", self.XResolution)
            print("YResolution=", self.YResolution)
            if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3:
                print("AIDelay=", self.AIDelay)
            print("Oversampling=", self.Oversampling)
            print("ZeissScanSpeed=", self.ZeissScanSpeed)
            print("DecimatingFactor=", self.DecimatingFactor)
            print("ScanRate=", self.ScanRate)
            print("FramelineRampdownRatio=", self.FramelineRampdownRatio)
            print("Xmin=", self.Xmin)
            print("Xmax=", self.Xmax)
            print("Detmin=", self.Detmin)
            print("Detmax=", self.Detmax)
            print("AI1=", self.AI1)
            print("AI2=", self.AI2)
            print("AI3=", self.AI3)
            print("AI4=", self.AI4)
            if self.FileVersion > 8:
                print("Sample_ID=", self.Sample_ID)
            print("Notes=", self.Notes)
            print("SEMShiftX=", self.SEMShiftX)
            print("SEMShiftY=", self.SEMShiftY)
            print("DetA=", self.DetA)
            print("DetB=", self.DetB)
            print("DetC=", self.DetC)
            print("DetD=", self.DetD)
            print("Mag=", self.Mag)
            print("PixelSize=", self.PixelSize)
            print("WD=", self.WD)
            print("EHT=", self.EHT)
            print("SEMApr=", self.SEMApr)
            print("HighCurrent=", self.HighCurrent)
            print("SEMCurr=", self.SEMCurr)
            print("SEMRot=", self.SEMRot)
            print("ChamVac=", self.ChamVac)
            print("GunVac=", self.GunVac)
            print("SEMStiX=", self.SEMStiX)
            print("SEMStiY=", self.SEMStiY)
            print("SEMAlnX=", self.SEMAlnX)
            print("SEMAlnY=", self.SEMAlnY)
            print("StageX=", self.StageX)
            print("StageY=", self.StageY)
            print("StageZ=", self.StageZ)
            print("StageT=", self.StageT)
            print("StageR=", self.StageR)
            print("StageM=", self.StageM)
            print("BrightnessA=", self.BrightnessA)
            print("ContrastA=", self.ContrastA)
            print("BrightnessB=", self.BrightnessB)
            print("ContrastB=", self.ContrastB)
            print("Mode=", self.Mode)
            print("FIBFocus=", self.FIBFocus)
            print("FIBProb=", self.FIBProb)
            print("FIBCurr=", self.FIBCurr)
            print("FIBRot=", self.FIBRot)
            print("FIBAlnX=", self.FIBAlnX)
            print("FIBAlnY=", self.FIBAlnY)
            print("FIBStiX=", self.FIBStiX)
            print("FIBStiY=", self.FIBStiY)
            print("FIBShiftX=", self.FIBShiftX)
            print("FIBShiftY=", self.FIBShiftY)
            if self.FileVersion > 4:
                print("MillingXResolution=", self.MillingXResolution)
                print("MillingYResolution=", self.MillingYResolution)
                print("MillingXSize=", self.MillingXSize)
                print("MillingYSize=", self.MillingYSize)
                print("MillingULAng=", self.MillingULAng)
                print("MillingURAng=", self.MillingURAng)
                print("MillingLineTime=", self.MillingLineTime)
                print("FIBFOV (um)=", self.FIBFOV)
                print("MillingPIDOn=", self.MillingPIDOn)
                print("MillingPIDMeasured=", self.MillingPIDMeasured)
                print("MillingPIDTarget=", self.MillingPIDTarget)
                print("MillingPIDTargetSlope=", self.MillingPIDTargetSlope)
                print("MillingPIDP=", self.MillingPIDP)
                print("MillingPIDI=", self.MillingPIDI)
                print("MillingPIDD=", self.MillingPIDD)
                print("MachineID=", self.MachineID)
                print("SEMSpecimenI=", self.SEMSpecimenI)
            if self.FileVersion > 5:
                print("Temperature=", self.Temperature)
                print("FaradayCupI=", self.FaradayCupI)
                print("FIBSpecimenI=", self.FIBSpecimenI)
                print("BeamDump1I=", self.BeamDump1I)
                print("MillingYVoltage=", self.MillingYVoltage)
                print("FocusIndex=", self.FocusIndex)
                print("FIBSliceNum=", self.FIBSliceNum)
            if self.FileVersion > 7:
                print("BeamDump2I=", self.BeamDump2I)
                print("MillingI=", self.MillingI)
            print("SEMSpecimenI=", self.SEMSpecimenI)
            print("FileLength=", self.FileLength)

    def display_images(self):
        """
        Display auto-scaled detector images without saving the figure into the file.

        """
        fig, axs = subplots(2, 1, figsize=(10, 5))
        axs[0].imshow(self.RawImageA, cmap="Greys")
        axs[1].imshow(self.RawImageB, cmap="Greys")
        ttls = [
            "Detector A: " + self.DetA.strip("\x00"),
            "Detector B: " + self.DetB.strip("\x00"),
        ]
        for ax, ttl in zip(axs, ttls):
            ax.axis(False)
            ax.set_title(ttl, fontsize=10)
        fig.suptitle(self.fname)

    def save_images_jpeg(self, **kwargs):
        """
        Display auto-scaled detector images and save the figure into JPEG file (s).

        Parameters
        ----------
        kwargs:
        images_to_save : str
            Images to save. options are: 'A', 'B', or 'Both' (default).
        invert : boolean
            If True, the image will be inverted.

        """
        images_to_save = kwargs.get("images_to_save", "Both")
        invert = kwargs.get("invert", False)

        if images_to_save == "Both" or images_to_save == "A":
            if self.ftype == 0:
                fname_jpg = (
                    os.path.splitext(self.fname)[0]
                    + "_"
                    + self.DetA.strip("\x00")
                    + ".jpg"
                )
            else:
                fname_jpg = os.path.splitext(self.fname)[0] + "DetA.jpg"
            Img = self.RawImageA_8bit_thresholds()[0]
            if invert:
                Img = uint8(255) - Img
            PILImage.fromarray(Img).save(fname_jpg)

        try:
            if images_to_save == "Both" or images_to_save == "B":
                if self.ftype == 0:
                    fname_jpg = (
                        os.path.splitext(self.fname)[0]
                        + "_"
                        + self.DetB.strip("\x00")
                        + ".jpg"
                    )
                else:
                    fname_jpg = os.path.splitext(self.fname)[0] + "DetB.jpg"
                Img = self.RawImageB_8bit_thresholds()[0]
                if invert:
                    Img = uint8(255) - Img
                PILImage.fromarray(Img).save(fname_jpg)
        except:
            print("No Detector B image to save")

    def save_images_tif(self, images_to_save="Both"):
        """
        Save the detector images into TIF file (s).

        Parameters
        ----------
        images_to_save : str
            Images to save. options are: 'A', 'B', or 'Both' (default).

        """
        if self.ftype == 0:
            if images_to_save == "Both" or images_to_save == "A":
                fnameA = (
                    os.path.splitext(self.fname)[0]
                    + "_"
                    + self.DetA.strip("\x00")
                    + ".tif"
                )
                tiff.imsave(fnameA, self.RawImageA)
            if self.DetB != "None":
                if images_to_save == "Both" or images_to_save == "B":
                    fnameB = (
                        os.path.splitext(self.fname)[0]
                        + "_"
                        + self.DetB.strip("\x00")
                        + ".tif"
                    )
                    tiff.imsave(fnameB, self.RawImageB)
        else:
            print("original File is already in TIF format")

    def get_image_min_max(
        self,
        image_name="ImageA",
        thr_min=1.0e-4,
        thr_max=1.0e-3,
        nbins=256,
        disp_res=False,
    ):
        """
        Calculates the data range of the EM data. ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

        Calculates histogram of pixel intensities of of the loaded image
        with number of bins determined by parameter nbins (default = 256)
        and normalizes it to get the probability distribution function (PDF),
        from which a cumulative distribution function (CDF) is calculated.
        Then given the threshold_min, threshold_max parameters,
        the minimum and maximum values for the image are found by finding
        the intensities at which CDF= threshold_min and (1- threshold_max), respectively.

        Parameters
        ----------
        image_name : string
            the name of the image to perform this operations (defaulut is 'RawImageA')
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        disp_res : boolean
            (default is False) - to plot/ display the results

        Returns:
            dmin, dmax: (float) minimum and maximum values of the data range.
        """
        if image_name == "ImageA":
            im = self.ImageA
        if image_name == "ImageB":
            im = self.ImageB
        if image_name == "RawImageA":
            im = self.RawImageA
        if image_name == "RawImageB":
            im = self.RawImageB
        return get_min_max_thresholds(
            im, thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=disp_res
        )

    def RawImageA_8bit_thresholds(
        self, thr_min=1.0e-3, thr_max=1.0e-3, data_min=-1, data_max=-1, nbins=256
    ):
        """
        Convert the Image A into 8-bit array

        Parameters
        ----------
        thr_min : float
            lower CDF threshold for determining the minimum data value
        thr_max : float
            upper CDF threshold for determining the maximum data value
        data_min : float
            If different from data_max, this value will be used as low bound for I8 data conversion
        data_max : float
            If different from data_min, this value will be used as high bound for I8 data conversion
        nbins : int
            number of histogram bins for building the PDF and CDF

        Returns
        dt, data_min, data_max
            dt : 2D uint8 array
                Converted data
            data_min : float
                value used as low bound for I8 data conversion
            data_max : float
                value used as high bound for I8 data conversion
        """
        if self.EightBit == 1:
            # print('8-bit image already - no need to convert')
            dt = self.RawImageA
        else:
            if data_min == data_max:
                data_min, data_max = self.get_image_min_max(
                    image_name="RawImageA",
                    thr_min=thr_min,
                    thr_max=thr_max,
                    nbins=nbins,
                    disp_res=False,
                )
            dt = (
                (np.clip(self.RawImageA, data_min, data_max) - data_min)
                / (data_max - data_min)
                * 255.0
            ).astype(np.uint8)
        return dt, data_min, data_max

    def RawImageB_8bit_thresholds(
        self, thr_min=1.0e-3, thr_max=1.0e-3, data_min=-1, data_max=-1, nbins=256
    ):
        """
        Convert the Image B into 8-bit array

        Parameters
        ----------
        thr_min : float
            lower CDF threshold for determining the minimum data value
        thr_max : float
            upper CDF threshold for determining the maximum data value
        data_min : float
            If different from data_max, this value will be used as low bound for I8 data conversion
        data_max : float
            If different from data_min, this value will be used as high bound for I8 data conversion
        nbins : int
            number of histogram bins for building the PDF and CDF

        Returns
        dt, data_min, data_max
            dt : 2D uint8 array
                Converted data
            data_min : float
                value used as low bound for I8 data conversion
            data_max : float
                value used as high bound for I8 data conversion
        """
        if self.EightBit == 1:
            # print('8-bit image already - no need to convert')
            dt = self.RawImageB
        else:
            if data_min == data_max:
                data_min, data_max = self.get_image_min_max(
                    image_name="RawImageB",
                    thr_min=thr_min,
                    thr_max=thr_max,
                    nbins=nbins,
                    disp_res=False,
                )
            dt = (
                (np.clip(self.RawImageB, data_min, data_max) - data_min)
                / (data_max - data_min)
                * 255.0
            ).astype(np.uint8)
        return dt, data_min, data_max

    def save_snapshot(self, **kwargs):
        """
        Builds an image that contains both the Detector A and Detector B (if present) images as well as a table with important FIB-SEM parameters.

        kwargs:
         ----------
        thr_min : float
            lower CDF threshold for determining the minimum data value. Default is 1.0e-3
        thr_max : float
            upper CDF threshold for determining the maximum data value. Default is 1.0e-3
        data_min : float
            If different from data_max, this value will be used as low bound for I8 data conversion
        data_max : float
            If different from data_min, this value will be used as high bound for I8 data conversion
        nbins : int
            number of histogram bins for building the PDF and CDF
        disp_res : True
            If True display the results
        dpi : int
            Default is 300
        snapshot_name : string
            the name of the image to perform this operations (defaulut is frame_name + '_snapshot.png').



        Returns
        dt, data_min, data_max
            dt : 2D uint8 array
                Converted data
            data_min : float
                value used as low bound for I8 data conversion
            data_max : float
                value used as high bound for I8 data conversion
        """
        thr_min = kwargs.get("thr_min", 1.0e-3)
        thr_max = kwargs.get("thr_max", 1.0e-3)
        nbins = kwargs.get("nbins", 256)
        disp_res = kwargs.get("disp_res", True)
        dpi = kwargs.get("dpi", 300)
        snapshot_name = kwargs.get(
            "snapshot_name", os.path.splitext(self.fname)[0] + "_snapshot.png"
        )

        ifDetB = self.DetB != "None"
        if ifDetB:
            try:
                dminB, dmaxB = self.get_image_min_max(
                    image_name="RawImageB",
                    thr_min=thr_min,
                    thr_max=thr_max,
                    nbins=nbins,
                    disp_res=False,
                )
                fig, axs = subplots(3, 1, figsize=(11, 8))
            except:
                ifDetB = False
                pass
        if not ifDetB:
            fig, axs = subplots(2, 1, figsize=(7, 8))
        fig.subplots_adjust(
            left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.15, hspace=0.1
        )
        dminA, dmaxA = self.get_image_min_max(
            image_name="RawImageA",
            thr_min=thr_min,
            thr_max=thr_max,
            nbins=nbins,
            disp_res=False,
        )
        axs[1].imshow(self.RawImageA, cmap="Greys", vmin=dminA, vmax=dmaxA)
        if ifDetB:
            axs[2].imshow(self.RawImageB, cmap="Greys", vmin=dminB, vmax=dmaxB)
        try:
            ttls = [
                self.Notes.strip("\x00"),
                "Detector A:  "
                + self.DetA.strip("\x00")
                + ",  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}".format(
                    dminA, dmaxA, thr_min, thr_max
                )
                + "    (Brightness: {:.1f}, Contrast: {:.1f})".format(
                    self.BrightnessA, self.ContrastA
                ),
                "Detector B:  "
                + self.DetB.strip("\x00")
                + ",  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}".format(
                    dminB, dmaxB, thr_min, thr_max
                )
                + "    (Brightness: {:.1f}, Contrast: {:.1f})".format(
                    self.BrightnessB, self.ContrastB
                ),
            ]
        except:
            ttls = ["", "Detector A", ""]
        for j, ax in enumerate(axs):
            ax.axis(False)
            ax.set_title(ttls[j], fontsize=10)
        fig.suptitle(self.fname)

        if self.FileVersion > 8:
            cell_text = [
                [
                    "Sample ID",
                    "{:s}".format(self.Sample_ID.strip("\x00")),
                    "",
                    "Frame Size",
                    "{:d} x {:d}".format(self.XResolution, self.YResolution),
                    "",
                    "Scan Rate",
                    "{:.3f} MHz".format(self.ScanRate / 1.0e6),
                ],
                [
                    "Machine ID",
                    "{:s}".format(self.MachineID.strip("\x00")),
                    "",
                    "Pixel Size",
                    "{:.1f} nm".format(self.PixelSize),
                    "",
                    "Oversampling",
                    "{:d}".format(self.Oversampling),
                ],
                [
                    "FileVersion",
                    "{:d}".format(self.FileVersion),
                    "",
                    "Working Dist.",
                    "{:.3f} mm".format(self.WD),
                    "",
                    "FIB Focus",
                    "{:.1f}  V".format(self.FIBFocus),
                ],
                [
                    "Bit Depth",
                    "{:d}".format(8 * (2 - self.EightBit)),
                    "",
                    "EHT Voltage\n\nSEM Current",
                    "{:.3f} kV \n\n{:.3f} nA".format(self.EHT, self.SEMCurr * 1.0e9),
                    "",
                    "FIB Probe",
                    "{:d}".format(self.FIBProb),
                ],
            ]
        else:
            if self.FileVersion > 0:
                cell_text = [
                    [
                        "",
                        "",
                        "",
                        "Frame Size",
                        "{:d} x {:d}".format(self.XResolution, self.YResolution),
                        "",
                        "Scan Rate",
                        "{:.3f} MHz".format(self.ScanRate / 1.0e6),
                    ],
                    [
                        "Machine ID",
                        "{:s}".format(self.MachineID.strip("\x00")),
                        "",
                        "Pixel Size",
                        "{:.1f} nm".format(self.PixelSize),
                        "",
                        "Oversampling",
                        "{:d}".format(self.Oversampling),
                    ],
                    [
                        "FileVersion",
                        "{:d}".format(self.FileVersion),
                        "",
                        "Working Dist.",
                        "{:.3f} mm".format(self.WD),
                        "",
                        "FIB Focus",
                        "{:.1f}  V".format(self.FIBFocus),
                    ],
                    [
                        "Bit Depth",
                        "{:d}".format(8 * (2 - self.EightBit)),
                        "",
                        "EHT Voltage",
                        "{:.3f} kV".format(self.EHT),
                        "",
                        "FIB Probe",
                        "{:d}".format(self.FIBProb),
                    ],
                ]
            else:
                cell_text = [
                    [
                        "",
                        "",
                        "",
                        "Frame Size",
                        "{:d} x {:d}".format(self.XResolution, self.YResolution),
                        "",
                        "Scan Rate",
                        "",
                    ],
                    [
                        "Machine ID",
                        "",
                        "",
                        "Pixel Size",
                        "{:.1f} nm".format(self.PixelSize),
                        "",
                        "Oversampling",
                        "",
                    ],
                    [
                        "FileVersion",
                        "{:d}".format(self.FileVersion),
                        "",
                        "Working Dist.",
                        " ",
                        "",
                        "FIB Focus",
                        "",
                    ],
                    [
                        "Bit Depth",
                        "{:d}".format(8 * (2 - self.EightBit)),
                        "",
                        "EHT Voltage",
                        "",
                        "",
                        "FIB Probe",
                        "",
                    ],
                ]
        llw0 = 0.3
        llw1 = 0.18
        llw2 = 0.02
        clw = [llw1, llw0, llw2, llw1, llw1, llw2, llw1, llw1]
        tbl = axs[0].table(
            cellText=cell_text,
            colWidths=clw,
            cellLoc="center",
            colLoc="center",
            bbox=[0.02, 0, 0.96, 1.0],
            # bbox = [0.45, 1.02, 2.8, 0.55],
            zorder=10,
        )

        fig.savefig(snapshot_name, dpi=dpi)
        if disp_res == False:
            plt.close(fig)

    def analyze_noise_ROIs(self, Noise_ROIs, Hist_ROI, **kwargs):
        """
        Analyses the noise statistics in the selected ROI's of the EM data.
        ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

        Calls Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs)
        Performs following:
        1. For each of the selected ROI's, this method will perfrom the following:
            1a. Smooth the data by 2D convolution with a given kernel.
            1b. Determine "Noise" as difference between the original raw and smoothed data.
            1c. Calculate the mean intensity value of the data and variance of the above "Noise"
        2. Plot the dependence of the noise variance vs. image intensity.
        3. One of the parameters is a DarkCount. If it is not explicitly defined as input parameter,
            it will be determined from the header data:
                for RawImageA it is self.Scaling[1,0]
                for RawImageB it is self.Scaling[1,1]
        4. The equation is determined for a line that passes through the point:
                Intensity=DarkCount and Noise Variance = 0
                and is a best fit for the [Mean Intensity, Noise Variance] points
                determined for each ROI (Step 1 above).
        5. Another ROI (defined by Hist_ROI parameter) is used to built an
            intensity histogram of the actual data. Peak of that histogram is determined.
        6. The data is plotted. Two values of SNR are defined from the slope of the line in Step 4:
            PSNR (Peak SNR) = Mean Intensity/sqrt(Noise Variance) at the intensity
                at the histogram peak determined in the Step 5.
            DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance),
                where Max and Min Intensity are determined by corresponding cummulative
                threshold parameters, and Noise Variance is taken at the intensity
                in the middle of the range (Min Intensity + Max Intensity)/2.0

        Parameters
        ----------
        Noise_ROIs : list of lists: [[left, right, top, bottom]]
            list of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the noise.
        Hist_ROI : list [left, right, top, bottom]
            coordinates (indices) of the boundaries of the image subset to evaluate the real data histogram.

        kwargs:
        image_name : string
            the name of the image to perform this operations (defaulut is 'RawImageA').
        DarkCount : float
            the value of the Intensity Data at 0.
        kernel : 2D float array
            a kernel to perfrom 2D smoothing convolution.
        filename : str
            filename - used for plotting the data. If not explicitly defined will use the instance attribute self.fname
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

        Returns:
        mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR
            mean_vals and var_vals are the Mean Intensity and Noise Variance values for the Noise_ROIs (Step 1)
            NF_slope is the slope of the linear fit curve (Step 4)
            PSNR and DSNR are Peak and Dynamic SNR's (Step 6)
        """
        image_name = kwargs.get("image_name", "RawImageA")

        if image_name == "RawImageA":
            ImgEM = self.RawImageA.astype(float)
            DarkCount = self.Scaling[1, 0]
        if image_name == "RawImageB" and self.DetB != "None":
            ImgEM = self.RawImageB.astype(float)
            DarkCount = self.Scaling[1, 1]

        if (image_name == "RawImageA") or (
            image_name == "RawImageB" and self.DetB != "None"
        ):
            st = 1.0 / np.sqrt(2.0)
            def_kernel = np.array(
                [[st, 1.0, st], [1.0, 1.0, 1.0], [st, 1.0, st]]
            ).astype(float)
            def_kernel = def_kernel / def_kernel.sum()
            kernel = kwargs.get("kernel", def_kernel)
            DarkCount = kwargs.get("DarkCount", DarkCount)
            nbins_disp = kwargs.get("nbins_disp", 256)
            thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
            nbins_analysis = kwargs.get("nbins_analysis", 100)
            thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
            Notes = kwargs.get("Notes", self.Notes.strip("\x00"))
            kwargs["kernel"] = kernel
            kwargs["DarkCount"] = DarkCount
            kwargs["img_label"] = image_name
            kwargs["res_fname"] = (
                os.path.splitext(self.fname)[0]
                + "_"
                + image_name
                + "_Noise_Analysis_ROIs.png"
            )
            kwargs["Notes"] = Notes
            mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR = Single_Image_Noise_ROIs(
                ImgEM, Noise_ROIs, Hist_ROI, **kwargs
            )

        else:
            print("No valid image name selected")
            mean_vals = 0.0
            var_vals = 0.0
            NF_slope = 0.0
            PSNR = 0.0
            MSNR = 0.0
            DSNR = 0.0

        return mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR

    def analyze_noise_statistics(self, **kwargs):
        """
        Analyses the noise statistics of the EM data image.
        ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

        Calls Single_Image_Noise_Statistics(img, **kwargs)
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
        8. The data is plotted. Two values of SNR are defined from the slope of the line in Step 7:
            PSNR (Peak SNR) = Intensity /sqrt(Noise Variance) at the intensity
                at the histogram peak determined in the Step 3.
            MSNR (Mean SNR) = Mean Intensity /sqrt(Noise Variance)
            DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance),
                where Max and Min Intensity are determined by corresponding cummulative
                threshold parameters, and Noise Variance is taken at the intensity
                in the middle of the range (Min Intensity + Max Intensity)/2.0

        Parameters
        ----------
            kwargs:
            image_name : str
                Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
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
            mean_vals and var_vals are the Mean Intensity and Noise Variance values for Step 5
            I0 is zero intercept (should be close to DarkCount)
            PSNR and DSNR are Peak and Dynamic SNR's (Step 8)
        """
        image_name = kwargs.get("image_name", "RawImageA")

        if image_name == "RawImageA":
            ImgEM = self.RawImageA.astype(float)
            DarkCount = self.Scaling[1, 0]
        if image_name == "RawImageB" and self.DetB != "None":
            ImgEM = self.RawImageB.astype(float)
            DarkCount = self.Scaling[1, 1]

        if (image_name == "RawImageA") or (
            image_name == "RawImageB" and self.DetB != "None"
        ):
            st = 1.0 / np.sqrt(2.0)
            evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
            def_kernel = np.array(
                [[st, 1.0, st], [1.0, 1.0, 1.0], [st, 1.0, st]]
            ).astype(float)
            def_kernel = def_kernel / def_kernel.sum()
            kernel = kwargs.get("kernel", def_kernel)
            DarkCount = kwargs.get("DarkCount", DarkCount)
            nbins_disp = kwargs.get("nbins_disp", 256)
            thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
            nbins_analysis = kwargs.get("nbins_analysis", 100)
            thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
            disp_res = kwargs.get("disp_res", True)
            save_res_png = kwargs.get("save_res_png", True)
            default_res_name = (
                os.path.splitext(self.fname)[0]
                + "_Noise_Analysis_"
                + image_name
                + ".png"
            )
            res_fname = kwargs.get("res_fname", default_res_name)
            img_label = kwargs.get("img_label", self.Sample_ID)
            Notes = kwargs.get("Notes", self.Notes.strip("\x00"))
            dpi = kwargs.get("dpi", 300)

            noise_kwargs = {
                "image_name": image_name,
                "evaluation_box": evaluation_box,
                "kernel": kernel,
                "DarkCount": DarkCount,
                "nbins_disp": nbins_disp,
                "thresholds_disp": thresholds_disp,
                "nbins_analysis": nbins_analysis,
                "thresholds_analysis": thresholds_analysis,
                "disp_res": disp_res,
                "save_res_png": save_res_png,
                "res_fname": res_fname,
                "Notes": Notes,
                "dpi": dpi,
            }

            (
                mean_vals,
                var_vals,
                I0,
                PSNR,
                DSNR,
                popt,
                result,
            ) = Single_Image_Noise_Statistics(ImgEM, **noise_kwargs)
        else:
            mean_vals, var_vals, I0, PSNR, DSNR, popt, result = (
                [],
                [],
                0.0,
                0.0,
                np.array((0.0, 0.0)),
                [],
            )
        return mean_vals, var_vals, I0, PSNR, DSNR, popt, result

    def analyze_SNR_autocorr(self, **kwargs):
        """
        Estimates SNR using auto-correlation analysis of a single image.
        ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

        Calculates SNR of a single image base on auto-correlation analysis of a single image, after [1].
        Calls function Single_Image_SNR(img, **kwargs)

        Parameters
        ---------
        kwargs:
        image_name : str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        edge_fraction : float
            fraction of the full autocetrrelation range used to calculate the "mean value" (default is 0.10)
        extrapolate_signal : boolean
            extrapolate to find signal autocorrelationb at 0-point (without noise). Default is True
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
        disp_res : boolean
            display results (plots) (default is True)
        save_res_png : boolean
            save the analysis output into a PNG file (default is True)
        res_fname : string
            filename for the sesult image ('SNR_result.png')
        img_label : string
            optional image label
        dpi : int
            dots-per-inch resolution for the output image

        Returns:
            xSNR, ySNR : float, float
                SNR determind using the method in [1] along X- and Y- directions.
                If there is a direction with slow varying data - that direction provides more accurate SNR estimate
                Y-streaks in typical FIB-SEM data provide slow varying Y-component becuase streaks
                usually get increasingly worse with increasing Y.
                So for typical FIB-SEM data use ySNR

        [1] J. T. L. Thong et al, Single-image signal-tonoise ratio estimation. Scanning, 328–336 (2001).
        """
        image_name = kwargs.get("image_name", "RawImageA")
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        edge_fraction = kwargs.get("edge_fraction", 0.10)
        extrapolate_signal = kwargs.get("extrapolate_signal", True)
        disp_res = kwargs.get("disp_res", True)
        save_res_png = kwargs.get("save_res_png", True)
        default_res_name = (
            os.path.splitext(self.fname)[0]
            + "_AutoCorr_Noise_Analysis_"
            + image_name
            + ".png"
        )
        res_fname = kwargs.get("res_fname", default_res_name)
        dpi = kwargs.get("dpi", 300)

        SNR_kwargs = {
            "edge_fraction": edge_fraction,
            "extrapolate_signal": extrapolate_signal,
            "disp_res": disp_res,
            "save_res_png": save_res_png,
            "res_fname": res_fname,
            "img_label": image_name,
            "dpi": dpi,
        }

        if image_name == "RawImageA":
            img = self.RawImageA
        if image_name == "RawImageB":
            img = self.RawImageB
        if image_name == "ImageA":
            img = self.ImageA
        if image_name == "ImageB":
            img = self.ImageB

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

        xSNR, ySNR, rSNR = Single_Image_SNR(
            img[yi_eval:ya_eval, xi_eval:xa_eval], **SNR_kwargs
        )

        return xSNR, ySNR, rSNR

    def show_eval_box(self, **kwargs):
        """
        Show the box used for noise analysis.
        ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

        kwargs
        ---------
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        image_name : str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        data_dir : str
            data directory (path)
        Sample_ID : str
            Sample ID
        invert_data : boolean
            If True - the data is inverted
        save_res_png  : boolean
            Save PNG image of the frame overlaid with with evaluation box
        """
        image_name = kwargs.get("image_name", "RawImageA")
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", os.path.dirname(self.fname))
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        nbins_disp = kwargs.get("nbins_disp", 256)
        thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
        invert_data = kwargs.get("invert_data", False)
        save_res_png = kwargs.get("save_res_png", False)

        if image_name == "RawImageA":
            img = self.RawImageA
        if image_name == "RawImageB":
            img = self.RawImageB
        if image_name == "ImageA":
            img = self.ImageA
        if image_name == "ImageB":
            img = self.ImageB

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

        range_disp = get_min_max_thresholds(
            img[yi_eval:ya_eval, xi_eval:xa_eval],
            thr_min=thresholds_disp[0],
            thr_max=thresholds_disp[1],
            nbins=nbins_disp,
            disp_res=False,
        )

        fig, ax = subplots(1, 1, figsize=(10.0, 11.0 * ysz / xsz))
        ax.imshow(img, cmap="Greys", vmin=range_disp[0], vmax=range_disp[1])
        ax.grid(True, color="cyan")
        ax.set_title(self.fname)
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
            fig.savefig(os.path.splitext(self.fname + "_evaluation_box.png", dpi=300))

    def determine_field_fattening_parameters(self, **kwargs):
        """
        Perfrom 2D parabolic fit (calls Perform_2D_fit(Img, estimator, **kwargs)) and determine the field-flattening parameters

        Parameters
        ----------
        kwargs:
        image_names : list of str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        estimator : RANSACRegressor(),
                    LinearRegression(),
                    TheilSenRegressor(),
                    HuberRegressor()
        bins : int
            binsize for image binning. If not provided, bins=10
        Analysis_ROIs : list of lists: [[left, right, top, bottom]]
            list of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the parabolic fit.
        calc_corr : boolean
            If True - the full image correction is calculated
        ignore_Y  : boolean
            If True - the parabolic fit to only X is perfromed
        Xsect : int
            X - coordinate for Y-crossection
        Ysect : int
            Y - coordinate for X-crossection
        disp_res : boolean
            (default is False) - to plot/ display the results
        save_res_png : boolean
            save the analysis output into a PNG file (default is False)
        save_correction_binary = boolean
            save the mage)name and img_correction_array data into a binary file
        res_fname : string
            filename for the result image ('**_Image_Flattening.png'). The binary image is derived from the same root, e.g. '**_Image_Flattening.bin'
        label : string
            optional image label
        dpi : int

        Returns:
        img_correction_coeffs, img_correction_arrays
        """
        image_names = kwargs.get("image_names", ["RawImageA"])
        estimator = kwargs.get("estimator", LinearRegression())
        if "estimator" in kwargs:
            del kwargs["estimator"]
        calc_corr = kwargs.get("calc_corr", False)
        ignore_Y = kwargs.get("ignore_Y", False)
        lbl = kwargs.get("label", "")
        disp_res = kwargs.get("disp_res", True)
        bins = kwargs.get("bins", 10)  # bins = 10
        Analysis_ROIs = kwargs.get("Analysis_ROIs", [])
        save_res_png = kwargs.get("save_res_png", False)
        res_fname = kwargs.get(
            "res_fname", os.path.splitext(self.fname)[0] + "_Image_Flattening.png"
        )
        save_correction_binary = kwargs.get("save_correction_binary", False)
        dpi = kwargs.get("dpi", 300)

        img_correction_arrays = []
        img_correction_coeffs = []
        for image_name in image_names:
            if image_name == "RawImageA":
                img = self.RawImageA - self.Scaling[1, 0]
            if image_name == "RawImageB":
                img = self.RawImageB - self.Scaling[1, 1]
            if image_name == "ImageA":
                img = self.ImageA
            if image_name == "ImageB":
                img = self.ImageB

            ysz, xsz = img.shape
            Xsect = kwargs.get("Xsect", xsz // 2)
            Ysect = kwargs.get("Ysect", ysz // 2)

            intercept, coefs, mse, img_correction_array = Perform_2D_fit(
                img, estimator, image_name=image_name, **kwargs
            )
            img_correction_arrays.append(img_correction_array)
            img_correction_coeffs.append(coefs)

        if calc_corr:
            self.image_correction_sources = image_names
            self.img_correction_arrays = img_correction_arrays
            if save_correction_binary:
                bin_fname = res_fname.replace("png", "bin")
                pickle.dump(
                    [image_names, img_correction_arrays], open(bin_fname, "wb")
                )  # saves source name and correction array into the binary file
                self.image_correction_file = res_fname.replace("png", "bin")
                print(
                    "Image Flattening Info saved into the binary file: ",
                    self.image_correction_file,
                )
        # self.intercept = intercept
        self.img_correction_coeffs = img_correction_coeffs
        return intercept, img_correction_coeffs, img_correction_arrays

    def flatten_image(self, **kwargs):
        """
        Flatten the image(s). Image flattening parameters must be determined (determine_field_fattening_parameters)

        Parameters
        ----------
        kwargs:
        image_correction_file : str
            full path to a binary filename that contains source names (image_correction_sources) and correction arrays (img_correction_arrays)
            if image_correction_file exists, the data is loaded from it.
        image_correction_sources : list of str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        img_correction_arrays : list of 2D arrays
            arrays containing field flatteting info

        Returns:
        flattened_images : list of 2D arrays
        """

        if hasattr(self, "image_correction_file"):
            image_correction_file = kwargs.get(
                "image_correction_file", self.image_correction_file
            )
        else:
            image_correction_file = kwargs.get("image_correction_file", "")

        try:
            # try loading the image correction data from the binary file
            with open(image_correction_file, "rb") as f:
                [image_correction_sources, img_correction_arrays] = pickle.load(f)
        except:
            #  if that did not work, see if the correction data was provided directly
            if hasattr(self, "image_correction_source"):
                image_correction_sources = kwargs.get(
                    "image_correction_sources", self.image_correction_sources
                )
            else:
                image_correction_sources = kwargs.get(
                    "image_correction_sources", [False]
                )

            if hasattr(self, "img_correction_arrays"):
                img_correction_arrays = kwargs.get(
                    "img_correction_arrays", self.img_correction_arrays
                )
            else:
                img_correction_arrays = kwargs.get("img_correction_arrays", [False])

        flattened_images = []
        for image_correction_source, img_correction_array in zip(
            image_correction_sources, img_correction_arrays
        ):
            if (image_correction_source is not False) and (
                img_correction_array is not False
            ):
                if image_correction_source == "RawImageA":
                    flattened_image = (
                        self.RawImageA - self.Scaling[1, 0]
                    ) * img_correction_array + self.Scaling[1, 0]
                if image_correction_source == "RawImageB":
                    flattened_image = (
                        self.RawImageB - self.Scaling[1, 1]
                    ) * img_correction_array + self.Scaling[1, 1]
                if image_correction_source == "ImageA":
                    flattened_image = self.ImageA * img_correction_array
                if image_correction_source == "ImageB":
                    flattened_image = self.ImageB * img_correction_array
            else:
                if image_correction_source == "RawImageA":
                    flattened_image = self.RawImageA
                if image_correction_source == "RawImageB":
                    flattened_image = self.RawImageB
                if image_correction_source == "ImageA":
                    flattened_image = self.ImageA
                if image_correction_source == "ImageB":
                    flattened_image = self.ImageB
            flattened_images.append(flattened_image)

        return flattened_images
