# %%
import os
import copy
import time

from functools import partial
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmap

from mpl_toolkits.mplot3d import Axes3D\

import scipy as sp
from scipy import ndimage
import scipy.optimize as optimize
import cPickle as cpl
import yaml

import pp_dexela

from hexrd import constants as cnst
from hexrd import config
from hexrd import imageseries
from hexrd import instrument
from hexrd import valunits
from hexrd import gridutil as gutil
from hexrd.xrd import distortion as dFuncs
from hexrd.xrd import transforms_CAPI as xfcapi
from hexrd.xrd import transforms as xf
from hexrd.xrd import indexer
from hexrd.xrd import experiment as expt
import hexrd.xrd.detector as det
from hexrd.xrd.transforms_CAPI import anglesToGVec, \
    makeRotMatOfExpMap, makeDetectorRotMat, makeOscillRotMat, \
    gvecToDetectorXY, detectorXYToGvec
import hexrd.xrd.xrdutil as xrdutil
from hexrd.xrd.xrdutil import angularPixelSize, make_reflection_patches, \
    simulateGVecs, _project_on_detector_plane

from hexrd.imageseries import omega
from hexrd.imageseries.omega import OmegaImageSeries
import hexrd.fitting.fitpeak
import hexrd.fitting.peakfunctions as pkfuncs
import hexrd.matrixutil as mutil
from hexrd.gridutil import cellIndices

# custom functions for this workflow found in dependent_functions.py
from dependent_functions_ti import *

# for applying processing options (flips, dark subtretc...)
PIS = pp_dexela.ProcessedDexelaIMS


# %%
# =============================================================================
# START USER INPUT
# =============================================================================

# CHOOSE PANEL, 0 = FF1, 1 = FF2
panel_id = 0

panel_fnames = ['ff2', 'ff1']
panel_keys = ['FF1', 'FF2']

matl_file = '/nfs/chess/user/lh644/ff_processing/materials.cpl'
det_file = '/nfs/chess/user/lh644/ff_processing_new/dexela_calibrated_instrument_22.yml'

mat_name = ['ni', 'chcr']

pktype = 'pvoigt'  # currently only psuedo voigt peak types work

# Fitting options
#ftol = 1e-6
#xtol = 1e-6

x_ray_energy = 55.618

# For 2021-2 setup, max_tth = 12.5 deg for 90 deg bins, max_tth = 9.5 deg for 0 deg bins
max_tth = 7
tth_bin_size = 0.22  # in degrees

# parameter
num_dark_frames = 1
num_frames = 89
# the first frame (0) seems to be coming out bright, start at 1 to keep things simple
start_frame = 1

spacing = 1  # there are lots of files might be better to not look at them all

diff_percentage = 0.25  # heuristic value to determine if a frame was scrambled, if two sequential frames varies by more than the percentage of total intensity, its assumed to be a garbage frame

# Sample Rotation parameters
delta_ome = 1
ome_bnds = [-44, 44]

num_processors = 16

# %% LOAD DETECTOR GEOMETRY
instr_cfg = yaml.load(open(det_file, 'r'))

tiltAngles = instr_cfg['detectors'][panel_keys[panel_id]
                                    ]['transform']['tilt_angles']
tVec_d = np.array(instr_cfg['detectors'][panel_keys[panel_id]]
                  ['transform']['t_vec_d']).reshape(3, 1)

chi = instr_cfg['oscillation_stage']['chi']
tVec_s = np.array(instr_cfg['oscillation_stage']['t_vec_s']).reshape(3, 1)

rMat_d = makeDetectorRotMat(tiltAngles)
rMat_s = makeOscillRotMat([chi, 0.])

pixel_size = instr_cfg['detectors'][panel_keys[0]]['pixels']['size']

nrows = instr_cfg['detectors'][panel_keys[panel_id]]['pixels']['rows']
ncols = instr_cfg['detectors'][panel_keys[panel_id]]['pixels']['columns']

row_dim = pixel_size[0]*nrows  # in mm
col_dim = pixel_size[1]*ncols  # in mm

x_col_edges = pixel_size[1]*(np.arange(ncols+1) - 0.5*ncols)
y_row_edges = pixel_size[0]*(np.arange(nrows+1) - 0.5*nrows)[::-1]

panel_dims = [(-0.5*ncols*pixel_size[1],
               -0.5*nrows*pixel_size[0]),
              (0.5*ncols*pixel_size[1],
               0.5*nrows*pixel_size[0])]

detector_params = np.hstack(
    [tiltAngles, tVec_d.flatten(), chi, tVec_s.flatten()])

distortion = None


# %% LOAD MATERIAL DATA AND ASSEMBLE MATERIALS
plane_data = [None]*len(mat_name)
mat_used = [None]*len(mat_name)

# grab plane data, and useful things hanging off of it
materials = cpl.load(open(matl_file, "rb"))
tth_vals = np.array([])

for jj in np.arange(len(mat_name)):

    check = np.zeros(len(materials))
    for ii in np.arange(len(materials)):
        # print materials[ii].name
        check[ii] = materials[ii].name == mat_name[jj]

    mat_used[jj] = materials[np.where(check)[0][0]]

    #niti_mart.beamEnergy = valunits.valWUnit("wavelength","ENERGY",61.332,"keV")
    mat_used[jj].beamEnergy = valunits.valWUnit(
        "wavelength", "ENERGY", x_ray_energy, "keV")
    mat_used[jj].planeData.exclusions = np.zeros(
        len(mat_used[0].planeData.exclusions), dtype=bool)

    mat_used[jj].planeData.tThMax = np.amax(np.radians(max_tth))

    plane_data[jj] = mat_used[jj].planeData

    tth_vals = np.hstack((tth_vals, plane_data[jj].getTTh()))

tth_vals = np.degrees(np.sort(tth_vals))  # Currently not working correctly

# Hard coded for Ti-TiB material
tth_vals = [3, 10.5]
tth_vals = np.sort(tth_vals)

# %% Use for Strains Along the loading

# note these are numbers in the unmasked portion of the plane_data list
hkl_nums_use = np.arange(len(tth_vals))


# %% Block of code for calculating 2theta bin widths and omega values

tth_bin_spacer = 0.1  # in degrees
tth_bnds = np.array([tth_vals[hkl_nums_use[0]]-tth_bin_spacer,
                     tth_vals[hkl_nums_use[-1]]+tth_bin_spacer])
print(tth_bnds)
tth_bin_size = tth_bnds[1]-tth_bnds[0]
tth_bin_cen = (tth_bnds[1]+tth_bnds[0])/2.
num_hkls = len(hkl_nums_use)
ome_cens = np.arange(ome_bnds[0]+delta_ome/2.,
                     ome_bnds[1]+delta_ome/2., delta_ome)


# %% Block of code for deciding on peaks bin widths
# Single Bin Processing
# FF1 bins spans from 0 -180, FF2 bins span a smaller range because they were sitting lower (216-324)
if panel_id == 0:
    azi_bin_size = 10  # size of the pie slize azimuthally
    eta_vals = np.arange(0, 181, 10)  # these can be whatever you want
    num_azi_bins = len(eta_vals)

else:
    azi_bin_size = 1.  # size of the pie slize azimuthally
    eta_vals = np.array([270.])  # these can be whatever you want
    num_azi_bins = len(eta_vals)

print(eta_vals)

# %% File Info and Macro Strain Info

data_loc = '/nfs/chess/raw/2022-2/id3a/hassani-2911-h/'
save_dir = '/nfs/chess/user/lh644/ti_tib_npz/'
file_header = 'Ti-TiB-1'

scans = np.array([4, 5, 6, 24, 25, 26, 29, 30, 31, 34, 35, 36, 54, 55, 56, 59, 60, 61, 64, 65, 66,
                  84, 85, 86, 89, 90, 91, 109, 110, 111, 114, 115, 116, 134, 135, 136])
darks = np.array([3, 3, 3, 23, 23, 23, 28, 28, 28, 33, 33, 33, 53, 53, 53, 58, 58, 58, 63, 63, 63,
                  83, 83, 83, 88, 88, 88, 108, 108, 108, 113, 113, 113, 133, 133, 133])
file_nos = np.array([141, 142, 143, 145, 146, 147, 149, 150, 151, 153, 154, 155, 157, 158, 159, 161, 162, 163, 165, 166, 167,
                     169, 170, 171, 173, 174, 175, 177, 178, 179, 181, 182, 183, 185, 186, 187])
dark_nos = np.array([140, 140, 140, 144, 144, 144, 148, 148, 148, 152, 152, 152, 156, 156, 156, 160, 160, 160, 164, 164, 164,
                     168, 168, 168, 172, 172, 172, 176, 176, 176, 180, 180, 180, 184, 184, 184])


num_scans = len(scans)

# %% Code for processing the data

scan_to_view = 0
frame_to_view = 10

num_azi_bins = len(eta_vals)
num_pks = num_hkls


pf_data = dict()
i_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
tth_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
eta_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
ome_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
width_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
strain_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])


# for ll in [scan_to_view]:
for ll in np.arange(num_scans):
    scan_no = ll
    print(' ')
    print('scan: ' + str(scans[scan_no]))

    file_name = data_loc + '%s/%d/ff/%s_%06.6d.h5' % (
        file_header, scans[scan_no], panel_fnames[panel_id], file_nos[scan_no])
    dark_name = data_loc + '%s/%d/ff/%s_%06.6d.h5' % (
        file_header, darks[scan_no], panel_fnames[panel_id], dark_nos[scan_no])

    raw_data = load_data(panel_keys, panel_id, num_frames,
                         file_name, num_dark_frames, dark_name)

    for kk in np.arange(num_frames):
        # for kk in [frame_to_view]:
        print('file: ' + str(kk))
        frame = raw_data[kk]

#        print('Fitting peaks using multiprocessing...')
#        pool = Pool(processes=num_processors)
#        func_part = partial(multi_spectrum_mp, frame, tth_bin_cen, detector_params, tth_bin_size, azi_bin_size,
#                            distortion, nrows, ncols, pixel_size, eta_vals, tth_vals[hkl_nums_use], pktype, num_hkls, None)
#        fit_data = pool.map(func_part, np.arange(len(eta_vals)), chunksize=1)
#        pool.close()
#        print('Reconsolidating data...')
        for jj in np.arange(num_azi_bins):

            cur_eta = eta_vals[jj]
            print(cur_eta)
#            tmp_p = fit_data[jj]
#            p_array = np.reshape(tmp_p[:4*num_hkls], [num_pks, 4])
#            f = np.zeros(len(tth_centers))
#            for ii in np.arange(num_pks):
#                f = f + pkfuncs._pvoigt1d_no_bg(p_array[ii], tth_centers)

            peak_data, tth_cen, eta_cen = pull_patch_data(raw_data[1], tth_bin_cen, cur_eta, detector_params,
                                                          tth_tol=tth_bin_size, eta_tol=azi_bin_size, npdiv=3,
                                                          distortion=distortion, pix_dims=[
                                                              nrows, ncols],
                                                          pixel_pitch=pixel_size)
#            print(peak_data)

            tth_centers = np.squeeze(tth_cen[0, :])
            intensity_1d = np.sum(peak_data, axis=0)
#            plt.close('all')
#            plt.plot(tth_centers, intensity_1d, 'x')
#            plt.plot(tth_centers, f)

            scan_str = str(scans[ll]).rjust(3, '0')
            frame_str = str(kk).rjust(2, '0')
            eta_str = str(eta_vals[jj]).rjust(3, '0')
            save_file_name = str(save_dir + file_header + "-scan-" +
                                 scan_str + '-frame-' + frame_str + '-eta-' + eta_str)
            np.savez(save_file_name + '.npz', tth_centers=tth_centers,
                     intensity_1d=intensity_1d)
#            print (np.size(tth_centers))

#            for ii in np.arange(num_hkls):
#                i_mat[ll, kk, ii, jj] = sp.integrate.simps(pkfuncs._pvoigt1d_no_bg(
#                    p_array[ii, :], tth_centers), tth_centers)  # fixed 6/9/18
#                # plt.plot(tth_centers,pkfuncs._pvoigt1d_no_bg(p_array[ii,:],tth_centers),'r')
#                # i_mat[ll,kk,ii,jj]=p_array[:,0]
#
#            tth_mat[ll, kk, :, jj] = p_array[:, 1]
#            eta_mat[ll, kk, :, jj] = np.ones(num_hkls)*cur_eta
#            ome_mat[ll, kk, :, jj] = np.ones(num_hkls)*ome_cens[kk]
#            width_mat[ll, kk, :, jj] = p_array[:, 2]

# %% File Locations

data_loc = '/nfs/chess/raw/2022-2/id3a/hassani-2911-h/'
save_dir = '/nfs/chess/user/lh644/ti_tib_npz/'
file_header = 'Ti-TiB-2'

scans = np.array([3, 4, 5, 23, 24, 25, 28, 29, 30, 33, 34, 35,
                  53, 54, 55, 58, 59, 60, 63, 64, 65, 83, 84, 85])
darks = np.array([2, 2, 2, 22, 22, 22, 27, 27, 27, 32, 32, 32,
                  52, 52, 52, 57, 57, 57, 62, 62, 62, 82, 82, 82])
file_nos = np.array([189, 190, 191, 193, 194, 195, 197, 198, 199, 201,
                     202, 203, 205, 206, 207, 209, 210, 211, 213, 214, 215, 217, 218, 219])
dark_nos = np.array([188, 188, 188, 192, 192, 192, 196, 196, 196, 200,
                     200, 200, 204, 204, 204, 208, 208, 208, 212, 212, 212, 216, 216, 216])


data_loc = '/nfs/chess/raw/2022-2/id3a/hassani-2911-h/'
save_dir = '/nfs/chess/user/lh644/ti_tib_npz/'
file_header = 'Ti-TiB-2b'

scans = np.array([9, 10, 11, 14, 15, 16, 19, 20, 21, 24, 25, 26,
                  29, 30, 31, 34, 35, 36, 39, 40, 41, 44, 45, 46, 49, 50, 51])
darks = np.array([8, 8, 8, 13, 13, 13, 18, 18, 18, 23, 23, 23,
                  28, 28, 28, 33, 33, 33, 38, 38, 38, 43, 43, 43, 48, 48, 48])
file_nos = np.array([227, 228, 229, 231, 232, 233, 235, 236, 237, 239, 240, 241,
                     243, 244, 245, 247, 248, 249, 251, 252, 253, 255, 256, 257, 259, 260, 261])
dark_nos = np.array([226, 226, 226, 230, 230, 230, 234, 234, 234, 238, 238, 238,
                     242, 242, 242, 246, 246, 246, 250, 250, 250, 254, 254, 254, 258, 258, 258])


data_loc = '/nfs/chess/raw/2022-2/id3a/hassani-2911-h/'
save_dir = '/nfs/chess/user/lh644/ti_tib_npz/'
file_header = 'Ti-TiB-3'

scans = np.array([10, 54, 95, 99, 102, 105, 108, 111,
                  114, 118, 121, 124, 127, 130, 133])
darks = np.array([9, 53, 94, 98, 101, 104, 107, 110,
                  113, 117, 120, 123, 126, 129, 132])
file_nos = np.array([13, 27, 29, 31, 33, 35, 37,
                     39, 41, 43, 45, 47, 49, 51, 53])
dark_nos = np.array([12, 26, 28, 30, 32, 34, 36,
                     38, 40, 42, 44, 46, 48, 50, 52])


data_loc = '/nfs/chess/raw/2022-2/id3a/hassani-2911-h/'
save_dir = '/nfs/chess/user/lh644/ti_tib_npz/'
file_header = 'Ti-TiB-4-p'

scans = np.array([3, 4, 5, 23, 24, 25, 28, 29, 30, 48, 49, 50, 53, 54, 73, 74, 75, 78, 79, 80, 98, 99, 100,
                  103, 104, 105, 129, 130, 131, 134, 135, 136, 154, 155, 156, 159, 160, 161])
darks = np.array([2, 2, 2, 22, 22, 22, 27, 27, 27, 47, 47, 47, 52, 52, 72, 72, 72, 77, 77, 77, 97, 97, 97,
                  102, 102, 102, 128, 128, 128, 133, 133, 133, 153, 153, 153, 158, 158, 158])
file_nos = np.array([88, 89, 90, 92, 93, 94, 96, 97, 98, 100, 101, 102, 104, 105, 106, 108, 109, 110, 112,
                     113, 114, 116, 117, 118, 120, 121, 122, 124, 125, 126, 128, 129, 130, 132, 133, 134, 136, 137, 138])
dark_nos = np.array([87, 87, 87, 91, 91, 91, 95, 95, 95, 99, 99, 99, 103, 103, 103, 107, 107, 107, 111,
                     111, 111, 115, 115, 115, 119, 119, 119, 123, 123, 123, 127, 127, 127, 131, 131, 131, 135, 135, 135])


data_loc = '/nfs/chess/raw/2022-2/id3a/hassani-2911-h/'
save_dir = '/nfs/chess/user/lh644/ti_tib_npz/'
file_header = 'Ti-TiB-4-v'

scans = np.array([8, 26, 29, 57, 60, 78, 81, 99, 102, 120, 123, 141, 144])
darks = np.array([7, 25, 28, 56, 59, 77, 80, 98, 101, 119, 122, 140, 143])
file_nos = np.array([60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84])
dark_nos = np.array([59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83])
