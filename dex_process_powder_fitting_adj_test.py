    # %%

# custom functions for this workflow found in dependent_functions.py
from dependent_functions_cu import *
import pp_dexela
from hexrd.imageseries import omega
import os
import copy
import time
from functools import partial
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D\

import matplotlib.cm as cmap


import cPickle as cpl
import yaml

import scipy.optimize as optimize

import hexrd.xrd.detector as det


import hexrd.fitting.fitpeak
import hexrd.fitting.peakfunctions as pkfuncs

from hexrd.xrd import distortion as dFuncs
from hexrd.xrd import transforms_CAPI as xfcapi


from hexrd.xrd import transforms as xf
from hexrd.xrd.transforms_CAPI import anglesToGVec, \
    makeRotMatOfExpMap, makeDetectorRotMat, makeOscillRotMat, \
    gvecToDetectorXY, detectorXYToGvec
from hexrd.xrd.xrdutil import angularPixelSize, make_reflection_patches, \
    simulateGVecs, _project_on_detector_plane


from hexrd import constants as cnst
from hexrd import config
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd import instrument
from hexrd.xrd import indexer


from hexrd import valunits

from hexrd.gridutil import cellIndices


from hexrd.xrd import experiment as expt
import hexrd.xrd.xrdutil as xrdutil
import hexrd.matrixutil as mutil

from hexrd.xrd import transforms as xf

from scipy import ndimage
import scipy as sp

from hexrd import gridutil as gutil


d2r = np.pi/180.
r2d = 180./np.pi


# for applying processing options (flips, dark subtretc...)
PIS = pp_dexela.ProcessedDexelaIMS


# %%
# =============================================================================
# START USER INPUT
# =============================================================================

# CHOOSE PANEL, 0 = FF1, 1 = FF2
panel_id = 0

panel_fnames = ['ff1', 'ff2']
panel_keys = ['FF1', 'FF2']

dark_name = '/nfs/chess/user/lh644/ceo2-0611/3/ff/%s_000084.h5' % (
    panel_fnames[panel_id])

matl_file = '/nfs/chess/user/lh644/ff_processing/materials.cpl'
det_file = '/nfs/chess/user/lh644/ff_processing_new/dexela_calibrated_instrument.yml'

mat_name = ['ni', 'chcr']

pktype = 'pvoigt'  # currently only psuedo voigt peak types work

# Fitting options
#ftol = 1e-6
#xtol = 1e-6

x_ray_energy = 61.332

# For 2021-2 setup, max_tth = 12.5 deg for 90 deg bins, max_tth = 9.5 deg for 0 deg bins
max_tth = 16
tth_bin_size = 0.22  # in degrees

# parameter
num_dark_frames = 1
num_frames = 1440
# the first frame (0) seems to be coming out bright, start at 1 to keep things simple
start_frame = 1

spacing = 1  # there are lots of files might be better to not look at them all

diff_percentage = 0.25  # heuristic value to determine if a frame was scrambled, if two sequential frames varies by more than the percentage of total intensity, its assumed to be a garbage frame

# Sample Rotation parameters
delta_ome = .25
ome_bnds = [-180, 180]

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


tth_vals = np.sort(tth_vals)*r2d  # Currently not working correctly


# Hard Coded for Ni-CrC Composites
# TTH values corresponding to the following peaks:
# Crc (420) CrC (422) CrC (511) Ni (111) CrC (440) CrC (531) Ni (200)
# CrC (622) CrC (822) Ni (220) CrC (662) CrC (911) CrC (844) Ni (311) Ni (222).

#tth_vals = [4.89, 5.36, 5.67, 5.73, 6.19, 6.47, 6.63,
#            7.25, 9.39, 9.41, 9.5, 10, 10.8, 11.06, 11.56]
#tth_vals = np.sort(tth_vals)

# Hard Coded for Ni-CrC Composites Transverse HT.HIP
# TTH values corresponding to the following peaks:
## Crc (420) CrC (422) CrC (511) Ni (111) CrC (440) CrC (531) Ni (200) CrC (622)
#
#tth_vals = [4.84, 5.31, 5.63, 5.68, 6.15, 6.41, 6.54, 6.58, 7.20]
#tth_vals = np.sort(tth_vals)

# Hard Coded for Ni-CrC Composites/ HT 9 specific
# TTH values corresponding to the following peaks:
# Crc (420) ???? CrC (422) ???? CrC (511) Ni (111) CrC (440) CrC (531) Ni (200)
# CrC (622) CrC (822) Ni (220) CrC (662) ???? CrC (911) CrC (844) Ni (311) Ni (222)

#tth_vals = [4.89, 5.1, 5.36, 5.5, 5.67, 5.73, 6.19, 6.47, 6.6,
#            7.25, 9.36, 9.41, 9.5, 9.85, 10, 10.8, 11., 11.5]
#tth_vals = np.sort(tth_vals)


# Hard coded for Cu-Ta material
tth_vals = [4.95, 5.22, 5.57, 6.44, 7.00, 7.39, 7.66, 8.60, 9, 9.14, 9.98, 10.49, 10.8, 11.28, 11.76, 14.32]
tth_vals = np.sort(tth_vals)

#tth_vals = [4.95, 5.11, 5.22, 5.54, 6.42]
#tth_vals = np.sort(tth_vals)
# %% Use for Strains Along the loading

# note these are numbers in the unmasked portion of the plane_data list
hkl_nums_use = np.arange(len(tth_vals))


# %% Block of code for calculating 2theta bin widths and omega values

tth_bin_spacer = 0.5  # in degrees
tth_bnds = np.array([tth_vals[hkl_nums_use[0]]-tth_bin_spacer,
                     tth_vals[hkl_nums_use[-1]]+tth_bin_spacer])
tth_bin_size = tth_bnds[1]-tth_bnds[0]
tth_bin_cen = (tth_bnds[1]+tth_bnds[0])/2.
num_hkls = len(hkl_nums_use)
ome_cens = np.arange(ome_bnds[0]+delta_ome/2.,
                     ome_bnds[1]+delta_ome/2., delta_ome)


# %% Block of code for deciding on peaks bin widths
# Single Bin Processing
# FF1 bins spans from 0 -180, FF2 bins span a smaller range because they were sitting lower (216-324)
if panel_id == 0:
    azi_bin_size = 1  # size of the pie slize azimuthally
    eta_vals = np.array([90.])  # these can be whatever you want
    num_azi_bins = len(eta_vals)

else:
    azi_bin_size = 5.  # size of the pie slize azimuthally
    eta_vals = np.array([270.])  # these can be whatever you want
    num_azi_bins = len(eta_vals)

print(np.arange(len(eta_vals)))

# %% File Info and Macro Strain Info
#data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
#file_header = 'ceo2-0611'
#scans = np.array([1, 2, 3])
#file_nos = scans + 81
#num_scans = len(scans)

data_loc = '/nfs/chess/user/lh644/'
file_header = 'Cu-1'
scans = np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19])
file_nos = scans + 84
num_scans = len(scans)


#data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
#file_header = 'Cu-2'
#scans = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18])
#file_nos = np.array([104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
#num_scans = len(scans)

# macro_strain = np.array([0., 0.000758, 0.00128, 0.001921, 0.002345, 0.002738, 0.003234,
#                         0.003684, 0.003944, 0.0042, 0.004576, 0.004989, 0.005237,
#                         0.005465, 0.005687, 0.006267, 0.0069, 0.007624])
# applied_load = np.arange(0, 851, 50)


# %% Code for processing the data

from dependent_functions_cu import *

scan_to_view = 0
frame_to_view = 0 

num_azi_bins = len(eta_vals)
num_pks = num_hkls


pf_data = dict()
i_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
tth_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
eta_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
ome_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
width_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])
strain_mat = np.zeros([num_scans, num_frames, num_hkls, num_azi_bins])


#for ll in [scan_to_view]:
for ll in np.arange(num_scans):
    print(' ')
    print('scan: '+str(ll))

    scan_no = ll
    file_name = data_loc + '%s/%d/ff/%s_%06.6d.h5' % (
        file_header, scans[scan_no], panel_fnames[panel_id], file_nos[scan_no])

    raw_data = load_data(panel_keys, panel_id, num_frames,
                         file_name, num_dark_frames, dark_name)

    peak_data, tth_cen, eta_cen = pull_patch_data(raw_data[1], tth_bin_cen, 90., detector_params,
                                                  tth_tol=tth_bin_size, eta_tol=azi_bin_size, npdiv=3,
                                                  distortion=distortion, pix_dims=[
                                                      nrows, ncols],
                                                  pixel_pitch=pixel_size)
#    plt.close('all')
    tth_centers = np.squeeze(tth_cen[0, :])
    intensity_1d = np.sum(peak_data, axis=0)
    plt.plot(tth_centers, intensity_1d, 'x')

    for kk in [frame_to_view]:  # np.arange(num_frames):
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

#            cur_eta = eta_vals[jj]
#            tmp_p = fit_data[jj]
#
#            p_array = np.reshape(tmp_p[:4*num_hkls], [num_pks, 4])
#
#            f = np.zeros(len(tth_centers))
#            for ii in np.arange(num_pks):
#
#                f = f+pkfuncs._pvoigt1d_no_bg(p_array[ii], tth_centers)
#
#            f = f+tth_centers**2*tmp_p[-3]+tth_centers * \
#                tmp_p[-2]+tmp_p[-1]  # Quadratic background

            plt.plot(tth_centers, intensity_1d, 'x')
            #plt.plot(tth_centers, f)
            
            save_file_name = 'Cu1-ff0-90-scan-' + str(ll) + '-frame-' + str (kk)
            np.savez(save_file_name+'.npz', tth_centers = tth_centers, intensity_1d = intensity_1d)
            print (np.size(tth_centers))

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

# %% File and folders

# data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
# file_header = 'Cu-1'
# scans = np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
# file_nos = scans + 84
# num_scans = len(scans)


#data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
#file_header = 'HIP-1/WAXS'
#scans = np.array([1, 17, 18, 34, 35, 36, 52, 53, 69, 70, 71, 87])
#file_nos = np.array([132, 133, 134, 135, 136, 137,
#                     138, 139, 140, 141, 142, 143])
#num_scans = len(scans)

#data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
#file_header = 'HT9-1/WAXS'
#scans = np.array([1, 17, 19, 37, 38, 39, 55, 56, 75, 76, 77, 108])
#file_nos = np.array([120, 121, 122, 123, 124, 125,
#                      126, 127, 128, 129, 130, 131])
#num_scans = len(scans)

#data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
#file_header = 'HIP-2'
#scans = np.array([1, 35, 36, 37, 53, 54, 55, 71, 102])
#file_nos = np.array([229, 263, 264,
#                     265, 266, 267, 268, 269, 270])
#num_scans = len(scans)

#data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
#file_header = 'HT6-1/WAXS'
#scans = np.array([16, 17, 18, 34, 35, 36, 52, 53, 69, 70])
#file_nos = np.array([159, 160, 161, 177, 178, 179, 195, 196, 212, 213])
#num_scans = len(scans)

#data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
#file_header = 'He-2/WAXS'
#scans = np.array([1, 17, 18, 34, 35, 36, 52, 53, 69, 71, 72, 88, 89, 120, 121])
#file_nos = np.array([144, 145, 146, 147, 148, 149, 150,
#                     151, 152, 153, 154, 155, 156, 157, 158])
#num_scans = len(scans)
            
#data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
#file_header = 'N-1/WAXS'
#scans = np.array([4, 41, 42, 66, 67, 83, 84, 100, 101, 118,
#                  119, 135, 136, 152, 153, 170, 171, 172, 188])
#file_nos = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21,
#                     22, 23, 24, 25, 26, 27, 28, 29, 30, 46])
#num_scans = len(scans)
            
#data_loc = '/nfs/chess/raw/2021-2/id3a/hassani-1000-5/'
#file_header = 'He-1/WAXS'
#scans = np.array([16, 17, 18, 19, 20, 21, 38, 39, 40, 41,
#                  42, 67, 68, 69, 70, 71, 87, 103, 104])
#file_nos = np.array([47, 48, 49, 50, 51, 52, 69, 70, 71,
#                     72, 73, 74, 75, 76, 77, 78, 79, 80, 81])
#num_scans = len(scans)
            