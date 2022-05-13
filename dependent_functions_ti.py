#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:21:09 2020

@author: dcp99
"""

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
# REQUIRED FUNCTIONS
# =============================================================================
# plane data


def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


# images
def load_images(yml):
    return imageseries.open(yml, format="frame-cache", style="npz")


# instrument
def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


# %%
# Multipeak Kludge
def fit_pk_obj_1d_mpeak(p, x, f0, pktype, num_pks):

    f = np.zeros(len(x))
    # THIS IS ALSO MESSED UP AND HARD CODED FOR PSUEDO VOIGTS
    p_fit = np.reshape(p[:4*num_pks], [num_pks, 4])
    for ii in np.arange(num_pks):
        if pktype == 'gaussian':
            f = f+pkfuncs._gaussian1d_no_bg(p_fit[ii], x)
        elif pktype == 'lorentzian':
            f = f+pkfuncs._lorentzian1d_no_bg(p_fit[ii], x)
        elif pktype == 'pvoigt':
            f = f+pkfuncs._pvoigt1d_no_bg(p_fit[ii], x)
        elif pktype == 'split_pvoigt':
            f = f+pkfuncs._split_pvoigt1d_no_bg(p_fit[ii], x)

    f = f+x**2*p[-3]+x*p[-2]+p[-1]  # Quadratic background
    resd = f-f0
    return resd


# Create an angular patch on the detector and pulls intensity values for the patch
# currently a kludge hacked together from pullspots functionality in hexrd.xrdutil
def pull_patch_data(frame, tth, eta, detector_params, tth_tol, eta_tol,
                    npdiv, distortion, pix_dims, pixel_pitch):
    # Experiment Geometry Extraction
    rMat_d = xfcapi.makeDetectorRotMat(detector_params[:3])
    tVec_d = np.ascontiguousarray(detector_params[3:6])
    chi = detector_params[6]
    tVec_s = np.ascontiguousarray(detector_params[7:10])
    rMat_s = xfcapi.makeOscillRotMat([chi, 0.])

    # crystal; this will be a list of things, computed from quaternions
    #  - trivial case here...
    rMat_c = np.eye(3)
    tVec_c = np.zeros((3, 1))

    # Detector Geometry Extraction
    num_rows = pix_dims[0]
    num_cols = pix_dims[1]

    # panel dimensions calculated from pixel pitches
    row_dim = pixel_pitch[0]*num_rows
    col_dim = pixel_pitch[1]*num_cols

    # panel is ( (xmin, ymin), (xmax, ymax) )
    panel_dims = (
        (-0.5*col_dim, -0.5*row_dim),
        (0.5*col_dim,  0.5*row_dim),
    )
    row_edges = (np.arange(num_rows+1)*pixel_pitch[0] + panel_dims[0][1])[::-1]
    col_edges = np.arange(num_cols+1)*pixel_pitch[1] + panel_dims[0][0]

    # Patch Center
    angs = [tth*d2r, eta*d2r, 0.]

    gVec_c_cen = xfcapi.anglesToGVec(angs,
                                     chi=chi,
                                     rMat_c=rMat_c)

    xy_cen = xfcapi.gvecToDetectorXY(gVec_c_cen,
                                     rMat_d, rMat_s, rMat_c,
                                     tVec_d, tVec_s, tVec_c)

    est_ang_pix_res = xrdutil.angularPixelSize(xy_cen, pixel_pitch,
                                               rMat_d, rMat_s,
                                               tVec_d, tVec_s, tVec_c,
                                               distortion=distortion)[0]

    ndiv_tth = npdiv*np.ceil(tth_tol/(est_ang_pix_res[0]*r2d))
    ndiv_eta = npdiv*np.ceil(eta_tol/(est_ang_pix_res[1]*r2d))

    # Make Patch, Bilinear Interpolation Would Be Nice
    tth_del = np.arange(0, ndiv_tth+1)*tth_tol/float(ndiv_tth) - 0.5*tth_tol
    eta_del = np.arange(0, ndiv_eta+1)*eta_tol/float(ndiv_eta) - 0.5*eta_tol

    # store dimensions for convenience
    #   * etas and tths are bin vertices, ome is already centers
    sdims = [len(eta_del)-1, len(tth_del)-1]

    # meshgrid args are (cols, rows), a.k.a (fast, slow)
    m_tth, m_eta = np.meshgrid(tth_del, eta_del)
    npts_patch = m_tth.size

    # calculate the patch XY coords from the (tth, eta) angles
    # * will CHEAT and ignore the small perturbation the different
    #   omega angle values causes and simply use the central value
    gVec_angs_vtx = np.tile(angs, (npts_patch, 1)) \
        + d2r*np.vstack([m_tth.flatten(),
                         m_eta.flatten(),
                         np.zeros(npts_patch)
                         ]).T

    # connectivity
    conn = gutil.cellConnectivity(sdims[0], sdims[1], origin='ll')

    # evaluation points...
    #   * for lack of a better option will use centroids
    tth_eta_cen = np.array(gutil.cellCentroids(
        np.atleast_2d(gVec_angs_vtx[:, :2]), conn))

    gVec_angs = np.hstack([tth_eta_cen,
                           np.tile(angs[2], (len(tth_eta_cen), 1))])
    gVec_c = xfcapi.anglesToGVec(gVec_angs,
                                 chi=chi,
                                 rMat_c=rMat_c)

    xy_eval = xfcapi.gvecToDetectorXY(gVec_c,
                                      rMat_d, rMat_s, rMat_c,
                                      tVec_d, tVec_s, tVec_c)
#    print xy_eval
#
#    plt.plot(xy_eval[:,0],xy_eval[:,1],'x')

    # Apply distortion correction
    if distortion is not None and len(distortion) == 2:
        xy_eval = distortion[0](xy_eval, distortion[1], invert=True)
        pass

    row_indices = gutil.cellIndices(row_edges, xy_eval[:, 1])
    col_indices = gutil.cellIndices(col_edges, xy_eval[:, 0])

    patch_j, patch_i = np.meshgrid(range(sdims[1]), range(sdims[0]))
    patch_i = patch_i.flatten()
    patch_j = patch_j.flatten()

#    plt.figure()
#    plt.imshow(frame)
    peak_data = frame[row_indices, col_indices].reshape(sdims[0], sdims[1])

    tth_cen = tth_eta_cen[:, 0].reshape(sdims[0], sdims[1])*r2d
    eta_cen = tth_eta_cen[:, 1].reshape(sdims[0], sdims[1])*r2d
    return peak_data, tth_cen, eta_cen


# %% Assemble Guess

def estimate_mpk_parms(tth_vals, x, f, pktype='pvoigt', bgtype='linear', fwhm_guess=0.07):

    num_pks = len(tth_vals)
    min_val = np.min(f)

    if pktype == 'gaussian' or pktype == 'lorentzian':
        p0tmp = np.zeros([num_pks, 3])
        p0tmp_lb = np.zeros([num_pks, 3])
        p0tmp_ub = np.zeros([num_pks, 3])

        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            pt = np.argmin(np.abs(x-tth_vals[ii]))
            p0tmp[ii, :] = [(f[pt]-min_val), tth_vals[ii], fwhm_guess]
            p0tmp_lb[ii, :] = [(f[pt]-min_val)*0.1,
                               tth_vals[ii]-0.02, fwhm_guess*0.5]
            p0tmp_ub[ii, :] = [(f[pt]-min_val)*10.0,
                               tth_vals[ii]+0.02, fwhm_guess*2.0]
    elif pktype == 'pvoigt':
        p0tmp = np.zeros([num_pks, 4])
        p0tmp_lb = np.zeros([num_pks, 4])
        p0tmp_ub = np.zeros([num_pks, 4])

        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            pt = np.argmin(np.abs(x-tth_vals[ii]))
            p0tmp[ii, :] = [np.abs(f[pt]-min_val),
                            tth_vals[ii], np.abs(fwhm_guess), 0.5]
            p0tmp_lb[ii, :] = [
                np.abs(f[pt]-min_val)*0.1, tth_vals[ii]-0.02, np.abs(fwhm_guess)*0.5, 0.0]
            p0tmp_ub[ii, :] = [
                np.abs(f[pt]-min_val)*10.0, tth_vals[ii]+0.02, np.abs(fwhm_guess)*2.0, 1.0]
    elif pktype == 'split_pvoigt':
        p0tmp = np.zeros([num_pks, 6])
        p0tmp_lb = np.zeros([num_pks, 6])
        p0tmp_ub = np.zeros([num_pks, 6])

        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_pks):
            pt = np.argmin(np.abs(x-tth_vals[ii]))
            p0tmp[ii, :] = [(f[pt]-min_val), tth_vals[ii],
                            fwhm_guess, fwhm_guess, 0.5, 0.5]
            p0tmp_lb[ii, :] = [(f[pt]-min_val)*0.1, tth_vals[ii] -
                               0.02, fwhm_guess*0.5, fwhm_guess*0.5, 0.0, 0.0]
            p0tmp_ub[ii, :] = [(f[pt]-min_val)*10.0, tth_vals[ii] +
                               0.02, fwhm_guess*2.0, fwhm_guess*2.0, 1.0, 1.0]

    if bgtype == 'linear':
        num_pk_parms = len(p0tmp.ravel())
        p0 = np.zeros(num_pk_parms+2)
        lb = np.zeros(num_pk_parms+2)
        ub = np.zeros(num_pk_parms+2)
        p0[:num_pk_parms] = p0tmp.ravel()
        lb[:num_pk_parms] = p0tmp_lb.ravel()
        ub[:num_pk_parms] = p0tmp_ub.ravel()

        p0[-2] = min_val

        lb[-2] = -float('inf')
        lb[-1] = -float('inf')

        ub[-2] = float('inf')
        ub[-1] = float('inf')

    elif bgtype == 'constant':
        num_pk_parms = len(p0tmp.ravel())
        p0 = np.zeros(num_pk_parms+1)
        lb = np.zeros(num_pk_parms+1)
        ub = np.zeros(num_pk_parms+1)
        p0[:num_pk_parms] = p0tmp.ravel()
        lb[:num_pk_parms] = p0tmp_lb.ravel()
        ub[:num_pk_parms] = p0tmp_ub.ravel()

        p0[-1] = min_val
        lb[-1] = -float('inf')
        ub[-1] = float('inf')

    elif bgtype == 'quadratic':
        num_pk_parms = len(p0tmp.ravel())
        p0 = np.zeros(num_pk_parms+3)
        lb = np.zeros(num_pk_parms+3)
        ub = np.zeros(num_pk_parms+3)
        p0[:num_pk_parms] = p0tmp.ravel()
        lb[:num_pk_parms] = p0tmp_lb.ravel()
        ub[:num_pk_parms] = p0tmp_ub.ravel()

        p0[-3] = min_val
        lb[-3] = -float('inf')
        lb[-2] = -float('inf')
        lb[-1] = -float('inf')
        ub[-3] = float('inf')
        ub[-2] = float('inf')
        ub[-1] = float('inf')

    bnds = (lb, ub)

    return p0, bnds


# %% Multiprocessing Dev
def multi_spectrum_mp(frame, tth_bin_cen, detector_params, tth_bin_size, azi_bin_size, distortion, nrows, ncols, pixel_size, eta_vals, tth_vals, pktype, num_hkls, fit_data, index):

    peak_data, tth_cen, eta_cen = pull_patch_data(frame, tth_bin_cen, eta_vals[index], detector_params,
                                                  tth_tol=tth_bin_size, eta_tol=azi_bin_size, npdiv=3,
                                                  distortion=distortion, pix_dims=[
                                                      nrows, ncols],
                                                  pixel_pitch=pixel_size)

    tth_centers = np.squeeze(tth_cen[0, :])

    intensity_1d = np.sum(peak_data, axis=0)

    x = tth_centers
    f = intensity_1d

    fitArgs = (x, f, pktype, num_hkls)

    # THIS IS HARD CODED FOR PSUEDO VOIGTS
    if fit_data == None:
        # make initial guess for the fit parameters and generate bounds
        p0tmp = np.zeros([num_hkls, 4])
        p0tmp_lb = np.zeros([num_hkls, 4])
        p0tmp_ub = np.zeros([num_hkls, 4])

        # x is just 2theta values
        # make guess for the initital parameters
        for ii in np.arange(num_hkls):
            pt = np.argmin(np.abs(x-tth_vals[ii]))

            # parameters from dcp
#            p0tmp[ii,:]=[f[pt],tth_vals[ii],0.05,0.5] #amplitude @ the 2theta0 position
#            p0tmp_lb[ii,:]=[f[pt]*0.1,tth_vals[ii]-0.03,0.0,0.0]
#            p0tmp_ub[ii,:]=[f[pt]*15.0,tth_vals[ii]+0.03,0.3,1.0]

            # these are the initial guesses for bounds for fitting
            # ordering 0: amplitude, 1: position(2theta mean), 2: FWHM, 3: mixing parameter
            # you should not be changing the bounds of the mixing parameters
            # amplitude @ the 2theta0 position
            p0tmp[ii, :] = [f[pt], tth_vals[ii], 0.05, 0.5]
            p0tmp_lb[ii, :] = [f[pt]*.1, tth_vals[ii]- 0.03, 0.0, 0.0]
            p0tmp_ub[ii, :] = [f[pt]*20, tth_vals[ii]+ 0.03, 1, 1.0]

        p0 = np.zeros(4*num_hkls+3)
        lb = np.zeros(4*num_hkls+3)
        ub = np.zeros(4*num_hkls+3)

        # bound guesses for background parameters
        p0[:4*num_hkls] = p0tmp.ravel()
        lb[:4*num_hkls] = p0tmp_lb.ravel()
        lb[-3] = -float('inf')
        lb[-2] = -float('inf')
        lb[-1] = -float('inf')
        ub[:4*num_hkls] = p0tmp_ub.ravel()
        ub[-3] = float('inf')
        ub[-2] = float('inf')
        ub[-1] = float('inf')
        bnds = (lb, ub)

        p_init = optimize.least_squares(fit_pk_obj_1d_mpeak, p0.ravel(
        ), bounds=bnds, args=fitArgs, ftol=1e-6, xtol=1e-6)

    else:
        guess = fit_data[index]
        guess_r = np.reshape(guess[:4*num_hkls], [num_hkls, 4])

        p0tmp_lb = np.zeros([num_hkls, 4])
        p0tmp_ub = np.zeros([num_hkls, 4])

        # x is just 2theta values
        # make guess for the initital parameters
        # this is not used

        for ii in np.arange(num_hkls):

            p0tmp_lb[ii, :] = [guess_r[ii, 0]*0.75,
                               guess_r[ii, 1]-0.015, 0.05, 0.0]
            p0tmp_ub[ii, :] = [guess_r[ii, 0]*1.5,
                               guess_r[ii, 1]+0.015, 0.2, 1.0]

        lb = np.zeros(4*num_hkls+3)
        ub = np.zeros(4*num_hkls+3)

        # bound guesses for background parameters
        lb[:4*num_hkls] = p0tmp_lb.ravel()
        lb[-3] = -float('inf')
        lb[-2] = -float('inf')
        lb[-1] = -float('inf')
        ub[:4*num_hkls] = p0tmp_ub.ravel()
        ub[-3] = float('inf')
        ub[-2] = float('inf')
        ub[-1] = float('inf')
        bnds = (lb, ub)

        p_init = optimize.least_squares(
            fit_pk_obj_1d_mpeak, guess, bounds=bnds, args=fitArgs, ftol=1e-6, xtol=1e-6)

    # p_array=np.reshape(p.x[:4*num_hkls],[num_pks,p.x.shape[0]/(num_pks)])

    guess = p_init.x
    guess_r = np.reshape(guess[:4*num_hkls], [num_hkls, 4])

    # x is just 2theta values
    # make guess for the initital parameters
    for ii in np.arange(num_hkls):
        #
        # p0tmp_lb[ii,:]=[guess_r[ii,0]*0.1,guess_r[ii,1]-0.015,0.0,0.0] #from dcp
        # p0tmp_ub[ii,:]=[guess_r[ii,0]*5.0,guess_r[ii,1]+0.015,guess_r[ii,2]*1.5,1.0]
        # these are the secondary guesses for bounds for fitting
        p0tmp_lb[ii, :] = [guess_r[ii, 0]*0.1, guess_r[ii, 1]-.015, 0.0, 0.0]
        p0tmp_ub[ii, :] = [guess_r[ii, 0]*5.0,
                           guess_r[ii, 1]+0.015, guess_r[ii, 2]*1.5, 1.0]

    lb = np.zeros(4*num_hkls+3)
    ub = np.zeros(4*num_hkls+3)

    # bound guesses for background parameters
    lb[:4*num_hkls] = p0tmp_lb.ravel()
    lb[-3] = -float('inf')
    lb[-2] = -float('inf')
    lb[-1] = -float('inf')
    ub[:4*num_hkls] = p0tmp_ub.ravel()
    ub[-3] = float('inf')
    ub[-2] = float('inf')
    ub[-1] = float('inf')
    bnds = (lb, ub)

    p = optimize.least_squares(
        fit_pk_obj_1d_mpeak, guess, bounds=bnds, args=fitArgs, ftol=1e-6, xtol=1e-6)

    return p.x


def load_data(panel_keys, panel_id, num_frames, file_name, num_dark_frames, dark_name):

    panel_opts = dict.fromkeys(panel_keys)

    panel_opts = dict.fromkeys(['FF1', 'FF2'])
    panel_opts['FF1'] = [('add-row', 1944), ('add-column',
                                             1296), ('flip', 'v'), ('flip', 'r90')]
    panel_opts['FF2'] = [('add-row', 1944), ('add-column',
                                             1296), ('flip', 'h'), ('flip', 'r90')]

    omw = omega.OmegaWedges(num_dark_frames)
    omw.addwedge(0., 360.0, num_dark_frames)
    ppd_dark = pp_dexela.PP_Dexela(
        dark_name,
        omw,
        panel_opts[panel_keys[panel_id]],
        panel_id=panel_keys[panel_id],
        frame_start=0)

    omw = omega.OmegaWedges(num_frames)
    omw.addwedge(0., 360.0, num_frames)
    ppd = pp_dexela.PP_Dexela(
        file_name,
        omw,
        panel_opts[panel_keys[panel_id]],
        panel_id=panel_keys[panel_id],
        frame_start=0)

    ppd._dark = ppd_dark.dark

    raw_data = ppd.processed()

    return raw_data
