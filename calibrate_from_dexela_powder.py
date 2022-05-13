#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:04:31 2017

@author: bernier2
"""
from __future__ import print_function

import os

import glob

import numpy as np

from scipy.optimize import leastsq

try:
    import dill as cpl
except(ImportError):
    import cPickle as cpl

import yaml
import pp_dexela

from hexrd import imageseries
from hexrd import instrument
from hexrd.fitting import fitpeak
import matplotlib.pyplot as plt

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
# =============================================================================
# USER INPUT
# =============================================================================

# Modified 6/21/2021 for 2021-2 Run

raw_data_dir_template = '/nfs/chess/raw/2022-2/id3a/%s/%s/%d/ff/*.h5'
expt_name = 'hassani-2911-h'
samp_name = 'ceo2-0429'

scan_number = 5

mat_filename = '/nfs/chess/user/lh644/ff_processing_new/materials_updated.cpl'
mat_key = 'ceo2'


instrument_filename = '/nfs/chess/user/lh644/ff_processing_new/dexela_calibrated_instrument_22.yml'

# !!!: hard coded options for each dexela for April 2017
# CAVEAT: these keys MUST match instrument file!
#
# in case of dark file:
#
#     panel_opts = [('dark', <dark image array>, ('flip', 'v'), ]
#
# note that panel options are applied in L-R order from the list.
# note that panel options are applied in L-R order from the list.
panel_opts = dict.fromkeys(['FF1', 'FF2'])
panel_opts['FF1'] = [('add-row', 1944), ('add-column', 1296),
                     ('flip', 'v'), ('flip', 'r90')]
panel_opts['FF2'] = [('add-row', 1944), ('add-column', 1296),
                     ('flip', 'h'), ('flip', 'r90')]

# tolerances for patches
tth_tol = 0.01
eta_tol = 10.

# output option
overwrite_fit = True

# %%
# =============================================================================
# IMAGESERIES
# =============================================================================

# for applying processing options (flips, dark subtretc...)
PIS = pp_dexela.ProcessedDexelaIMS

# load instrument
instr = load_instrument(instrument_filename)
det_keys = instr.detectors.keys()

# grab imageseries filenames
file_names = glob.glob(
    raw_data_dir_template % (expt_name, samp_name, scan_number)
)
check_files_exist = [os.path.exists(file_name) for file_name in file_names]
if not np.all(check_files_exist):
    raise RuntimeError("files don't exist!")

img_dict = dict.fromkeys(det_keys)
for file_name in file_names:
    ims = imageseries.open(file_name, format='hdf5', path='/imageseries')
    this_key = file_name.split('/')[-1].split('_')[0].upper()

    tmp = np.zeros(PIS(ims, panel_opts[this_key]).shape)

    for ii in np.arange(len(PIS(ims, panel_opts[this_key]))):
        tmp = tmp+PIS(ims, panel_opts[this_key])[ii]

    # the back of this command panel_opts is applying all of the flipping and dark substraction, ORDER SENSITIVE
    img_dict[this_key] = tmp

# %%
# =============================================================================
# INSTRUMENT
# =============================================================================

plane_data = load_pdata(mat_filename, mat_key)

instr = load_instrument(instrument_filename)

# get powder line profiles
#
# output is as follows:
#
# patch_data = {<detector_key>:[ringset_index][patch_index]}
# patch_data[0] = [two_theta_edges, ref_eta]
# patch_data[1] = [intensities]
##
powder_lines = instr.extract_line_positions(
    plane_data, img_dict,
    tth_tol=tth_tol, eta_tol=eta_tol,
    npdiv=2, collapse_tth=False,
    do_interpolation=True)


# ideal tth
tth_ideal = plane_data.getTTh()
tth0 = []
for idx in plane_data.getMergedRanges()[0]:
    tth0.append(tth_ideal[idx[0]])

# GRAND LOOP OVER PATCHES
rhs = dict.fromkeys(det_keys)
for det_key in det_keys:
    rhs[det_key] = []
    panel = instr.detectors[det_key]
    for i_ring, ringset in enumerate(powder_lines[det_key]):
        for angs, intensities in ringset:
            tth_centers = np.average(
                np.vstack([angs[0][:-1], angs[0][1:]]),
                axis=0)
            eta_ref = angs[1]

            p0 = fitpeak.estimate_pk_parms_1d(
                tth_centers, np.squeeze(intensities), 'pvoigt'
            )

            p = fitpeak.fit_pk_parms_1d(
                p0, tth_centers, np.squeeze(intensities), 'pvoigt'
            )

            tth_meas = p[1]

            #
            xy_meas = panel.angles_to_cart([[tth_meas, eta_ref], ])
            rhs[det_key].append(
                np.hstack([xy_meas.squeeze(), tth0[i_ring], eta_ref])
            )
    rhs[det_key] = np.array(rhs[det_key])

# build parameter list
x0 = []
for k, v in instr.detectors.iteritems():
    x0.append(np.hstack([v.tilt, v.tvec]))
x0 = np.hstack(x0)
# %%S


def multipanel_powder_objfunc(param_list, data_dict, instr):
    """
    """
    npp = 6
    ii = 0
    jj = npp
    resd = []
    for det_key, panel in instr.detectors.iteritems():

        params = param_list[ii:ii+jj]
        print(params)
        panel.tilt[:2] = params[:2]
        panel.tvec = params[3:]
        ii += npp
        jj += npp

        calc_xy = panel.angles_to_cart(data_dict[det_key][:, 2:])
        calc_r = np.sqrt(calc_xy[:, 0]**2+calc_xy[:, 1]**2)
        meas_r = np.sqrt(data_dict[det_key][:, 0] **
                         2+data_dict[det_key][:, 1]**2)

        tth_eta, g_vec = panel.cart_to_angles(data_dict[det_key][:, 0:2])

        strain = (np.sin(data_dict[det_key][:, 2] /
                         2.)/np.sin(tth_eta[:, 0]/2.)-1.)
        print(np.max(np.abs(strain)))
        plt.plot(strain)
        resd.append(
            #(data_dict[det_key][:, :2].flatten() - calc_xy.flatten())**2
            (np.sin(data_dict[det_key][:, 2]/2.) / \
             np.sin(tth_eta[:, 0]/2.)-1.)**2
        )
    return np.hstack(resd)


result = leastsq(multipanel_powder_objfunc, x0,
                 args=(rhs, instr), ftol=1e-8, xtol=1e-8)
# %%

if overwrite_fit:
    calibration_dict = {'grain_id': 0, 'inv_stretch': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        'orientation': [0.0, 0.0, 0.0], 'position': [0.0, 0.0, 0.0]}

    instr.write_config(instrument_filename, calibration_dict=calibration_dict)
