import csv
import os
import math
import random


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def csv_column_to_matrix(csv_file_path, has_header_row, column_index, matrix_row_length):
    matrix = []
    with open(csv_file_path, mode="r") as input_file:
        csv_reader = csv.reader(input_file, delimiter=",")
        is_first_row = True
        row_count = 0
        matrix_row = []
        for row in csv_reader:
            if is_first_row:
                is_first_row = False
                if has_header_row:
                    continue
            row_count += 1
            row = [float(x) for x in row]
            matrix_row.append(row[column_index])
            if row_count % matrix_row_length == 0:
                matrix.append(matrix_row)
                matrix_row = []
    return np.array(matrix)


def get_tth(matrix):
    tth = np.sin((matrix * (10.2665586536948 - 1.57211584737215) /
                  2999 + 1.57211584737215)/2*np.pi/180)
    return tth


def save_csv(save_file_path, current_file_name, data_type, data):

    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)

    csv_file_path = str(
        save_file_path + current_file_name + data_type + '.csv')
    with open(csv_file_path, "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(data)


def make_plot(matrix, output_folder, current_file_name, note, lower_bound, upper_bound):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    plt.imshow(matrix, interpolation='none',
               vmin=lower_bound, vmax=upper_bound, cmap='bwr')
    plt.colorbar(orientation='vertical')
    plt.savefig(str(output_folder + current_file_name + note),
                bbox_inches='tight')
    plt.clf()


def get_mapping_strain(sample, data_path, tth, crystal_plane):

    name = [data_path, "/", sample, "_2d_mapping_s000_90deg_center.csv"]
    input_file_name = "".join(name)
    array = csv_column_to_matrix(input_file_name, True, crystal_plane, 56)
    calibrated_tth = get_tth(array)
    strain = np.zeros((37, 56))

    for k in range(0, 37):
        for j in range(0, 56):
            strain[k][j] = calibrated_tth[k][j]/tth[k][j]-1

    return strain


def get_mapping_intensity(sample, data_path, inten, crystal_plane):

    name = [data_path, "/", sample, "_2d_mapping_s000_90deg_amplitude.csv"]
    input_file_name = "".join(name)
    array = csv_column_to_matrix(input_file_name, True, crystal_plane, 56)
    intensity = np.zeros((37, 56))

    for k in range(0, 37):
        for j in range(0, 56):
            intensity[k][j] = inten[k][j]/array[k][j]

    return intensity


def get_mapping_fwhm(sample, data_path, fwhm, crystal_plane):

    name = [data_path, "/", sample, "_2d_mapping_s000_90deg_sigma.csv"]
    input_file_name = "".join(name)
    array = csv_column_to_matrix(input_file_name, True, crystal_plane, 56)
    relative_sigma = np.zeros((37, 56))

    for k in range(0, 37):
        for j in range(0, 56):
            relative_sigma[k][j] = fwhm[k][j]/array[k][j]

    return relative_sigma


def polt_strain(sample, lb, ub, crystal_plane, data_path, debug_path, image_path):

    input_file_names = [f for f in os.listdir(data_path) if (
        os.path.isfile(os.path.join(data_path, f)) and f.endswith(".csv"))]
    for input_file_name in input_file_names:
        input_file_path = os.path.join(data_path, input_file_name)
        center = csv_column_to_matrix(input_file_path,
                                      has_header_row=True,
                                      column_index=crystal_plane,
                                      matrix_row_length=56)
        dot_index = input_file_name.rindex(".")
        csv_name = input_file_name[:dot_index] + "_"
        note = str('strain' + str(crystal_plane))
        tth = get_tth(center)
        strain = get_mapping_strain(sample, data_path, tth, crystal_plane)
        # save_csv(debug_path, csv_name, note, strain)
        path = [image_path, "/strain", str(crystal_plane).rjust(2, '0'), "/"]
        plot_path = "".join(path)
        make_plot(strain, plot_path, csv_name, note, lb, ub)


def polt_intensity(sample, lb, ub, crystal_plane, data_path, debug_path, image_path):

    csv_file_path = data_path
    input_file_names = [f for f in os.listdir(csv_file_path) if (
        os.path.isfile(os.path.join(csv_file_path, f)) and f.endswith(".csv"))]
    for input_file_name in input_file_names:
        input_file_path = os.path.join(csv_file_path, input_file_name)
        inten = csv_column_to_matrix(input_file_path,
                                     has_header_row=True,
                                     column_index=crystal_plane,
                                     matrix_row_length=56)
        dot_index = input_file_name.rindex(".")
        csv_name = input_file_name[:dot_index] + "_"
        note = str('intensity' + str(crystal_plane))
        relative_intensity = get_mapping_intensity(
            sample, data_path, inten, crystal_plane)
        # save_csv(debug_path, csv_name, note, relative_intensity)
        path = [image_path, "/intensity", str(crystal_plane).rjust(2, '0'), "/"]
        plot_path = "".join(path)
        make_plot(relative_intensity, plot_path,
                  csv_name, note, lb, ub)


def polt_sigma(sample, lb, ub, crystal_plane, data_path, debug_path, image_path):

    csv_file_path = data_path
    input_file_names = [f for f in os.listdir(csv_file_path) if (
        os.path.isfile(os.path.join(csv_file_path, f)) and f.endswith(".csv"))]
    for input_file_name in input_file_names:
        input_file_path = os.path.join(csv_file_path, input_file_name)
        sigma = csv_column_to_matrix(input_file_path,
                                     has_header_row=True,
                                     column_index=crystal_plane,
                                     matrix_row_length=56)
        dot_index = input_file_name.rindex(".")
        csv_name = input_file_name[:dot_index] + "_"
        note = str('fwhm' + str(crystal_plane))
        relative_FWHM = get_mapping_fwhm(
            sample, data_path, sigma, crystal_plane)
        # save_csv(debug_path, csv_name, note, relative_FWHM)
        path = [image_path, "/fwhm", str(crystal_plane).rjust(2, '0'), "/"]
        plot_path = "".join(path)
        make_plot(relative_FWHM, plot_path, csv_name,
                  note, lb, ub)


def mapping(type, sample, data_path, debug_path, image_path):

    # Peak 0: Ni (111)/Crc(511) Peak 1: Ni (200) Peak 2: Ni (220) Peak 3: Ni (311) Peak 4: CrC (420) Peak 5: CrC (422)	Peak 6: CrC (440) 2020-3
    # Peak 0/CrC (420) 1/CrC (422) 2/Ni (111) 3/CrC (440) 4/CrC (531) 5/Ni (200) 6/Ni (220) 7/Ni (311) 8/Ni (222)

    if not os.path.exists(debug_path):
        os.mkdir(debug_path)
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    crystal_plane = np.arange(0,9)
    for crystal_plane in crystal_plane:
        if type == "strain":
            polt_strain(sample, -.003, 0.003, crystal_plane, data_path, debug_path, image_path)
        elif type == "intensity":
            polt_intensity(sample, 0.5, 1.5, crystal_plane, data_path, debug_path, image_path)
        elif type == "fwhm":
            polt_sigma(sample, 0.8, 1.2, crystal_plane, data_path, debug_path, image_path)
        else:
            print("error")
            quit()

def main():
    # mapping("strain", "He1_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/He_1-1/90deg_center/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
    #         "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_1-1/")
    # mapping("fwhm", "He1_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/He_1-1/90deg_sigma/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
    #         "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_1-1/")
    # mapping("intensity", "He1_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/He_1-1/90deg_amp/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
    #         "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_1-1/")

    mapping("strain", "He1_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/He_1-2/90deg_center/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
            "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_1-2/")
    mapping("fwhm", "He1_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/He_1-2/90deg_sigma/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
            "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_1-2/")
    mapping("intensity", "He1_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/He_1-2/90deg_amp/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
            "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_1-2/")

    mapping("strain", "N1_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/N_1-2/90deg_center/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
            "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_1-2/")
    mapping("fwhm", "N1_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/N_1-2/90deg_sigma/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
            "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_1-2/")
    mapping("intensity", "N1_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/N_1-2/90deg_amp/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
            "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_1-2/")

    mapping("strain", "N1_3", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/N_1-3/90deg_center/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
            "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_1-3/")
    mapping("fwhm", "N1_3", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/N_1-3/90deg_sigma/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
            "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_1-3/")
    mapping("intensity", "N1_3", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/N_1-3/90deg_amp/", "Y:/APS/2021-3_1IDC/WAXS_fitting/mapping/debug/",
            "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_1-3/")

if __name__ == "__main__":
    main()
