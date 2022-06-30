import csv
import os
import math
import random
import string


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


# Function to save csv
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
    
    fig, ax = plt.subplots(1,1)    
    img = ax.imshow(matrix, interpolation='none',
               vmin=lower_bound, vmax=upper_bound, extent=[0,90,0,360], cmap='bwr')
    ax.set_xticks([0,30,60,90])
    ax.set_yticks([0,90,180,270,360])
    fig.colorbar(img, orientation='vertical')
    ax.set_aspect('auto')

    plt.savefig(str(output_folder + current_file_name + note),
                bbox_inches='tight')
    plt.clf()
    plt.close(fig)


def get_tth(matrix):
    tth = np.sin((matrix * (10.2665586536948 - 1.57211584737215) /
                  2999 + 1.57211584737215)/2*np.pi/180)
    return tth


def get_pole_figure_strain(sample, data_path, tth, ome, azm, strain_index):

    name = [sample, "_pf_s000_", str(azm).rjust(3, '0'), "deg_center.csv"]
    input_file_name = "".join(name)
    input_file_path = os.path.join(data_path, input_file_name)
    array = csv_column_to_matrix(input_file_path, True, strain_index, 90)
    calibrated_tth = get_tth(array)
    strain = calibrated_tth[0, ome]/tth-1

    return strain


def get_pole_figure_fwhm(sample, data_path, fwhm, ome, azm, strain_index):

    name = [sample, "_pf_s000_", str(azm).rjust(3, '0'), "deg_sigma.csv"]
    input_file_name = "".join(name)
    input_file_path = os.path.join(data_path, input_file_name)
    array = csv_column_to_matrix(input_file_path, True, strain_index, 90)
    relative_fwhm = fwhm/array[0, ome]

    return relative_fwhm


def get_pole_figure_intensity(sample, data_path, inten, ome, azm, strain_index):

    name = [sample, "_pf_s000_", str(azm).rjust(3, '0'), "deg_amplitude.csv"]
    input_file_name = "".join(name)
    input_file_path = os.path.join(data_path, input_file_name)
    array = csv_column_to_matrix(input_file_path, True, strain_index, 90)
    relative_intensity = inten/array[0, 1]

    return relative_intensity


def polt_pole_figure_strain(lb, ub, strain_index, sample, data_path, debug_path, image_path):

    input_file_names = [f for f in os.listdir(data_path) if (
        os.path.isfile(os.path.join(data_path, f)) and f.endswith(".csv"))]
    input_file_names.sort()
    counter = 0
    strain_output = []

    for input_file_name in input_file_names:

        input_file_path = os.path.join(data_path, input_file_name)

        center = csv_column_to_matrix(input_file_path,
                                      has_header_row=True,
                                      column_index=strain_index,
                                      matrix_row_length=90)
        dot_index = input_file_name.rindex(".")
        csv_name = input_file_name[:dot_index] + "_"
        note = str('strain' + str(strain_index))
        counter = counter + 1

        tth = get_tth(center)
        strain_data = []

        for i in range(1, 90):
            omega = i
            azm = ((counter-1)*10) % 360
            strain_to_calculate = tth[0, i]
            strain = get_pole_figure_strain(sample,
                                            data_path, strain_to_calculate, omega, azm, strain_index)
            strain_data.append(strain)

        strain_output.append(strain_data)
        #  save_csv(debug_path, csv_name, note, strain_output)
        if counter % 36 == 0:
            make_plot(strain_output, image_path, csv_name, note, lb, ub)
            strain_output = []


def polt_pole_figure_fwhm(lb, ub, strain_index, sample, data_path, debug_path, image_path):

    input_file_names = [f for f in os.listdir(data_path) if (
        os.path.isfile(os.path.join(data_path, f)) and f.endswith(".csv"))]
    input_file_names.sort()
    counter = 0
    fwhm_output = []

    for input_file_name in input_file_names:

        input_file_path = os.path.join(data_path, input_file_name)

        center = csv_column_to_matrix(input_file_path,
                                      has_header_row=True,
                                      column_index=strain_index,
                                      matrix_row_length=90)
        dot_index = input_file_name.rindex(".")
        csv_name = input_file_name[:dot_index] + "_"
        note = str('fwhm' + str(strain_index))
        counter = counter + 1

        tth = center
        fwhm_data = []

        for i in range(1, 90):
            omega = i
            azm = ((counter-1)*10) % 360
            fwhm_to_calculate = tth[0, i]
            fwhm = get_pole_figure_fwhm(sample,
                                        data_path, fwhm_to_calculate, omega, azm, strain_index)
            fwhm_data.append(fwhm)

        fwhm_output.append(fwhm_data)
        #  save_csv(debug_path, csv_name, note, strain_output)
        if counter % 36 == 0:
            make_plot(fwhm_output, image_path, csv_name, note, lb, ub)
            fwhm_output = []


def polt_pole_figure_intensity(lb, ub, strain_index, sample, data_path, debug_path, image_path):

    input_file_names = [f for f in os.listdir(data_path) if (
        os.path.isfile(os.path.join(data_path, f)) and f.endswith(".csv"))]
    input_file_names.sort()
    counter = 0
    intensity_output = []

    for input_file_name in input_file_names:

        input_file_path = os.path.join(data_path, input_file_name)

        center = csv_column_to_matrix(input_file_path,
                                      has_header_row=True,
                                      column_index=strain_index,
                                      matrix_row_length=90)
        dot_index = input_file_name.rindex(".")
        csv_name = input_file_name[:dot_index] + "_"
        note = str('intensity' + str(strain_index))
        counter = counter + 1

        tth = center
        intensity_data = []

        for i in range(1, 90):
            omega = i
            azm = ((counter-1)*10) % 360
            intensity_to_calculate = tth[0, i]
            intensity = get_pole_figure_intensity(sample,
                                                  data_path, intensity_to_calculate, omega, azm, strain_index)
            intensity_data.append(intensity)

        intensity_output.append(intensity_data)
        #  save_csv(debug_path, csv_name, note, strain_output)
        if counter % 36 == 0:
            make_plot(intensity_output, image_path, csv_name, note, lb, ub)
            intensity_output = []


def pole_figure(type, lb, ub, peaks, sample, data_path, debug_path, image_path):
    lower_bound = lb
    upper_bound = ub

    if not os.path.exists(debug_path):
        os.mkdir(debug_path)
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    for peaks in peaks:
        strain_index = peaks
        text = [image_path, "/", type, "_", str(strain_index), "/"]
        image_folder = ''.join(text)
        if type == "strain":
            polt_pole_figure_strain(lower_bound, upper_bound, strain_index, sample,
                                    data_path, debug_path, image_folder)
        elif type == "intensity":
            polt_pole_figure_intensity(lower_bound, upper_bound, strain_index, sample,
                                       data_path, debug_path, image_folder)
        elif type == "fwhm":
            polt_pole_figure_fwhm(lower_bound, upper_bound, strain_index, sample,
                                  data_path, debug_path, image_folder)
        else:
            print("error")
            quit()


def main():
    # pole_figure("strain", -.003, .003, np.arange(0, 9), "N3_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_3-1-HT9/pf_center/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_3-1-HT9/")
    # pole_figure("intensity", 0.8, 1.2, np.arange(0, 9), "N3_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_3-1-HT9/pf_amp/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_3-1-HT9/")
    # pole_figure("fwhm", 0.5, 1.5, [2, 5, 6, 7, 8], "N3_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_3-1-HT9/pf_sigma/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_3-1-HT9/")

    # pole_figure("strain", -.003, .003, np.arange(0, 9), "N3_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_3-2-HT9/pf_center/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_3-2-HT9/")
    # pole_figure("intensity", 0.8, 1.2, np.arange(0, 9), "N3_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_3-2-HT9/pf_amp/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_3-2-HT9/")
    # pole_figure("fwhm", 0.5, 1.5, [2, 5, 6, 7, 8], "N3_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_3-2-HT9/pf_sigma/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_3-2-HT9/")

    # pole_figure("strain", -.003, .003, np.arange(0, 9), "N2_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_2-1-HT6/pf_center/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_2-1-HT6/")
    # pole_figure("intensity", 0.8, 1.2, np.arange(0, 9), "N2_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_2-1-HT6/pf_amp/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_2-1-HT6/")
    # pole_figure("fwhm", 0.5, 1.5, [2, 5, 6, 7, 8], "N2_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_2-1-HT6/pf_sigma/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_2-1-HT6/")

    # pole_figure("strain", -.003, .003, np.arange(0, 9), "N2_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_2-2-As-Is/pf_center/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_2-2-As-Is/")
    # pole_figure("intensity", 0.8, 1.2, np.arange(0, 9), "N2_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_2-2-As-Is/pf_amp/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_2-2-As-Is/")
    # pole_figure("fwhm", 0.5, 1.5, [2, 5, 6, 7, 8], "N2_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/N_2-2-As-Is/pf_sigma/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/N_2-2-As-Is/")

    pole_figure("strain", -.003, .003, np.arange(0, 9), "He2_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/He_2-1-HT9/pf_center/",
                "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_2-1-HT9/")
    # pole_figure("intensity", 0.8, 1.2, np.arange(0, 9), "He2_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/He_2-1-HT9/pf_amp/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_2-1-HT9/")
    pole_figure("fwhm", 0.5, 1.5, [2, 5, 6, 7, 8], "He2_1", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/He_2-1-HT9/pf_sigma/",
                "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_2-1-HT9/")

    pole_figure("strain", -.003, .003, np.arange(0, 9), "He2_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/He_2-2-As-Is/pf_center/",
                "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_2-2-As-Is/")
    # pole_figure("intensity", 0.8, 1.2, np.arange(0, 9), "He2_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/He_2-2-As-Is/pf_amp/",
    #             "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_2-2-As-Is/")
    pole_figure("fwhm", 0.5, 1.5, [2, 5, 6, 7, 8], "He2_2", "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/He_2-2-As-Is/pf_sigma/",
                "Y:/APS/2021-3_1IDC/WAXS_fitting/pf/debug/", "Y:/APS/2021-3_1IDC/WAXS_fitting/plots/He_2-2-As-Is/")

if __name__ == "__main__":
    main()
