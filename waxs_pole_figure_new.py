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

    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(matrix, interpolation='none',
                    vmin=lower_bound, vmax=upper_bound, extent=[0, 90, 0, 360], cmap='bwr')
    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([0, 90, 180, 270, 360])
    fig.colorbar(img, orientation='vertical')
    ax.set_aspect('auto')

    plt.savefig(str(output_folder + current_file_name + note),
                bbox_inches='tight')
    plt.clf()
    plt.close(fig)


def extract_parameter(peak_index, data_path, debug_path):

    input_file_names = [f for f in os.listdir(data_path) if (
        os.path.isfile(os.path.join(data_path, f)) and f.endswith(".csv"))]
    input_file_names.sort()

    center_list = []

    for input_file_name in input_file_names:

        input_file_path = os.path.join(data_path, input_file_name)
        centers = pd.read_csv(input_file_path, header=0)
        center = centers.loc[0].iat[peak_index]
        center_list.append(center)

    print(len(center_list))
    return center_list


def compose_strains(tth, eta, omega, scans, peak, sample, debug_path):
    if not os.path.exists(debug_path):
        os.mkdir(debug_path)

    strain = np.zeros([scans, omega, eta])
    tth_mat = np.zeros([scans, omega, eta])
    
    index = 0
    for i in range(scans):
        for j in range(omega):
            for k in range(eta):
                # print(index)
                tth_mat[i, j, k] = tth[index]
                index += 1
    
    for i in range(scans):
        for j in range(omega):
            for k in range(eta):             
                strain[i, j, k] = tth_mat[0, j, k]/tth_mat[i, j, k]-1
        # Code to save CSV files
        data_type = "strains_" + str(i).rjust(2, '0') + "_"+ str(peak).rjust(2, '0')
        save_csv(debug_path, sample, data_type, strain[i, :, :])

    return strain


def make_plots(image_path, sample, scans, peak, data, lower_bound, upper_bound):
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    for i in range(scans):
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(data[i, :, :], interpolation='none',
                        vmin=lower_bound, vmax=upper_bound, extent=[0, 180, 0, 90], cmap='bwr')
        ax.set_xticks([0, 90, 180])
        ax.set_yticks([0, 30, 60, 90])
        fig.colorbar(img, orientation='vertical')
        ax.set_aspect('auto')
        path_comp = str(image_path) + sample + "_" + str(peak).rjust(2, '0') + "_" + str(i).rjust(2, '0')
        plt.savefig(path_comp, bbox_inches='tight')
        plt.clf()
        plt.close(fig)


def pole_figure_ti(type, lb, ub, peaks, sample, scans, data_path, image_path, debug_path):

    if not os.path.exists(debug_path):
        os.mkdir(debug_path)
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    eta = 19
    omega = 89

    for peak in peaks:
        image_folder_path = ''.join(
            [image_path, "\\", type, "_", str(peak), "\\"])
        debug_folder_path = ''.join(
            [debug_path, "\\", type, "_", str(peak), "\\"])
        parameter = extract_parameter(peak, data_path, debug_folder_path)

        if type == "strain":
            strain = compose_strains(
                parameter, eta, omega, scans, peak, sample, debug_folder_path)
            make_plots(image_folder_path, sample, scans, peak, strain, lb, ub)
        # elif type == "intensity":
        #     polt_pole_figure_intensity(lower_bound, upper_bound, peak, sample,
        #                                data_path, debug_path, image_folder_path)
        # elif type == "fwhm":
        #     polt_pole_figure_fwhm(lower_bound, upper_bound, peak, sample,
        #                           data_path, debug_path, image_folder_path)
        else:
            print("error")
            quit()


def main():
    # # For CHESS 2022-2 Ti-TiB samples
    pole_figure_ti("strain", -.01, .01, np.arange(0, 6), "Ti-TiB-4v", 13, r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4v-eta-all\results\center",
                   r"Y:\CHESS\ID3A_2022-2\polefigures\ti-tib-4v", r"Y:\CHESS\ID3A_2022-2\debug\ti-tib-4v")
    pole_figure_ti("strain", -.01, .01, np.arange(0, 6), "Ti-TiB-4p", 38, r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4p-eta-all\results\center",
                   r"Y:\CHESS\ID3A_2022-2\polefigures\ti-tib-4p", r"Y:\CHESS\ID3A_2022-2\debug\ti-tib-4p")
    pole_figure_ti("strain", -.01, .01, np.arange(0, 6), "Ti-TiB-3", 15, r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-3-eta-all\results\center",
                   r"Y:\CHESS\ID3A_2022-2\polefigures\ti-tib-3", r"Y:\CHESS\ID3A_2022-2\debug\ti-tib-3")
    pole_figure_ti("strain", -.01, .01, np.arange(0, 6), "Ti-TiB-2b", 27, r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2b-eta-all\results\center",
                   r"Y:\CHESS\ID3A_2022-2\polefigures\ti-tib-2b", r"Y:\CHESS\ID3A_2022-2\debug\ti-tib-2b")
    pole_figure_ti("strain", -.01, .01, np.arange(0, 6), "Ti-TiB-2", 24, r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2-eta-all\results\center",
                   r"Y:\CHESS\ID3A_2022-2\polefigures\ti-tib-2", r"Y:\CHESS\ID3A_2022-2\debug\ti-tib-2")
    pole_figure_ti("strain", -.01, .01, np.arange(0, 6), "Ti-TiB-1", 36, r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-1-eta-all\results\center",
                   r"Y:\CHESS\ID3A_2022-2\polefigures\ti-tib-1", r"Y:\CHESS\ID3A_2022-2\debug\ti-tib-1")

if __name__ == "__main__":
    main()
