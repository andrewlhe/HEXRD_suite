import numpy as np
import csv
import os
import math
import random
import matplotlib.pyplot as plt


def save_csv(save_file_path, current_file_name, data):

    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)

    csv_file_path = str(
        save_file_path + current_file_name + '.csv')
    with open(csv_file_path, "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(data)


def read_npz(data_path):

    npz_file_path = data_path
    input_file_names = [f for f in os.listdir(npz_file_path) if (
        os.path.isfile(os.path.join(npz_file_path, f)) and f.endswith(".npz"))]

    for input_file_name in input_file_names:
        input_file_path = os.path.join(npz_file_path, input_file_name)
        data = np.load(input_file_path)
        tth_dat = np.array([data['tth_centers']])
        int_dat = np.array([data['intensity_1d']])
        dat = np.concatenate((tth_dat.T, int_dat.T), axis=1)
        print(dat)
        save_csv(data_path, input_file_name, dat)


def main(): 
    read_npz(r'Y:\CHESS\ID3A_2022-2\test\\')

if __name__ == "__main__":
    main()