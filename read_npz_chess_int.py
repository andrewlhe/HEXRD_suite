import numpy as np
import csv
import os


def save_csv(save_file_path, current_file_name, data):

    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)

    csv_file_path = str(
        save_file_path + current_file_name + '.csv')
    with open(csv_file_path, "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(data)


def read_npz(data_path, save_path):

    npz_file_path = data_path
    input_file_names = [f for f in os.listdir(npz_file_path) if (
        os.path.isfile(os.path.join(npz_file_path, f)) and f.endswith("090.npz"))]

    for input_file_name in input_file_names:
        input_file_path = os.path.join(npz_file_path, input_file_name)
        data = np.load(input_file_path)
        tth_dat = np.array([data['tth_centers']])
        int_dat = np.array([data['intensity_1d']])
        dat = np.concatenate((tth_dat.T, int_dat.T), axis=1)
        print(dat)
        save_csv(save_path, input_file_name, dat)

    # Fitting Aid
    length = np.size(tth_dat)
    fitting_aid = np.zeros((1, length))

    for input_file_name in input_file_names:
        input_file_path = os.path.join(npz_file_path, input_file_name)
        data = np.load(input_file_path)
        int_dat = np.array([data['intensity_1d']])
        fitting_aid += int_dat
        fitting_data = np.concatenate((tth_dat.T, fitting_aid.T), axis=1)
    
    aid_file_name = input_file_name + "aid"
    save_csv(save_path, aid_file_name, fitting_data)


def main():
    read_npz(r'Z:\CHESS\ID3A_2022-2\lineouts\ti-tib-1\\',
             r'Z:\CHESS\ID3A_2022-2\lineouts\ti-tib-1-eta-090-1\\')
    # read_npz(r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2\\',
    #          r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2-eta-090\\')
    # read_npz(r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2b\\',
    #          r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2b-eta-090\\')
    # read_npz(r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-3\\',
    #          r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-3-eta-090\\')
    # read_npz(r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4p\\',
    #          r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4p-eta-090\\')
    # read_npz(r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4v\\',
    #          r'Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4v-eta-090\\')

if __name__ == "__main__":
    main()
