import csv
import os
import random

import numpy as np
import pandas as pd
from scipy import optimize, signal

from lmfit import models

DEBUG = False
K = 0.2 #tth boundary

# Function to generate models
def generate_model(spec):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        # for now VoigtModel has gamma constrained to sigma
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(
                f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(
            **default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params


# Function to consolidate information for each peak
def make_spec_voigt(x, y, model_type, center, height, sigma, gamma, center_min, center_max):
    return {
        'x': x,
        'y': y,
        'model': [
            {
                'type': model_type,
                'params': {'center': center, 'height': height, 'sigma': sigma, 'gamma': gamma},
                # 'help': {'center': {'min': center_min, 'max': center_max}}
            }
        ]
    }


# Function to run the fitting (and to provide plots of fittings)
def process_spec(spec):
    model, params = generate_model(spec)
    output = model.fit(spec['y'], params, x=spec['x'])
    print_best_values(spec, output)
    return output.best_values


# Function to display best values
def print_best_values(spec, output):
    model_params = {
        'GaussianModel':   ['amplitude', 'sigma'],
        'LorentzianModel': ['amplitude', 'sigma'],
        'VoigtModel':      ['amplitude', 'sigma', 'gamma']
    }
    best_values = output.best_values
    if DEBUG:
        print('center    model          amplitude     sigma      gamma')
    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        values = ', '.join(
            f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
        if DEBUG:
            print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')
            print()


# Function to process each row of data 
def process_row_tialphatib(row):

    # # for Ti-alpha-TiB samples
    # # Peaks to fit: TiB (200)*** TiB (211) TiB (301) Ti-a (100) Ti-a (002) Ti-a (103) fitting using Voigt models - Axial Parameters
    centers = [4.20, 6.540, 6.868, 5.000, 5.466, 9.633]
    lower_bounds = [x - K for x in centers]
    upper_bounds = [x + K for x in centers]

    # # Peaks to fit: TiB (200) TiB (211) TiB (301) Ti-a (100) Ti-a (002) Ti-a (103) fitting using Voigt models - Transverse Parameters
    # centers = [4.165, 6.530, 6.850, 4.974, 5.428, 9.565]
    # lower_bounds = [x - K for x in centers]
    # upper_bounds = [x + K for x in centers]


    # # for Ti-alpha-Ti-beta-TiB samples
    # # Peaks to fit: TiB (101) TiB (301) Ti-a (100) Ti-a (002) Ti-a (103) Ti-b (200) fitting using Voigt models - Axial Parameters
    # centers = [3.473, 6.868, 5.000, 5.466, 9.633, 7.900]
    # lower_bounds = [x - K for x in centers]
    # upper_bounds = [x + K for x in centers]


    types = ['VoigtModel', 'VoigtModel', 'VoigtModel', 'VoigtModel', 'VoigtModel',
             'VoigtModel']
    heights = [100000, 100000, 100000, 200000, 100000, 200000]
    sigmas = [1, 1, 1, 1, 1, 1]
    gammas = [1, 1, 1, 1, 1, 1]
    center_min = []
    center_max = []

    amplitude_row = []
    center_row = []
    sigma_row = []
    gamma_row = []

    for lower_bound, upper_bound, model_type, center, height, sigma, gamma in list(zip(lower_bounds, upper_bounds, types, centers, heights, sigmas, gammas)):
        np_x = row.iloc[:,0].to_numpy()
        np_y = row.iloc[:,1].to_numpy()
        difference_array_lower_bound = np.absolute(np_x-lower_bound)
        difference_array_upper_bound = np.absolute(np_x-upper_bound)
        lower_bound_index = difference_array_lower_bound.argmin()
        upper_bound_index = difference_array_upper_bound.argmin()
        x = np_x[lower_bound_index:upper_bound_index]
        y = np_y[lower_bound_index:upper_bound_index]
        spec = make_spec_voigt(x, y, model_type, center,
                               height, sigma, gamma, center_min, center_max)
        best_values = process_spec(spec)

        amplitude_row.append(best_values["m0_amplitude"])
        center_row.append(best_values["m0_center"])
        sigma_row.append(best_values["m0_sigma"])
        gamma_row.append(best_values["m0_gamma"])

    return amplitude_row, center_row, sigma_row, gamma_row


# Function to process entire dataset
def process_file_tialphatib(input_directory_path, input_file_name, output_directory_path):

    
    if not os.path.exists(output_directory_path):
        os.mkdir(output_directory_path)


    # for use with Ti-alpha-TiB
    header_row = ['TiB (200)', 'TiB (211)', 'TiB (301)',
                  'Ti-a (100)', 'Ti-a (002)', 'Ti-a (103)']

    # # for use with Ti-alpha/beta-TiB
    # header_row = ['TiB (101)', 'TiB (301)', 'Ti-a (100)',
    #               'Ti-a (002)', 'Ti-a (103)', 'Ti-b (200)']

    amplitude_data = []
    center_data = []
    sigma_data = []
    gamma_data = []

    input_file_path = os.path.join(input_directory_path, input_file_name)

    df = pd.read_csv(input_file_path, header=None)

    amplitude_row, center_row, sigma_row, gamma_row = process_row_tialphatib(
        df)
    amplitude_data.append(amplitude_row)
    center_data.append(center_row)
    sigma_data.append(sigma_row)
    gamma_data.append(gamma_row)

    for type, data in list(zip(["amplitude", "center", "sigma", "gamma"], [amplitude_data, center_data, sigma_data, gamma_data])):
        dot_index = input_file_name.rindex(".npz")
        output_file_name = input_file_name[:dot_index] + "_" + type + ".csv"
        output_file_path = os.path.join(output_directory_path, type)
        save_csv(output_file_path, output_file_name, header_row, data)


def save_csv(csv_file_path, csv_file_name, headers, data):

    if not os.path.exists(csv_file_path):
        os.mkdir(csv_file_path)
    csv_file = os.path.join(csv_file_path, csv_file_name)
    with open(csv_file, "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(headers)
        csv_writer.writerows(data)


def main():  # for 2022-2-ID3A testing

    # Location to save images (if selected)
    image_dir = "C:/Users/helew/Documents/Fitting"

    # # Import Data & set output directory

    # Single phase Ti
    input_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-1-eta-all"
    output_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-1-eta-all\results"

    input_file_names = [f for f in os.listdir(input_directory_path) if (
        os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

    for input_file_name in input_file_names:
        process_file_tialphatib(input_directory_path, input_file_name,
                                output_directory_path)

    input_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2-eta-all"
    output_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2-eta-all\results"

    input_file_names = [f for f in os.listdir(input_directory_path) if (
        os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

    for input_file_name in input_file_names:
        process_file_tialphatib(input_directory_path, input_file_name,
                                output_directory_path)

    input_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2b-eta-all"
    output_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-2b-eta-all\results"

    input_file_names = [f for f in os.listdir(input_directory_path) if (
        os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

    for input_file_name in input_file_names:
        process_file_tialphatib(input_directory_path, input_file_name,
                                output_directory_path)

    # # Dual phase Ti
    # input_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-3-eta-all"
    # output_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-3-eta-all\results"

    # input_file_names = [f for f in os.listdir(input_directory_path) if (
    #     os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

    # for input_file_name in input_file_names:
    #     process_file_tialphatib(input_directory_path, input_file_name,
    #                             output_directory_path)

    # input_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4p-eta-all"
    # output_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4p-eta-all\results"

    # input_file_names = [f for f in os.listdir(input_directory_path) if (
    #     os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

    # for input_file_name in input_file_names:
    #     process_file_tialphatib(input_directory_path, input_file_name,
    #                             output_directory_path)

    # input_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4v-eta-all"
    # output_directory_path = r"Y:\CHESS\ID3A_2022-2\lineouts\ti-tib-4v-eta-all\results"

    # input_file_names = [f for f in os.listdir(input_directory_path) if (
    #     os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

    # for input_file_name in input_file_names:
    #     process_file_tialphatib(input_directory_path, input_file_name,
    #                             output_directory_path)


if __name__ == "__main__":
    main()

# def process_row_nicrc(row):

#     # Function to process each row of data for Ni-CrC samples
#     # # Peaks to fit: CrC (420) CrC (422) Ni (111)/Crc(511), CrC (440) CrC (531) Ni (200)，Ni (220), Ni (311) Ni (222) fitting using Voigt models
#     # centers = [895, 1029, 1130, 1272, 1355, 1390, 2194, 2669, 2812] # He HT AX Specific

#     # # Peaks to fit: CrC (420) CrC (422) Ni (111)/Crc(511), CrC (440) CrC (531) Ni (200)，Ni (220), Ni (311) Ni (222) fitting using Voigt models
#     # centers = [894, 1031, 1136, 1274, 1355, 1395, 2200, 2673, 2820] # He Untreated AX Specific

#     # # Peaks to fit: CrC (420) CrC (422) Ni (111)/Crc(511), CrC (440) CrC (531) Ni (200)，Ni (220), Ni (311) Ni (222) fitting using Voigt models
#     centers = [893, 1030, 1137, 1273, 1356, 1397, 2202,
#                2676, 2819]  # N 600C / N Untreated AX Specific
#     lower_bounds = [856, 990, 1070, 1235, 1325, 1370, 2130, 2625, 2755]
#     upper_bounds = [936, 1070, 1215, 1320, 1370, 1470, 2225, 2715, 2897]

#     # Peaks to fit: CrC (420) CrC (422) Ni (111)/Crc(511), CrC (440) CrC (531) Ni (200)，Ni (220), Ni (311) Ni (222) fitting using Voigt models
#     # centers = [896, 1030, 1140, 1275, 1355, 1394, 2200, 2675, 2819] # N 3-1 900C AX Specific
#     # lower_bounds = [856, 990, 1070, 1235, 1325, 1370, 2130, 2625, 2755]
#     # upper_bounds = [936, 1070, 1215, 1320, 1370, 1470, 2225, 2715, 2897]

#     # # Peaks to fit: CrC (420) CrC (422) Ni (111)/Crc(511), CrC (440) CrC (531) Ni (200)，Ni (220), Ni (311) Ni (222) fitting using Voigt models
#     # centers = [894, 1031, 1131, 1275, 1360, 1392, 2195, 2670, 2812] # He HT TR Specific

#     # # Peaks to fit: CrC (420) CrC (422) Ni (111)/Crc(511), CrC (440) CrC (531) Ni (200)，Ni (220), Ni (311) Ni (222) fitting using Voigt models
#     # centers = [894, 1032, 1137, 1275, 1361, 1399, 2203, 2678, 2821] # He HT TR Specific

#     # # Peaks to fit: CrC (420) CrC (422) Ni (111)/Crc(511), CrC (440) CrC (531) Ni (200)，Ni (220), Ni (311) Ni (222) fitting using Voigt models
#     # centers = [894, 1032, 1134, 1274, 1359, 1395,
#     #            2200, 2674, 2819]  # N 900C TR Specific

#     # # Peaks to fit: CrC (420) CrC (422) Ni (111)/Crc(511), CrC (440) CrC (531) Ni (200)，Ni (220), Ni (311) Ni (222) fitting using Voigt models
#     # centers = [894, 1031, 1137, 1274, 1359, 1398, 2202, 2677, 2821] # N As/600C Specific

#     types = ['VoigtModel', 'VoigtModel', 'VoigtModel', 'VoigtModel', 'VoigtModel',
#              'VoigtModel', 'VoigtModel', 'VoigtModel', 'VoigtModel']
#     heights = [2000, 2000, 200000, 2000, 2000, 20000, 2000, 2000, 2000]
#     sigmas = [1, 1, 1, 1, 1, 1, 1, 1, 1]
#     gammas = [1, 1, 1, 1, 1, 1, 1, 1, 1]
#     center_min = []
#     center_max = []

#     amplitude_row = []
#     center_row = []
#     sigma_row = []
#     gamma_row = []

#     for lower_bound, upper_bound, model_type, center, height, sigma, gamma in list(zip(lower_bounds, upper_bounds, types, centers, heights, sigmas, gammas)):
#         x = np.array(range(lower_bound, upper_bound))
#         y = np.array(row[lower_bound:upper_bound])
#         spec = make_spec_voigt(x, y, model_type, center,
#                                height, sigma, gamma, center_min, center_max)
#         best_values = process_spec(spec)

#         amplitude_row.append(best_values["m0_amplitude"])
#         center_row.append(best_values["m0_center"])
#         sigma_row.append(best_values["m0_sigma"])
#         gamma_row.append(best_values["m0_gamma"])

#     return amplitude_row, center_row, sigma_row, gamma_row

# def process_file_nicrc_moving_nicrc(input_directory_path, input_file_name, output_directory_path, num_steps):
#     # Function to set peaks' locations and other parameters

#     header_row = ["CrC (420)", "CrC (422)", "Ni (111)/Crc(511)", "CrC (440)",
#                   "CrC (531)", "Ni (200)", "Ni (220)", "Ni (311)", "Ni (222)"]
#     amplitude_data = []
#     center_data = []
#     sigma_data = []
#     gamma_data = []

#     input_file_path = os.path.join(input_directory_path, input_file_name)
#     with open(input_file_path, mode="r") as input_file:
#         csv_reader = csv.reader(input_file, delimiter=",")
#         row_count = 0
#         row_start = None
#         row_end = None
#         sum_row = []
#         for row in csv_reader:
#             row_count += 1
#             print("********** Row {} **********".format(row_count))
#             if (row_start is not None and row_count < row_start) or (row_end is not None and row_count > row_end):
#                 print("Skipped")
#                 print()
#                 continue
#             row = [float(x) for x in row]
#             sum_row.append(row)
#             if len(sum_row) > num_steps:
#                 sum_row.pop(0)
#             sum_row_np = np.array(sum_row)
#             avgrow = sum_row_np.mean(axis=0)
#             amplitude_row, center_row, sigma_row, gamma_row = process_row_nicrc(
#                 avgrow)
#             amplitude_data.append(amplitude_row)
#             center_data.append(center_row)
#             sigma_data.append(sigma_row)
#             gamma_data.append(gamma_row)

#     for name, data in list(zip(["amplitude.csv", "center.csv", "sigma.csv", "gamma.csv"], [amplitude_data, center_data, sigma_data, gamma_data])):
#         dot_index = input_file_name.rindex(".")
#         output_file_name = input_file_name[:dot_index] + "_" + name
#         output_file_path = os.path.join(
#             output_directory_path, output_file_name)
#         save_csv(output_file_path, header_row, data)


# def process_file_nicrc(input_directory_path, input_file_name, output_directory_path):
#     # Function to set peaks' locations and other parameters
#     # for use with Ni/CrC

#     header_row = ["CrC (420)", "CrC (422)", "Ni (111)/Crc(511)", "CrC (440)",
#                "CrC (531)", "Ni (200)", "Ni (220)", "Ni (311)", "Ni (222)"]
#     amplitude_data = []
#     center_data = []
#     sigma_data = []
#     gamma_data = []

#     input_file_path = os.path.join(input_directory_path, input_file_name)
#     with open(input_file_path, mode="r") as input_file:
#         csv_reader = csv.reader(input_file, delimiter=",")
#         row_count = 0
#         row_start = None
#         row_end = None
#         for row in csv_reader:
#             row_count += 1
#             print("********** Row {} **********".format(row_count))
#             if (row_start is not None and row_count < row_start) or (row_end is not None and row_count > row_end):
#                 print("Skipped")
#                 print()
#                 continue

#             row = [float(x) for x in row]
#             amplitude_row, center_row, sigma_row, gamma_row = process_row_nicrc(row)
#             amplitude_data.append(amplitude_row)
#             center_data.append(center_row)
#             sigma_data.append(sigma_row)
#             gamma_data.append(gamma_row)

#     for name, data in list(zip(["amplitude.csv", "center.csv", "sigma.csv", "gamma.csv"], [amplitude_data, center_data, sigma_data, gamma_data])):
#         dot_index = input_file_name.rindex(".")
#         output_file_name = input_file_name[:dot_index] + "_" + name
#         output_file_path = os.path.join(
#             output_directory_path, output_file_name)
#         save_csv(output_file_path, header_row, data)

# def main(): ## for 2021-3-1IDC

#     # Location to save images (if selected)
#     image_dir = "C:/Users/helew/Documents/Fitting"

#     # Import Data & set output directory

#     input_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_extract\mapping\He_1-1\0deg"
#     output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\He_1-1"

#     input_file_names = [f for f in os.listdir(input_directory_path) if (
#         os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

#     for input_file_name in input_file_names:
#         process_file_nicrc(input_directory_path, input_file_name,
#                             output_directory_path)

#     input_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_extract\mapping\He_1-1\90deg"
#     output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\He_1-1"

#     input_file_names = [f for f in os.listdir(input_directory_path) if (
#         os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

#     for input_file_name in input_file_names:
#         process_file_nicrc(input_directory_path, input_file_name,
#                             output_directory_path)
#     input_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_extract\mapping\He_1-2\0deg"
#     output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\He_1-2"

#     input_file_names = [f for f in os.listdir(input_directory_path) if (
#         os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

#     for input_file_name in input_file_names:
#         process_file_nicrc(input_directory_path, input_file_name,
#                             output_directory_path)
#     input_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_extract\mapping\He_1-2\90deg"
#     output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\He_1-2"

#     input_file_names = [f for f in os.listdir(input_directory_path) if (
#         os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

#     for input_file_name in input_file_names:
#         process_file_nicrc(input_directory_path, input_file_name,
#                             output_directory_path)
#     input_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_extract\mapping\N_1-2\0deg"
#     output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\N_1-2"

#     input_file_names = [f for f in os.listdir(input_directory_path) if (
#         os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

#     for input_file_name in input_file_names:
#         process_file_nicrc(input_directory_path, input_file_name,
#                             output_directory_path)
#     input_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_extract\mapping\N_1-2\90deg"
#     output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\N_1-2"

#     input_file_names = [f for f in os.listdir(input_directory_path) if (
#         os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

#     for input_file_name in input_file_names:
#         process_file_nicrc(input_directory_path, input_file_name,
#                             output_directory_path)
#     input_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_extract\mapping\N_1-3\0deg"
#     output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\N_1-3"

#     input_file_names = [f for f in os.listdir(input_directory_path) if (
#         os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

#     for input_file_name in input_file_names:
#         process_file_nicrc(input_directory_path, input_file_name,
#                             output_directory_path)
#     input_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_extract\mapping\N_1-3\90deg"
#     output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\N_1-3"

#     input_file_names = [f for f in os.listdir(input_directory_path) if (
#         os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".csv"))]

#     for input_file_name in input_file_names:
#         process_file_nicrc(input_directory_path, input_file_name,
#                             output_directory_path)
