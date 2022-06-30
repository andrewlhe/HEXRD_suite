import os
import csv
from statistics import mean
import pandas as pd
import numpy as np


def save_csv(csv_file_path, headers, data):
    with open(csv_file_path, "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(headers)
        csv_writer.writerows(data)


input_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\pf\N_3-2\90deg_sigma"
only_csv_files = [f for f in os.listdir(input_path) if
                  (os.path.isfile(os.path.join(input_path, f)) and f.endswith(".csv"))]
only_csv_files.sort()
# print(only_csv_files)

headers = []
avg = []
stdev = []
adj_std = []
adj_mean = []
adj_avg = []
adj_stdev = []
norm_avg = []

for csv_file in only_csv_files:
    df = pd.read_csv(os.path.join(input_path, csv_file), header=0)
    mean = df.mean()
    # adj_mean = mean * (10.2665586536948 - 1.57211584737215) / \
    #     2999 + 1.57211584737215
    std = df.std()
    # adj_std = std * (10.2665586536948 - 1.57211584737215) / 2999
    if headers == []:
        headers = list(df.columns)
    avg.append(list(mean))
    stdev.append(list(std))
    # adj_avg.append(list(adj_mean))
    # adj_stdev.append(list(adj_std))


output_path = os.path.join(input_path, "output")
if not os.path.exists(output_path):
    os.mkdir(output_path)


save_csv(os.path.join(output_path, "avg.csv"), headers, avg)
save_csv(os.path.join(output_path, "std.csv"), headers, stdev)
# save_csv(os.path.join(output_path, "adjavg.csv"), headers, adj_avg)
# save_csv(os.path.join(output_path, "adjstd.csv"), headers, adj_stdev)