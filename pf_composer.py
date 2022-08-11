import os
import csv
from statistics import mean, stdev
import pandas as pd
import numpy as np
from numpy import append as np_append


def save_csv(csv_file_path, headers, data):
    with open(csv_file_path, "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(headers)
        csv_writer.writerows(data)


input_path = r"Y:\CHESS\ID3A_2022-2\ti-tib-1-eta-090\results\center"
only_csv_files = [f for f in os.listdir(input_path) if
                  (os.path.isfile(os.path.join(input_path, f)) and f.endswith(".csv"))]
only_csv_files.sort()
k = np.size(only_csv_files)
# print(only_csv_files)

headers = []
avg = np.empty([36,6])
standard_deviation = np.empty([36,6])
adj_std = []
adj_mean = []
adj_avg = []
adj_stdev = []
norm_avg = []
output = np.empty([k,6])

i = 0
for csv_file in only_csv_files:
    df = pd.read_csv(os.path.join(input_path, csv_file), header=0)
    x = df.iloc[0].to_numpy()
    for j in range(0, len(x)):
        output[i,j]  = x[j]
    i += 1

for j in range (0,36):
    for i in range (0,6):
        lower_bound = int(89 * j)
        upper_bound = int(89 * (j + 1))
        calc_element = output[lower_bound:upper_bound,i]
        avg[j,i] = mean(calc_element)
        standard_deviation[j,i] = stdev(calc_element)

print(avg)
print(standard_deviation)

if headers == []:
    headers = list(df.columns)
# avg.append(list(mean))
# stdev.append(list(std))

# # adj_mean = mean * (10.2665586536948 - 1.57211584737215) / \
# #     2999 + 1.57211584737215
# # adj_std = std * (10.2665586536948 - 1.57211584737215) / 2999
# adj_avg.append(list(adj_mean))
# adj_stdev.append(list(adj_std))

output_path = os.path.join(input_path, "output")
if not os.path.exists(output_path):
    os.mkdir(output_path)


save_csv(os.path.join(output_path, "avg.csv"), headers, avg)
save_csv(os.path.join(output_path, "std.csv"), headers, standard_deviation) 
# save_csv(os.path.join(output_path, "adjavg.csv"), headers, adj_avg)
# save_csv(os.path.join(output_path, "adjstd.csv"), headers, adj_stdev)