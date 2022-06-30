import numpy as np
import matplotlib.pyplot as plt

data = np.load('Y:/CHESS/ID3A_2021-2/results_40bin/45deg/npz/He-2-ff0-45-attempt1.npz')
# strain_mat=np.zeros([num_scans,num_frames,num_hkls,num_azi_bins])
str_mat = data['strain_mat']
twotheta_mat = data['tth_mat']
eta_mat = data['eta_mat']
ome_mat = data['ome_mat']
width_mat = data['width_mat']
i_mat = data['i_mat']

scans = np.size(str_mat, 0)
frame = np.size(str_mat, 1)
hlk = np.size(str_mat, 2)
bins = np.size(str_mat, 3)
print(hlk)

mean_strain = np.zeros((scans, hlk))
std_strain = np.zeros((scans, hlk))
mean_i = np.zeros((scans, hlk))
std_i = np.zeros((scans, hlk))
mean_twotheta = np.zeros((scans, hlk))
std_twotheta = np.zeros((scans, hlk))
mean_width = np.zeros((scans, hlk))
std_width = np.zeros((scans, hlk))


for i in range(0, scans):
    for j in range(0, hlk):
        mean_strain[i, j] = np.mean([str_mat[i, x, j, 0] for x in range(40, 1445, 40)])
        std_strain[i, j] = np.std([str_mat[i, x, j, 0] for x in range(40, 1445, 40)])
        mean_i[i, j] = np.mean([i_mat[i, x, j, 0] for x in range(40, 1445, 40)])
        std_i[i,j] = np.std([i_mat[i, x, j, 0] for x in range(40, 1445, 40)])
        mean_twotheta[i, j] = np.mean([twotheta_mat[i, x, j, 0] for x in range(40, 1445, 40)])
        std_twotheta[i,j]=np.std([twotheta_mat[i, x, j, 0] for x in range(40, 1445, 40)])
        mean_width[i, j] = np.mean([width_mat[i, x, j, 0] for x in range(40, 1445, 40)])
        std_width[i, j] = np.std([width_mat[i, x, j, 0] for x in range(40, 1445, 40)])

# np.savetxt('Y:/CHESS/ID3A_2021-2/results_40bin/45deg/npz/He-2-ff0-45-attempt1-mean_strain.csv',
#            mean_strain, delimiter=',', fmt='%f')
# np.savetxt('Y:/CHESS/ID3A_2021-2/results_40bin/45deg/npz/He-2-ff0-45-attempt1-std_strain.csv',
#            std_strain, delimiter=',', fmt='%f')
np.savetxt('Y:/CHESS/ID3A_2021-2/results_40bin/45deg/npz/He-2-ff0-45-attempt1-mean_intensity.csv',
           mean_i, delimiter=',', fmt='%f')
np.savetxt('Y:/CHESS/ID3A_2021-2/results_40bin/45deg/npz/He-2-ff0-45-attempt1-std_intensity.csv',
           std_i, delimiter=',', fmt='%f')
np.savetxt('Y:/CHESS/ID3A_2021-2/results_40bin/45deg/npz/He-2-ff0-45-attempt1-mean_2theta.csv',
           mean_twotheta, delimiter=',', fmt='%f')
np.savetxt('Y:/CHESS/ID3A_2021-2/results_40bin/45deg/npz/He-2-ff0-45-attempt1-std_2theta.csv',
           std_twotheta, delimiter=',', fmt='%f')
np.savetxt('Y:/CHESS/ID3A_2021-2/results_40bin/45deg/npz/He-2-ff0-45-attempt1-mean_width.csv',
           mean_width, delimiter=',', fmt='%f')
np.savetxt('Y:/CHESS/ID3A_2021-2/results_40bin/45deg/npz/He-2-ff0-45-attempt1-std_width.csv',
           std_width, delimiter=',', fmt='%f')
