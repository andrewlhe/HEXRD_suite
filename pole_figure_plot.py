import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmap

from hexrd.xrd import transforms_CAPI as xfcapi

d2r=np.pi/180.
r2d=180./np.pi

num_frames=60

#data = np.load('/nfs/chess/aux/user/lh644/ff_processing/cs-h-l1-s2_pole.npz')
data = np.load('/nfs/chess/aux/user/lh644/ff_processing/cs-n-l1-s2_pole2.npz')
strain_mat = data['strain_mat'] 
tth_mat = data['tth_mat']
eta_mat = data['eta_mat']
ome_mat = data['ome_mat']
width_mat = data['width_mat']
i_mat = data['i_mat']
          
#%% Use for Pole Figures

hkl_nums_use=np.arange(0,20) #note these are numbers in the unmasked portion of the plane_data list

#%% Pole Figure Bin Configuration (Top Detector)

num_azi_bins=30
azi_bin_size=180./num_azi_bins
eta_vals=np.arange(0,180*1.001,180/num_azi_bins)

 #%% Listify data for pole figures, this is dumb

scan_no=0

tth_list=np.zeros([num_frames*num_azi_bins,len(hkl_nums_use)])
eta_list=np.zeros([num_frames*num_azi_bins,len(hkl_nums_use)])
ome_list=np.zeros([num_frames*num_azi_bins,len(hkl_nums_use)])
i_list=np.zeros([num_frames*num_azi_bins,len(hkl_nums_use)])

strain_list=np.zeros([num_frames*num_azi_bins,len(hkl_nums_use)])

for ii in np.arange(len(hkl_nums_use)):#note the conversion to degrees
    counter=0
    for kk in np.arange(num_frames):
    

        for jj in np.arange(num_azi_bins):
            if np.isnan(tth_mat[scan_no,kk,ii,jj]):
                tth_list[counter,ii]=0.
            else:
                tth_list[counter,ii]=tth_mat[scan_no,kk,ii,jj]
                eta_list[counter,ii]=eta_mat[scan_no,kk,ii,jj]
                ome_list[counter,ii]=ome_mat[scan_no,kk,ii,jj]
                i_list[counter,ii]=i_mat[scan_no,kk,ii,jj]
                strain_list[counter,ii]=strain_mat[scan_no,kk,ii,jj]
                counter+=1
            
            
            
#% Transform diffraction angles into directions in the sample coordinate system
n_s=np.zeros([tth_list.shape[0],3,len(hkl_nums_use)])

for hkl_num in np.arange(len(hkl_nums_use)):

    angs=np.array([np.radians(tth_list[:,hkl_num]),np.radians(eta_list[:,hkl_num]),np.radians(ome_list[:,hkl_num])])
    n_s[:,:,hkl_num] = xfcapi.anglesToGVec(angs.T)


#%% Plot a pole figure
output_folder='/nfs/chess/aux/user/lh644/ff_processing/200715_pole_figures_trimmed/'
file_header='cs-n-l1-s2'
cur_hkl=16

plt.close('all')

norm_i_list=np.copy(i_list)

for ii in np.arange(len(hkl_nums_use)):
    norm_i_list[:,ii]=i_list[:,ii]/np.max(i_list[:,ii])

cm=plt.get_cmap('jet')

cnorm=colors.Normalize(vmin=0., vmax=1., clip=False)
scalarMap_i = cmap.ScalarMappable(norm=cnorm,cmap=cm)
scalarMap_i.set_array([0.,1.])


color_val = scalarMap_i.to_rgba(norm_i_list[:,cur_hkl])

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

for ii in range(len(n_s)):
    #print(ii)
    ax.scatter(n_s[ii,2,cur_hkl],n_s[ii,0,cur_hkl],n_s[ii,1,cur_hkl], c=color_val[ii,:],marker='o',s=50,edgecolor='none')


ax.scatter(0.,0.,0.,c=[0.8,0.8,0.8,1.0],marker='o',s=10000,edgecolor='none')

plt.colorbar(scalarMap_i)
ax.set_aspect('equal')

ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')    
    
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])

plt.savefig(output_folder + '%s_layer_1_ipf_load_%0.02d_hkl_%0.02d.png' %(file_header,scan_no,cur_hkl), bbox_inches='tight')

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

for ii in range(len(n_s)):
    #print(ii)
    ax.scatter(n_s[ii,2,cur_hkl],n_s[ii,0,cur_hkl],n_s[ii,1,cur_hkl], c=color_val[ii,:],marker='o',s=50,edgecolor='none')


ax.scatter(0.,0.,0.,c=[0.8,0.8,0.8,1.0],marker='o',s=10000,edgecolor='none')



plt.colorbar(scalarMap_i)
ax.set_aspect('equal')

ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')    
    
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])

ax.view_init(elev=90., azim=0.)


plt.savefig(output_folder + '%s_layer_1_ipf_top_load_%0.02d_hkl_%0.02d.png' %(file_header,scan_no,cur_hkl), bbox_inches='tight')

#% Plot Strain Pole Figures

min_val=-2.5e-3
max_val=2.5e-3
cm=plt.get_cmap('bwr')
cnorm=colors.Normalize(vmin=min_val, vmax=max_val, clip=False)
scalarMap_s = cmap.ScalarMappable(norm=cnorm,cmap=cm)
scalarMap_s.set_array([min_val,max_val])


color_val = scalarMap_s.to_rgba(strain_list[:,cur_hkl])

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

for ii in range(len(n_s)):
    #print(ii)
    ax.scatter(n_s[ii,2,cur_hkl],n_s[ii,0,cur_hkl],n_s[ii,1,cur_hkl], c=color_val[ii,:],marker='o',s=50,edgecolor='none')


ax.scatter(0.,0.,0.,c=[0.8,0.8,0.8,1.0],marker='o',s=10000,edgecolor='none')



plt.colorbar(scalarMap_s)
ax.set_aspect('equal')

ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')    
    
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])

plt.savefig(output_folder + '%s_layer_1_spf_load_%0.02d_hkl_%0.02d.png' %(file_header,scan_no,cur_hkl), bbox_inches='tight')

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

for ii in range(len(n_s)):
    #print(ii)
    ax.scatter(n_s[ii,2,cur_hkl],n_s[ii,0,cur_hkl],n_s[ii,1,cur_hkl], c=color_val[ii,:],marker='o',s=50,edgecolor='none')


ax.scatter(0.,0.,0.,c=[0.8,0.8,0.8,1.0],marker='o',s=10000,edgecolor='none')



plt.colorbar(scalarMap_s)
ax.set_aspect('equal')

ax.set_xlabel('z')
ax.set_ylabel('x')
ax.set_zlabel('y')    
    
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_zlim([-1.5,1.5])

ax.view_init(elev=90., azim=0.)

plt.savefig(output_folder + '%s_layer_1_spf_top_load_%0.02d_hkl_%0.02d.png' %(file_header,scan_no,cur_hkl), bbox_inches='tight')   