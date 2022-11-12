from distutils import dist
from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import *

dir_list = ['wusi','wusi_nocut']
body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)

hand_list = []
var_list = []
for i in range(2):
    dir_base = f'/home1/zhuwentao/projects/MRT/data_wusi/{dir_list[i]}/'
    # print(dir_base)
    file = os.listdir(dir_base)
    for folder in file:
        p = dir_base + folder + '/'
        sequences = os.listdir(p)
        print(folder)
        for sequence in sequences:
            # print(sequence)
            clip = []
            p1 = p + sequence + '/'
            poses_dir = p1 + 'poses.npy'
            poses = np.load(poses_dir,allow_pickle=True)
            poses = poses.reshape(-1,5,15,3) # sequence_len * 5peole * 15 joints * xyz
            # print('current pose npy shape= ',poses.shape)
            poses = poses.transpose((1,0,2,3))
            poses = poses[:,:,[2,6,7,8,12,13,14,1,0,3,4,5,9,10,11],:]
            length = poses.shape[1]
            # print('current length = ',poses.shape[1])
            
            poses = poses.reshape(-1,15,3)
            # print(poses.shape)
            for j in range(poses.shape[0]): # every person/frame
                xyz = poses[j]
                lhand = xyz[11]
                rhand = xyz[14]
                root = xyz[0]
                ldist = np.sqrt(np.sum((lhand - root)**2))/1000
                rdist = np.sqrt(np.sum((rhand - root)**2))/1000
                # print(ldist)
                hand_list.append(ldist)
                hand_list.append(rdist)
                clip.append(ldist)
                clip.append(rdist)
                
            var = np.var(clip)
            # print(var)
            var_list.append(var)
            # print(var)
            # print(xyz.shape)
                # quit()
            
print('into saving')
intensity_dir = './intensity'
if not os.path.exists(intensity_dir):
    os.mkdir(intensity_dir)
    
hand = plt.violinplot(hand_list,showmedians=True)
plt.ylabel('meters',fontsize=11)
filename = intensity_dir + '/hand.eps'
plt.savefig(filename,dpi=600)            
plt.close()

var = plt.violinplot(var_list,showmedians=True)
# plt.ylabel('meters',fontsize=11)
filename = intensity_dir + '/variance_hands_offset.eps'
plt.savefig(filename,dpi=600)            
plt.close()
quit() 
