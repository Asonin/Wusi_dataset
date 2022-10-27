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
angles = np.array(
    [[0,1,2],[1,2,3],[1,0,4],[0,4,5],[4,5,6],[1,0,7],[4,0,7],[0,7,9],[0,7,12],
     [7,9,10],[9,10,11],[7,12,13],[12,13,14],[9,7,8],[8,7,12]]
)

total_length = 0 # total length for all data in frames
angle_list = []
# root_offset = []
for i in range(2):
    dir_base = f'/home1/zhuwentao/projects/MRT/data_wusi/{dir_list[i]}/'
    # print(dir_base)
    file = os.listdir(dir_base)
    for folder in file:
        p = dir_base + folder + '/'
        sequences = os.listdir(p)
        for sequence in sequences:
            p1 = p + sequence + '/'
            poses_dir = p1 + 'poses.npy'
            poses = np.load(poses_dir,allow_pickle=True)
            poses = poses.reshape(-1,5,15,3)
            # print('current pose npy shape= ',poses.shape)
            poses = poses.transpose((1,0,2,3))
            poses = poses[:,:,[2,6,7,8,12,13,14,1,0,3,4,5,9,10,11],:]
            length = poses.shape[1]
            # print('current length = ',poses.shape[1])
            # calculate total length
            total_length += length
            
            poses = poses.reshape(-1,15,3)
            # print(poses.shape)
            for j in range(poses.shape[0]): # every person/frame
                xyz = poses[j]
                # print(xyz.shape)
                # quit()
                for angle in angles:
                    # print(angle)
                    # quit()
                    a = np.array(xyz[angle[0]] - xyz[angle[1]])
                    # print(a.shape)
                    # print(a)
                    # quit()
                    b = np.array(xyz[angle[2]] - xyz[angle[1]])
                    
                    a /= np.linalg.norm(a)
                    b /= np.linalg.norm(b)
                    # 夹角cos值
                    cos_ = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
                    
                    arccos = np.arccos(cos_)
                    angle_list.append(arccos/np.pi)
            
            
part = plt.violinplot(angle_list,showmedians=True)
filename = './angle_distribution.jpg'
plt.savefig(filename)            
            # quit()
            