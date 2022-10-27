from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import *

dir_list = ['one_test_4seconds_30.npy','one_train_4seconds_30.npy',
            'two_test_4seconds_2.npy','two_train_4seconds_2.npy']
body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)
use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27]
# data_list=data_list.reshape(-1,120,15,3)
# data_list=data_list[:,:,[0,1,4,7,2,5,8,12,15,16,18,20,17,19,21],:]
# body_edges = np.array(
# [[0,1], [1,2],[2,3],[0,4],
# [4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
# )
angles = np.array(
    [[0,1,2],[1,2,3],[1,0,4],[0,4,5],[4,5,6],[1,0,7],[4,0,7],[0,7,9],[0,7,12],
     [7,9,10],[9,10,11],[7,12,13],[12,13,14],[9,7,8],[8,7,12]]
)

total_length = 0 # total length for all data in frames
angle_list = []
# root_offset = []
for i in range(4):
    dir_base = f'/home1/zhuwentao/projects/MRT/mocap/{dir_list[i]}'
    # print(dir_base)
    poses = np.load(dir_base,allow_pickle=True)
    # print(poses.shape)
    # continue
    poses=poses[:,:,:,use,:]
    poses=poses.reshape(-1,15,3)
    # print('current pose npy shape= ',poses.shape)
    # quit()
    # poses = poses.transpose((1,0,2,3))
    length = poses.shape[0]
    total_length += length
            
    # poses = poses.reshape(-1,15,3)
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
                    eps = 1e-6
                    if 1.0 < cos_ < 1.0 + eps:
                        cos_ = 1.0
                    elif -1.0 - eps < cos_ < -1.0:
                        cos_ = -1.0
                    arccos = np.arccos(cos_)
                    # print(arccos/np.pi)
                    angle_list.append(arccos/np.pi)
            
            
part = plt.violinplot(angle_list,showmedians=True)
filename = './angle_distribution.jpg'
plt.savefig(filename)            
            # quit()
            