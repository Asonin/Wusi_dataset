from locale import normalize
from tkinter import OFF
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
angles = np.array(
    [[0,1,2],[1,2,3],[1,0,4],[0,4,5],[4,5,6],[1,0,7],[4,0,7],[0,7,9],[0,7,12],
     [7,9,10],[9,10,11],[7,12,13],[12,13,14],[9,7,8],[8,7,12]]
)
use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27]

total_length = 0 # total length for all data in frames

root_offset = []
root_offset_0 = []
hands_offset= []


for i in range(4):
    dir_base = f'/home1/zhuwentao/projects/MRT/mocap/{dir_list[i]}'
    # print(dir_base)
    poses = np.load(dir_base,allow_pickle=True)
    print(poses.shape)
    # continue
    poses=poses[:,:,:,use,:]
    poses = poses.transpose((1,0,2,3,4))
    print(poses.shape)
    # continue
    if i<2:
        poses=poses.reshape(1,-1,15,3)
    else:
        poses = poses.reshape(2,-1,15,3)
    # poses=poses.reshape(-1,15,3)
    print('current pose npy shape= ',poses.shape)
    # continue
    for j in range(poses.shape[0]): # every person
                xyz = poses[j]
                # print(xyz.shape)
                root_pos = xyz[:,0]
                # hands aligned
                xyz[:,:,0] = xyz[:,:,0] - np.mean(xyz[:,:,0])
                xyz[:,:,2] = xyz[:,:,2] - np.mean(xyz[:,:,2])
                hands_pos1 = xyz[:,11]
                # hands_pos1[:,0] = hands_pos1[:,0] - np.mean(hands_pos1[:,0])
                # hands_pos1[:,2] = hands_pos1[:,2] - np.mean(hands_pos1[:,2])
                hands_pos2 = xyz[:,14]
                # print(root_pos.shape)
                # quit()
                for k in range(1,root_pos.shape[0]):
                    # root_off = root_pos[k] - root_pos[k-1]
                    root_off= np.sqrt(np.sum((root_pos[k] - root_pos[k-1])**2))
                    root_off_0 = np.sqrt(np.sum((root_pos[k] - root_pos[0])**2))
                    # print(root_off)
                    # print(root_off_0)
                    root_offset.append(root_off)
                    root_offset_0.append(root_off_0)
                    
                    hands1_off= np.sqrt(np.sum((hands_pos1[k] - hands_pos1[k-1])**2))
                    hands2_off = np.sqrt(np.sum((hands_pos2[k] - hands_pos2[0])**2))
                    hands_offset.append(hands1_off)
                    hands_offset.append(hands2_off)
                    # quit()
                # for k in range(poses.shape[1]): # every frame
                #     xyz = poses[j][k]
                #     print(xyz.shape)
                #     quit()
                    
            
            
part = plt.violinplot(root_offset,showmedians=True)
filename = './root_offset_mocap.jpg'
plt.savefig(filename)         
plt.close()
part = plt.violinplot(root_offset_0,showmedians=True)
filename = './root_offset_0_mocap.jpg'
plt.savefig(filename)         
plt.close()
part = plt.violinplot(hands_offset,showmedians=True)
filename = './hands_velocity_aligned_mocap.jpg'
plt.savefig(filename)         
plt.close()
            # quit()
            