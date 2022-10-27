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
angles = np.array(
    [[0,1,2],[1,2,3],[1,0,4],[0,4,5],[4,5,6],[1,0,7],[4,0,7],[0,7,9],[0,7,12],
     [7,9,10],[9,10,11],[7,12,13],[12,13,14],[9,7,8],[8,7,12]]
)
use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27]
hands_angles = np.array(
    [[11,10,7,9],[14,13,7,12]]
)
shoulder_angles = np.array(
    [[10,9,7],[13,12,7]]
)
foot_angles = np.array(
    [[3,2,0,1],[6,5,0,4]]
)
leg_angles = np.array(
    [[2,1,0],[5,4,0]]
)
total_length = 0 # total length for all data in frames
hands_angle_list = []
shoulder_angle_list = []
foot_angle_list = []
leg_angle_list = []
# root_offset = []
for i in range(4):
    dir_base = f'/home1/zhuwentao/projects/MRT/mocap/{dir_list[i]}'
    poses = np.load(dir_base,allow_pickle=True)
    # print(poses.shape)
    # continue
    poses=poses[:,:,:,use,:]
    poses=poses.reshape(-1,15,3)
    # print('current pose npy shape= ',poses.shape)
    # quit()
    # poses = poses.transpose((1,0,2,3))
    length = poses.shape[0]
    # print(poses.shape)
    # quit()
    for j in range(poses.shape[0]): # every person/frame
                # for k in range(poses.shape[1]):
                    xyz = poses[j]
                    for angle in hands_angles:
                        a = np.array(xyz[angle[0]] - xyz[angle[1]])
                        b = np.array(xyz[angle[2]] - xyz[angle[3]])
                        a /= np.linalg.norm(a)
                        b /= np.linalg.norm(b)
                        # 夹角cos值
                        cos_ = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
                        arccos = np.arccos(cos_)
                        hands_angle_list.append(arccos/np.pi)
                    for angle in shoulder_angles:
                        a = np.array(xyz[angle[0]] - xyz[angle[1]])
                        b = np.array(xyz[angle[2]] - xyz[angle[1]])
                        a /= np.linalg.norm(a)
                        b /= np.linalg.norm(b)
                        # 夹角cos值
                        cos_ = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
                        arccos = np.arccos(cos_)
                        shoulder_angle_list.append(arccos/np.pi)
                    for angle in foot_angles:
                        a = np.array(xyz[angle[0]] - xyz[angle[1]])
                        b = np.array(xyz[angle[2]] - xyz[angle[3]])
                        a /= np.linalg.norm(a)
                        b /= np.linalg.norm(b)
                        # 夹角cos值
                        cos_ = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
                        arccos = np.arccos(cos_)
                        foot_angle_list.append(arccos/np.pi)
                    for angle in leg_angles:
                        a = np.array(xyz[angle[0]] - xyz[angle[1]])
                        b = np.array(xyz[angle[2]] - xyz[angle[1]])
                        a /= np.linalg.norm(a)
                        b /= np.linalg.norm(b)
                        # 夹角cos值
                        cos_ = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
                        arccos = np.arccos(cos_)
                        leg_angle_list.append(arccos/np.pi)
                    # print(xyz.shape)
                    # quit()
                
part = plt.violinplot(hands_angle_list,showmedians=True)
filename = './hands_distribution_mocap.jpg'
plt.savefig(filename)            
plt.close()
part = plt.violinplot(shoulder_angle_list,showmedians=True)
filename = './shoulder_distribution_mocap.jpg'
plt.savefig(filename)            
plt.close()
part = plt.violinplot(foot_angle_list,showmedians=True)
filename = './foot_distribution_mocap.jpg'
plt.savefig(filename)            
plt.close()
part = plt.violinplot(leg_angle_list,showmedians=True)
filename = './leg_distribution.jpg'
plt.savefig(filename)            
plt.close()
            # quit()
            