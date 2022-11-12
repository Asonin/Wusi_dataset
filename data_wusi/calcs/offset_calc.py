from distutils import dist
from locale import normalize
from xxlimited import foo
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
    [[1,2,3],[4,5,6],[2,1,0],[0,4,5],[1,0,4],[8,7,0],[10,9,7],[7,12,13],[9,10,11],[12,13,14]]
    # [[0,1,2],[1,2,3],[1,0,4],[0,4,5],[4,5,6],[1,0,7],[4,0,7],[0,7,9],[0,7,12],
    #  [7,9,10],[9,10,11],[7,12,13],[12,13,14],[9,7,8],[8,7,12]]
)

# change from all joints to calc per joint
# knees = np.array([[1,2,3],[4,5,6]])
# thighs = np.array([[2,1,0],[0,4,5]])
# crotch = np.array([[1,0,4]])
# head = np.array([[8,7,0]])
# shoulder = np.array([[10,9,7],[7,12,13]])
# elbow = np.array([[9,10,11],[12,13,14]])

# waist = np.array([[7],[1,0,4]]) # platform waist


foot_list = []
hand_list = []

# thigh_list = []
# crotch_list = []
# head_list = []
# shoulder_list = []
# elbow_list = []
# waist_list = []

# root_offset = []
for i in range(2):
    dir_base = f'/home1/zhuwentao/projects/MRT/data_wusi/{dir_list[i]}/'
    # print(dir_base)
    file = os.listdir(dir_base)
    for folder in file:
        print(folder)
        p = dir_base + folder + '/'
        sequences = os.listdir(p)
        for sequence in sequences:
            p1 = p + sequence + '/'
            poses_dir = p1 + 'poses.npy'
            poses = np.load(poses_dir,allow_pickle=True)
            poses = poses.reshape(-1,5,15,3) # sequence_len * 5peole * 15 joints * xyz
            # print('current pose npy shape= ',poses.shape)
            poses = poses.transpose((1,0,2,3))
            poses = poses[:,:,[2,6,7,8,12,13,14,1,0,3,4,5,9,10,11],:]
            length = poses.shape[1]
            # print('current length = ',poses.shape[1])
            # calculate total length
            # print(poses.shape)
            # quit()
            # poses = poses.reshape(-1,15,3)
            # print(poses.shape)
            for j in range(poses.shape[0]): # every frame
                if j > 2:
                    break
                xyz = poses[j]
                # print(xyz.shape)
                # quit()
                for k in range(xyz.shape[0]):
                    for l in range(k+1,xyz.shape[0]):
                        xyz1 = xyz[k]
                        xyz2 = xyz[l]
                        # print(xyz1.shape)
                        # quit()
                        lfoot_off= np.sqrt(np.sum((xyz1[3] - xyz2[3])**2))*25/(15*1000)
                        rfoot_off= np.sqrt(np.sum((xyz1[6] - xyz2[6])**2))*25/(15*1000)
                        lhand_off= np.sqrt(np.sum((xyz1[11] - xyz2[11])**2))*25/(15*1000)
                        rhand_off= np.sqrt(np.sum((xyz1[14] - xyz2[14])**2))*25/(15*1000)
                        
                        # root_off= np.sqrt(np.sum((xyz1[0] - xyz2[0])**2))
                        # root_off_0 = np.sqrt(np.sum((root_pos[k] - root_pos[0])**2))
                        # print(root_off)
                        # print(root_off_0)
                        foot_list.append(lfoot_off)
                        foot_list.append(rfoot_off)
                        hand_list.append(lhand_off)
                        hand_list.append(rhand_off)
                        # root_offset.append(root_off)
                        # root_offset_0.append(root_off_0)
                    # angle_list.append(arccos/np.pi)
# quit()

distribution_dir = './offset'
if not os.path.exists(distribution_dir):
    os.mkdir(distribution_dir)


# hand_offset = plt.violinplot(hand_list,showmedians=True)
# plt.ylabel('m/s',fontsize=11)
# filename = distribution_dir + '/hand_velocity.eps'
# plt.savefig(filename,dpi=600)            
# plt.close()
# foot_list = plt.violinplot(foot_list,showmedians=True)
# plt.ylabel('m/s',fontsize=11)
# filename = distribution_dir + '/foot_velocity.eps'
# plt.savefig(filename,dpi=600)            
# plt.close()

hand_list = np.array(hand_list)
acch = (hand_list[1:] - hand_list[:-1])*25/15
foot_list = np.array(foot_list)
print(foot_list.shape)
accf = (foot_list[1:] - foot_list[:-1])*25/15

hand_acc = plt.violinplot(acch,showmedians=True)
plt.ylabel('m/s^2',fontsize=11)
filename = distribution_dir + '/hand_acc.eps'
plt.savefig(filename,dpi=600)            
plt.close()
foot_list = plt.violinplot(accf,showmedians=True)
plt.ylabel('m/s^2',fontsize=11)
filename = distribution_dir + '/foot_acc.eps'
plt.savefig(filename,dpi=600)            
plt.close()