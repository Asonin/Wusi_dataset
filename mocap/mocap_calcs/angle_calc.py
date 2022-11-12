from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import *
import seaborn as sns

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
    [[1,2,3],[4,5,6],[2,1,0],[0,4,5],[1,0,4],[8,7,0],[10,9,7],[7,12,13],[9,10,11],[12,13,14]]
    
    # [[0,1,2],[1,2,3],[1,0,4],[0,4,5],[4,5,6],[1,0,7],[4,0,7],[0,7,9],[0,7,12],
    #  [7,9,10],[9,10,11],[7,12,13],[12,13,14],[9,7,8],[8,7,12]]
)

knees = np.array([[1,2,3],[4,5,6]])
thighs = np.array([[2,1,0],[0,4,5]])
crotch = np.array([[1,0,4]])
head = np.array([[8,7,0]])
shoulder = np.array([[10,9,7],[7,12,13]])
elbow = np.array([[9,10,11],[12,13,14]])

total_length = 0 # total length for all data in frames
angle_list = []
knee_list = []
thigh_list = []
crotch_list = []
head_list = []
shoulder_list = []
elbow_list = []
waist_list = []

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
                # if j > 100:
                #     break
                xyz = poses[j]
                # print(xyz.shape)
                # quit()
                a = np.array(xyz[1] - xyz[0])
                b = np.array(xyz[4] - xyz[0])
                n = np.cross(a,b)
                c = np.array(xyz[7] - xyz[0])
                n /= np.linalg.norm(n)
                c /= np.linalg.norm(c)
                cos_ = np.dot(n,c)/(np.linalg.norm(n)*np.linalg.norm(c))
                arccos = np.arccos(cos_)
                ang = arccos/np.pi
                waist_list.append(ang)
                for i, angle in enumerate(angles):
                    
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
                    ang = arccos / np.pi
                    # print(arccos/np.pi)
                    if i < 2:
                        knee_list.append(ang)
                        # print(np.array(knee_list).shape)
                    elif i < 4:
                        thigh_list.append(ang)
                    elif i < 5:
                        crotch_list.append(ang)
                    elif i < 6:
                        head_list.append(ang)
                    elif i < 8:
                        shoulder_list.append(ang)
                    else:
                        elbow_list.append(ang)
                    # angle_list.append(arccos/np.pi)
            
            
print('total_length = ',total_length)

distribution_dir = './mocap_distribution'
if not os.path.exists(distribution_dir):
    os.mkdir(distribution_dir)

# knee = plt.violinplot(knee_list,showmedians=True)
knee = sns.violinplot(y=knee_list,color='orange')
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/knee_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
# quit()
thigh = sns.violinplot(y=thigh_list,color='orange')
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/thigh_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
crotch = sns.violinplot(y=crotch_list,color='orange')
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/crotch_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
head = sns.violinplot(y=head_list,color='orange')
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/head_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
shoulder = sns.violinplot(y=shoulder_list,color='orange')
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/shoulder_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
elbow = sns.violinplot(y=elbow_list,color='orange')
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/elbow_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
waist = sns.violinplot(y=waist_list,color='orange')
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/waist_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
            # quit()
            