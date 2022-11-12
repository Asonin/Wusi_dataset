from distutils import dist
from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import *
import seaborn as sns
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
knees = np.array([[1,2,3],[4,5,6]])
thighs = np.array([[2,1,0],[0,4,5]])
crotch = np.array([[1,0,4]])
head = np.array([[8,7,0]])
shoulder = np.array([[10,9,7],[7,12,13]])
elbow = np.array([[9,10,11],[12,13,14]])

# waist = np.array([[7],[1,0,4]]) # platform waist


total_length = 0 # total length for all data in frames
angle_list = []
knee_list = []
thigh_list = []
crotch_list = []
head_list = []
shoulder_list = []
elbow_list = []
waist_list = []
seq_cnt = 0
seq_max = 0
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
            
            seq_cnt += 1
            p1 = p + sequence + '/'
            poses_dir = p1 + 'poses.npy'
            poses = np.load(poses_dir,allow_pickle=True)
            poses = poses.reshape(-1,5,15,3) # sequence_len * 5peole * 15 joints * xyz
            seq_len = poses.shape[0]
            # if seq_len > 100:
            #     seq_max += 1
            # continue
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
                    # print(i)
                    # print(angle)
                    # quit()
                    a = np.array(xyz[angle[0]] - xyz[angle[1]])
                    b = np.array(xyz[angle[2]] - xyz[angle[1]])
                    a /= np.linalg.norm(a)
                    b /= np.linalg.norm(b)
                    # 夹角cos值
                    cos_ = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
                    arccos = np.arccos(cos_)
                    ang = arccos/np.pi
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
# quit()
print('total_length = ',total_length)
print(seq_cnt)
print(seq_max)
# quit()
distribution_dir = './distribution'
if not os.path.exists(distribution_dir):
    os.mkdir(distribution_dir)

knee = sns.violinplot(y=knee_list,showmedians=True)
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/knee_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
thigh = sns.violinplot(y=thigh_list,showmedians=True)
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/thigh_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
crotch = plt.violinplot(crotch_list,showmedians=True)
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/crotch_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
head = plt.violinplot(head_list,showmedians=True)
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/head_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
shoulder = plt.violinplot(shoulder_list,showmedians=True)
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/shoulder_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
elbow = plt.violinplot(elbow_list,showmedians=True)
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/elbow_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
waist = plt.violinplot(waist_list,showmedians=True)
plt.ylabel('pi',fontsize=11)
filename = distribution_dir + '/waist_distribution.eps'
plt.savefig(filename,dpi=600)            
plt.close()
            # quit()
            