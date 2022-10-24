from ossaudiodev import SNDCTL_SEQ_GETINCOUNT
import numpy as np
import os

# currently, we do not use ball position

data = []
dir_base = '/home1/zhuwentao/projects/MRT/output/wusi/'
file = os.listdir(dir_base)
for folder in file:
    p = dir_base + folder + '/'
    sequences = os.listdir(p)
    for sequence in sequences:
        p1 = p + sequence + '/'
        # ball_dir = p1 + 'ball_pos.npy'
        # pose_id_dir = p1 + 'pose_id.npy'
        poses_dir = p1 + 'poses.npy'
        # ball_pos = np.load(ball_dir, allow_pickle=True)
        # pose_id = np.load(pose_id_dir, allow_pickle=True)
        # print(pose_id)
        poses = np.load(poses_dir,allow_pickle=True)
        poses = poses.reshape(-1,5,15,3)
        poses = poses.transpose((1,0,2,3))
        # poses = np.expand_dims(poses, 0)
        # data.append(poses)
        # print(poses.shape)
        length = poses.shape[1]
        # length = int(poses.shape[1]/50) # cut sequences
        for i in range(length):
            if i + 120 >= length:
                break
            # print(i)
            pose = poses[:,i:i+120,:,:]
            # print(pose.shape)
            data.append(pose)
            
dir_base = '/home1/zhuwentao/projects/MRT/output/wusi_nocut/'
file = os.listdir(dir_base)
for folder in file:
    p = dir_base + folder + '/'
    sequences = os.listdir(p)
    for sequence in sequences:
        p1 = p + sequence + '/'
        # ball_dir = p1 + 'ball_pos.npy'
        # pose_id_dir = p1 + 'pose_id.npy'
        poses_dir = p1 + 'poses.npy'
        # ball_pos = np.load(ball_dir, allow_pickle=True)
        # pose_id = np.load(pose_id_dir, allow_pickle=True)
        # print(pose_id)
        poses = np.load(poses_dir,allow_pickle=True)
        poses = poses.reshape(-1,5,15,3)
        poses = poses.transpose((1,0,2,3))
        # poses = np.expand_dims(poses, 0)
        # data.append(poses)
        # print(poses.shape)
        
        # length = int(poses.shape[1]/50) # cut sequences
        length = poses.shape[1]
        for i in range(length):
            if i + 120 >= length:
                break
            # print(i)
            pose = poses[:,i:i+120,:,:]
            # pose = poses[:,(i)*50:(i+1)*50,:,:]
            # print(pose.shape)
            data.append(pose)
data = np.array(data)
print(data.shape)
save_path = 'data1.npy'
np.save(save_path,data)