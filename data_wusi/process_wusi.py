from operator import length_hint
import numpy as np
import os
import argparse

dir_list = ['wusi','wusi_nocut']

def process_data(stride, sequence_len):
    # currently, we do not use ball position
    data = []
    # cnt_50 = 0
    # cnt_100 = 0
    # cnt_0 = 0
    # length_calc = 0
    # cnt = 0
    for i in range(2):
        dir_base = f'/home1/zhuwentao/projects/MRT/data_wusi/{dir_list[i]}/'
        print(dir_base)
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
                poses = np.load(poses_dir,allow_pickle=True)
                poses = poses.reshape(-1,5,15,3)
                # print(poses.shape)
                # cnt += 1
                # if poses.shape[0] >= 50 and poses.shape[0] < 100:
                #     cnt_50 += 1
                #     length_calc += poses.shape[0]
                # else:
                #     cnt_0 += 1
                # if poses.shape[0] >= 100:
                #     cnt_100 += 1
                # quit()
                poses = poses.transpose((1,0,2,3))
                # poses = np.expand_dims(poses, 0)
                # data.append(poses)
                # print(poses[0][0])
                # [2,6,7,8,12,13,14,1,0,3,4,5,9,10,11]
                # swaped = poses[:, [2,6,7,8,12,13,14,1,0,3,4,5,9,10,11]]
                poses = poses[:,:,[2,6,7,8,12,13,14,1,0,3,4,5,9,10,11],:]
                # print(poses[0][0])
                # print(poses.shape)
                length = poses.shape[1]
                # print(length)
                # length = int(poses.shape[1]/50) # cut sequences
                for j in range(0, length, stride):
                    # print(i)
                    if j + sequence_len > length:
                        break
                    # print(i)
                    pose = poses[:, j:j + sequence_len, :, :]
                    # print(pose.shape)
                    data.append(pose)
                    
    data = np.array(data)
    # print('<50', cnt_0)
    # print(cnt_50)
    # print(cnt)
    # print('length', length_calc)
    print(data.shape)
    save_path = f'data_undivided.npy'
    print(save_path)
    np.save(save_path,data)
    # quit()
    return data


def divide_data(ori_data, ratio):
    len_training = int(ratio * ori_data.shape[0])
    data = ori_data[:len_training,:,:,:,:]
    # data=np.array(data) 
    # print(data.shape) #(2265, 5, 120, 15, 3)
    # data[:,:,:,:,0]=data[:,:,:,:,0]/np.max(data[:,:,:,:,0])
    # print(np.max(data[:,:,:,:,0]))
    data[:,:,:,:,0] = data[:,:,:,:,0] - np.mean(data[:,:,:,:,0])
    data[:,:,:,:,0] = data[:,:,:,:,0] / 1000
    # print(data[0,0,0,:,0])
    # print(np.max(data[:,:,:,:,1]))
    # print(data[0,0,0,:,0])
    data[:,:,:,:,1] = data[:,:,:,:,1]/1000
    # print(data[0,0,0,:,1])
    data[:,:,:,:,2] = data[:,:,:,:,2] - np.mean(data[:,:,:,:,2])
    data[:,:,:,:,2] = data[:,:,:,:,2] / 1000
    # print(data[0,0,0,:,2])
    # print(np.max(data[:,:,:,:,2]))
    # print(data[0][0][0])
    # print(data.shape)
    data=data.reshape(data.shape[0],5,-1,45)
    print('tarining set length = ',data.shape)
    np.save(f'training.npy',data)
    np.save(f'discriminator.npy',data)

    # quit()
    # ###########################################################################

    # #test data
    data = ori_data[len_training:,:,:,:,:]
    # data = np.array(data) 
    # print(data.shape)
    data[:,:,:,:,0] = data[:,:,:,:,0] - np.mean(data[:,:,:,:,0])
    data[:,:,:,:,0] = data[:,:,:,:,0] / 1000
    # print(data[0,0,0,:,0])
    # print(np.max(data[:,:,:,:,1]))
    # print(data[0,0,0,:,0])
    data[:,:,:,:,1] = data[:,:,:,:,1] / 1000
    # print(data[0,0,0,:,1])
    data[:,:,:,:,2] = data[:,:,:,:,2] - np.mean(data[:,:,:,:,2])
    data[:,:,:,:,2] = data[:,:,:,:,2] / 1000
    # print(data[0,0,0,:,2])
    data = data.reshape(data.shape[0],5,-1,45)
    print('testing set length = ',data.shape)
    np.save(f'test.npy',data)
    


if __name__ == '__main__':

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default='5')
    parser.add_argument('--sequence_len', type=int, default='50')
    parser.add_argument('--ratio', type=float, default='0.5')
    args = parser.parse_args()
    print(args.stride, args.sequence_len)
    data = process_data(args.stride, args.sequence_len)
    divide_data(data, args.ratio)