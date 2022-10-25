import numpy as np
import os
import argparse

train=np.load('data_25fps_50frames.npy',allow_pickle=True)
# print(train.shape)
data=train[:2265,:,:,:,:]
data=np.array(data) 
# print(data.shape) #(2265, 5, 120, 15, 3)
# data[:,:,:,:,0]=data[:,:,:,:,0]/np.max(data[:,:,:,:,0])
# print(np.max(data[:,:,:,:,0]))
data[:,:,:,:,0]=data[:,:,:,:,0]-np.mean(data[:,:,:,:,0])
data[:,:,:,:,0]=data[:,:,:,:,0]/1000
# print(data[0,0,0,:,0])
# print(np.max(data[:,:,:,:,1]))
# print(data[0,0,0,:,0])
data[:,:,:,:,1]=data[:,:,:,:,1]/1000
# print(data[0,0,0,:,1])
data[:,:,:,:,2]=data[:,:,:,:,2]-np.mean(data[:,:,:,:,2])
data[:,:,:,:,2]=data[:,:,:,:,2]/1000
# print(data[0,0,0,:,2])
# print(np.max(data[:,:,:,:,2]))
# print(data[0][0][0])
# print(data.shape)
data=data.reshape(data.shape[0],5,-1,45)
print(data.shape)
# np.save('train_2265_meter.npy',data)

# quit()
# ###########################################################################

# #test data
data=train[2265:,:,:,:,:]
data=np.array(data) 
print(data.shape)
data[:,:,:,:,0]=data[:,:,:,:,0]-np.mean(data[:,:,:,:,0])
data[:,:,:,:,0]=data[:,:,:,:,0]/1000
print(data[0,0,0,:,0])
# print(np.max(data[:,:,:,:,1]))
# print(data[0,0,0,:,0])
data[:,:,:,:,1]=data[:,:,:,:,1]/1000
print(data[0,0,0,:,1])
data[:,:,:,:,2]=data[:,:,:,:,2]-np.mean(data[:,:,:,:,2])
data[:,:,:,:,2]=data[:,:,:,:,2]/1000
print(data[0,0,0,:,2])
data=data.reshape(data.shape[0],5,-1,45)
print(data.shape)
np.save('test_312_meter.npy',data)



# ###########################################################################

# #discriminator data
# data=train[20000:26000,:,:,:,:]
# data=np.array(data) 
# print(data.shape)
# data[:,:,:,0]=data[:,:,:,0]-np.mean(data[:,:,:,0])
# data[:,:,:,2]=data[:,:,:,2]-np.mean(data[:,:,:,2])
# data=data.reshape(data.shape[0],5,-1,45)
# print(data.shape)
# # np.save('dis_wusi.npy',data)