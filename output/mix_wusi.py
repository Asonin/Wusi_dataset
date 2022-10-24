import numpy as np
import os

train=np.load('data1.npy',allow_pickle=True)
print(train.shape)
data=train[:6000,:,:,:,:]
data=np.array(data) 
print(data.shape)

data=data.reshape(data.shape[0],5,-1,45)
print(data.shape)
np.save('train_wusi.npy',data)



# ###########################################################################

# #test data
data=train[10000:10800,:,:,:,:]
data=np.array(data) 
print(data.shape)

data=data.reshape(data.shape[0],5,-1,45)
print(data.shape)
np.save('test_wusi.npy',data)



# ###########################################################################

# #discriminator data
data=train[20000:26000,:,:,:,:]
data=np.array(data) 
print(data.shape)

data=data.reshape(data.shape[0],5,-1,45)
print(data.shape)
np.save('dis_wusi.npy',data)