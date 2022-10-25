import torch.utils.data as data
import torch
import numpy as np

class D_DATA(data.Dataset):
    def __init__(self,joints=15):
        self.data=np.load('./data_wusi/discriminator.npy',allow_pickle=True)
        self.len=len(self.data)
        

    def __getitem__(self, index):
        # input_seq=self.data[index][:,:30,:][:,::2,:]
        # output_seq=self.data[index][:,30:,:][:,::2,:]
        input_seq=self.data[index][:,:25,:]#input, 25 fps
        output_seq=self.data[index][:,25:,:]#output, 25 fps
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)
        return input_seq,output_seq
        

    def __len__(self):
        return self.len

