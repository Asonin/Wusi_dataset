import torch.utils.data as data
import torch
import numpy as np

class DATA(data.Dataset):
    def __init__(self):
        self.data=np.load('./data_wusi/training.npy',allow_pickle=True)
        self.len=len(self.data)
        

    def __getitem__(self, index):
        input_seq=self.data[index][:,:25,:]#input, 25 fps
        output_seq=self.data[index][:,25:,:]#output, 25 fps
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)
        return input_seq,output_seq
        
        
    def __len__(self):
        return self.len



class TESTDATA(data.Dataset):
    def __init__(self,dataset='wusi'):
        if dataset=='wusi':
            self.data=np.load('./data_wusi/test.npy',allow_pickle=True)
        self.len=len(self.data)


    def __getitem__(self, index):
        input_seq=self.data[index][:,:25,:]#input, 25 fps
        output_seq=self.data[index][:,25:,:]#output, 25 fps
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)
        return input_seq,output_seq


    def __len__(self):
        return self.len