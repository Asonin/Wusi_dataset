# from importlib.metadata import requires
import torch
import torch.optim as optim
import numpy as np
import torch_dct as dct #https://github.com/zh217/torch-dct
import time

from MRT.Models import Transformer,Discriminator
from utils import disc_l2_loss,adv_disc_l2_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import os


from data import DATA
dataset = DATA()
batch_size=32

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

from discriminator_data import D_DATA
real_=D_DATA()

real_motion_dataloader = torch.utils.data.DataLoader(real_, batch_size=batch_size, shuffle=True)
real_motion_all=list(enumerate(real_motion_dataloader))

device='cuda'

model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64,device=device).to(device)

discriminator = Discriminator(d_word_vec=45, d_model=45, d_inner=256,
            n_layers=3, n_head=8, d_k=32, d_v=32,device=device).to(device)

lrate=0.0003
lrate2=0.0005

params = [
    {"params": model.parameters(), "lr": lrate}
]
optimizer = optim.Adam(params)
params_d = [
    {"params": discriminator.parameters(), "lr": lrate}
]
optimizer_d = optim.Adam(params_d)



for epoch in range(100):
    total_loss=0
    
    for j,data in enumerate(dataloader,0):
                
        use=None
        input_seq,output_seq=data 
        # print(batch_size) # currently 1
        # print('input_seq_ori', input_seq.shape) # 1,5,25,45
        # print('output_seq_ori', output_seq.shape) # 1,5,76,45
        # quit()
        input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device) # batch, N_person, 25 (25 fps 1 second), 45 (15joints xyz) 
        output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device) # batch, N_persons, 46 (last frame of input + future 3 seconds), 45 (15joints xyz) 
        
        # first 1 second predict future 1 second
        input_=input_seq.view(-1,25,input_seq.shape[-1]) # batch x n_person ,25: 25 fps, 1 second, 45: 15joints x 3
        # print('input_ reshaped', input_.shape) # batch_size*5,25,45
        # quit()
        output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])
        # print('output_ reshaped', output_.shape) # batch_size*5,76,45
        
        input_ = dct.dct(input_)
        # print('input_ after dct', input_.shape) # batch_size*5,25,45
        # quit()
        rec_=model.forward(input_[:,1:25,:]-input_[:,:24,:],dct.idct(input_[:,-1:,:]),input_seq,use)
        # print('1-1, rec_', rec_.shape) #batch*5,25,45
        rec=dct.idct(rec_)
        # print('1-1, rec_ after idct', rec.shape) # batch*5,25,45
        # quit()
        # first 2 seconds predict 1 second
        new_input=torch.cat([input_[:,1:25,:]-input_[:,:24,:],dct.dct(rec_)],dim=-2)
        # print('2-1, new_input', new_input.shape)
        new_input_seq=torch.cat([input_seq,output_seq[:,:,1:26]],dim=-2)
        # print('2-1, new_input_seq', new_input_seq.shape)
        new_input_=dct.dct(new_input_seq.reshape(-1,50,45))
        # print('2-1, new_input_seq after dct', new_input_.shape)
        # # print(new_input_.shape)
        new_rec_=model.forward(new_input_[:,1:,:]-new_input_[:,:49,:],dct.idct(new_input_[:,-1:,:]),new_input_seq,use)
        # print('2-1, new_rec_', new_rec_.shape)
        new_rec=dct.idct(new_rec_)
        # print('2-1, new_rec_ after idct', new_rec.shape)
        # first 3 seconds predict 1 second
        # # print(input_seq.shape, output_seq.shape)
        new_new_input_seq=torch.cat([input_seq,output_seq[:,:,1:51]],dim=-2)
        # print('3-1, new_new_input_seq', new_new_input_seq.shape)
        # # print(new_new_input_seq.shape)
        new_new_input_=dct.dct(new_new_input_seq.reshape(-1,75,45))
        # print('3-1, new_new_input_ after dct', new_new_input_.shape)
        new_new_rec_=model.forward(new_new_input_[:,1:,:]-new_new_input_[:,:74,:],dct.idct(new_new_input_[:,-1:,:]),new_new_input_seq,use)
        # print('3-1, new_new_rec_', new_new_rec_.shape)
        new_new_rec=dct.idct(new_new_rec_)
        # print('3-1, new_new_rec_ after idct', new_new_rec.shape)
        # # print(rec.shape)
        rec=torch.cat([rec,new_rec,new_new_rec],dim=-2)
        # print('after 3 preds, rec', rec.shape)
        # quit()
        # # print(rec.shape, new_rec.shape, new_new_rec.shape)
        
        results=output_[:,:1,:]
        for i in range(1,51+25):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]
        # print('results', results.shape)
        
        loss=torch.mean((rec[:,:,:]-(output_[:,1:76,:]-output_[:,:75,:]))**2) # offset
        
        
        if (j+1)%2==0:
            
            fake_motion=results
            # print('fake motion', fake_motion.shape)
            disc_loss=disc_l2_loss(discriminator(fake_motion))
            loss=loss+0.0005*disc_loss
            
            fake_motion=fake_motion.detach()

            real_motion=real_motion_all[int(j/2)][1][1]
            real_motion=real_motion.view(-1,76,45)[:,1:76,:].float().to(device)

            fake_disc_value = discriminator(fake_motion)
            real_disc_value = discriminator(real_motion)

            d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = adv_disc_l2_loss(real_disc_value, fake_disc_value)
            
            optimizer_d.zero_grad()
            d_motion_disc_loss.backward()
            optimizer_d.step()
        
       
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()
 
        total_loss=total_loss+loss

    print('epoch:',epoch,'loss:',total_loss/(j+1))
    if (epoch+1)%5==0:
        save_path=f'./saved_model_wusi/{epoch}.model'
        torch.save(model.state_dict(),save_path)


        
        
