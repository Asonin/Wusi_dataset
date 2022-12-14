from lib2to3.pytree import Base
from locale import bind_textdomain_codeset
import torch
import numpy as np
import torch_dct as dct
import time
from MRT.Models import Transformer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from data import TESTDATA
import ffmpeg


# colors for visualization
colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']

dataset_name='wusi'
test_dataset = TESTDATA(dataset=dataset_name)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
device='cpu'
batch_size=32
model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64,device=device).to(device)
plot=False

gt=False
model.load_state_dict(torch.load('./saved_model_wusi/99.model',map_location=device)) 

body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)

losses=[]
total_loss=0
loss_list1=[]
loss_list2=[]
loss_list3=[]

aligned_loss_list1 = []
aligned_loss_list2 = []
aligned_loss_list3 = []

root_loss_list1=[]
root_loss_list2=[]
root_loss_list3=[]

base_dir = 'save_results_wusi/'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
with torch.no_grad():
    model.eval()
    loss_list=[]
    for jjj,data in enumerate(test_dataloader,0):
        print(jjj)
        if jjj%5==0:
            folder_dir = base_dir + str(jjj) + '/'
            if not os.path.exists(folder_dir):
                os.mkdir(folder_dir)
        #if jjj!=20:
        #    continue
        input_seq,output_seq=data
        input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device) # 1,5,25,45
        output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device) # 1,5,76,45
        n_joints=int(input_seq.shape[-1]/3)
        # print(n_joints)
        use=[input_seq.shape[1]]
        
        input_=input_seq.view(-1,25,input_seq.shape[-1])
  
    
        output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])

        input_ = dct.dct(input_)
        output__ = dct.dct(output_[:,:,:])
        
        
        rec_=model.forward(input_[:,1:25,:]-input_[:,:24,:],dct.idct(input_[:,-1:,:]),input_seq,use)
        
        rec=dct.idct(rec_)

        results=output_[:,:1,:]
        for i in range(1,26):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]

        new_input_seq=torch.cat([input_seq,results.reshape(input_seq.shape)],dim=-2)
        new_input=dct.dct(new_input_seq.reshape(-1,50,45))
        
        new_rec_=model.forward(new_input[:,1:,:]-new_input[:,:-1,:],dct.idct(new_input[:,-1:,:]),new_input_seq,use)
        
        
        new_rec=dct.idct(new_rec_)
        
        new_results=new_input_seq.reshape(-1,50,45)[:,-1:,:]
        for i in range(1,26):
            new_results=torch.cat([new_results,new_input_seq.reshape(-1,50,45)[:,-1:,:]+torch.sum(new_rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        new_results=new_results[:,1:,:]
        
        results=torch.cat([results,new_results],dim=-2)

        rec=torch.cat([rec,new_rec],dim=-2)

        results=output_[:,:1,:]

        for i in range(1,26+25):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]

        new_new_input_seq=torch.cat([input_seq,results.unsqueeze(0)],dim=-2)
        new_new_input=dct.dct(new_new_input_seq.reshape(-1,75,45))
        
        new_new_rec_=model.forward(new_new_input[:,1:,:]-new_new_input[:,:-1,:],dct.idct(new_new_input[:,-1:,:]),new_new_input_seq,use)

        new_new_rec=dct.idct(new_new_rec_)
        rec=torch.cat([rec,new_new_rec],dim=-2)

        results=output_[:,:1,:]

        for i in range(1,51+25):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]
        
        prediction_1=results[:,:25,:].view(results.shape[0],-1,n_joints,3) #people, frames, 15, 3
        # print(prediction_1.shape)
        # quit()
        prediction_2=results[:,:50,:].view(results.shape[0],-1,n_joints,3)
        prediction_3=results[:,:75,:].view(results.shape[0],-1,n_joints,3)

        gt_1=output_seq[0][:,1:26,:].view(results.shape[0],-1,n_joints,3)
        gt_2=output_seq[0][:,1:51,:].view(results.shape[0],-1,n_joints,3)
        gt_3=output_seq[0][:,1:76,:].view(results.shape[0],-1,n_joints,3)

        if dataset_name=='mocap':
            #match the scale with the paper, also see line 63 in mix_mocap.py
            loss1=torch.sqrt(((prediction_1/1.8 - gt_1/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss2=torch.sqrt(((prediction_2/1.8 - gt_2/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss3=torch.sqrt(((prediction_3/1.8 - gt_3/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()

            #pose with align
            loss1=torch.sqrt((((prediction_1 - prediction_1[:,:,0:1,:] - gt_1 + gt_1[:,:,0:1,:])/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss2=torch.sqrt((((prediction_2 - prediction_2[:,:,0:1,:] - gt_2 + gt_2[:,:,0:1,:])/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss3=torch.sqrt((((prediction_3 - prediction_3[:,:,0:1,:] - gt_3 + gt_3[:,:,0:1,:])/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()

        if dataset_name=='mupots':
            loss1=torch.sqrt(((prediction_1 - gt_1) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss2=torch.sqrt(((prediction_2 - gt_2) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss3=torch.sqrt(((prediction_3 - gt_3) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            
            #pose with align
            loss1=torch.sqrt(((prediction_1 - prediction_1[:,:,0:1,:] - gt_1 + gt_1[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss2=torch.sqrt(((prediction_2 - prediction_2[:,:,0:1,:] - gt_2 + gt_2[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss3=torch.sqrt(((prediction_3 - prediction_3[:,:,0:1,:] - gt_3 + gt_3[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()

        if dataset_name == 'wusi':
            loss1=torch.sqrt(((prediction_1 - gt_1) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss2=torch.sqrt(((prediction_2 - gt_2) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss3=torch.sqrt(((prediction_3 - gt_3) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            
            #pose with align
            aligned_loss1=torch.sqrt(((prediction_1 - prediction_1[:,:,0:1,:] - gt_1 + gt_1[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            aligned_loss2=torch.sqrt(((prediction_2 - prediction_2[:,:,0:1,:] - gt_2 + gt_2[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            aligned_loss3=torch.sqrt(((prediction_3 - prediction_3[:,:,0:1,:] - gt_3 + gt_3[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            
            root_loss1=torch.sqrt(((prediction_1[:,:,0,:] - gt_1[:,:,0,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            # print(root_loss1)
            root_loss2=torch.sqrt(((prediction_2[:,:,0,:] - gt_2[:,:,0,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            # print(root_loss2)
            root_loss3=torch.sqrt(((prediction_3[:,:,0,:] - gt_3[:,:,0,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            

        
        loss_list1.append(np.mean(loss1))#+loss1
        loss_list2.append(np.mean(loss2))#+loss2
        loss_list3.append(np.mean(loss3))#+loss3
        
        aligned_loss_list1.append(np.mean(aligned_loss1))#+loss1
        aligned_loss_list2.append(np.mean(aligned_loss2))#+loss2
        aligned_loss_list3.append(np.mean(aligned_loss3))#+loss3
        
        root_loss_list1.append(np.mean(root_loss1))#+loss1
        root_loss_list2.append(np.mean(root_loss2))#+loss2
        root_loss_list3.append(np.mean(root_loss3))#+loss3
        
        loss=torch.mean((rec[:,:,:]-(output_[:,1:76,:]-output_[:,:75,:]))**2)
        losses.append(loss)
        
        rec=results[:,:,:]
        
        rec=rec.reshape(results.shape[0],-1,n_joints,3)
        
        input_seq=input_seq.view(results.shape[0],25,n_joints,3)
        pred=torch.cat([input_seq,rec],dim=1)
        output_seq=output_seq.view(results.shape[0],-1,n_joints,3)[:,1:,:,:]
        all_seq=torch.cat([input_seq,output_seq],dim=1)
        
        pred=pred[:,:,:,:].cpu()
        all_seq=all_seq[:,:,:,:].cpu()
        
        if plot and jjj % 5 == 0:
            fig = plt.figure(figsize=(10, 4.5))
            # fig.tight_layout()
            ax = fig.add_subplot(111, projection='3d')
            
            # plt.ion()
            length=100
            length_=100
            i=0

            p_x=np.linspace(-10,10,25)
            p_y=np.linspace(-10,10,25)
            X,Y=np.meshgrid(p_x,p_y)
            
            while i < length_:
                
                ax.lines = []

                for x_i in range(p_x.shape[0]):
                    temp_x=[p_x[x_i],p_x[x_i]]
                    temp_y=[p_y[0],p_y[-1]]
                    z=[0,0]
                    ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)

                for y_i in range(p_x.shape[0]):
                    temp_x=[p_x[0],p_x[-1]]
                    temp_y=[p_y[y_i],p_y[y_i]]
                    z=[0,0]
                    ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)

                for j in range(results.shape[0]):
                    
                    xs=pred[j,i,:,0].numpy()
                    # print(xs)
                    ys=pred[j,i,:,1].numpy()
                    zs=pred[j,i,:,2].numpy()
                    
                    alpha=1
                    ax.plot( xs, ys,zs, 'y.',alpha=alpha)
                    
                    if gt:
                        x=all_seq[j,i,:,0].numpy()
                        
                        y=all_seq[j,i,:,1].numpy()
                        z=all_seq[j,i,:,2].numpy()
                    
                    
                        ax.plot( x, y, z, 'y.')

                    plot_edge=True
                    if plot_edge:
                        for edge in body_edges:
                            x=[pred[j,i,edge[0],0],pred[j,i,edge[1],0]]
                            y=[pred[j,i,edge[0],1],pred[j,i,edge[1],1]]
                            z=[pred[j,i,edge[0],2],pred[j,i,edge[1],2]]
                            if i>=15:
                                ax.plot(x, y, z,zdir='z',c='blue',alpha=alpha)
                                
                            else:
                                ax.plot(x, y, z,zdir='z',c='green',alpha=alpha)
                            
                            if gt:
                                x=[all_seq[j,i,edge[0],0],all_seq[j,i,edge[1],0]]
                                y=[all_seq[j,i,edge[0],1],all_seq[j,i,edge[1],1]]
                                z=[all_seq[j,i,edge[0],2],all_seq[j,i,edge[1],2]]
                            
                                if i>=15:
                                    ax.plot( x, y, z,'yellow',alpha=0.8)
                                else:
                                    ax.plot(  x, y, z,'green')
                            
                   
                    ax.set_xlim3d([-2 , 5])
                    ax.set_ylim3d([-4 , 3])
                    ax.set_zlim3d([0,4])
                    # ax.set_xlim3d([-8 , 8])
                    # ax.set_ylim3d([-8 , 8])
                    # ax.set_zlim3d([0,5])
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    # ax.set_zticklabels([])
                    ax.set_axis_off()
                    #ax.patch.set_alpha(1)
                    #ax.set_aspect('equal')
                    #ax.set_xlabel("x")
                    #ax.set_ylabel("y")
                    #ax.set_zlabel("z")
                    plt.title(str(i),y=-0.1)
                plt.pause(0.1)
                prefix = '{:02}'.format(i)
                filename = folder_dir + prefix
                plt.savefig(filename)
                
                i += 1


            # plt.ioff()
            # plt.show()
            plt.savefig(filename)
            plt.close()
            
            pics_3d = folder_dir + '/%2d.png'
            out_dir_3d = base_dir + f'/{jjj}_3d.mp4'
            if os.path.exists(out_dir_3d):
                continue
            (
                ffmpeg
                .input(pics_3d, framerate=25)
                .output(out_dir_3d)
                .run()
            )

            
    print('avg 1 second',np.mean(loss_list1))
    print('avg 2 seconds',np.mean(loss_list2))
    print('avg 3 seconds',np.mean(loss_list3))
    
    print('avg 1 second aligned',np.mean(aligned_loss_list1))
    print('avg 2 seconds aligned',np.mean(aligned_loss_list2))
    print('avg 3 seconds aligned',np.mean(aligned_loss_list3))
    
    print('avg 1 second root',np.mean(root_loss_list1))
    print('avg 2 seconds root',np.mean(root_loss_list2))
    print('avg 3 seconds root',np.mean(root_loss_list3))
