import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import ffmpeg

# colors for visualization
colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']

data=np.load('training.npy',allow_pickle=True)
print(data.shape)
# quit()
eg=1
data_list=data[eg]

# quit()
data_list=data_list.reshape(5,100,15,3)
print(data_list.shape)
# quit()
body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)


fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(111, projection='3d')


length_=data_list.shape[1]

i=0
folder_dir = './test_vis/'
if not os.path.exists(folder_dir):
    os.mkdir(folder_dir)
while i < length_:
    ax.lines = []
    for j in range(len(data_list)):
        
        xs=data_list[j,i,:,0]
        ys=data_list[j,i,:,1]
        zs=data_list[j,i,:,2]
        #print(xs)
        ax.plot( xs, ys,zs, 'y.')
        
        
        plot_edge=True
        if plot_edge:
            for edge in body_edges:
                x=[data_list[j,i,edge[0],0],data_list[j,i,edge[1],0]]
                y=[data_list[j,i,edge[0],1],data_list[j,i,edge[1],1]]
                z=[data_list[j,i,edge[0],2],data_list[j,i,edge[1],2]]
                if i>=30:
                    ax.plot(x, y, z, 'green')
                else:
                    ax.plot(x, y,z, 'blue')
        
        
        ax.set_xlim3d([-2 , 5])
        ax.set_ylim3d([-4 , 3])
        ax.set_zlim3d([-0, 4])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
    plt.pause(0.01)
    i += 1
    prefix = '{:02}'.format(i)
    
    filename = folder_dir + prefix
    plt.savefig(filename)
    
pics_3d = folder_dir + '/%2d.png'
out_dir_3d = './3d.mp4'

if os.path.exists(out_dir_3d):
    quit()
(
    ffmpeg
    .input(pics_3d, framerate=25)
    .output(out_dir_3d)
    .run()
) 