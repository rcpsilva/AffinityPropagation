import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generateTData(nsamples):

    timeStamps = np.linspace(10, 1000, num=nsamples)*0.05+50;
    positions = np.ones(nsamples);
    irradiance = np.sin(timeStamps)*timeStamps+timeStamps*5

    data1 = np.concatenate((np.transpose([positions]),np.transpose([timeStamps]),np.transpose([irradiance])), axis = 1)

    positions = np.ones(nsamples)*2;
    irradiance = irradiance/10 -30

    data2 = np.concatenate((np.transpose([positions]),np.transpose([timeStamps]),np.transpose([irradiance])), axis = 1)


    positions = np.ones(nsamples)*3;
    irradiance = np.sin(timeStamps/2)*timeStamps+timeStamps*2

    data3 = np.concatenate((np.transpose([positions]),np.transpose([timeStamps]),np.transpose([irradiance])), axis = 1)


    data = np.concatenate((data1,data2,data3),axis = 0)

    #plt.plot(data[0:nsamples,1],data[0:nsamples,2],'r.')
    #plt.plot(data[nsamples:2*nsamples,1],data[nsamples:2*nsamples,2],'g.')
    #plt.plot(data[2*nsamples:3*nsamples,1],data[2*nsamples:3*nsamples,2],'b.') 
    
    return data

def plotTData(data):
    
    nsamples = data.shape[0]
    print(nsamples)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for c, p in [('r',0),('g',1),('b',2)]:
        xs = data[p*nsamples:(p+1)*nsamples,1]
        ys = data[p*nsamples:(p+1)*nsamples,0]
        zs = data[p*nsamples:(p+1)*nsamples,2]
        ax.scatter(xs, ys, zs, c=c)

    ax.set_xlabel('time')
    ax.set_ylabel('position')
    ax.set_zlabel('irradiance')

    plt.show() 

data = generateTData(25)
print(data)
plotTData(data)