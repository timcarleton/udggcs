import h5py
from scipy.optimize import minimize_scalar
import numpy as np
import imp
import transformtonormal
import crossmanyvectors
imp.reload(transformtonormal)
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
# From Mistani 2016


def msfrr(coords,r,mass):
    cmx=np.mean(coords[:,0]*mass)/np.mean(mass)
    cmy=np.mean(coords[:,1]*mass)/np.mean(mass)
    w=np.where((coords[:,0]-cmx)**2+(coords[:,1]-cmy)**2<r**2)[0]
    return np.sum(mass[w])

def getgcfrac(f,redshift,normal=[0,0,1],haradiusfactor=0.95):

#    print(cutout)


    #sfgas=np.where(f['PartType0']['StarFormationRate'][:]>0)[0]
    sfgas=np.where(f['PartType0']['StarFormationRate'][:]>10**-3.1)[0]
    if len(sfgas)<=1:
        return 0

    ngas=len(f['PartType0']['Masses'])
#    gascm=np.mean(f['PartType0']['Coordinates'][:,:]*np.repeat(f['PartType0']['Masses'][:],3).reshape([ngas,3]),axis=0)/np.mean(f['PartType0']['Masses'][:])
    gascm=np.mean(f['PartType0']['Coordinates'][:,:][sfgas,:]*np.repeat(f['PartType0']['Masses'][:][sfgas],3).reshape([len(sfgas),3]),axis=0)/np.mean(f['PartType0']['Masses'][:][sfgas])
    gasvector=f['PartType0']['Coordinates'][:,:]-gascm

    #newgasvector=transformtonormal.transformtonormal(gasvector,np.array(normal)/np.sqrt(np.sum(np.array(normal)**2)))
    newgasvector=crossmanyvectors.crossmanyvectors(gasvector,np.array(normal)/np.sqrt(np.sum(np.array(normal)**2)))



    totsfgasmass=np.sum(f['PartType0']['Masses'][:][sfgas])

    rsf=10**minimize_scalar(lambda x: abs(msfrr(newgasvector[sfgas,:],10**x,f['PartType0']['Masses'][:][sfgas])-haradiusfactor*totsfgasmass),method='bounded',bounds=[np.log10(np.min(newgasvector[sfgas,0]**2+newgasvector[sfgas,1]**2)**.5),np.log10(np.max(newgasvector[sfgas,0]**2+newgasvector[sfgas,1]**2)**.5)])['x']/.6774/(1+redshift)



#    print(rsf,msfrr(newgasvector[sfgas,:],rsf,f['PartType0']['Masses'][:][sfgas]),totsfgasmass)
    sigmasfr=np.sum(f['PartType0']['StarFormationRate'][:])/np.pi/rsf**2

#    print(redshift,rsf,np.sum(f['PartType0']['StarFormationRate']),sigmasfr)
#    print(rsf,f['PartType0']['StarFormationRate'][:])
    #plt.clf()
    #plt.scatter(newgasvector[sfgas,0],newgasvector[sfgas,1],alpha=.2,c=np.log10(np.array(f['PartType0']['StarFormationRate'][:])[sfgas]))
    #plt.colorbar()
#    plt.scatter(gasvector[sfgas,0],gasvector[sfgas,1],alpha=.2,c=np.log10(np.array(f['PartType0']['Masses'][:])[sfgas]))
    #circle=plt.Circle((0,0),rsf*.6774*(1+redshift),fill=False)
    #plt.gca().add_artist(circle)
    #plt.xlim(-max(newgasvector[sfgas,0]**2+newgasvector[sfgas,1]**2)**.5*1.1,max(newgasvector[sfgas,0]**2+newgasvector[sfgas,1]**2)**.5*1.1)
    #plt.ylim(-max(newgasvector[sfgas,0]**2+newgasvector[sfgas,1]**2)**.5*1.1,max(newgasvector[sfgas,0]**2+newgasvector[sfgas,1]**2)**.5*1.1)

    #plt.show()
    
    return 0.29*sigmasfr**0.24
