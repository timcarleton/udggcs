import numpy as np
from python import downloadcutout
import h5py

def getmwithinrsim(particleids,h,x=-1,y=-1,z=-1,justdm=True):

    if x==-1:
        x=np.mean(h['PartType1']['Coordinates'][:,0])
    if y==-1:
        y=np.mean(h['PartType1']['Coordinates'][:,1])
    if z==-1:
        z=np.mean(h['PartType1']['Coordinates'][:,2])
        
    distgas=((h['PartType0']['Coordinates'][:,0]-x)**2+(h['PartType0']['Coordinates'][:,1]-y)**2+(h['PartType0']['Coordinates'][:,2]-z)**2)**.5
    distdm=((h['PartType1']['Coordinates'][:,0]-x)**2+(h['PartType1']['Coordinates'][:,1]-y)**2+(h['PartType1']['Coordinates'][:,2]-z)**2)**.5
    diststar=((h['PartType4']['Coordinates'][:,0]-x)**2+(h['PartType4']['Coordinates'][:,1]-y)**2+(h['PartType4']['Coordinates'][:,2]-z)**2)**.5
    distbh=((h['PartType5']['Coordinates'][:,0]-x)**2+(h['PartType5']['Coordinates'][:,1]-y)**2+(h['PartType5']['Coordinates'][:,2]-z)**2)**.5

    mwithinr=[]
    for i in range(len(particleids)):
        w=np.where(h['PartType1']['ParticleIDs'][:]==particleids[i])[0]
        if len(w)>0:
            totmass=0
            totmass+=len(np.where(distdm<distdm[w[0]])[0])*7.5E6
            if not justdm:
                wingas=np.where(distgas<distdm[w[0]])[0]
                totmass+=h['PartType0']['Masses'][wingas]

                winstar=np.where(diststar<distdm[w[0]])[0]
                totmass+=h['PartType4']['Masses'][winstar]

                winbh=np.where(distbh<distdm[w[0]])[0]
                totmass+=h['PartType5']['BH_Mass'][winbh]

            mwithinr.append(totmass)
        else:
            mwithinr.append(0)
    mwithinr=np.array(mwithinr)
    return mwithinr
