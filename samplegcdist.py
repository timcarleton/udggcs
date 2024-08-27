import numpy as np
import sampleschecter

def samplegcdist(n,mc=2E5):
    return np.random.gamma(-1,mc/1E5,size=n)*1E5


def samplegcfrommass(mass,stop='before',mc=2E5,mmin=10000,slope=-1.999):

    if not np.isfinite(mc):
        mc=2E5

    if mass<mmin:
        return np.array([])

    if mass<mc/1000:
        return np.array([])
    mean=2086.02180795*np.log10(mc)**2-5809.87607421*np.log10(mc)+2371.73620262
    mean=mean/5
    #print('m',mass,mc,mass/mean)
#    print(mass/mean)
#    allgcs=samplegcdist(int(mass/mean))
    allgcs=sampleschecter.sampleschecter(slope, mc, mmin,int(mass/mean))

    if stop=='before':
        if np.sum(allgcs)<=mass:
            allgcs=np.append(allgcs,sampleschecter.sampleschecter( slope, mc, mmin,1))
        while np.sum(allgcs)>=mass:
            allgcs=allgcs[0:-1]
        return allgcs
        
    else:
        if np.sum(allgcs)>mass:
            while np.sum(allgcs)>=mass:
                allgcs=allgcs[0:-1]
            return np.append(allgcs,sampleschecter.sampleschecter( slope, mc, mmin,1))
        else:
            while np.sum(allgcs)<=mass:
                allgcs=np.append(allgcs,sampleschecter.sampleschecter(slope, mc, mmin,1))
            return allgcs
