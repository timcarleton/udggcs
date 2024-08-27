import numpy as np

def getgcdmdt(m,r,vcirc,alpha=0.67,withiso=False):
    
    if withiso:
        tiso=17*(m/2E5)
    else:
        try:
            tiso=np.zeros_like(m)+np.inf
        except:
            tiso=np.inf

    pr=41.4*r/vcirc

    ttide=10*(m/2E5)**alpha*pr
    #print(pr)

    dmdt=-m/np.min([tiso,ttide],axis=0)
    try:
        if m<=0:
            return 0
    except:
        dmdt[np.where(m<0)]=0

    try:
        if not np.isfinite(vcirc):
            return 0
    except:
        dmdt[np.where(~np.isfinite(vcirc))[0]]=0
        
    return dmdt
#    return np.zeros_like(m)
