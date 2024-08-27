import numpy as np

def updatepositions(positions,gcparticles,subdat,pos,redshift):

    dists=[]
    for gci in range(len(gcparticles)):
        if 'PartType1' in subdat.keys():
            wparticleidm=np.where(gcparticles[gci]==np.array(subdat['PartType1']['ParticleIDs'][:]))[0]
        else:
            wparticleidm=[]
        if 'PartType4' in subdat.keys():
            wparticleist=np.where(gcparticles[gci]==np.array(subdat['PartType4']['ParticleIDs'][:]))[0]
        else:
            wparticleist=[]
        if len(wparticleidm)!=0:
            positions[gci]=np.sqrt((subdat['PartType1']['Coordinates'][wparticleidm[0],0]/1000/.6774/(1+redshift)-pos[0]/.6774/(1+redshift))**2+
                                 (subdat['PartType1']['Coordinates'][wparticleidm[0],1]/1000/.6774/(1+redshift)-pos[1]/.6774/(1+redshift))**2+
                                 (subdat['PartType1']['Coordinates'][wparticleidm[0],2]/1000/.6774/(1+redshift)-pos[2]/.6774/(1+redshift))**2)

        elif len(wparticleist)!=0:
            positions[gci]=np.sqrt((subdat['PartType4']['Coordinates'][wparticleist[0],0]/1000/.6774/(1+redshift)-pos[0]/.6774/(1+redshift))**2+
                                (subdat['PartType4']['Coordinates'][wparticleist[0],1]/1000/.6774/(1+redshift)-pos[1]/.6774/(1+redshift))**2+
                                 (subdat['PartType4']['Coordinates'][wparticleist[0],2]/1000/.6774/(1+redshift)-pos[2]/.6774/(1+redshift))**2)
        
