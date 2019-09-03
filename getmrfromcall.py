import numpy as np
import profileclass

from colossus.cosmology import cosmology as colcosmology
from colossus.halo import concentration,mass_adv
colcosmology.setCosmology('planck18')


def getmr(r,mstar,rstar,m200,r200,deltac,z,alpha,beta,gamma):

    totmass=np.zeros(len(r))
    mstr=np.zeros(len(r))
    massr=np.zeros(len(r))
    
    wrg0=np.where(r>0)[0]

    mstr[wrg0]=mstar*r[wrg0]**3/(rstar**2+r[wrg0]**2)**1.5

    c=concentration.modelPrada12(m200,z)*10**(deltac)

    rs=r200/c

    massr[wrg0]=profileclass.getmassfromzhao0(alpha,beta,gamma,1,rs,r[wrg0])*m200/profileclass.getmassfromzhao0(alpha,beta,gamma,1,rs,r200)

    return massr+mstr

def getmrstripped(r,mstar,rstar,rho0,rs,alpha,beta,gamma):

    totmass=np.zeros(len(r))
    mstr=np.zeros(len(r))
    massr=np.zeros(len(r))

    wrg0=np.where(r>0)[0]

    mstr[wrg0]=mstar*r[wrg0]**3/(rstar**2+r[wrg0]**2)**1.5

    massr[wrg0]=profileclass.getmassfromzhao0(alpha,beta,gamma,rho0,rs,r[wrg0])

    return massr+mstr

def getmwithinr(particleids,h,mstar,rstar,r0=[-1],m200=0,r200=0,deltac=0,redshift=0,alpha=1,beta=3,gamma=0,rho0=0,rscale=0,x=-1,y=-1,z=-1,stripped=False,part_type=['dm','star']):
    if x==-1:
        x=np.mean(h['PartType1']['Coordinates'][:,0])
    if y==-1:
        y=np.mean(h['PartType1']['Coordinates'][:,1])
    if z==-1:
        z=np.mean(h['PartType1']['Coordinates'][:,2])

    if len(r0)==0:
        return [],[]

    if r0[0]==-1:
        xcoordsgc=np.zeros(len(particleids))
        ycoordsgc=np.zeros(len(particleids))
        zcoordsgc=np.zeros(len(particleids))
        for itype in range(len(part_type)):
            if part_type[itype]=='dm':
                for ii in range(len(particleids)):
                    wgcdm=np.where(np.array(h['PartType1']['ParticleIDs'][:]).astype(np.uint64)==np.uint64(int(particleids[ii])))[0]
                    if len(wgcdm)==1:
                        xcoordsgc[ii]=h['PartType1']['Coordinates'][wgcdm[0],0]
                        ycoordsgc[ii]=h['PartType1']['Coordinates'][wgcdm[0],1]
                        zcoordsgc[ii]=h['PartType1']['Coordinates'][wgcdm[0],2]
            elif part_type[itype]=='star':
                for ii in range(len(particleids)):
                    wgcst=np.where(np.uint64(int(particleids[ii]))==np.array(h['PartType4']['ParticleIDs'][:]).astype(np.uint64))[0]
                    if len(wgcst)==1:
                        xcoordsgc[ii]=h['PartType4']['Coordinates'][wgcst[0],0]
                        ycoordsgc[ii]=h['PartType4']['Coordinates'][wgcst[0],1]
                        zcoordsgc[ii]=h['PartType4']['Coordinates'][wgcst[0],2]

        rs=np.sqrt((xcoordsgc-x)**2+(ycoordsgc-y)**2+(zcoordsgc-z)**2)
    else:
        rs=r0
   
    if stripped:
        return getmrstripped(rs,mstar,rstar,rho0,rscale,alpha,beta,gamma),rs
    else:
        return getmr(rs,mstar,rstar,m200,r200,deltac,redshift,alpha,beta,gamma),rs
