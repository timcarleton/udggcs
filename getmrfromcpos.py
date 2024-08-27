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

def getmwithinr(positions,mstar,rstar,m200=0,r200=0,deltac=0,redshift=0,alpha=1,beta=3,gamma=0,rho0=0,rscale=0,stripped=False):
    if len(positions)==0:
        return []

    rs=np.array(positions)
    
    if stripped:
        return getmrstripped(rs,mstar,rstar,rho0,rscale,alpha,beta,gamma)
    else:
        return getmr(rs,mstar,rstar,m200,r200,deltac,redshift,alpha,beta,gamma)
