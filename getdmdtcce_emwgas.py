import numpy as np
from astropy import coordinates
import crossmanyvectors
import matplotlib.pyplot as plt

def getgcdmdtcce(mclust,mgas,mstar,redshift,rclust=1.5):

       #sigmagas=5.07
    #sigmagas=100
    #rhogas=1
    
    if mstar==0:
        sigmagas=10**(.14*np.log10(1E6)+.91)
        rhogas=10**(.147*np.log10(1E6)-1.07)
    else:
        #expectedmgas=10**(0.57260384+0.87775058*np.log10(mstar)+0.18328601*redshift)
        expectedmgas=10**(-0.82730848+(0.9853019-0.01295845*redshift)*np.log10(mstar)+2.69667971*np.log10(1+redshift))
        sigmagas=10**(.14*np.log10(mstar)+.91)*(mgas/expectedmgas)
        rhogas=10**(.147*np.log10(mstar)-1.07)*(mgas/expectedmgas)

    #rhoh=.5*mclust/(4/3*np.pi*1.5**3)
    rhoh=.5*mclust/(4/3*np.pi*rclust**3)
    phiad=(1+9*(rhoh/rhogas/1E4))**-1.5

    fmol=(1+.025*(sigmagas/(1E2))**-2)**-1
    fsigma=3.92*((10-8*fmol)/2)**0.5

    tcce=0.176*(fsigma/4)**-1*(rhogas)**-1.5*(mclust/1E5)*phiad**-1
    #tcce=0.176*(fsigma/4)**-1*(rhogas)**-1.5*(mclust/1E5)

    #t5evap=10*10**(-(1.03+zgc)/.5)
    #dmdt=(-m/tcce).value-(m/(t5evap*(m/1E5)**.7))
    t5evap=17/2
    dmdt=(-mclust/tcce)-(mclust/(t5evap*(mclust/1E5)**.7))
    #dmdt=(-m/tcce).value-m/(t5evap*(m/1E5))
   # print('f',fsigma,rhogas,m,phiad,d3d3,f['PartType0']['Density'][:][sfgas]*1E10/h*h**3*(1+redshift))

    print('tc',sigmagas,rhogas)
    #print('tc',sigmagas,rhogas,rhoh,phiad,0.176*(fsigma/4)**-1*(rhogas)**-1.5)
    #print('tc',np.histogram(np.log10(tcce),range=[-2,-1],bins=50))
    #print('rhog',np.median(rhogas))

    try:
        if mclust<=0:
            return 0
    except:
        dmdt[np.where(mclust<=0)[0]]=0
    #return dmdt
    return tcce
        
