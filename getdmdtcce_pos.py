import numpy as np
from astropy import coordinates
import crossmanyvectors
import matplotlib.pyplot as plt

def getgcdmdtcce(f,distance,m,redshift,zgc,nthneighbor=3,h=0.6774 ,normal=np.array([0,0,1]),angles='random'):

    ngas=len(f['PartType0']['Masses'])
    sfgas=np.where(f['PartType0']['StarFormationRate'][:]>0)[0]
    #sfgas=np.arange(len(f['PartType0']['StarFormationRate']))
    if len(sfgas)<=nthneighbor:
        return 0

    
    if angles=='random':
        u=np.random.rand(len(distance))
        v=np.random.rand(len(distance))
        theta=2*np.pi*u
        phi=np.arccos(2*v-1)
        xgc=distance*np.sin(phi)*np.cos(theta)
        ygc=distance*np.sin(phi)*np.sin(theta)
        zgc=distance*np.cos(phi)

    else:
        try:
            wstar=np.where(f['PartType4']['GFM_StellarFormationTime'][:]>0)[0]
            if len(wstar)==0:
                u=np.random.rand(len(distance))
                v=np.random.rand(len(distance))
                theta=2*np.pi*u
                phi=np.arccos(2*v-1)
                xgc=distance*np.sin(phi)*np.cos(theta)
                ygc=distance*np.sin(phi)*np.sin(theta)
                zgc=distance*np.cos(phi)
            else:
                availablex=f['PartType4']['Coordinates'][:,0][wstar]-np.mean(f['PartType1']['Coordinates'][:,0])
                availabley=f['PartType4']['Coordinates'][:,1][wstar]-np.mean(f['PartType1']['Coordinates'][:,1])
                availablez=f['PartType4']['Coordinates'][:,2][wstar]-np.mean(f['PartType1']['Coordinates'][:,2])
                
                availabletheta=np.arctan2(availabley,availablex)
                availablephi=np.arccos(availablez/np.sqrt(availablex**2+availabley**2+availablez**2))
                print(distance,availabletheta,availablephi)
            
                chosenangles=np.random.choice(len(availabletheta),len(distance))
                xgc=distance*np.sin(availablephi[chosenangles])*np.cos(availabletheta[chosenangles])
                ygc=distance*np.sin(availablephi[chosenangles])*np.sin(availabletheta[chosenangles])
                zgc=distance*np.cos(availablephi[chosenangles])
        except TypeError:
            u=np.random.rand(len(distance))
            v=np.random.rand(len(distance))
            theta=2*np.pi*u
            phi=np.arccos(2*v-1)
            xgc=distance*np.sin(phi)*np.cos(theta)
            ygc=distance*np.sin(phi)*np.sin(theta)
            zgc=distance*np.cos(phi)

    print(xgc,ygc,zgc)
    
    gccoords=np.array([xgc+np.mean(f['PartType0']['Coordinates'][:,0]),ygc+np.mean(f['PartType0']['Coordinates'][:,1]),zgc+np.mean(f['PartType0']['Coordinates'][:,2])]).T

    
    gascm=np.mean(f['PartType0']['Coordinates'][sfgas,:]*np.repeat(f['PartType0']['Masses'][:][sfgas],3).reshape([len(sfgas),3]),axis=0)/np.mean(f['PartType0']['Masses'][:][sfgas])
    gasvector=np.array(f['PartType0']['Coordinates'][sfgas,:])-gascm
    gcvector=gccoords-gascm
    #plt.scatter(gasvector[:,0],gasvector[:,1],c=f['PartType0']['Masses'][:][sfgas])
    #plt.scatter(gasvector[:,0],gasvector[:,1],c=f['PartType0']['Density'][:][sfgas])
    #plt.plot(f['PartType4']['Coordinates'][:,0]-gascm[0],f['PartType4']['Coordinates'][:,1]-gascm[1],'o')
    #plt.plot(xgc,ygc,'o')
    #plt.show()
    #print('gas',np.mean(np.array(f['PartType0']['Coordinates'][sfgas,0])-gascm[0]))

    newgasvector=crossmanyvectors.crossmanyvectors(gasvector,np.array(normal)/np.sqrt(np.sum(np.array(normal)**2)))
    newgcvector=crossmanyvectors.crossmanyvectors(gcvector,np.array(normal)/np.sqrt(np.sum(np.array(normal)**2)))

    gascoords2=coordinates.CartesianRepresentation(newgasvector[:,0],newgasvector[:,1],np.zeros(len(sfgas)))
    gccoords2=coordinates.CartesianRepresentation(newgcvector[:,0],newgcvector[:,1],np.zeros(len(distance)))

    gascoords3=coordinates.CartesianRepresentation(gasvector[:,0],gasvector[:,1],gasvector[:,2])
    gccoords3=coordinates.CartesianRepresentation(gcvector[:,0],gcvector[:,1],gcvector[:,2])
    
    basegascoords=coordinates.BaseCoordinateFrame(gascoords2)
    basegccoords=coordinates.BaseCoordinateFrame(gccoords2)

    basegascoords3=coordinates.BaseCoordinateFrame(gascoords3)
    basegccoords3=coordinates.BaseCoordinateFrame(gccoords3)

    print(basegccoords,basegascoords)
    matches2,d2d2,d3d2=coordinates.match_coordinates_3d(basegccoords,basegascoords,nthneighbor=nthneighbor)
    matches3,d2d3,d3d3=coordinates.match_coordinates_3d(basegccoords3,basegascoords3,nthneighbor=nthneighbor)

    sigmagas=3*np.array(f['PartType0']['Masses'][:][sfgas])[matches2]/h*1E10/np.pi/(d3d2/(1+redshift))**2 #Msun/kpc^2
    rhogas=3*np.array(f['PartType0']['Masses'][:][sfgas])[matches3]/h*1E10/(4*np.pi/3*(d3d3/(1+redshift))**3) #Msun/kpc^3

    #print('mt',np.sum(np.array(f['PartType0']['Masses'][:][sfgas])),np.sum(np.array(f['PartType4']['Masses'][:])))
    rhogas*=5E2
    rhoh=.5*m/(4/3*np.pi*1.5E-3**3)
    phiad=(1+9*(rhoh/rhogas/1E4))**-1.5

    fmol=(1+.025*(sigmagas/(1E2*1E6))**-2)**-1
    fsigma=3.92*((10-8*fmol)/2)**0.5

    tcce=0.176*(fsigma/4)**-1*(rhogas/(1E9))**-1.5*(m/1E5)*phiad**-1


    #t5evap=10*10**(-(1.03+zgc)/.5)
    #dmdt=(-m/tcce).value-(m/(t5evap*(m/1E5)**.7))
    t5evap=17/2
    dmdt=(-m/tcce).value-(m/(t5evap*(m/1E5)**.7))
   # print('f',fsigma,rhogas,m,phiad,d3d3,f['PartType0']['Density'][:][sfgas]*1E10/h*h**3*(1+redshift))

    #print('tc',0.176*(fsigma/4)**-1*(rhogas/(1E9))**-1.5*phiad**-1)
    #print('rhog',np.median(rhogas))

    try:
        if m<=0:
            return 0
    except:
        dmdt[np.where(m<=0)[0]]=0
    return dmdt
        
