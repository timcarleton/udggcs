import numpy as np
from astropy import coordinates
import crossmanyvectors
import matplotlib.pyplot as plt

def getgcdmdtcce(f,ids,m,redshift,zgc,part_type=['dm','star'],nthneighbor=3,h=0.6774 ,normal=np.array([0,0,1])):

    sfgas=np.where(f['PartType0']['StarFormationRate'][:]>0)[0]
    if len(sfgas)<=1:
        return 0

    xcoordsgc=np.zeros(len(ids))
    ycoordsgc=np.zeros(len(ids))
    zcoordsgc=np.zeros(len(ids))
    for itype in range(len(part_type)):
        if part_type[itype]=='dm':
            for ii in range(len(ids)):
                wgcdm=np.where(np.array(f['PartType1']['ParticleIDs'][:]).astype(np.uint64)==np.uint64(int(ids[ii])))[0]
                if len(wgcdm)==1:
                    xcoordsgc[ii]=f['PartType1']['Coordinates'][wgcdm[0],0]
                    ycoordsgc[ii]=f['PartType1']['Coordinates'][wgcdm[0],1]
                    zcoordsgc[ii]=f['PartType1']['Coordinates'][wgcdm[0],2]
        elif part_type[itype]=='star':
            for ii in range(len(ids)):
                wgcst=np.where(np.uint64(int(ids[ii]))==np.array(f['PartType4']['ParticleIDs'][:]).astype(np.uint64))[0]
                if len(wgcst)==1:
                    xcoordsgc[ii]=f['PartType4']['Coordinates'][wgcst[0],0]
                    ycoordsgc[ii]=f['PartType4']['Coordinates'][wgcst[0],1]
                    zcoordsgc[ii]=f['PartType4']['Coordinates'][wgcst[0],2]

    gccoords=np.array([xcoordsgc,ycoordsgc,zcoordsgc]).T
    ngas=len(f['PartType0']['Masses'])
    if len(sfgas)<=nthneighbor:
        return 0
    
    gascm=np.mean(f['PartType0']['Coordinates'][sfgas,:]*np.repeat(f['PartType0']['Masses'][:][sfgas],3).reshape([len(sfgas),3]),axis=0)/np.mean(f['PartType0']['Masses'][:][sfgas])
    gasvector=np.array(f['PartType0']['Coordinates'][sfgas,:])-gascm
    gcvector=gccoords-gascm

    newgasvector=crossmanyvectors.crossmanyvectors(gasvector,np.array(normal)/np.sqrt(np.sum(np.array(normal)**2)))
    newgcvector=crossmanyvectors.crossmanyvectors(gcvector,np.array(normal)/np.sqrt(np.sum(np.array(normal)**2)))

    gascoords2=coordinates.CartesianRepresentation(newgasvector[:,0],newgasvector[:,1],np.zeros(len(sfgas)))
    gccoords2=coordinates.CartesianRepresentation(newgcvector[:,0],newgcvector[:,1],np.zeros(len(ids)))

    gascoords3=coordinates.CartesianRepresentation(gasvector[:,0],gasvector[:,1],gasvector[:,2])
    gccoords3=coordinates.CartesianRepresentation(gcvector[:,0],gcvector[:,1],gcvector[:,2])
    
    basegascoords=coordinates.BaseCoordinateFrame(gascoords2)
    basegccoords=coordinates.BaseCoordinateFrame(gccoords2)

    basegascoords3=coordinates.BaseCoordinateFrame(gascoords3)
    basegccoords3=coordinates.BaseCoordinateFrame(gccoords3)

    matches2,d2d2,d3d2=coordinates.match_coordinates_3d(basegccoords,basegascoords,nthneighbor=nthneighbor)
    matches3,d2d3,d3d3=coordinates.match_coordinates_3d(basegccoords3,basegascoords3,nthneighbor=nthneighbor)

    sigmagas=3*np.array(f['PartType0']['Masses'][:][sfgas])[matches2]/h*1E10/np.pi/(d3d2/(1+redshift))**2 #Msun/kpc^2
    rhogas=3*np.array(f['PartType0']['Masses'][:][sfgas])[matches3]/h*1E10/(4*np.pi/3*(d3d3/(1+redshift))**3) #Msun/kpc^3

    rhoh=.5*m/(4/3*np.pi*1.5E-3**3)
    phiad=(1+9*(rhoh/rhogas/1E4))**-1.5

    fmol=(1+.025*(sigmagas/(1E2*1E6))**-2)**-1
    fsigma=3.92*((10-8*fmol)/2)**0.5

    tcce=0.176*(fsigma/4)**-1*(rhogas/(1E9))**-1.5*(m/1E5)*phiad**-1


    #t5evap=10*10**(-(1.03+zgc)/.5)
    #dmdt=(-m/tcce).value-(m/(t5evap*(m/1E5)**.7))
    t5evap=17/2
    dmdt=(-m/tcce).value-m/(t5evap*(m/1E5))
    print(tcce)

    try:
        if m<=0:
            return 0
    except:
        dmdt[np.where(m<=0)[0]]=0
    return dmdt
        
