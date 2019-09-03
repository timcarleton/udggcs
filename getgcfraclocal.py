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
from astropy import coordinates
import getgcfrac as globalgcfrac
# From Mistani 2016



def getgcfrac(f,redshift,normal=[0,0,1],nthneighbor=3):

    sfgas=np.where(f['PartType0']['StarFormationRate'][:]>0)[0]
    if len(sfgas)<=1:
        return 0

    ngas=len(f['PartType0']['Masses'])
    if len(sfgas)<=nthneighbor:
        return globalgcfrac.getgcfrac(f,redshift,normal)
    gascm=np.mean(f['PartType0']['Coordinates'][:,:][sfgas,:]*np.repeat(f['PartType0']['Masses'][:][sfgas],3).reshape([len(sfgas),3]),axis=0)/np.mean(f['PartType0']['Masses'][:][sfgas])
    gasvector=f['PartType0']['Coordinates'][:,:]-gascm
    newgasvector=crossmanyvectors.crossmanyvectors(gasvector,np.array(normal)/np.sqrt(np.sum(np.array(normal)**2)))

    coords=coordinates.CartesianRepresentation(newgasvector[sfgas,0],newgasvector[sfgas,1],np.zeros(len(sfgas)))
    basecoords=coordinates.BaseCoordinateFrame(coords)
    matches,d2d,d3d=coordinates.match_coordinates_3d(basecoords,basecoords,nthneighbor=nthneighbor+1)

    sigmasfr=np.sum(3*f['PartType0']['StarFormationRate'][:][sfgas]*f['PartType0']['StarFormationRate'][:][sfgas]/np.pi/(d3d/(1+redshift))**2)/np.sum(f['PartType0']['StarFormationRate'][:][sfgas])
    
    
    
    return 0.29*sigmasfr.value**0.24
