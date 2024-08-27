import numpy as np
import h5py
import downloadcutout
import samplepower
import copy

def getrfromnumber(mgcs,rmin=1E-1,rmax=10,power=-3.5):

    totmass=len(mgcs)

    rho0=totmass*(power+3)/((rmax**(power+3)-rmin**(power+3)))
    
    randtst=np.random.rand(len(mgcs))*totmass
    return (randtst/rho0*(power+3)+rmin**(power+3))**(1/(power+3))


def getrfromnumber_plummer(mgcs,rh):

    randtst=np.random.rand(len(mgcs))
    return rh*randtst**(1.0/3)/np.sqrt(1-randtst**(2.0/3))

def getrfromnumberwithpos(mgcs,rs,rmin=1E-1,rmax=10,power=-3.5):

    totmass=len(mgcs)

    rho0=totmass*(power+3)/((rmax**(power+3)-rmin**(power+3)))
    
    randtst=np.random.rand(len(mgcs))*totmass
    #positions=(randtst/rho0*(power+3)+rmin**(power+3))**(1/(power+3))
    positions=rmax-rmax*np.random.power(-power,size=len(mgcs))

    repositions=np.repeat(positions,len(rs)).reshape(len(positions),len(rs))
    rers=np.tile(rs,len(positions)).reshape(len(positions),len(rs))
    #print(abs(repositions-rers))
    chosenrs=np.argmin(abs(repositions-rers),axis=1)
    return rs[chosenrs],chosenrs


def getrfromm(mgcs,rmin=1E-1,rmax=10,power=-3.5):

#    r=samplepower.samplepower(-3.5,xmin=1E-3,xmax=10,size=len(mgcs))
    
#    sortr=np.sort(r)
    r=np.zeros(len(mgcs))
    totmass=np.sum(mgcs)

    rho0=totmass*(power+3)/(rmax**(power+3)-rmin**(power+3))
#    masses=np.random.permutation(mgcs)
    cummass=0
    for i in range(len(mgcs)):
        cummass=cummass+mgcs[i]
        
        if i==0:
            r[i]=(rmin**(power+3)+mgcs[i]/totmass*(rmax**(power+3)-rmin**(power+3)))**(1/(power+3))
        else:
            r[i]=(r[i-1]**(power+3)+mgcs[i]/totmass*(rmax**(power+3)-rmin**(power+3)))**(1/(power+3))
    return r

def getrfrommwithpos(mgcs,rs,rmin=1E-1,rmax=10,power=-3.5,solve='cummass'):

#    r=samplepower.samplepower(-3.5,xmin=1E-3,xmax=10,size=len(mgcs))
    
#    sortr=np.sort(r)
    r=np.zeros(len(mgcs))
    rssort=np.argsort(rs)
    totmass=np.sum(mgcs)

    available=(np.ones(len(rs))+1).astype(np.bool)
    rho0=totmass*(power+3)/(rmax**(power+3)-rmin**(power+3))
#    masses=np.random.permutation(mgcs)
    cummass=0
    besti=0
    bestis=[]
    for i in range(len(mgcs)):
        cummass=cummass+mgcs[i]
#        print(cummass)
        if i==0:
            rbest=(rmin**(power+3)+mgcs[i]/totmass*(rmax**(power+3)-rmin**(power+3)))**(1/(power+3))
            besti=np.argmin(abs(rs[rssort]-rbest))
            bestis.append(rssort[besti])
            r[i]=rs[rssort[besti]]
            available[rssort[besti]]=False
        else:
            if solve=='cummass':
 #               print(cummass/totmass)
                rbest=(rmin**(power+3)+cummass/totmass*(rmax**(power+3)-rmin**(power+3)))**(1/(power+3))
            else:
                rbest=(r[i-1]**(power+3)+mgcs[i]/totmass*(rmax**(power+3)-rmin**(power+3)))**(1/(power+3))
            besti=np.argmin(abs(rs[rssort]-rbest))
            if len(mgcs)<=len(rs):
                for j in range(besti,len(rs[rssort])-1):
                    if available[rssort[besti]]:
                        break
                    besti=besti+1
            bestis.append(rssort[besti])
            r[i]=rs[rssort[besti]]
            available[rssort[besti]]=False
    return r,np.array(bestis)

def getgcparticles(h,mgcs,x=-1,y=-1,z=-1,rmin=1E-3,rmax=10,power=-3.5,alreadygc=[],part_type=['dm'],replace=False,redshift=0):

    if len(mgcs)==0:
        return np.array([]),np.array([])
    
    rs=getrfromm(mgcs)

    xcoords=np.array([])
    ycoords=np.array([])
    zcoords=np.array([])
    ids=np.array([])
    for itype in range(len(part_type)):
        if part_type[itype]=='dm':
            xcoords=np.append(xcoords,h['PartType1']['Coordinates'][:,0])
            ycoords=np.append(ycoords,h['PartType1']['Coordinates'][:,1])
            zcoords=np.append(zcoords,h['PartType1']['Coordinates'][:,2])
            ids=np.append(ids,h['PartType1']['ParticleIDs'][:])
        elif part_type[itype]=='star' and 'PartType4' in h.keys():
            wstar=np.where(h['PartType4']['GFM_StellarFormationTime'][:]>0)[0]
            xcoords=np.append(xcoords,h['PartType4']['Coordinates'][:,0][wstar])
            ycoords=np.append(ycoords,h['PartType4']['Coordinates'][:,1][wstar])
            zcoords=np.append(zcoords,h['PartType4']['Coordinates'][:,2][wstar])
            ids=np.append(ids,h['PartType4']['ParticleIDs'][:][wstar])
        if 'PartType4' not in h.keys():
            wstar=[]

            
    if x==-1:
        #x=np.mean(coords[:,0])
        if len(wstar)>0:
            x=np.mean(h['PartType4']['Coordinates'][:,0][wstar])
        else:
            xcoords=np.append(xcoords,h['PartType1']['Coordinates'][:,0])
            ycoords=np.append(ycoords,h['PartType1']['Coordinates'][:,1])
            zcoords=np.append(zcoords,h['PartType1']['Coordinates'][:,2])
            ids=np.append(ids,h['PartType1']['ParticleIDs'][:])
            x=np.mean(xcoords)
    if y==-1:
        if len(wstar)>0:
            y=np.mean(h['PartType4']['Coordinates'][:,1][wstar])
        else:
            xcoords=np.append(xcoords,h['PartType1']['Coordinates'][:,0])
            ycoords=np.append(ycoords,h['PartType1']['Coordinates'][:,1])
            zcoords=np.append(zcoords,h['PartType1']['Coordinates'][:,2])
            ids=np.append(ids,h['PartType1']['ParticleIDs'][:])
            y=np.mean(ycoords)
        #y=np.mean(coords[:,1])
    if z==-1:
        if len(wstar)>0:
            z=np.mean(h['PartType4']['Coordinates'][:,2][wstar])
        else:
            xcoords=np.append(xcoords,h['PartType1']['Coordinates'][:,0])
            ycoords=np.append(ycoords,h['PartType1']['Coordinates'][:,1])
            zcoords=np.append(zcoords,h['PartType1']['Coordinates'][:,2])
            ids=np.append(ids,h['PartType1']['ParticleIDs'][:])
            z=np.mean(zcoords)
        #z=np.mean(coords[:,2])

    wtochoose=np.arange(len(ids[:]))
    coords=np.array([xcoords,ycoords,zcoords]).T
    ids=np.array(ids)

    if not replace:
        for i in alreadygc:
            wi=np.where(i==ids[:][wtochoose])[0]
            if len(wi)==1:
                wtochoose=np.delete(wtochoose,wi[0])
            
        wtochoose=np.array([i for i in range(len(coords[:,0])) if ids[:][i] not in alreadygc])

        if len(wtochoose)<len(mgcs):
            reusegcs=np.random.choice(len(ids[:]),len(mgcs)-len(wtochoose))
            wtochoose=np.append(wtochoose,reusegcs).astype(np.int)

    distdm=((np.array(coords)[wtochoose,0]-x)**2+(np.array(coords)[wtochoose,1]-y)**2+(np.array(coords)[wtochoose,2]-z)**2)**.5

    particleids=[]
    #positions,bis=getrfromnumberwithpos(mgcs,distdm,power=-3.5,rmin=rmin,rmax=30)
    positions,bis=getrfromnumberwithpos(np.ones_like(mgcs),distdm,power=power,rmin=rmin,rmax=rmax)

    positions=positions/.6774/(1+redshift)

    particleids=np.array(ids[:][wtochoose[bis]])

    return particleids,positions
