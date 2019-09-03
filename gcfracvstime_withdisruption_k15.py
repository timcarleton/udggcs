from python import getgcfraclocal as getgcfrac
import imp
imp.reload(getgcfrac)
from astropy.io import fits
import numpy as np
from python import downloadcutout
imp.reload(downloadcutout)
from requests import HTTPError
from urllib3.exceptions import ReadTimeoutError
import samplegcdist
imp.reload(samplegcdist)
import getgcparticles_type
imp.reload(getgcparticles_type)
import h5py
import getgcdmdt
imp.reload(getgcdmdt)
import getdmdtcce
imp.reload(getdmdtcce)
import profileclass
import getmwithinr
import os
imp.reload(getmwithinr)
import getperi
import profileclass
import getmrfromcall
imp.reload(getmrfromcall)
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology as colcosmology
from colossus.halo import concentration,mass_adv,mass_defs
colcosmology.setCosmology('planck18')

gn=4.30091727e-6

#fsubs=fits.open('tngsubs1save.fits')
fsubs=fits.open('tngsubs1mass.fits')
#fsubs=fits.open('tngsubs1_udg.fits')
#fsubs=fits.open('isodwarfstng.fits')

p,po,poi,avgperi,omatperi=getperi.getperi(fsubs)

coredat=np.loadtxt('finalcoredatid.txt',skiprows=1)
coredatinf=np.loadtxt('initialcoredatid.txt',skiprows=1)

zmax=[]
imax=[]
for i in range(len(fsubs[1].data)):

    if fsubs[1].data.mvir[i,0]>0:
        imax.append(np.nanargmax(fsubs[1].data.mvir[i]))
        zmax.append(fsubs[1].data.redshift[i,np.nanargmax(fsubs[1].data.mvir[i])])
    else:
        imax.append(0)
        zmax.append(0)
imax=np.array(imax)
zmax=np.array(zmax)
w8=np.where((fsubs[1].data.mstar[:,0]/.7*1E10<1E9) & (fsubs[1].data.mstar[:,0]/.7*1E10>1E7) & (fsubs[1].data.id0!=18727)) [0][500:]
#w8=np.where((coredat[:,0]<1E9) & (coredat[:,0]>1E7)) [0]
deltatime=np.array([fsubs[1].data.time[0,i]-fsubs[1].data.time[0,i+1] for i in range(len(fsubs[1].data.time[0])-1)])

if os.path.isfile('tngsubgcdat_any_k15_newgcmc6r0a.txt'):
    gcdat=np.loadtxt('tngsubgcdat_any_k15_newgcmc6r0a.txt')
    
    totgcmassall=gcdat[:,1].tolist()
    finalgcfrac=gcdat[:,2].tolist()
    donesubs=w8[0:len(gcdat)].tolist()
    gcsurvive=gcdat[:,3].tolist()
    mediangcmass=gcdat[:,4].tolist()
    mediangcage=gcdat[:,5].tolist()

else:
    gcdat=[]
    totgcmassall=[]
    finalgcfrac=[]
    donesubs=[]
    gcsurvive=[]
    mediangcmass=[]
    mediangcage=[]
    
for sub in w8[len(gcdat):]:

    fracvstime=[]
    extramass=0
    disruptedmass=0
    mgclist=[]
    gcparticles=[]
    gcpositions=[]
    mgcborn=[]
    gcage=[]
    zgc=[]
    mgcs=0
    
    rstarinf=coredatinf[sub,2]
    alphasubinf=coredatinf[sub,3]
    betasubinf=coredatinf[sub,4]
    gammasubinf=coredatinf[sub,5]

    mvirinf=profileclass.getmassfromzhao0(alphasubinf,betasubinf,gammasubinf,coredatinf[sub,6],coredatinf[sub,7],coredatinf[sub,8])
    m2new,r2new,c2new=mass_defs.changeMassDefinition(mvirinf,coredatinf[sub,8]/coredatinf[sub,7],zmax[sub],'vir','200c')
    deltacinf=np.log10(c2new)-np.log10(concentration.modelPrada12(m2new,zmax[sub]))

    mstarfinal=coredat[sub,1]
    rstarfinal=coredat[sub,2]
    alphasubfinal=coredat[sub,3]
    betasubfinal=coredat[sub,4]
    gammasubfinal=coredat[sub,5]
    rho0final=coredat[sub,6]
    rsfinal=coredat[sub,7]
    
    for cutout in np.arange(1,100):
    #for cutout in np.arange(1,30):

        if len(poi[sub])==0:
            mstarsubi=fsubs[1].data.mstar[sub,99-cutout]/.68*1E10
            m200subi=fsubs[1].data.m200[sub,99-cutout]/.68
            r200subi=fsubs[1].data.r200[sub,99-cutout]/.68/(1+fsubs[1].data.redshift[sub,99-cutout])
            rstarsubi=rstarinf
            alphasubi=alphasubinf
            betasubi=betasubinf
            gammasubi=gammasubinf
            deltacsubi=deltacinf
            stripped=False

        else:
            if cutout<99-poi[sub][0]:

                mstarsubi=fsubs[1].data.mstar[sub,99-cutout]/.68*1E10
                m200subi=fsubs[1].data.m200[sub,99-cutout]/.68
                r200subi=fsubs[1].data.r200[sub,99-cutout]/.68/(1+fsubs[1].data.redshift[sub,99-cutout])
                rstarsubi=rstarinf
                alphasubi=alphasubinf
                betasubi=betasubinf
                gammasubi=gammasubinf
                deltacsubi=deltacinf
                stripped=False

            else:
                mstarsubi=mstarfinal
                rstarsubi=rstarfinal
                alphasubi=alphasubfinal
                betasubi=betasubfinal
                gammasubi=gammasubfinal
                rho0subi=rho0final
                rssubi=rsfinal
                stripped=True

        
        if (~np.isfinite(fsubs[1].data.sfr[sub,99-cutout]) and mgcs==0) or (fsubs[1].data.sfr[sub,99-cutout]==0 and mgcs==0):
            fracvstime.append(0)
            continue

        if (fsubs[1].data.id[sub,99-cutout]==-99 and mgcs>0):
            if stripped:
                gcmwithinr,currentdist=getmrfromcall.getmwithinr(gcparticles,subdat,mstarsubi,rstarsubi,r0=np.array(gcpositions),alpha=alphasubi,beta=betasubi,gamma=gammasubi,rho0=rho0subi,rscale=rssubi,stripped=True)
            else:
                gcmwithinr,currentdist=getmrfromcall.getmwithinr(gcparticles,subdat,mstarsubi,rstarsubi,r0=np.array(gcpositions),alpha=alphasubi,beta=betasubi,gamma=gammasubi,m200=m200subi,r200=r200subi,deltac=deltacsubi,redshift=fsubs[1].data.redshift[sub,99-cutout],stripped=False)
            dmdt=getgcdmdt.getgcdmdt(np.array(mgclist),np.array(gcpositions),np.sqrt(gn*gcmwithinr/gcpositions),withiso=False)
            disruptedmass-=np.nansum(dmdt*deltatime[99-cutout])
            mgclist=(np.array(mgclist)+dmdt*deltatime[99-cutout]).tolist()
            continue

        
        try:
            dir=downloadcutout.getcutout(fsubs[1].data.id0[sub],cutout,fsubs[1].data.id[sub,99-cutout])
            try:
                subdat=h5py.File(dir)
                tmp=subdat['PartType1']
            except:
                os.system('rm '+dir)
                dir=downloadcutout.getcutout(fsubs[1].data.id0[sub],cutout,fsubs[1].data.id[sub,99-cutout])
                subdat=h5py.File(dir)
            print(sub,cutout,dir)
        
        except (HTTPError,ReadTimeoutError) as httpexception:
            print(fsubs[1].data.id0[i],cutout)
            try:
                fracvstime.append(fracvstime[-1])
            except:
                fracvstime.append(np.nan)
            continue

        if (fsubs[1].data.sfr[sub,99-cutout]==0 and fsubs[1].data.mstar[sub,99-cutout]>0):
            if stripped:
                gcmwithinr,currentdist=getmrfromcall.getmwithinr(gcparticles,subdat,mstarsubi,rstarsubi,r0=np.array(gcpositions),alpha=alphasubi,beta=betasubi,gamma=gammasubi,rho0=rho0subi,rscale=rssubi,stripped=True)
            else:
                gcmwithinr,currentdist=getmrfromcall.getmwithinr(gcparticles,subdat,mstarsubi,rstarsubi,r0=np.array(gcpositions),alpha=alphasubi,beta=betasubi,gamma=gammasubi,m200=m200subi,r200=r200subi,deltac=deltacsubi,redshift=fsubs[1].data.redshift[sub,99-cutout],stripped=False)


            if len(mgclist)>0:
                dmdt=getgcdmdt.getgcdmdt(np.array(mgclist),np.array(gcpositions),np.sqrt(gn*gcmwithinr/gcpositions),withiso=False)
                dmdt+=getdmdtcce.getgcdmdtcce(subdat,gcparticles,np.array(mgclist),fsubs[1].data.redshift[sub,99-cutout],np.max([np.zeros_like(zgc)-1.1,np.array(zgc)],axis=0))
                disruptedmass-=np.nansum(dmdt*deltatime[99-cutout])
                mgclist=(np.array(mgclist)+dmdt*deltatime[99-cutout]).tolist()
            continue
        
#        dir='tngcutouts/cutout_'+str(fsubs[1].data.id0[sub])+'/tngcutout_'+str(99-cutout)+'.hdf5'

        nangles=5
        try:
            fracs=[getgcfrac.getgcfrac(subdat,fsubs[1].data.redshift[sub,99-cutout],normal=[np.random.rand(),np.random.rand(),np.random.rand()]) for i in range(nangles)]
            fracvstime.append(np.mean(fracs))
        except:
            try:
                fracvstime.append(fracvstime[-1])
            except:
                fracvstime.append(np.nan)

        mgcs=fracvstime[-1]*fsubs[1].data.sfr[sub,99-cutout]*deltatime[99-cutout]*1E9
        if np.isfinite(mgcs):
            wsf=np.where(subdat['PartType0']['StarFormationRate'][:]>0)[0]
            #gcs=samplegcdist.samplegcfrommass(mgcs+extramass,mc=max([10**(6.5+.7*np.log10(np.mean(subdat['PartType0']['GFM_Metallicity'][:][wsf]/.0127))),2E5]))
            gcs=samplegcdist.samplegcfrommass(mgcs+extramass,mc=9E6)
            
            mgclist=mgclist+gcs.tolist()
            ngcs=len(gcs)
            extramass=mgcs+extramass-sum(gcs)
            mgcborn=mgcborn+gcs.tolist()
            zgc=zgc+[np.log10(np.mean(subdat['PartType0']['GFM_Metallicity'][:][wsf]))]*ngcs
        if np.isfinite(mgcs):
            particles,positions=getgcparticles_type.getgcparticles(subdat,gcs,alreadygc=gcparticles,part_type=['dm','star'])
            gcparticles=gcparticles+particles.tolist()
            gcpositions=gcpositions+positions.tolist()
            if stripped:
                gcmwithinr,currentpositions=getmrfromcall.getmwithinr(gcparticles,subdat,mstarsubi,rstarsubi,r0=np.array(gcpositions),alpha=alphasubi,beta=betasubi,gamma=gammasubi,rho0=rho0subi,rscale=rssubi,stripped=True)
            else:
                gcmwithinr,currentpositions=getmrfromcall.getmwithinr(gcparticles,subdat,mstarsubi,rstarsubi,r0=np.array(gcpositions),alpha=alphasubi,beta=betasubi,gamma=gammasubi,m200=m200subi,r200=r200subi,deltac=deltacsubi,redshift=fsubs[1].data.redshift[sub,99-cutout],stripped=False)

            if len(mgclist)>0:
                dmdt=getgcdmdt.getgcdmdt(np.array(mgclist),np.array(gcpositions),np.sqrt(gn*gcmwithinr/gcpositions),withiso=False)
                print('a',dmdt)
                dmdt+=getdmdtcce.getgcdmdtcce(subdat,gcparticles,np.array(mgclist),fsubs[1].data.redshift[sub,99-cutout],np.max([np.zeros_like(zgc)-1.1,np.array(zgc)],axis=0))
                print('b',dmdt)
                disruptedmass-=np.nansum(dmdt*deltatime[99-cutout])
                
                gcage=gcage+np.repeat(fsubs[1].data.time[sub,99-cutout],len(gcs)).tolist()
                
                mgclist=(np.array(mgclist)+dmdt*deltatime[99-cutout]).tolist()

    
    #finaldir=downloadcutout.getcutout(fsubs[1].data.id0[sub],99,fsubs[1].data.id[sub,0])
    #finalh=h5py.File(finaldir)
    if len(mgclist)==0:
        totgcmassall.append(0)
        finalgcfrac.append(0)
        gcsurvive.append(0)
        mediangcmass.append(0)
        mediangcage.append(0)

    else:
        
        finalh=subdat
        mgclist=np.array(mgclist)
        notstripped=np.array([gci for gci in range(len(gcparticles)) if gcparticles[gci] in np.array(finalh['PartType1']['ParticleIDs'][:])])
        if len(notstripped)==0:
            gcsurvive.append(0)
            totgcmass=0
        else:
            wnotdisolved=np.where(mgclist[notstripped]>0)[0]
            gcsurvive.append(len(wnotdisolved))
            if len(wnotdisolved)==0:
                totgcmass=0
            else:
                totgcmass=np.sum(np.array(mgclist)[notstripped[wnotdisolved]])

        
        #plt.hist(np.log10(np.array(mgclist)[notstripped[wnotdisolved]]),range=[3,6],histtype='step')
        #plt.hist(np.log10(np.array(mgcborn)),range=[3,6],histtype='step')
        #plt.yscale('log')
        #plt.show()
        mediangcage.append(np.median(fsubs[1].data.time[0,0]-np.array(gcage)))
        if len(wnotdisolved)==0:
            mediangcmass.append(0)
        else:
            mediangcmass.append(np.median(np.array(mgclist)[notstripped[wnotdisolved]]))
    
        fracvstime=np.array(fracvstime)
    #totgcmass=np.nancumsum(fracvstime*fsubs[1].data.sfr[sub,0:-1][::-1]*deltatime[::-1]*1E9)
        totgcmassall.append(totgcmass)
#    np.savetxt('tnggcmass_tngsubs_a.txt',totgcmassall)
        totstellarmass=np.nancumsum(fsubs[1].data.sfr[sub,0:-1][::-1]*deltatime[::-1]*1E9)
    #finalgcfrac.append(totgcmass[-1]/totnongcmass[-1])
        finalgcfrac.append(totgcmass/(totstellarmass[-1]+disruptedmass))

    donesubs.append(sub)

#    np.savetxt('tngsubgcfrac_tngsubs_a.txt',finalgcfrac)
    np.savetxt('tngsubgcdat_any_k15_newgcmc9r0b.txt',np.array([fsubs[1].data.id0[np.array(donesubs)],totgcmassall,finalgcfrac,gcsurvive,np.array(mediangcmass),np.array(mediangcage)]).T,header='id0\tTot GC Mass\tGC mass/M*\tNGC\tMedian Mass\tMedian Age')
