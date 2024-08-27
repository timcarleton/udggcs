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
import getdmdtcce_pos
imp.reload(getdmdtcce_pos)
import getdmdtcce_em
import getdmdtcce_emwgas
imp.reload(getdmdtcce_em)
imp.reload(getdmdtcce_emwgas)
import profileclass
import getmwithinr
import os
imp.reload(getmwithinr)
import getperi
import profileclass
import getmrfromcall
imp.reload(getmrfromcall)
import getmrfromcpos
import matplotlib.pyplot as plt
import getfracabove
from colossus.cosmology import cosmology as colcosmology
from colossus.halo import concentration,mass_adv,mass_defs
import updatepositions
colcosmology.setCosmology('planck18')

gn=4.30091727e-6

#fsubs=fits.open('tngsubs1save.fits')
#fsubs=fits.open('tngsubsmass.fits')
#fsubs=fits.open('tngsubs1_udg.fits')
fsubs=fits.open('isodwarfstng.fits')

#p,po,poi,avgperi,omatperi=getperi.getperi(fsubs)

#coredat=np.loadtxt('finalcoredatid.txt',skiprows=1)
#coredatinf=np.loadtxt('initialcoredatid.txt',skiprows=1)

np.random.seed(2)
#gcobjects=np.array([22169,17648,17535,17602,17902, 801466,17429, 740794,17563, 17579,19776, 747214,17424, 17481,17370, 17338,17488, 17379,17470, 17292])
#gcobjects=np.array([17648,17535,17602,17902, 801466,17429, 740794,17563, 17579, 747214,17424, 17481,17370, 17338,17488, 17379,17470, 17292,17323,17470])
#gcobjects=np.array([17648,17535,17602,17902, 801466,17429, 740794,17563, 17579, 747214,17424, 17481,17370, 17338,17488, 17379])

np.random.seed(4)
mvbins=np.arange(9.2,10.5,.2)
gcobjects=[]

#for i in range(len(mvbins)-1):
#    winbin=np.where((fsubs[1].data.mvir[:,0]/.6774>10**mvbins[i]) & (fsubs[1].data.mvir[:,0]/.6774<10**mvbins[i+1]) & (zpk<.5) & (peaksfr<5))[0]
#    gcobjects=gcobjects+np.random.choice(fsubs[1].data.id0[winbin],10).tolist()

#w8=np.where((fsubs[1].data.mvir[:,0]/.6774>1.7E9) & (zpk<.5) & (peaksfr<5) & (fsubs[1].data.mstar[:,0]/.6774*1E10>1E7) & (fsubs[1].data.mstar[:,0]/.6774*1E10<1E9))[0]

#w8=[]
#for i in range(len(gcobjects)):
#    wid=np.where(gcobjects[i]==fsubs[1].data.id0)[0]
#    w8.append(wid[0])
#gcobjects=np.array([fsubs[1].data.id0[i] for i in w8])
#w8=np.where((fsubs[1].data.mstar[:,0]/.7*1E10<1E9) & (fsubs[1].data.mstar[:,0]/.7*1E10>1E7) & (fsubs[1].data.id0!=18727) & (peaksfr<5))[0]
w8=np.where((fsubs[1].data.mstar[:,0]/.7*1E10<1E9) & (fsubs[1].data.mstar[:,0]/.7*1E10>1E7) & (fsubs[1].data.id0!=18727) & (peaksfr<5) & (fsubs[1].data.m200host[:,0]/.7>2.5E14))[0]
#w8=np.where((fsubs[1].data.mstar[:,0]/.7*1E10<1E9) & (fsubs[1].data.mstar[:,0]/.7*1E10>5E8) & (fsubs[1].data.id0!=18727) & (peaksfr<5) )[0]
c1_mgas=1
c2_mgas=10
print('w',w8)

# w8=np.array([402,  180,  266,  191,  688, 1113,  272,  887,  590,  395, 1072,
#         299,  272,  486,  232,  757,  650,  492, 1042,  344,  334,  672,
#         401,  454,  306,  584, 1065,  893,  238,  882,  219,  213,  248,
#         967,  576,  772,  776,  290,  455,  695, 1095,  863,  483,  341,
#         207, 1090,  279,  239,  533, 1102,  235,  355,  492,  757,  472,
#         234,  679, 1069,  360, 1073,  615,  526, 1007,  560, 1037, 1045,
#         563,  940,  268,  430,  889,  909,  570, 1086,  284,  300,  228,
#         656,  198, 1007,  273,  575,  362,  680,  215,  753,  361,  554,
#         778,  287, 1095,  185,  246, 1113,  987,  773,  402,  280,  254,
#        1095])

#w8=np.array([198])

#w8=np.random.choice(w8,100)
#w9=np.random.choice(np.where((fsubs[1].data.mstar[:,0]/.7*1E10>1E9) & (fsubs[1].data.mstar[:,0]/.7*1E10<1E10))[0],5)
#w8=np.where((coredat[:,0]<1E9) & (coredat[:,0]>1E7)) [0]
deltatime=np.array([fsubs[1].data.time[0,i]-fsubs[1].data.time[0,i+1] for i in range(len(fsubs[1].data.time[0])-1)])
print('startinng')
def rungcs(c1,c2,c3):
    if os.path.isfile('tngsubgcdat_bymass_10e_tst.txt'):
        gcdat=np.loadtxt('tngsubgcdat_bymass_10e_tst.txt')
    
        totgcmassall=gcdat[:,1].tolist()
        finalgcfrac=gcdat[:,2].tolist()
        donesubs=w8[0:len(gcdat)].tolist()
        gcsurvive=gcdat[:,3].tolist()
        mediangcmass=gcdat[:,4].tolist()
        mediangcage=gcdat[:,5].tolist()
        peakgcmass=gcdat[:,6].tolist()

    else:
        r12number=[]
        r12inf=[]
        gcdat=[]
        totgcmassall=[]
        finalgcfrac=[]
        donesubs=[]
        gcsurvive=[]
        mediangcmass=[]
        mediangcage=[]
        peakgcmass=[]
        gcdat=[]
        totgcmassall=[]
        finalgcfrac=[]
        donesubs=[]
        gcsurvive=[]
        mediangcmass=[]
        mediangcage=[]
        peakgcmass=[]
        fracgtmedian=[]
        mdetect=[]
        ndetect=[]
    
    for sub in w8[len(gcdat):]:
        print(sub)

        totmass=[]
        fracvstime=[]
        extramass=0
        disruptedmass=0
        mgclist=[]
        gcparticles=[]
        gcpositions=[]
        mgcborn=[]
        gcage=[]
        zgc=[]
        mgasvstime=[]
        m1=[]
        r1=[]
        mgcvstime=[]
        
        
        mgcs=0
        wdat=np.where(coredatinf[:,0]==fsubs[1].data.id0[sub])[0][0]
        rstarinf=coredatinf[wdat,2]
        alphasubinf=coredatinf[wdat,3]
        betasubinf=coredatinf[wdat,4]
        gammasubinf=coredatinf[wdat,5]

        mvirinf=profileclass.getmassfromzhao0(alphasubinf,betasubinf,gammasubinf,coredatinf[wdat,6],coredatinf[wdat,7],coredatinf[wdat,8])
        m2new,r2new,c2new=mass_defs.changeMassDefinition(mvirinf,coredatinf[wdat,8]/coredatinf[wdat,7],zmax[sub],'vir','200c')
        deltacinf=np.log10(c2new)-np.log10(concentration.modelPrada12(m2new,zmax[sub]))

        mstarfinal=coredat[wdat,1]
        rstarfinal=coredat[wdat,2]
        alphasubfinal=coredat[wdat,3]
        betasubfinal=coredat[wdat,4]
        gammasubfinal=coredat[wdat,5]
        rho0final=coredat[wdat,6]
        rsfinal=coredat[wdat,7]
        
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

            #print('c1',rstarsubi)
            #timestep option 1: no gcs and no sfr
            if (~np.isfinite(fsubs[1].data.sfr[sub,99-cutout]) and len(gcpositions)==0) or (fsubs[1].data.sfr[sub,99-cutout]==0 and len(gcpositions)==0):
                fracvstime.append(0)
                mgasvstime.append(0)
                m1.append(0)
                r1.append(1)
                mgcvstime.append(0)
                if (99-cutout==imax[sub]):
                    peakgcmass.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))
                continue

            #for other timestep options, we need cutout data
            try:
                dir=downloadcutout.getcutout(fsubs[1].data.id0[sub],cutout,fsubs[1].data.id[sub,99-cutout])
                try:
                    subdat=h5py.File(dir)
                    tmp=subdat['PartType1']
                except:
                    os.system('rm '+dir)
                    try:
                        dir=downloadcutout.getcutout(fsubs[1].data.id0[sub],cutout,fsubs[1].data.id[sub,99-cutout])
                    except:
                        try:
                            dir=downloadcutout.getcutout(fsubs[1].data.id0[sub],cutout,fsubs[1].data.id[sub,99-cutout])
                        except:
                            fracvstime.append(np.nan)
                            mgasvstime.append(np.nan)
                            m1.append(np.nan)
                            r1.append(np.nan)
                            continue
                    subdat=h5py.File(dir)
                #print(sub,cutout,dir,fsubs[1].data.sfr[sub,99-cutout])

            #timestep option 3: no cutout info
            except (HTTPError,ReadTimeoutError,KeyError) as httpexception:
                print(fsubs[1].data.id0[i],cutout)
                #if this is after timestep 1, use previous info
                try:
                    fracvstime.append(fracvstime[-1])
                    mgasvstime.append(mgasvstime[-1])
                    m1.append(m1[-1])
                    r1.append(r1[-1])
                #else, use nan
                except:
                    fracvstime.append(np.nan)
                    mgasvstime.append(np.nan)
                    m1.append(np.nan)
                    r1.append(np.nan)
                if (99-cutout==imax[sub]):
                    peakgcmass.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))
                mgcvstime.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))
                continue

            
            #timestep option 2: some gcs, but indeterminate sfr
            if (fsubs[1].data.id[sub,99-cutout]==-99 and mgcs>0 or (~np.isfinite(fsubs[1].data.sfr[sub,99-cutout]))):
                if stripped:
                    gcmwithinr=getmrfromcpos.getmwithinr(gcpositions,mstarsubi,rstarsubi,alpha=alphasubi,beta=betasubi,gamma=gammasubi,rho0=rho0subi,rscale=rssubi,stripped=True)
                    
                else:
                    gcmwithinr=getmrfromcpos.getmwithinr(gcpositions,mstarsubi,rstarsubi,alpha=alphasubi,beta=betasubi,gamma=gammasubi,m200=m200subi,r200=r200subi,deltac=deltacsubi,redshift=fsubs[1].data.redshift[sub,99-cutout],stripped=False)
                #print(gcmwithinr,fsubs[1].data.mstar[sub,99-cutout]*np.array(gcpositions)**3/(np.array(gcpositions)**2+rstarsubi**2)**1.5)
                dmdt=getgcdmdt.getgcdmdt(np.array(mgclist),np.array(gcpositions),np.zeros_like(gcpositions),withiso=True)
                m1.append(np.median(np.sqrt(gn*gcmwithinr/gcpositions)))
                r1.append(np.median(gcpositions))
                disruptedmass-=np.nansum(dmdt*deltatime[99-cutout])
                mgclist=(np.array(mgclist)+dmdt*deltatime[99-cutout]).tolist()

                fracvstime.append(0)
                mgasvstime.append(0)
                if (99-cutout==imax[sub]):
                    peakgcmass.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))
                mgcvstime.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))
                continue


            updatepositions.updatepositions(gcpositions,gcparticles,subdat,[fsubs[1].data.x[sub,99-cutout],fsubs[1].data.y[sub,99-cutout],fsubs[1].data.z[sub,99-cutout]],fsubs[1].data.redshift[sub,99-cutout])
            
            #timestep option 4: no sfr, but some stars/gcs
            if (fsubs[1].data.sfr[sub,99-cutout]==0 and fsubs[1].data.mstar[sub,99-cutout]>0):
                if stripped:
                    gcmwithinr=getmrfromcpos.getmwithinr(gcpositions,mstarsubi,rstarsubi,alpha=alphasubi,beta=betasubi,gamma=gammasubi,rho0=rho0subi,rscale=rssubi,stripped=True)
                    
                else:
                    gcmwithinr=getmrfromcpos.getmwithinr(gcpositions,mstarsubi,rstarsubi,alpha=alphasubi,beta=betasubi,gamma=gammasubi,m200=m200subi,r200=r200subi,deltac=deltacsubi,redshift=fsubs[1].data.redshift[sub,99-cutout],stripped=False)

                if len(mgclist)>0:
                    m1.append(np.median(np.sqrt(gn*gcmwithinr/gcpositions)))
                    r1.append(np.median(gcpositions))
                else:
                    m1.append(np.median(np.sqrt(gn*mstarsubi/2/rstarsubi)))
                    r1.append(np.median(rstarsubi))
                if len(mgclist)>0:
                    dmdt=getgcdmdt.getgcdmdt(np.array(mgclist),np.array(gcpositions),np.zeros_like(gcpositions),withiso=True)
                    if 'PartType0' in subdat.keys():
                        if(len(subdat['PartType0']['ParticleIDs'])>0):
                            wsfgas=np.where(subdat['PartType0']['StarFormationRate'][:]>0)[0]
                            if len(wsfgas)>0:
                                dmdt-=np.array(mgclist)/(10**(c1*np.log10(np.sum(subdat['PartType0']['Masses'][:][wsfgas])/fsubs[1].data.mstar[sub,99-cutout])+c2*(fsubs[1].data.mstar[sub,99-cutout]/.6774*1E10)+c3))
                                wz=np.where(np.array(mgclist)<0)[0]
                                dmdt[wz]=0
                    disruptedmass-=np.nansum(dmdt*deltatime[99-cutout])
                    #wnz=np.where(np.array(mgclist)>0)[0]
                    mgclist=(np.array(mgclist)+dmdt*deltatime[99-cutout]).tolist()
                if (99-cutout==imax[sub]):
                    peakgcmass.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))
                mgcvstime.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))
                totmass.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))
                fracvstime.append(0)
                if 'PartType0' in subdat.keys():
                    if(len(subdat['PartType0']['ParticleIDs'])>0):
                        wsfgas=np.where(subdat['PartType0']['StarFormationRate'][:]>0)[0]
                        mgasvstime.append(np.sum(subdat['PartType0']['Masses'][:][wsfgas]))
                    else:
                        mgasvstime.append(0)
                else:
                    mgasvstime.append(0)
                continue

    #        dir='tngcutouts/cutout_'+str(fsubs[1].data.id0[sub])+'/tngcutout_'+str(99-cutout)+'.hdf5'

            #timestep option 5: we have covered the sfr=0 cases, so there is sfr, so we get the fraction born in gcs
            nangles=5
            try:
                fracs=[getgcfrac.getgcfrac(subdat,fsubs[1].data.redshift[sub,99-cutout],normal=[np.random.rand(),np.random.rand(),np.random.rand()]) for i in range(nangles)]
                fracvstime.append(np.mean(fracs))
            except:
                try:
                    fracvstime.append(fracvstime[-1])
                except:
                    fracvstime.append(np.nan)

            if 'PartType0' in subdat.keys():
                if(len(subdat['PartType0']['ParticleIDs'])>0):
                    wsfgas=np.where(subdat['PartType0']['StarFormationRate'][:]>0)[0]

#                    mc=max([10**(6-.5*np.log10(np.mean(subdat['PartType0']['GFM_Metallicity'][:][wsfgas]/.0127))),10**7])
                    mc=10**6
                    fracabove=getfracabove.getmfracabove(mc)
                    mgasvstime.append(fracabove*np.sum(subdat['PartType0']['Masses'][:][wsfgas]))
                    fracvstime[-1]=fracvstime[-1]*fracabove
                else:
                    wsfgas=[]
                    mgasvstime.append(0)
            else:
                mgasvstime.append(0)

            mgcs=fracvstime[-1]*fsubs[1].data.sfr[sub,99-cutout]*deltatime[99-cutout]*1E9
            if np.isfinite(mgcs):
#                gcs=samplegcdist.samplegcfrommass(mgcs+extramass,mc=max([10**(6.5+.7*np.log10(np.mean(subdat['PartType0']['GFM_Metallicity'][:][wsfgas]/.0127))),10**5.8]))
                if len(wsfgas)>0:
                    gcs=samplegcdist.samplegcfrommass(mgcs,mc=mc)
                else:
                    gcs=samplegcdist.samplegcfrommass(mgcs,mc=10**6.5)
                #gcs=samplegcdist.samplegcfrommass(mgcs+extramass,mc=10**6.5)
#                gcs=samplegcdist.samplegcfrommass(mgcs,mc=10**6.5)
                #gcs=np.array([mgcs])

                mgclist=mgclist+gcs.tolist()
                ngcs=len(gcs)
                extramass=mgcs+extramass-sum(gcs)
                mgcborn=mgcborn+gcs.tolist()

                if 'PartType0' in subdat.keys():
                    wsf=np.where(subdat['PartType0']['StarFormationRate'][:]>0)[0]
                    zgc=zgc+[np.log10(np.mean(subdat['PartType0']['GFM_Metallicity'][:][wsf]))]*ngcs

            if np.isfinite(mgcs):
                #positions=getgcparticles_type.getrfromnumber_plummer(gcs,2.5*rstarsubi)
                gcids,positions=getgcparticles_type.getgcparticles(subdat,gcs,rmin=1E-3,rmax=10,power=-3.5,alreadygc=gcpositions,part_type=['dm','star'],replace=True)
                gcparticles=gcparticles+gcids.tolist()
                gcpositions=gcpositions+positions.tolist()
                if stripped:
                    gcmwithinr=getmrfromcpos.getmwithinr(gcpositions,mstarsubi,rstarsubi,alpha=alphasubi,beta=betasubi,gamma=gammasubi,rho0=rho0subi,rscale=rssubi,stripped=True)
                else:
                    gcmwithinr=getmrfromcpos.getmwithinr(gcpositions,mstarsubi,rstarsubi,alpha=alphasubi,beta=betasubi,gamma=gammasubi,m200=m200subi,r200=r200subi,deltac=deltacsubi,redshift=fsubs[1].data.redshift[sub,99-cutout],stripped=False)
                if len(mgclist)>0:
                    m1.append(np.median(np.sqrt(gn*gcmwithinr/gcpositions)))
                    r1.append(np.median(gcpositions))
                else:
                    m1.append(np.median(np.sqrt(gn*mstarsubi/2/rstarsubi)))
                    r1.append(np.median(rstarsubi))
                if len(mgclist)>0:
                    dmdt=getgcdmdt.getgcdmdt(np.array(mgclist),np.array(gcpositions),np.zeros_like(gcpositions),withiso=True)

                    if 'PartType0' in subdat.keys():
                        if(len(subdat['PartType0']['ParticleIDs'])>0 and fsubs[1].data.mstar[sub,99-cutout]>0):
                            wsfgas=np.where(subdat['PartType0']['StarFormationRate'][:]>0)[0]
                            if len(wsfgas)>0:
                                dmdt-=np.array(mgclist)/(10**(c1*np.log10(np.sum(subdat['PartType0']['Masses'][:][wsfgas])/fsubs[1].data.mstar[sub,99-cutout])+c2*np.log10(fsubs[1].data.mstar[sub,99-cutout]/.6774*1E10)+c3))
                                wz=np.where(np.array(mgclist)<0)[0]
                                dmdt[wz]=0
                    disruptedmass-=np.nansum(dmdt*deltatime[99-cutout])

                    gcage=gcage+np.repeat(fsubs[1].data.time[sub,99-cutout],len(gcs)).tolist()
                    mgclist=(np.array(mgclist)+dmdt*deltatime[99-cutout]).tolist()
            totmass.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))

            if (99-cutout==imax[sub]):
                peakgcmass.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))
            mgcvstime.append(np.sum(np.array(mgclist)[np.where(np.array(mgclist)>0)[0]]))

            #plt.hist(np.log10(mgclist),range=[3,6])
            #plt.yscale('log')
            #plt.show()

        #finaldir=downloadcutout.getcutout(fsubs[1].data.id0[sub],99,fsubs[1].data.id[sub,0])
        #finalh=h5py.File(finaldir)
        if len(mgclist)==0 or len(np.where(np.array(mgclist)>0)[0])==0:
            totgcmassall.append(0)
            finalgcfrac.append(0)
            gcsurvive.append(0)
            mediangcmass.append(0)
            mediangcage.append(0)
            fracgtmedian.append(0)
            mdetect.append(0)
            ndetect.append(0)
            r12number.append(0)
            if len(gcpositions)>0:
                r12inf.append(np.sort(np.array(gcpositions))[len(gcpositions)//2])
            else:
                r12inf.append(0)
            print('none',r12number)
        else:

            finalh=subdat
            mgclist=np.array(mgclist)
            notstripped=np.array([gci for gci in range(len(gcparticles)) if (gcparticles[gci] in np.array(finalh['PartType1']['ParticleIDs'][:]) or gcparticles[gci] in np.array(finalh['PartType4']['ParticleIDs'][:]))])
            r12inf.append(np.sort(np.array(gcpositions))[len(gcpositions)//2])
            #notstripped=np.array([gci for gci in range(len(gcpositions)) ])
            if len(notstripped)==0:
                gcsurvive.append(0)
                totgcmass=0
                wnotdisolved=np.array([])
            else:
                wnotdisolved=np.where(mgclist[notstripped]>0)[0]
                gcsurvive.append(len(wnotdisolved))
                if len(wnotdisolved)==0:
                    totgcmass=0
                    print('nodisolved')
                    r12number.append(0)
                else:
                    totgcmass=np.sum(np.array(mgclist)[notstripped[wnotdisolved]])
                    dists=[]
                    for gci in range(len(wnotdisolved)):
                        wparticleidm=np.where(gcparticles[notstripped[wnotdisolved[gci]]]==np.array(finalh['PartType1']['ParticleIDs'][:]))[0]
                        wparticleist=np.where(gcparticles[notstripped[wnotdisolved[gci]]]==np.array(finalh['PartType4']['ParticleIDs'][:]))[0]
                        if len(wparticleidm)!=0:
                            dists.append(np.sqrt((finalh['PartType1']['Coordinates'][wparticleidm[0],0]/1000/.6774-fsubs[1].data.x[sub,0]/.6774)**2+
                                                 (finalh['PartType1']['Coordinates'][wparticleidm[0],1]/1000/.6774-fsubs[1].data.y[sub,0]/.6774)**2+
                                                 (finalh['PartType1']['Coordinates'][wparticleidm[0],2]/1000/.6774-fsubs[1].data.z[sub,0]/.6774)**2))

                        elif len(wparticleist)!=0:
                            dists.append(np.sqrt((finalh['PartType4']['Coordinates'][wparticleist[0],0]/1000/.6774-fsubs[1].data.x[sub,0]/.6774)**2+
                                                 (finalh['PartType4']['Coordinates'][wparticleist[0],1]/1000/.6774-fsubs[1].data.y[sub,0]/.6774)**2+
                                                 (finalh['PartType4']['Coordinates'][wparticleist[0],2]/1000/.6774-fsubs[1].data.z[sub,0]/.6774)**2))

                    sortedr=np.sort(dists)
                    wwithin10=np.where(np.array(sortedr)<10)[0]
                    sortedr=np.array(sortedr)[wwithin10]
                    r12number.append(sortedr[int(len(sortedr)/2)])
                    print('r12',r12number[-1],r12inf[-1])
                    print(len(gcparticles),len(notstripped))
            
            #plt.hist(np.log10(np.array(mgclist)[notstripped[wnotdisolved]]),range=[3,6],histtype='step')
            #plt.hist(np.log10(np.array(mgcborn)),range=[3,6],histtype='step')
            #plt.yscale('log')
            #plt.show()
            mediangcage.append(np.median(fsubs[1].data.time[0,0]-np.array(gcage)))
            fracgtmedian.append(len(np.where(mgclist[np.where(mgclist>0)[0]]>1.7E5)[0])/len(np.where(mgclist>0)[0]))
            mdetect.append(np.sum(mgclist[np.where(mgclist>1.5E5)[0]]))
            ndetect.append(len(np.where(mgclist>1.5E5)[0]))
                          
            print(len(np.where(mgclist[np.where(mgclist>0)[0]]>1.7E5)[0]),len(np.where(mgclist>0)[0]))
            if len(wnotdisolved)==0:
                mediangcmass.append(0)
            else:
                mediangcmass.append(np.median(np.array(mgclist)[notstripped[wnotdisolved]]))

            fracvstime=np.array(fracvstime)
            mgasvstime=np.array(mgasvstime)/.6774*1E10
            print(len(m1),len(mgasvstime))
            np.savetxt('gcinfo/subinfovstime_tst_'+str(fsubs[1].data.id0[sub])+'.txt',np.array([fracvstime,mgasvstime,fsubs[1].data.mstar[sub,0:-1][::-1]/.6774*1E10,fsubs[1].data.sfr[sub,0:-1][::-1],np.array(m1),np.array(r1),fsubs[1].data.redshift[sub,0:-1][::-1]]).T,header='fracgc\tmgas\tmstar\tsfr\tmed_period\tmed_pos\tz')
            np.savetxt('gcinfo/gcmassfunc_'+str(fsubs[1].data.id0[sub])+'.txt',mgclist[np.where(mgclist>0)[0]])
            np.savetxt('gcinfo/gcmassvstime_'+str(fsubs[1].data.id0[sub])+'.txt',np.array([fsubs[1].data.time[sub,0:-1][::-1],mgcvstime]).T,header='time\tmgc')
            print(fracvstime,mgasvstime)
        #totgcmass=np.nancumsum(fracvstime*fsubs[1].data.sfr[sub,0:-1][::-1]*deltatime[::-1]*1E9)
            totgcmassall.append(totgcmass)
    #    np.savetxt('tnggcmass_tngsubs_a.txt',totgcmassall)
            totstellarmass=np.nancumsum(fsubs[1].data.sfr[sub,0:-1][::-1]*deltatime[::-1])
        #finalgcfrac.append(totgcmass[-1]/totnongcmass[-1])
            finalgcfrac.append(totgcmass/(totstellarmass[-1]+disruptedmass))

        donesubs.append(sub)

        print(sub,len(totgcmassall),len(r12number))
    #    np.savetxt('tngsubgcfrac_tngsubs_a.txt',finalgcfrac)
        np.savetxt('tngsubgcdat_bymass_10_103_tst'+str(c1)+'_'+str(c2)+'_'+str(c3)+'_notide_6_dmsttrace_10_within10_update_wm_all.txt',np.array([fsubs[1].data.id0[np.array(donesubs)],totgcmassall,finalgcfrac,gcsurvive,np.array(mediangcmass),np.array(mediangcage),np.array(peakgcmass),np.array(fracgtmedian),np.array(mdetect),np.array(ndetect),np.array(r12number),np.array(r12inf)]).T)
    return totgcmassall

finalcoredat=np.loadtxt('finalcoredatid.txt',skiprows=1)
infcoredat=np.loadtxt('initialcoredatid.txt',skiprows=1)
wstrip=[]
wgc=[]
wgctides=[]

#gcobjects=np.array([794522, 17681, 18152, 18221, 18426, 18854, 17808, 18924, 19008, 17815, 17989, 17677, 17685, 18085, 17604, 17914, 17571, 703407, 17299, 17415])

for i in range(len(gcobjects)):
    wi=np.where(gcobjects[i]==fsubs[1].data.id0)[0]
    wgc.append(int(wi[0]))
    
for i in range(len(fsubs[1].data)):
    w=np.where(fsubs[1].data.id0[i]==finalcoredat[:,0])[0]
    wstrip.append(w[0])
wstrip=np.array(wstrip)
rmaxfinal=finalcoredat[wstrip,9]
vmaxfinal=finalcoredat[wstrip,10]
alpha=finalcoredat[wstrip,3]
beta=finalcoredat[wstrip,4]
gamma=finalcoredat[wstrip,5]
mstar=finalcoredat[wstrip,1]
stripped=np.ones(len(mstar))
for i in range(len(wgc)):
    stripped[wgc[i]]=finalcoredat[wstrip[wgc[i]],1]/infcoredat[wstrip[wgc[i]],1]
rstar=finalcoredat[wstrip,2]

mpeak=np.array([np.nanmax(fsubs[1].data.mvir[i]/.6774) for i in wgc])

from scipy.optimize import minimize

#bestfit=minimize(lambda p0:fitfunc(p0[0],p0[1]),[1,10])
#print(bestfit)
#minimize(lambda params: (np.log10(rungcs(params[0],params[1])*stripped[wgc])-np.log10(2.9E-5*mpeak))**2,[1,10])

#print(rungcs(-0.29605814,  0.28728233))
#print(rungcs(-0.75649842,  0.02320715,  0.14671275))
#print(rungcs(.5158,.3626,-.6211))
#print(rungcs( 1.48125487,  0.45817664, -0.73612078))
#print(rungcs(-1.42293966,  0.05432084, -0.24169949))
#print(rungcs(-0.158, 0.0815, -0.024))
#print(rungcs(0.14593688,  0.08848594, -0.09143344))
#print(rungcs(-0.24963473, -0.30311568,  3.01296165))
#print(rungcs(-0.16164743,  0.04189552, -0.00364701))
#print(rungcs(-1.29652035, -0.00778135,  0.10287339))
#print(rungcs(-0.12058949,  0.01970674,  0.03343717))
#print(rungcs(-0.12058949,  0.01970674,  0.3343717))
#print(rungcs(-0.4,  0,  0.3294))
print(rungcs(-0.4,  0,  0.25))
#print(rungcs(0.91548437, -0.21605789,  3.29890385))
#print(rungcs(0.93223353,  0.13345228,  0.39622082))
#print(rungcs(-0.77284531, -0.49819154,  4.14583802))
#print(rungcs(1.37744168, -0.05213953,  2.12363548))
#print(rungcs(-1.06581037, -0.36165371,  1.39982398))
#print(rungcs(-0.51774569, -0.13486223,  1.52751605))
#print(rungcs(1.29576834, -0.30828729,  3.9814053))
#print(rungcs(1.43397479, -0.49223985,  4.36038537))
#print(rungcs(2.38676354 ,-0.17759756 , 3.86963132))
#print(rungcs(0,0,4))
#print(rungcs(.13,.15,4.25))
c1=[-.5]
c2=[-.9]
c3=[.1]
c4=[9]

#fit=fitfunc(c1[0],c2[0],c3[0],c4[0])
