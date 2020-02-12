from astropy.io import fits
import numpy as np
import sampleschecter
import samplegcdist
import imp
imp.reload(samplegcdist)
import getgcdmdt
import bindata
import profileclass
import bindata_clip
from astropy.io import ascii
from scipy import stats
from astropy import stats as astats

def getmrcore(r,rho0,rs,alpha=1,beta=3):

    if alpha==1 and beta==3:
        return 4*np.pi*rho0*rs**3*(-(r*(3*r + 2*rs))/(2.*(r + rs)**2) + np.log((r + rs)/rs))
    else:
        return profileclass.getmassfromzhao0(alpha,beta,0,rho0,rs,r)

    
fsubs=fits.open('tngsubs1mass.fits')

wok=np.where(fsubs[1].data.mvir[:,0]>1E10)[0]
wdo=np.random.choice(wok,100)
time=fsubs[1].data.time[0]
deltatime=np.array([fsubs[1].data.time[0,i]-fsubs[1].data.time[0,i+1] for i in range(len(fsubs[1].data.time[0])-1)])

deltatime=deltatime[::-1]
finalcoredat=np.loadtxt('finalcoredatid.txt',skiprows=1)
coredatinf=np.loadtxt('initialcoredatid.txt',skiprows=1)

lim=ascii.read('limgcswother.csv')
limgc=2.3E5*lim['N_GC']
limgc[np.where(limgc<=0)[0]]=0
limmstar=0.6721434841599994-((lim['M_V']-4.83)/2.5)
limre=lim['R_egal']
sigmastarlim=10**limmstar/np.pi/limre**2
wudglim=np.where((limre>1.5) & (limre<7) & (sigmastarlim>1.73E6) & (sigmastarlim<1.73E7))[0]
wnotudglim=np.array([i for i in range(len(limmstar)) if i not in wudglim])
wnotudglim=wnotudglim[np.where((10**limmstar[wnotudglim]>1E7) & (10**limmstar[wnotudglim]<1E9) )[0]]


np.random.seed(2)
def getgcs(c1,c2,c3,obj):

    mgci=0

    try:
        objinfo=np.loadtxt('gcinfo/subinfovstime_'+str(obj)+'.txt')
    except:
        return np.nan

    gcs=np.array([])
    gcvstime=[]
    timei=[]
    for i in range(len(objinfo)):
        newgcs=objinfo[i][0]*objinfo[i][3]*deltatime[i]*1E9
        if newgcs>0:
            if newgcs>0:
                #gcsj=sampleschecter.sampleschecter(-1.99,10**6.5,10000,int(1.94E-5*newgcs))
            #np.random.seed(2)
                gcsj=samplegcdist.samplegcfrommass(newgcs,mc=10**6.5,slope=-1.99)
                gcs=np.append(gcs,gcsj)
            #print(i,objinfo[i][0],deltatime[i]/1E9,newgcs,objinfo[i][3],gcsj)
        wnz=np.where(gcs>0)
        #print(np.sum(gcs[wnz]))
        mwithinr=objinfo[i][4]
        dmdt=getgcdmdt.getgcdmdt(np.array(gcs),objinfo[i][5],objinfo[i][4],withiso=True)
        #print(gcs)
        if objinfo[i][1]>0 and objinfo[i][2]>0:
            #print(c1*np.log10(objinfo[i][1]/objinfo[i][2])+c2*np.log10(objinfo[i][2])+c3)
            #print(objinfo[i][1])
            dmdt-=gcs/(10**(c1*np.log10(objinfo[i][1]/objinfo[i][2])+c2*np.log10(objinfo[i][2])+c3))


        try:
            gcs[wnz]+=dmdt[wnz]*deltatime[i]
        except TypeError:
            gcs+=dmdt*deltatime[i]
        gcvstime.append(np.sum(gcs))
        timei.append(time[i])
    wnz=np.where(gcs>0)[0]
    np.savetxt('gcinfo/gcmassvstime_testb_'+str(obj)+'.txt',np.array([timei,gcvstime]).T)
    return np.sum(gcs[wnz])

np.random.seed(2)
#objects=np.random.choice(objects,20)


wstrip=[]
for i in range(len(fsubs[1].data)):
    w=np.where(fsubs[1].data.id0[i]==finalcoredat[:,0])[0]
    wstrip.append(w[0])

zmax=[]
imax=[]
peakgcmass=[]
peaksfr=[]
for i in range(len(fsubs[1].data)):

    if fsubs[1].data.mvir[i,0]>0:
        imax.append(np.nanargmax(fsubs[1].data.mvir[i]))
        zmax.append(fsubs[1].data.redshift[i,np.nanargmax(fsubs[1].data.mvir[i])])
        peaksfr.append(np.nanmax(fsubs[1].data.sfr[i]))
    else:
        imax.append(0)
        zmax.append(0)
        peaksfr.append(0)
imax=np.array(imax)
zmax=np.array(zmax)
peaksfr=np.array(peaksfr)

    
wstrip=np.array(wstrip)
rmaxfinal=finalcoredat[wstrip,9]
vmaxfinal=finalcoredat[wstrip,10]
rmaxinf=coredatinf[wstrip,9]
vmaxinf=coredatinf[wstrip,10]
mstarinf=coredatinf[wstrip,1]
rstarinf=coredatinf[wstrip,2]
alphainf=coredatinf[wstrip,3]
betainf=coredatinf[wstrip,4]
gammainf=coredatinf[wstrip,5]
rvirinf=coredatinf[wstrip,8]
rho0inf=coredatinf[wstrip,6]
rscaleinf=coredatinf[wstrip,7]
alpha=finalcoredat[wstrip,3]
beta=finalcoredat[wstrip,4]
gamma=finalcoredat[wstrip,5]
mstar=finalcoredat[wstrip,1]
#mstar=fsubs[1].data.mstar[:,0]*1E10/.7
rstar=finalcoredat[wstrip,2]
rvir=finalcoredat[wstrip,8]
rho0=finalcoredat[wstrip,6]
rscale=finalcoredat[wstrip,7]
mstar=finalcoredat[wstrip,1]
rstar=finalcoredat[wstrip,2]
stripped=np.ones(len(mstar))
for i in range(len(stripped)):
    stripped[i]=finalcoredat[wstrip[i],1]/coredatinf[wstrip[i],1]


mhaloinf=np.array([getmrcore(rvirinf[i],rho0inf[i],rscaleinf[i],alpha=1,beta=betainf[i]) for i in range(len(wstrip))])
mhalo=np.array([getmrcore(rvir[i],rho0[i],rscale[i],alpha=1,beta=beta[i]) for i in range(len(wstrip))])

#w8=w8[np.where(fsubs[1].data.mvir[w8,0]/.6774*2.9E-5>3E5)[0]]


wudgtides=np.where((mstar/np.pi/rstar**2>1.73E6) & (mstar/np.pi/rstar**2<1.73E7) & (rstar>1.5) & (rstar<7) & (fsubs[1].data.id0!=18727) & (peaksfr<5) & (fsubs[1].data.id0!=17625) & (fsubs[1].data.id0!=506884) & (mstar<1E9) & (mstar>1E7) & (fsubs[1].data.id0!=1120045) & (fsubs[1].data.mstar[:,0]/.7*1E10<1E9) & (fsubs[1].data.mstar[:,0]/.7*1E10>1E7)& (fsubs[1].data.id0!=19776))[0]
print(fsubs[1].data.mstar[wudgtides,0]/.7*1E10<1E9)
wnotudg=np.array([i for i in range(len(mstar)) if i not in wudgtides])
wnotudg=wnotudg[np.where( (fsubs[1].data.id0[wnotudg]!=18727) & (peaksfr[wnotudg]<5) & (fsubs[1].data.id0[wnotudg]!=17625) & (fsubs[1].data.id0[wnotudg]!=506884)& (mstar[wnotudg]<1E9) & (mstar[wnotudg]>1E7) & (fsubs[1].data.id0[wnotudg]!=1120045) & (fsubs[1].data.mstar[wnotudg,0]/.7*1E10<1E9)& (fsubs[1].data.mstar[wnotudg,0]/.7*1E10>1E7)& (fsubs[1].data.id0[wnotudg]!=19776))[0]]


#wnotsample=np.random.choice(len(wnotudg),100)
wnotsample=np.arange(len(wnotudg))

objectsnot=np.array([fsubs[1].data.id0[j] for j in wnotudg[wnotsample]])
objectsudg=np.array([fsubs[1].data.id0[j] for j in wudgtides])
print(np.nanmedian(limgc[wudglim]/np.median(limgc[wnotudglim])))

import matplotlib.pyplot as plt
plt.clf()
hist=np.histogram(limgc[wudglim]/np.median(limgc[wnotudglim]),range=[0,4.5],bins=12,density=1)
print(hist)
histstep=hist[1][1]-hist[1][0]
mstarbins=np.linspace(7,9,num=11)

def chi2(params):
    print('')
    print(params)
    if abs(params[0])>1.5 or abs(params[1])>.5 or (params[2]>10) or (params[2]<-10):
        return -np.inf
    deltas=np.random.randn(5)*.001
    kss=[]
    lnlike=[]
    #for j in range(3):
    for j in range(1):
        mgcsudg=[]
        mgcsnon=[]
        bu=np.zeros(len(mstarbins)-1)
        bn=np.zeros(len(mstarbins)-1)

        for i in range(len(objectsudg)):
        #getgcs(-.5,-.9,.1,19.5,objects[i])/(2.9E-5*fsubs[1].data.mvir[w8[i],0]/.6774))
            mgcsudg.append(getgcs(params[0]+deltas[0],params[1]+deltas[1],params[2]+deltas[2],objectsudg[i]))
        for i in range(len(objectsnot)):
            mgcsnon.append(getgcs(params[0]+deltas[0],params[1]+deltas[1],params[2]+deltas[2],objectsnot[i]))

        mgcsudg=np.array(mgcsudg)
        mgcsnon=np.array(mgcsnon)


        if len(np.where(np.isfinite(mgcsudg))[0])<20:
            kss.append(np.nan)
            continue

        if len(np.where(np.isfinite(mgcsnon))[0])<20:
            kss.append(np.nan)
            continue

        #wgt=np.where(mgcsudg/np.median(mgcsnon)>np.max(hist[1]))[0]
        #wlt=np.where(mgcsudg/np.median(mgcsnon)<np.min(hist[1]))[0]
        #welse=np.where((mgcsudg/np.median(mgcsnon)>=np.min(hist[1])) & (mgcsudg/np.median(mgcsnon)<=np.max(hist[1])))[0]
        #lnlikei=np.sum(np.log(hist[0][np.digitize(mgcsudg[welse]/np.median(mgcsnon),hist[1])-1]))
        #lnlikei+=np.sum(np.log(hist[0][-1])-2*(np.log(abs(mgcsudg[wgt]/np.median(mgcsnon)-np.max(hist[1])))/histstep))
        #lnlikei+=np.sum(np.log(hist[0][0])-2*(np.log(abs(mgcsudg[wlt]/np.median(mgcsnon)-np.min(hist[1]))/histstep)))
        #lnlike.append(lnlikei)

        #ksudg=stats.anderson_ksamp([mgcsudg/np.median(mgcsnon),limgc[wudglim]/np.median(limgc[wnotudglim])])
        #print(ksudg)
        
        #ksudg=stats.ks_2samp(mgcsudg/np.median(mgcsnon),limgc[wudglim]/np.median(limgc[wnotudglim]))
        #ksudg=stats.anderson_ksamp([mgcsudg/np.median(mgcsnon),limgc[wudglim]/np.median(limgc[wnotudglim])])
        #plt.hist(mgcsudg/np.mean(mgcsnon),histtype='step',label='My UDGs',range=[-5,50],bins=20)
        #plt.hist(mgcsudg,histtype='step',label='My UDGs',range=[-1E6,1E7],bins=20)
        #plt.hist(limgc[wudglim]/np.mean(limgc[wnotudglim]),histtype='step',label='Lim UDGs',range=[-5,50],bins=20)
        #plt.hist(limgc[wudglim],histtype='step',label='Lim UDGs',range=[-1E6,1E7],bins=20)
        #plt.legend()
        #print(ksudg)
        #plt.savefig('mgctstks.png')
        #plt.show()
        #kss.append(ksudg[0])
        #kss.append((np.nanmedian(mgcsudg/np.mean(mgcsnon))-np.nanmedian(limgc[wmudglim]/np.mean(limgc[wnotudglim])))**2+(np.nanpercentile(mgcsudg/np.mean(mgcsnon),75)-np.nanpercentile(limgc[wudglim]/np.mean(limgc[wnotudglim]),75))**2+(np.nanpercentile(mgcsudg/np.mean(mgcsnon),90)-np.nanpercentile(limgc[wudglim]/np.mean(limgc[wnotudglim]),90))**2)


        #sigclippedudg=astats.sigma_clipped_stats(mgcsudg,cenfunc='median',maxiters=1)
        #sigclippednon=astats.sigma_clipped_stats(mgcsnon,cenfunc='median',maxiters=1)
        for m in range(len(mstarbins)-1):
            wbi=np.where((np.log10(mstar[wudgtides])>mstarbins[m]) & (np.log10(mstar[wudgtides])<mstarbins[m+1]))[0]
            bu[m]=np.log10(np.sum(mgcsudg[wbi]*stripped[wudgtides[wbi]]))
            wbn=np.where((np.log10(mstar[wnotudg[wnotsample]])>mstarbins[m]) & (np.log10(mstar[wnotudg[wnotsample]])<mstarbins[m+1]))[0]
            bn[m]=np.log10(np.sum(mgcsnon[wbn]*stripped[wnotudg[wnotsample[wbn]]]))
        wcomp=np.where(np.isfinite(bu-bn))[0]
        kss.append(np.nanmean(bu[wcomp]-bn[wcomp]))
#        if np.median(mgcsudg)==0 or np.median(mgcsnon)==0 or not np.isfinite(np.median(mgcsudg)) or not np.isfinite(np.median(mgcsnon)):
#            kss.append(-np.inf)
#        else:
#            kss.append(np.log10(np.median(mgcsudg))-np.log10(np.median(mgcsnon)))
        
        #kss.append(np.log10(sigclippedudg[0])-np.log10(sigclippednon[0]))
        #kss.append(np.log10(sigclippedudg[0]/np.median(limgc[wudglim]))**2+.3*np.log10(sigclippednon[0]/np.median(limgc[wnotudglim]))**2)
        #kss.append(np.log10(sigclippedudg[0]/np.median(limgc[wudglim]))**2+.3*np.log10(sigclippednon[0]/np.median(limgc[wnotudglim]))**2)
        print(np.nanmedian(np.median(mgcsudg)-np.median(mgcsnon)))
        print(kss)

    if np.isfinite(np.nanmedian(kss)):
        return np.nanmedian(kss)
    else:
        return -np.inf


from scipy.optimize import minimize
#bestfit=minimize(lambda vals: -np.log10(chi2(vals)),[-.78,-.5,4.1],options={'eps':.01})
#print(bestfit)
import emcee
ndim, nwalkers = 3, 8
p0 = np.zeros([nwalkers, ndim])

p0[:,0]+=1.3
p0[:,1]+=-.3
p0[:,2]+=4
p0[:,0]+=np.random.randn(nwalkers)*.1
p0[:,1]+=np.random.randn(nwalkers)*.07
p0[:,2]+=np.random.randn(nwalkers)*.1
print(p0)
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lambda vals: chi2(vals))
#sampler.run_mcmc(p0, 200,progress=True)
#np.savetxt('mgctstks_deltasc_med2.txt',np.hstack([sampler.flatchain,np.array([sampler.flatlnprobability]).T]))
#for i in range(len(c1s)):
#    for j in range(len(c2s)):
#        for k in range(len(c3s)):
#            for l in range(len(c4s)):
#                fit[i,j,k,l]=chi2([c1s[i],c2s[j],len(c3s),len(c4s)])
#print(bestfit)
#print(chi2([0.34479191,  0.57688821]))
#print(chi2([-.7,0,-.1]))
#print(chi2([-.837,.078]))
#print(chi2([-1.957,1.896,.307]))
#print(chi2([.0563,.033,.053]))
#print(chi2([-.116,.030,-.042]))
#print(chi2([3.64773865, 0.22302668, 0.59843306]))
#print(chi2([-0.93144149, -0.09469492,  0.95599977]))
#print(chi2([-0.75649842,  0.02320715,  0.14671275]))
#print(chi2([.516,.3623,-.621]))
#print(chi2([1.48,.458,-0.736]))
#print(chi2([-.169,.422,-.583]))
#print(chi2([1.61904368, -0.40032135,  5.7667425]))
#print(chi2([.851,-.11,2.6]))
#print(chi2([-1.42293966,  0.05432084, -0.24169949]))
#print(chi2([.516,.3623,-.621]))
#print(chi2([-.158,.0815,-.024]))
#print(chi2([0.14593688,  0.08848594, -0.09143344]))
#print(chi2([-0.24963473, -0.30311568,  3.01296165]))
#print(chi2([-1.24321953e+00,  1.45864292e-01, -9.62877426e-01]))
#print(chi2([-0.16164743,  0.04189552, -0.00364701]))
#print(chi2([-0.12058949,  0.01970674,  0.03343717]))
#print(chi2([0.91548437, -0.21605789,  3.29890385]))
#print(chi2([1.04302558, -0.11318541,  2.48632237]))
#print(chi2([0.93223353,  0.13345228,  0.39622082]))
#print(chi2([1.19051542, 0.03466657, 1.48395427]))
#print(chi2([-0.53211829, -0.37927836,  2.88089601]))
#print(chi2([-0.77284531, -0.49819154,  4.14583802]))
print(chi2([1.43397479, -0.49223985,  4.36038537]))
