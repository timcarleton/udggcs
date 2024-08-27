import numpy as np
import matplotlib.pyplot as plt
from astropy import units
import getomega
from astropy.convolution import convolve, Box1DKernel
kminmpc=1*units.Mpc.to(units.km)

def getperi(f,buff=[1,1],withinrvir=True,physical=True,getomegaperi=True,h=.7,ntodo='all'):

    peri=[]
    peribyobj=[[] for i in range(len(f[1].data))]
    avgperibyobj=np.zeros(len(f[1].data))+np.nan
    periindbyobj=[[] for i in range(len(f[1].data))]
    omatperi=[[] for i in range(len(f[1].data))]
    
    nts=f[1].data.dhost.shape[1]
    dofit=False

    if ntodo=='all':
        nt=len(f[1].data)
    else:
        nt=ntodo
    for i in range(nt):
        dhostsmooth=convolve(f[1].data.dhost[i]/.7/(1+f[1].data.redshift[i]),Box1DKernel(4))

        ddist=(dhostsmooth[1:]-dhostsmooth[:-1])/h #dr/dt
        #print ddist
        #plt.plot(f[1].data.time[i,:],f[1].data.dhost[i])
        wout=np.append(np.where(ddist<=0)[0]+1,[len(ddist)-1]) #where not infalling
        #print wout
        #plt.plot(f[1].data.time[i,wout],f[1].data.dhost[i,wout])
        
        #find non-consecutive outfalls
        if len(wout)<buff[0]+buff[1]:
            continue
        dtout=wout[1:]-wout[:-1]
        #print dtout
        wmin=np.where((dtout!=1) & (wout[:-1]>buff[0]+buff[1]) & (wout[:-1]<nts-buff[0]-buff[1]))[0]
        #plt.plot(f[1].data.time[i,wout[wmin]],f[1].data.dhost[i,wout[wmin]])
        #print wout[wmin]
        for j in wmin:
            pre=np.arange(wout[j]+buff[0],wout[j]+buff[0]+buff[1])
            post=np.arange(wout[j]-buff[0]-buff[1],wout[j]-buff[0])
            all=np.arange(wout[j]-buff[0]-buff[1],wout[j]+buff[0]+buff[1])
            if np.mean(ddist[pre])>=0 and np.mean(ddist[post])<=0 and np.median(ddist[pre])>=0 and np.median(ddist[post])<=0:
                if withinrvir and f[1].data.dhost[i,wout[j]]*1000>f[1].data.rvirhost[i,wout[j]]:
                    continue
                #print np.mean(f[1].data.vxwrthost[pre]+f[1].data.vywrthost[pre]+f[1].data.vzwrthost[pre])
                #print np.mean(f[1].data.vxwrthost[post]+f[1].data.vywrthost[post]+f[1].data.vzwrthost[post])
                ninterp=20
                
                xtime=np.linspace(np.min(f[1].data.time[i,all]),np.max(f[1].data.time[i,all]),num=ninterp)
                
                dxhostall=np.interp(xtime,f[1].data.time[i,all][::-1],f[1].data.dxhost[i,all][::-1]/h/(1+f[1].data.redshift[i,all]))
                dyhostall=np.interp(xtime,f[1].data.time[i,all][::-1],f[1].data.dyhost[i,all][::-1]/h/(1+f[1].data.redshift[i,all]))
                dzhostall=np.interp(xtime,f[1].data.time[i,all][::-1],f[1].data.dzhost[i,all][::-1]/h/(1+f[1].data.redshift[i,all]))
                vxhostall=np.interp(xtime,f[1].data.time[i,all][::-1],f[1].data.vxwrthost[i,all][::-1])
                vyhostall=np.interp(xtime,f[1].data.time[i,all][::-1],f[1].data.vywrthost[i,all][::-1])
                vzhostall=np.interp(xtime,f[1].data.time[i,all][::-1],f[1].data.vzwrthost[i,all][::-1])

                dhostall=(dxhostall**2+dyhostall**2+dzhostall**2)**.5
                periindbyobj[i].append(wout[j])
                #plt.plot(f[1].data.time[i,pre],f[1].data.dhost[i,pre],'o')
                #plt.plot(f[1].data.time[i,post],f[1].data.dhost[i,post],'o')
                if dofit:
                    fit=np.polyfit(xtime,dhostall,2,w=1/(3+abs(xtime-np.median(xtime))).astype(np.float))
                    tmin=-fit[1]/2.0/fit[0]
                if getomegaperi:
                    if physical:
                        omall=getomega.getomega_direct(vxhostall,vyhostall,vzhostall,dxhostall*kminmpc,dyhostall*kminmpc,dzhostall*kminmpc)
                    else:
                        omall=getomega.getomega_direct(vxhostall,vyhostall,vzhostall,dxhostall*(1+f[1].data.redshift[i,wout[j]])*kminmpc,dyhostall*(1+f[1].data.redshift[i,wout[j]])*kminmpc,dzhostall*(1+f[1].data.redshift[i,wout[j]])*kminmpc)
                    if dofit:
                        omfit=np.polyfit(xtime,omall,2,w=1/(3+abs(xtime-np.median(xtime))).astype(np.float))
                    #plt.plot(xtime,omall)
                    #plt.plot(xtime,omfit[0]*xtime**2+omfit[1]*xtime+omfit[2])
                    #plt.show()
                    
                if physical:
                    if dofit:
                        peri.append((fit[0]*tmin**2+fit[1]*tmin+fit[2]))
                        peribyobj[i].append((fit[0]*tmin**2+fit[1]*tmin+fit[2]))
                    else:
                        peri.append(np.min(dhostall))
                        peribyobj[i].append(np.min(dhostall))
                else:
                    if dofit:
                        peri.append((fit[0]*tmin**2+fit[1]*tmin+fit[2])*(1+f[1].data.redshift[i,wout[j]]))
                        peribyobj[i].append((fit[0]*tmin**2+fit[1]*tmin+fit[2])*(1+f[1].data.redshift[i,wout[j]]))
                    else:
                        peri.append(np.min(dhostall)*(1+f[1].data.redshift[i,wout[j]]))
                        peribyobj[i].append(np.min(dhostall)*(1+f[1].data.redshift[i,wout[j]]))
                if getomegaperi:
                    if dofit:
                        omatperi[i].append(omfit[0]*tmin**2+omfit[1]*tmin+omfit[2])
                    else:
                        omatperi[i].append(np.max(omall))
                #plt.plot(tmin,(fit[0]*tmin**2+fit[1]*tmin+fit[2]),'o')
        #plt.show()

    for i in range(len(peribyobj)):
        if len(peribyobj[i])>0:
            avgperibyobj[i]=np.mean(peribyobj[i])

    if getomegaperi:
        return peri,peribyobj,periindbyobj,avgperibyobj,omatperi
    else:
        return peri,peribyobj,periindbyobj,avgperibyobj
    
