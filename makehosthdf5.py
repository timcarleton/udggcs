import numpy as np
from astropy import table,units,cosmology
from astropy.io import fits
import datetime
import h5py

def getcenter(tab,boxsize,edgebuff):
    wcenter=np.where((tab['x']*units.Mpc/astrocosmo.h>edgebuff) &
                     (tab['x']*units.Mpc/astrocosmo.h<boxsize-edgebuff) &
                     (tab['y']*units.Mpc/astrocosmo.h>edgebuff) &
                     (tab['y']*units.Mpc/astrocosmo.h<boxsize-edgebuff) &
                     (tab['z']*units.Mpc/astrocosmo.h>edgebuff) &
                     (tab['z']*units.Mpc/astrocosmo.h<boxsize-edgebuff))[0]
    return wcenter

def makehostfile(cat,mhostlim=[14,15],edgebuff=5*units.Mpc,onlymostmassive=True,isolation=2,nts=100,cutiso=False,savename='cuthosts.fits',maxhosts=-1,inid=[]):
    print(datetime.datetime.now())
    id=[]
    mvir=[]
    xpos=[]
    ypos=[]
    zpos=[]
    vx=[]
    vy=[]
    vz=[]
    rs=[]
    vmax=[]
    rvir=[]
    vrms=[]
    spin=[]
    tu=[]
    z0id=[]
    rskyp=[]
    menc=[]
    mbe=[]
    mdi=[]
    mpk=[]
    vpk=[]
    macc=[]
    vacc=[]
    aacc=[]
    aacc=[]
    apk=[]
    m200=[]
    m500=[]
    
    edgebuffmpc=edgebuff.to(units.Mpc).value

    catf=open(cat)
    catf.seek(0)
    counter=0
    countercenter=0
    countermass=0
    countermm=0
    for i in catf:
        if (counter % 1000000)==10:
            print(counter)
        fields=i.split()
        if fields[0]=='#Omega_M':
            astrocosmo=cosmology.LambdaCDM(100*float(fields[8]),float(fields[2][:-1]),float(fields[5][:-1]))
        if fields[0]=='#Full' and fields[1]=='box' and fields[2]=='size':
            boxsize=float(fields[4])/astrocosmo.h
        try:
            tid=float(fields[0])
        except:
            continue
        scalef=float(fields[0])
        
        if (float(fields[17])/astrocosmo.h>edgebuffmpc and float(fields[17])/astrocosmo.h<(boxsize-edgebuffmpc) and float(fields[18])/astrocosmo.h>edgebuffmpc and float(fields[18])/astrocosmo.h<(boxsize-edgebuffmpc) and float(fields[19])/astrocosmo.h>edgebuffmpc and float(fields[19])/astrocosmo.h<(boxsize-edgebuffmpc)) or (fields[0] in inid):
            countercenter=countercenter+1
            if float(fields[10])/astrocosmo.h>10**mhostlim[0] and float(fields[10])/astrocosmo.h<=10**mhostlim[1]:
                countermass=countermass+1
                if int(fields[6])==-1:
                    countermm=countermm+1
                    id.append(int(fields[1]))
                    mvir.append(float(fields[10]))
                    rvir.append(float(fields[11]))
                    rs.append(float(fields[12]))
                    vmax.append(float(fields[16]))
                    xpos.append(float(fields[17]))
                    ypos.append(float(fields[18]))
                    zpos.append(float(fields[19]))
                    vx.append(float(fields[20]))
                    vy.append(float(fields[21]))
                    vz.append(float(fields[22]))
                    vrms.append(float(fields[13]))
                    spin.append(float(fields[26]))
                    tu.append(float(fields[54]))
                    z0id.append(int(fields[29]))
                    rskyp.append(float(fields[35]))
                    menc.append(float(fields[36]))
                    mbe.append(float(fields[55]))
                    mdi.append(float(fields[56]))
                    mpk.append(float(fields[58]))
                    macc.append(float(fields[57]))
                    vpk.append(float(fields[60]))
                    vacc.append(float(fields[59]))
                    apk.append(float(fields[67]))
                    aacc.append(float(fields[68]))
                    m200.append(float(fields[38]))
                    m500.append(float(fields[39]))
                    
        counter=counter+1
    catf.close()
    print()
    print()
    print(datetime.datetime.now())
    print("\a")

    id=np.array(id)
    mvir=np.array(mvir)
    rvir=np.array(rvir)
    rs=np.array(rs)
    vmax=np.array(vmax)
    xpos=np.array(xpos)
    ypos=np.array(ypos)
    zpos=np.array(zpos)
    vx=np.array(vx)
    vy=np.array(vy)
    vz=np.array(vz)
    vrms=np.array(vrms)
    spin=np.array(spin)
    tu=np.array(tu)
    z0id=np.array(z0id)
    rskyp=np.array(rskyp)
    menc=np.array(menc)
    mbe=np.array(mbe)
    mdi=np.array(mdi)
    mkp=np.array(mpk)
    m200=np.array(m200)
    m500=np.array(m500)
    r200=(3*m200/astrocosmo.h/4.0/np.pi/(((astrocosmo.critical_density0.to(units.M_sun/units.kpc**3).value)*200)))**(1.0/3)*astrocosmo.h
    r500=(3*m500/4.0/np.pi/(((astrocosmo.h*astrocosmo.critical_density0.to(units.M_sun/units.kpc**3).value)*500)))**(1.0/3)
    vpk=np.array(vpk)
    macc=np.array(macc)
    mpk=np.array(mpk)
    vacc=np.array(vacc)
    zpk=1.0/np.array(apk)-1
    zacc=1.0/np.array(aacc)-1

    isohosts=[]
    nnearbyhosts=np.zeros(len(id)).astype(np.int)
    for i in range(len(id)):
        d2=(xpos-xpos[i])**2+(ypos-ypos[i])**2+(zpos-zpos[i])**2
        d2[i]=10*rvir[i]
        if np.min(d2)<isolation**2*(1000*rvir[i])**2:
            isohosts.append(i)
        else:
            w=np.where(d2<isolation**2*(1000*rvir[i])**2)[0]
            nnearbyhosts[i]=len(w)

    isohosts=np.array(isohosts)


    if cutiso:
        okhosts=isohosts
    else:
        okhosts=np.arange(len(id))
    print()

    if np.isfinite(maxhosts) and maxhosts>0:
        finalhosts=np.random.choice(range(len(okhosts)),size=maxhosts)
        okhosts=okhosts[finalhosts]
        isohosts=isohosts[finalhosts]
        nnearbyhosts=nnearbyhosts[finalhosts]
        print(len(isohosts))
    #make columns


    hdf=h5py.File(savename)
    hdf.create_dataset('id0',data=z0id[okhosts])
    hdf.create_dataset('m_peak',data=mpk[okhosts])
    hdf.create_dataset('m_acc',data=macc[okhosts])
    hdf.create_dataset('v_peak',data=vpk[okhosts])
    hdf.create_dataset('v_acc',data=vacc[okhosts])
    hdf.create_dataset('z_peak',data=zpk[okhosts])
    hdf.create_dataset('z_acc',data=zacc[okhosts])

    hdf.create_dataset('id',data=np.zeros([len(okhosts),nts]).astype(np.int64)-99,chunks=True)
    hdf.create_dataset('mvir',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('rvir',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('rs',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('vmax',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('x',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('y',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('z',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('vx',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('vy',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('vz',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('nnearbyhosts',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('scale',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('redshift',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('time',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('vrms',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('spin',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('tu',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('treefile',data=np.array(['' for i in range(len(okhosts))]).astype('S14'),chunks=True)
    hdf.create_dataset('treebytelocation',data=np.zeros([len(okhosts),nts]).astype(np.int)-99,chunks=True)
    hdf.create_dataset('rs_klypin',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('m_enclosed',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('m_behroozi',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('m_diemer',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('m200',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('m500',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('r200',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)
    hdf.create_dataset('r500',data=np.zeros([len(okhosts),nts])+np.nan,chunks=True)

    hdf['id'][:,0]=id[okhosts]
    hdf['mvir'][:,0]=mvir[okhosts]
    hdf['rvir'][:,0]=rvir[okhosts]
    hdf['rs'][:,0]=rs[okhosts]
    hdf['vmax'][:,0]=vmax[okhosts]
    hdf['x'][:,0]=xpos[okhosts]
    hdf['y'][:,0]=ypos[okhosts]
    hdf['z'][:,0]=zpos[okhosts]
    hdf['vx'][:,0]=vx[okhosts]
    hdf['vy'][:,0]=vy[okhosts]
    hdf['vz'][:,0]=vz[okhosts]
    hdf['vrms'][:,0]=vrms[okhosts]
    hdf['spin'][:,0]=spin[okhosts]
    hdf['tu'][:,0]=tu[okhosts]
    hdf['rs_klypin'][:,0]=rskyp[okhosts]
    hdf['m_enclosed'][:,0]=menc[okhosts]
    hdf['m_behroozi'][:,0]=mbe[okhosts]
    hdf['m_diemer'][:,0]=mdi[okhosts]
    hdf['m200'][:,0]=m200[okhosts]
    hdf['r200'][:,0]=r200[okhosts]
    hdf['m500'][:,0]=m500[okhosts]
    hdf['r500'][:,0]=r500[okhosts]

    hdf['scale'][:,0]=scalef
    hdf['redshift'][:,0]=1.0/scalef-1
    hdf['time'][:,0]=astrocosmo.age(1.0/scalef-1).value

    hdf.attrs['id0']='Halo id at z=0'
    hdf.attrs['id']='Halo id for all redshifts'
    hdf.attrs['mvir']='Mvir: Halo mass (Msun/h)'
    hdf.attrs['rvir']='Halo radius (kpc/h comoving)'
    hdf.attrs['rs']='Scale radius (kpc/h comoving)'
    hdf.attrs['vmax']='Maxmimum circular velocity (km/s physical)'
    hdf.attrs['x']='Halo x position (Mpc/h comoving)'
    hdf.attrs['y']='Halo y position (Mpc/h comoving)'
    hdf.attrs['z']='Halo z position (Mpc/h comoving)'
    hdf.attrs['vx']='Halo x velocity (km/s physical)'
    hdf.attrs['vy']='Halo y velocity (km/s physical)'
    hdf.attrs['vz']='Halo z velocity (km/s physical)'
    hdf.attrs['redshift']='redshift of all points'
    hdf.attrs['time']='age of universe of all points'
    hdf.attrs['vrms']='Velocity dispersion (km/s physical)'
    hdf.attrs['spin']='Halo spin parameter'
    hdf.attrs['m200']='Mass within R200 (M_sun/h)'
    hdf.attrs['r200']='Average Density within R200=200*matter density (kpc/h)'

    hdf.attrs['m500']='Mass within R500 (M_sun/h)'
    hdf.attrs['r500']='Average Density within R500=500*matter density (kpc/h)'
    hdf.attrs['T/U']='ratio of kinetic to potential energies'
    hdf.attrs['cosmology']='H0='+str(astrocosmo.h*100)+', Om0='+str(astrocosmo.Om0)+', Ode0='+str(astrocosmo.Om0)
    hdf.attrs['nnearbyhosts']='number of other hosts within '+str(isolation)+' rvir'
    hdf.attrs['masslowlim']=mhostlim[0]
    hdf.attrs['masshighlim']=mhostlim[1]
    hdf.attrs['edgebuff']=edgebuff.value
    hdf.attrs['treefile']='file where tree is'
    hdf.attrs['treebytelocation']='byte within tree file where halo is'
    hdf.attrs['rs_klypin']='Klypin 2011 sacle radius (kpc/h comoving)'
    hdf.attrs['m_enclosed']='total mass within overdensity (Msun/h)'
    hdf.attrs['m_beehrozi']='psudoevolution corrected mass (Behroozi), (Msun/h)'
    hdf.attrs['m_diemer']='psudoevolution corrected mass (Diemer), (Msun/h)'
    hdf.attrs['m_peak']='peak mass over accretion history (Msun/h)'
    hdf.attrs['m_acc']='mass at accretion (Msun/h)'
    hdf.attrs['v_peak']='maximum Vmax over accretion history (km/s)'
    hdf.attrs['v_acc']='Vmax at accretion (km/s)'
    hdf.attrs['z_peak']='redshift of m_peak'
    hdf.attrs['z_acc']='redshift of m_acc'

    hdf.close()
