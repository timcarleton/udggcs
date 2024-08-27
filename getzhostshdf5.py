import numpy as np
from astropy.io import fits
from astropy import units,table,cosmology
import datetime
import h5py
import os

def getprojs(fileobj,cat,gal,step):
    cat['treebytelocation'][gal,step]=fileobj.tell()
    line=fileobj.readline().split()
    try:
        tmp=float(line[0])
    except:
        return
    else:
        cat['id'][gal,step]=int(line[1])
        cat['scale'][gal,step]=float(line[0])
        cat['mvir'][gal,step]=float(line[10])
        cat['rvir'][gal,step]=float(line[11])
        cat['rs'][gal,step]=float(line[12])
        cat['vmax'][gal,step]=float(line[16])
        cat['x'][gal,step]=float(line[17])
        cat['y'][gal,step]=float(line[18])
        cat['z'][gal,step]=float(line[19])
        cat['vx'][gal,step]=float(line[20])
        cat['vy'][gal,step]=float(line[21])
        cat['vz'][gal,step]=float(line[22])
        cat['vrms'][gal,step]=float(line[13])
        cat['spin'][gal,step]=float(line[26])
        cat['tu'][gal,step]=float(line[54])
        cat['rs_klypin'][gal,step]=float(line[35])
        cat['m_enclosed'][gal,step]=float(line[36])
        cat['m_behroozi'][gal,step]=float(line[55])
        cat['m_diemer'][gal,step]=float(line[56])
        cat['m200'][gal,step]=float(line[38])
        if int(line[4])==0:
            return
        while True:
            x=fileobj.tell()
            tmp=fileobj.readline().split()
            try:
                if int(tmp[3])==int(line[1]):
                    fileobj.seek(x)
                    getprojs(fileobj,cat,gal,step+1)
            except:
                return


def getzhosts(hostfile,nts=100,folder='.'):

    print(datetime.datetime.now())
    
    tfiles=os.listdir(folder+'/trees')
    treefiles=[i for i in tfiles if i[0:4]=='tree']

    treef=[open(folder+'/trees/'+i) for i in treefiles]
    print()
    locations=np.genfromtxt(folder+'/trees/locations.dat',dtype=(np.long,np.long,np.long,'S20'))
    print()
    hd=treef[0].readline()
    cosmo=treef[0].readline().split()
    print(cosmo)
    astrocosmo=cosmology.LambdaCDM(100*float(cosmo[8]),float(cosmo[2][:-1]),float(cosmo[5][:-1]))
    treef[0].seek(0)
    hostf=h5py.File(hostfile)

    for i in range(len(hostf['x'][:])):
        if (i%5000)==10:
            print(i)
        w=np.where(locations['f0']==hostf['id0'][i])[0]
        for j in w:
            step=0
            infile=treef[locations['f1'][j]]
            infile.seek(locations['f2'][j])
            hostf['treefile'][i]=locations['f3'][w][0]
            while True:
                x=infile.tell()
                tmp=infile.readline().split()
                if len(tmp)>1:
                    if int(tmp[1])==hostf['id'][i,0]:
                        infile.seek(x)
                        break
            getprojs(infile,hostf,i,0)
            hostf.flush()
    for ii in treef:
        ii.close()
    print()
    print(datetime.datetime.now())
    print("\a")

    for i in range(nts):
        hostf['redshift'][:,i]=1.0/hostf['scale'][:,i]-1
        hostf['time'][:,i]=astrocosmo.age(1.0/hostf['scale'][:,i]-1)
    hostf['r200'][:,:]=(3*hostf['m200'][:,:]/astrocosmo.h/4.0/np.pi/(((astrocosmo.critical_density(hostf['redshift'][:,:]).to(units.M_sun/units.kpc**3).value)*200)))**(1.0/3)*astrocosmo.h
    hostf.close()
