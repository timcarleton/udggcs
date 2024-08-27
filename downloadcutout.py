import os
import tngget
def getcutout(id0,snap,idz=-1):
    if idz==-1:
        id=id0
    else:
        id=idz
    udgcuts=os.listdir('/Volumes/Elements/illustriscats/udgtngcutouts')
    othercuts=os.listdir('/Volumes/Elements/illustriscats/tngcutouts')


    strid=str(id0)
    stridz=str(id)
    strsnap=str(snap)


    if 'cutout_'+strid+',' in ','.join(udgcuts) or udgcuts[-1]=='cutout_'+strid:
        files=os.listdir('/Volumes/Elements/illustriscats/udgtngcutouts/cutout_'+strid)
        if 'tngcutout_'+strsnap+'.hdf5' in ','.join(files):
            return '/Volumes/Elements/illustriscats/udgtngcutouts/cutout_'+strid+'/tngcutout_'+strsnap+'.hdf5'
        else:
            tngget.get('http://www.tng-project.org/api/TNG100-1/snapshots/'+strsnap+'/subhalos/'+stridz+'/cutout.hdf5',headers=tngget.headers,savename='/Volumes/Elements/illustriscats/udgtngcutouts/cutout_'+strid+'/tngcutout_'+strsnap+'.hdf5')
            return '/Volumes/Elements/illustriscats/udgtngcutouts/cutout_'+strid+'/tngcutout_'+strsnap+'.hdf5'
        
    elif 'cutout_'+strid+',' in ','.join(othercuts) or othercuts[-1]=='cutout_'+strid:
        files=os.listdir('/Volumes/Elements/illustriscats/tngcutouts/cutout_'+strid)
        if 'tngcutout_'+strsnap+'.hdf5' in ','.join(files):
            return '/Volumes/Elements/illustriscats/tngcutouts/cutout_'+strid+'/tngcutout_'+strsnap+'.hdf5'
        else:
            tngget.get('http://www.tng-project.org/api/TNG100-1/snapshots/'+strsnap+'/subhalos/'+stridz+'/cutout.hdf5',headers=tngget.headers,savename='/Volumes/Elements/illustriscats/tngcutouts/cutout_'+strid+'/tngcutout_'+strsnap+'.hdf5')
            return '/Volumes/Elements/illustriscats/tngcutouts/cutout_'+strid+'/tngcutout_'+strsnap+'.hdf5'
      
    else:
        os.system('mkdir /Volumes/Elements/illustriscats/tngcutouts/cutout_'+strid)
        tngget.get('http://www.tng-project.org/api/TNG100-1/snapshots/'+strsnap+'/subhalos/'+stridz+'/cutout.hdf5',headers=tngget.headers,savename='/Volumes/Elements/illustriscats/tngcutouts/cutout_'+strid+'/tngcutout_'+strsnap+'.hdf5')
        return '/Volumes/Elements/illustriscats/tngcutouts/cutout_'+strid+'/tngcutout_'+strsnap+'.hdf5'
