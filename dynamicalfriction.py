#from Gnedin 2014, following Binney 2014
def dyanmicalfriction(r,vc,m,deltat,fecc=0.5):

    tdf=.45*r**2*vc/(m/1E5)*fecc
    drdt=-r/2/tdf

    return r+drdt*deltat
