from scipy.special import gamma, gammaincc
from numpy import exp
gammanorm=gamma(-.999)
gammanorm2=gamma(.001)
gammanorm3=gamma(-.999+1)

def getnfracabove(mc):
    c=mc/1E4
    
    a1=exp(-1.01/c)/(1/c)**.999*(99.6402*exp(1/c)-1.001*exp(.01/c))
    a2=-1.001*gammaincc(.001,.01/c)*gammanorm2
    a3=1.001*gammaincc(.001,1/c)*gammanorm2

    #    bottom=gammaincc(-.999,1/c)*gammanorm
    top=gammanorm-1/(-.999)*(gammanorm3*(1-gammaincc(-.999+1,1/c))+1/c**(-.999)*exp(-1/c))
    return top/(top+a1+a2+a3)

def getmfracabove(mc):
    c=mc/1E4
    
    top=gammaincc(.001,1/c)*gammanorm2
    bottom=(gammaincc(.001,.01/c)-gammaincc(.001,1/c))*gammanorm2
    
    return top/(top+bottom)
