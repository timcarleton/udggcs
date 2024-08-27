import copy
from sys import stdout
from scipy.optimize import minimize_scalar,minimize,fsolve
from numpy import log10
from scipy.integrate import quad
from scipy.misc import derivative
from astropy import units
from astropy.constants import G
from astropy.constants import c as speedoflight
from scipy.integrate import odeint,quad
from numpy import pi,sqrt,where,zeros,exp,shape,inf,log10,nan,log,isfinite
from inspect import getargspec
from scipy.special import hyp2f1,gammainc,betainc
from scipy.special import beta as betafunc
from scipy.special import gamma as gammafunc
import matplotlib.pyplot as plt
import numpy as np

GN=G.to(units.kpc**3/units.M_sun/(units.s)**2)

def betaincfunc(p,q,x):
    return x**p/p*hyp2f1(p,1-q,p+1,x)

def nfwfx(x):
    return np.log(1+x)-x/(1+x)

def getnfwrho0(mvir,rs,rvir):
    c=rvir/rs
    return mvir/nfwfx(c)/4/np.pi/rs**3

def getdneinasto(alpha):
    return 3*alpha-1.0/3+8.0/1215.0/alpha+184.0/229635.0/alpha**2+1048.0/31000725.0/alpha**3-17557576.0/1242974068875.0/alpha**4

def getmtoteinasto(alpha,rho0,rs):
    h=rs/getdneinasto(alpha)
    return 4*pi*rho0*h**3*alpha*gammafunc(3*alpha)

def getmasseinasto(rho0,rs,r,alpha):
    s=(getdneinasto(alpha)**alpha*r/rs).decompose().value
    return getmtoteinasto(alpha,rho0,rs)*(1-gammainc(3*alpha,s**(1.0/alpha))/gammafunc(3*alpha))

def getxfromrhonfw(rho,rhos):

    topa=(2.0)**(1.0/3)*rho
    bottoma1=2*rho**3
    bottoma2=27.0*rho**2*rhos
    bottoma3=3*sqrt(3)*sqrt(4*rho**5*rhos+27*rho**4*rhos**2)

    first=topa/(bottoma1+bottoma2+bottoma3)**(1.0/3)
    second=(bottoma1+bottoma2+bottoma3)**(1.0/3)/topa

    return 1.0/3*(first+second)-2.0/3


def getmassfromzhao0(alpha,beta,gamma,rhos,rs,r):

    a=1.0*(3-gamma)/alpha
    b=1.0*(beta-gamma)/alpha
    c=1.0*(alpha-gamma+3)/alpha
    try:
        v=(r/rs).decompose()
        try:
            y=(v.value)**alpha
        except OverflowError:
            return np.inf
        x=-y
    except:
        x=-(r/rs)**alpha
    f=hyp2f1(a,b,c,x)

    return 4*pi*rhos*r**3*(r/rs)**(-gamma)*f/(3-gamma)


def getxmaxzhao0(alpha,beta,gamma):

    if alpha==1 and beta==3 and gamma==1:
        return 2.1625816019114059
    if alpha==1 and beta==3 and gamma==0:
        return 4.4247006595498686
    if alpha==1 and beta==4 and gamma==1:
        return 1.0
    if alpha==1 and beta==4 and gamma==0:
        return 2.0
    if alpha==1 and beta==5 and gamma==0:
        return 1.2749172096233057
    if alpha==1 and beta==5 and gamma==1:
        return 0.645751312789415

    if alpha==2 and beta==3 and gamma==0:
        return 2.919847688299687
    if alpha==2 and beta==3 and gamma==1:
        return 1.9802913004322131

    a=float(alpha)
    b=float(beta)
    g=float(gamma)

#    x=minimize_scalar(lambda x: np.log10(abs((-1.*(3 - g)*x**(1 - g)*((1 + x**a)**(-(b - g)/a) - hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a)))/(-3. + g) - (1.*(2 - g)*x**(1 - g)*hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a))/(-3. + g))),bounds=[.001,20],method='bounded')
    x=minimize_scalar(lambda x: -getmassfromzhao0(a,b,g,1.0,1.0,x)/x,bounds=[.001,20],method='bounded')['x']
#    print x
    
#    if b<=4.5:
#        x=fsolve(lambda x: (-1.*(3 - g)*x**(1 - g)*((1 + x**a)**(-(b - g)/a) - hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a)))/(-3. + g) - (1.*(2 - g)*x**(1 - g)*hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a))/(-3. + g),2)[0]
#    if b>4.5:
#        x0=fsolve(lambda x: (-1.*(3 - g)*x**(1 - g)*((1 + x**a)**(-(4 - g)/a) - hyp2f1((3 - g)/a,(4 - g)/a,(3 + a - g)/a,-x**a)))/(-3. + g) - (1.*(2 - g)*x**(1 - g)*hyp2f1((3 - g)/a,(4 - g)/a,(3 + a - g)/a,-x**a))/(-3. + g),1)[0]
#        print x0
#        x=fsolve(lambda x: (-1.*(3 - g)*x**(1 - g)*((1 + x**a)**(-(b - g)/a) - hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a)))/(-3. + g) - (1.*(2 - g)*x**(1 - g)*hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a))/(-3. + g),x0)[0]
    return x

def getrho0frommvirzhao(alpha,beta,gamma,mvir,rs,delta):
    rvir=(mvir/(4*pi/3*delta))**(1.0/3)
    rho0=mvir/getmassfromzhao0(alpha,beta,gamma,1,rs,rvir)
    return rho0

def getmassfromnfw(rhos,rs,r):

    return 4*pi*rhos*rs**3*(log(1+r/rs)-r/(rs+r))

def getmassfromprofile(rhofunc,r,*args):
    if r==0:
            return 0*units.M_sun
    return 4*pi*quad(lambda x: (x*r.unit*x*r.unit*(rhofunc(x,*args).to(units.M_sun/r.unit**3))).value,0,r.value)[0]*units.M_sun

def zhaoc(alpha,beta,gamma):
    return 1.0/4/np.pi/betainc(alpha*(3-gamma),alpha*(beta-3),1)

def zhaoq(i,j):
    if i>=j and j>=0:
        return (-1)**(j)*gammafunc(i+1)/(gammafunc(j+1)*gammafunc(i-j+1))
    else:
        return 0

def zhaoa(i,alpha,beta,gamma,rho0):
    return 4*np.pi*alpha*rho0/(alpha*(3-gamma)+i)*zhaoq(alpha*(beta-3)-1,i)

def zhaobi(i,alpha,beta,gamma,rho0):
    b=0
    for j in range(i+1):
        b=b+zhaoq(alpha-1,j)*zhaoa(i-j,alpha,beta,gamma,rho0)
    return alpha*b

def zhaosx(i,x):

    if i==0:
        return -np.log(x)
    else:
        return (1-x**(-i))/i

def zhaof00(r,alpha,beta,gamma):
    chi=r**(1.0/alpha)/(r**(1.0/alpha)+1)

    c0=alpha*(beta-gamma)
    q0=alpha*(beta-3)
    p0=alpha*(-gamma+2)

    first=alpha*betainc(c0-q0,q0,chi)/(chi**(c0-q0)*(1-chi)**q0)*betafunc(c0-q0,q0)
    second=alpha*betainc(c0-p0,p0,1-chi)/((1-chi)**(c0-p0)*chi**p0)*betafunc(c0-p0,p0)
    return first+second

def zhaorho(x,rho0,alpha,beta,gamma):
    return rho0/(x**gamma*(1+x**1.0/alpha)**((beta-gamma)/alpha))

def zhaopotential(r,rho0,rs,alpha,beta,gamma):

    try:
        x=(r/rs).decompose().value
    except:
        x=r/rs

    b=float(beta)
    a=float(alpha)
    g=float(gamma)
    

    phir=-4*GN*np.pi*rho0*r**2*(hyp2f1(-2/a + b/a,b/a - g/a,1 - 2/a + b/a,-(1/x)**a)/((-2 + b)*(x)**b) + hyp2f1(3/a - g/a,b/a - g/a,1 + 3/a - g/a,-(x)**a)/((3 - g)*(x)**g))
    return phir

def zhaofi(i,alpha,beta,gamma):
    return 2*zhaoc(alpha,beta,gamma)*zhaoq(alpha*(2-gamma),i)

def zhaoxx(x):
    if x==1:
        return 1
    elif x<1:
        return np.arccosh(1.0/x)/np.sqrt(1-x**2)
    else:
        return np.arccos(1.0/x)/np.sqrt(x**2-1)

def zhaozx(x):
    return x*zhaoxx(x)-1

def zhaou(i,r,alpha):
    a2=1.0/alpha
    return quad(lambda x: (np.sin(x)**a2/(r**a2+np.sin(x)**a2))**i,0,np.pi/2)[0]


def nfwpotential(rho0,rs,r):
    try:
        tmp=rho0.unit
        x=(r/rs).decompose().value
        return -4*np.pi*GN*rho0*rs**3/r*np.log(1+x)
    except:
        return -4*np.pi*(GN.value)*rho0*rs**3/r*np.log(1+r/rs)

def plummerfe(m,a,e):
    try:
        tmp=m.unit
        return 24*np.sqrt(2)/7/np.pi**3*a**2/GN**5/m**4*e**3.5
    except:
        return 24*np.sqrt(2)/7/np.pi**3*a**2/(GN.value)**5/m**4*e**3.5

def getge(pot,e):

    rmax=fsolve(lambda x: pot(x)+e,1)
    return quad(lambda r: np.sqrt(2*(pot(r)-e))*r**2,0,rmax)[0]

def dehnengecore(m,rs,e):

    try:
        tmp=m.unit
        enorm=-(e/(GN*m/rs)).decompose().value
        pre=8*np.pi**2*np.sqrt(GN*m*rs**5)
    except:
        enorm=-(e/(GN*m/rs)).decompose().value
        pre=8*np.pi**2*np.sqrt((GN.value)*m*rs**5)

    first=np.sqrt(1-2*enorm)*(3-14*enorm-8*enorm**2)/(12*enorm**2)-np.pi
    second=(1-6*enorm+16*enorm**2)/(2*enorm)**2.5*np.arccos(-np.sqrt(1-2*enorm))
    return pre*(first+second)

def dehnenfecore(m,rs,e):
    try:
        tmp=m.unit
        enorm=-(e/(GN*m/rs)).decompose().value
        pre=3*m/2/np.pi**3/(GN*m*rs)**1.5
    except:
        enorm=-(e/((GN.to(units.kpc*units.km**2/(units.s**2*units.M_sun)))*m/rs)).decompose().value
        pre=3*m/2/np.pi**3/((GN.to(units.kpc*units.km**2/units.s**2/units.M_sun)).value*m*rs)**1.5

    first=np.sqrt(2*enorm)*(3-4*enorm)/(1-2*enorm)
    second=3*np.arcsinh(np.sqrt(2*enorm/(1-2*enorm)))
    return pre*(first-second)


def dehnenfe(m,rs,gamma,e):

    try:
        tmp=m.unit
        pre=(3-gamma)*m/2.0/(2*np.pi**2*GN*m*rs)**(1.5)
        enorm=-(e/(GN.to(units.kpc*units.km**2/(units.s**2*units.M_sun))*m/rs)).decompose().value

    except:
        pre=(3-gamma)*m/2.0/(2*np.pi**2*(GN.value)*rs)**(1.5)
        enorm=-e/((GN.to(units.kpc*units.km**2/(units.s**2*units.M_sun)).value)*m/rs)
    # print enorm
    p=1.0/(1.0+enorm*(gamma-2.0))
    pf=(gamma-2)*enorm/(1+enorm*(gamma-2.0))

    pre2=2*np.sqrt(enorm)*p**((gamma-4.0)/(gamma-2.0))


    first=-gamma*hyp2f1(0.5,((gamma-4.0)/(gamma-2.0)),1.5,pf)
    second=2*(gamma-1)*hyp2f1(0.5,((gamma-3.0)/(gamma-2.0)),1.5,pf)
    third=-2*(gamma-3)*hyp2f1(0.5,((gamma-1.0)/(gamma-2.0)),1.5,pf)
    fourth=(gamma-4)*hyp2f1(0.5,1.0*gamma/(gamma-2.0),1.5,pf)

    return pre*pre2*(first+
                     p**(1.0/(gamma-2.0))*(second+
                                           p**(2.0/(gamma-2.0))*(third+
                                                                 p**(1.0/(gamma-2.0))*fourth)))

#     first=(4-gamma)*hyp2f1(1,-gamma/(2.0-gamma),1.5,(2.0-gamma)*e)
#     second=-2*(3.0-gamma)*hyp2f1(1,(1.0-gamma)/(2.0-gamma),1.5,(2.0-gamma)*e)
#     third=2*(1.0-gamma)*hyp2f1(1.0,(3.0-gamma)/(2.0-gamma),1.5,(2.0*gamma)*e)
#     fourth=gamma*hyp2f1(1,(4.0-gamma)/(2.0-gamma),1.5,(2.0-gamma)*e)

#     return pre*(first+second+third+fourth)


class RhoProfile():

    rho_func=0
    min_r=10**(-4)
    max_r=10**(4)
    nargs=1
    rhounit=0
    munit=0
    vunit=0
    runit=0
    mfunc=0

    def __init__(self,rho_func,min_r=10**(-4)*units.kpc,max_r=10**(4)*units.kpc,rhounit=units.M_sun/(units.kpc)**3,munit=units.M_sun,vunit=units.km/units.s,runit=units.kpc,mfunc=None,phifunc=None):

        #rhofunc: given r, gives rho
        self.rho_func=rho_func
        self.min_r=min_r
        self.max_r=max_r
#        self.nargs=len(getargspec(rho_func).args)
        self.nargs=1
        self.rhounit=rhounit
        self.munit=munit
        self.vunit=vunit
        self.runit=runit
        self.mfunc=mfunc
        self.phifunc=phifunc

    def get_rho(self,r,*args,**kwargs):

        if self.nargs==1:
            try:
                return self.rho_func(r,**kwargs).to(self.rhounit)
            except AttributeError:
                return self.rho_func(r,**kwargs)*self.rhounit
        else:
            arglist=list(args)[0:self.nargs-1]
            argtp=tuple(arglist)
            return self.rho_func(r,*argtp,**kwargs).to(self.rhounit)

    def get_mass(self,r,*args,**kwargs):
        
        if self.mfunc!=None:
            return self.mfunc(r)*self.munit
        else:
            if r==0:
                return 0*self.munit
            m=4*pi*quad(lambda x: x*x*(self.get_rho(x*r.unit,*args,**kwargs).to(self.munit/(r.unit)**3)).value,self.min_r.to(r.unit).value,r.value)[0]
            return m*self.munit

    def phiode(self,y,r,*args):
        phi,r2dpdr=y
        return [(1.0/r/r*r2dpdr).value,(r*r*4*pi*GN*self.get_rho(r,args)).value]
    def get_phi(self,r,*args,**kwargs):#DOES NOT WORK
#        print('this function doesn\'t work yet')
#        final=odeint(lambda y,r,*args: self.phiode([y[0]*GN.unit*self.munit/self.runit,y[1]*GN.unit*self.munit/self.runit/self.runit],r*self.runit,args),[(-4*pi*GN*self.rho0*self.rs**2).value,0],[self.min_r.value,r.value])
#        return -4*pi*GN*derivative(lambda y,*args: y*y*d1(y,args),r,args=args,dx=.1*r)/r/r

        # BT 2.122

        if self.phifunc!=None:
            print('phi',self.phifunc(r))
            return self.phifunc(r)*self.vunit**2
        else:
            
            i1=1.0/(r.to(units.kpc).value)*quad(lambda rp:self.get_rho(rp*units.kpc).to(units.M_sun/units.kpc**3).value*rp**2,0,r.to(units.kpc).value)[0]*units.M_sun/units.kpc
            i2=quad(lambda nu: self.get_rho(1.0/np.sqrt(nu)*units.kpc).to(units.M_sun/units.kpc**3).value/nu**2/2.0,0,1.0/(r.to(units.kpc).value)**2)[0]*units.M_sun/units.kpc

            return -4*np.pi*GN*(i1+i2)

    def get_sigma(self,r):
        intg=quad(lambda nu: self.get_mass(1.0/nu*units.kpc).value*self.get_rho(1.0/nu*units.kpc).value*nu**4,0,(1.0/r).value)[0]
        return GN*intg/self.get_rho(r)

    def get_vcirc(self,r,*args,**kwargs):
        if r==0:
            return 0*self.vunit
        return sqrt(GN*self.get_mass(r,*args,**kwargs)/r).to(self.vunit)

    def get_vmax(self,*args,**kwargs):
        return -minimize_scalar(lambda x: -self.get_vcirc(x*self.runit,*args,**kwargs).value)['fun']*(self.vunit)

    def get_rmax(self,*args,**kwargs):
        return minimize_scalar(lambda x: -self.get_vcirc(x*self.runit,*args,**kwargs).value)['x']*(self.runit)

    def get_meanrho(self,r,*args,**kwargs):
        return self.get_mass(r,*args,**kwargs)/(4*pi/3*r**3)

    def get_projected(self,r,maxr):
        smallnum=1E-10
        projint=lambda x:(self.get_rho(x*r.unit).to(self.munit/(r.unit**3))*(x*r.unit)/(sqrt((x*r.unit)**2-r**2))).value
        mr=maxr.to(r.unit).value
        return 2*quad(projint,r.value+smallnum,mr-smallnum)[0]*self.munit/(r.unit**2)

class Zhao(RhoProfile):

    alpha=0
    beta=0
    gamma=0
    rho0=0
    rs=0
    rvir=0
    mvir=0
    deltavirrhou=0
    c=0
    xmax=-1
    rdecrvir=0

    def __init__(self,alpha,beta,gamma,rho0=0,rs=0,rvir=0,mvir=0,deltavirrhou=0,c=0,rdecrvir=0,vatr=0,rforv=0,vmax=0,rmax=0,rhounit=units.M_sun/(units.kpc)**3,munit=units.M_sun,vunit=units.km/units.s,runit=units.kpc):

        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.rhounit=rhounit
        self.munit=munit
        self.vunit=vunit
        self.runit=runit
        self.rdecrvir=rdecrvir


        if rs==0 and c!=0 and rvir!=0:
            rs=rvir/c
        if rvir==0 and c!=0 and rs!=0:
            rvir=rs*c

        if vmax!=0 and rmax!=0:
            vatr=vmax
            rforv=rmax
        if rho0!=0 and rs!=0 and rvir!=0:
            self.rho0=rho0
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            rhof=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhof)
            self.mvir=self.get_mass(rvir)
            self.deltavirrhou=self.mvir/(4*pi/3*rvir**3)
        elif mvir!=0 and rs!=0 and rvir!=0:
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            self.deltavirrhou=mvir/(4*pi/3*rvir**3)
            self.mvir=mvir
            if rdecrvir==0:
                m1=getmassfromzhao0(alpha,beta,gamma,1*mvir.unit/(rs.unit**3),rs,rvir)
                self.rho0=mvir/m1*mvir.unit/(rs.unit**3)
            else:
                rhof1=lambda r,rhos: self.rhofunc(r*rs.unit,self.alpha,self.beta,self.gamma,rhos*(mvir.unit/(rs.unit**3)),self.rs,self.rvir,rdecrvir=rdecrvir)
                m1=lambda rho0: getmassfromprofile(rhof1,rvir,rho0)
                x=minimize_scalar(lambda y: abs(log10(m1(y)/mvir)))['x']
                self.rho0=x*mvir.unit/(rs.unit**3)
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)
        elif deltavirrhou!=0 and rs!=0 and rvir!=0:
            mvir=4.0*pi/3*deltavirrhou*rvir**3
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            self.mvir=mvir
            self.deltavirrhou=deltavirrhou
            if rdecrvir==0:
                m1=getmassfromzhao0(alpha,beta,gamma,1*mvir.unit/(rs.unit**3),rs,rvir)
                self.rho0=mvir/m1*mvir.unit/(rs.unit**3)
            else:
                rhof1=lambda r,rhos: self.rhofunc(r*rs.unit,self.alpha,self.beta,self.gamma,rhos*(mvir.unit/(rs.unit**3)),self.rs,self.rvir,rdecrvir=rdecrvir)
                m1=lambda rho0: getmassfromprofile(rhof1,rvir,rho0)
                x=minimize_scalar(lambda y: abs(log10(m1(y)/mvir)))['x']
                self.rho0=x*mvir.unit/(rs.unit**3)
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)
        elif rho0!=0 and rvir!=0 and mvir!=0:
            self.rho0=rho0
            self.rvir=rvir
            self.mvir=mvir
            self.deltavirrhou=mvir/(4*pi/3*rvir**3)
            if rdecrvir==0:
                m1=lambda rsa: getmassfromzhao0(alpha,beta,gamma,rho0,rsa,rvir)
            else:
                rhof1=lambda r,rsa: self.rhofunc(r*rvir.unit,self.alpha,self.beta,self.gamma,self.rho0,rsa*rvir.unit,self.rvir,rdecrvir=rdecrvir)
                m1=lambda rsa: getmassfromprofile(rhof1,rvir,rsa)
            x=minimize_scalar(lambda y: abs(log10(m1(y)/mvir)))['x']
            self.rs=x*rvir.unit
            self.c=self.rvir/self.rs
            rhofn=lambda r: self.rhofunc(r*rvir.unit,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)
        elif rho0!=0 and rs!=0 and mvir!=0:
            self.rho0=rho0
            self.mvir=mvir
            self.rs=rs
            if rdecrvir==0:
                m1=lambda rvira: getmassfromzhao0(alpha,beta,gamma,rho0,rs,rvira)
            else:
                rhof1=lambda r,rvira: self.rhofunc(r*rs.unit,self.alpha,self.beta,self.gamma,self.rho0,self.rs,rvira*rs.unit,rdecrvir=rdecrvir)
                m1=lambda rvira: getmassfromprofile(rhof1,rvira,rvira)
            x=minimize_scalar(lambda y: abs(log10((m1(y)/mvir).decompose().value)))['x']
            self.rvir=x*rs.unit
            self.c=self.rvir/self.rs
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif rho0!=0 and rs!=0 and deltavirrhou!=0:
            self.rho0=rho0
            self.rs=rs
            self.deltavirrhou=deltavirrhou
            self.c=self.rvir/self.rs
            if rdecrvir==0:
                m1=lambda rvira: getmassfromzhao0(alpha,beta,gamma,rho0,rs,rvira)
            else:
                rhof1=lambda r,rvira: self.rhofunc(r*rs.unit,self.alpha,self.beta,self.gamma,self.rho0,self.rs,rvira*rs.unit,rdecrvir=rdecrvir)
                m1=lambda rvira: getmassfromprofile(rhof1,rvira,rvira)
            x=minimize_scalar(lambda y: abs((m1(y*rs.unit)/(4*pi/3*(y*rs.unit)**3)).value-(deltavirrhou.to(units.M_sun/rs.unit**3).value)))['x']
            self.rvir=x*rs.unit
            self.mvir=4*pi/3*(x*rs.unit)**3*deltavirrhou
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)
        elif rho0!=0 and rvir!=0 and deltavirrhou!=0:
            self.rho0=rho0
            self.rvir=rvir
            self.deltavirrhou=deltavirrhou
            mvir=4*pi/3*rvir**3
            self.mvir=mvir
            if rdecrvir==0:
                m1=lambda rsa: getmassfromzhao0(alpha,beta,gamma,rho0,rsa,rvir)
            else:
                rhof1=lambda r,rsa: self.rhofunc(r*rvir.unit,self.alpha,self.beta,self.gamma,self.rho0,rsa*rvir.unit,self.rvir,rdecrvir=rdecrvir)
                m1=lambda rsa: getmassfromprofile(rhof1,rvir,rsa)
            x=minimize_scalar(lambda y: abs(log10(m1(y)/mvir)))['x']
            self.rs=x*rvir.unit
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)
        elif rho0!=0 and mvir!=0 and deltavirrhou!=0:
            self.rho0=rho0
            self.mvir=mvir
            self.deltavirrhou=deltavirrhou
            rvir=(mvir/(4*pi/3*deltavirrhou))**(1.0/3)
            self.rvir=rvir
            self.c=self.rvir/self.rs
            if rdecrvir==0:
                m1=lambda rsa: getmassfromzhao0(alpha,beta,gamma,rho0,rsa,rvir)
            else:
                rhof1=lambda r,rsa: self.rhofunc(r*rvir.unit,self.alpha,self.beta,self.gamma,self.rho0,rsa*rvir.unit,self.rvir,rdecrvir=rdecrvir)
                m1=lambda rsa: getmassfromprofile(rhof1,rvir,rsa)
            x=minimize_scalar(lambda y: abs(log10(m1(y)/mvir)))['x']
            self.rs=x*rvir.unit
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif rs!=0 and mvir!=0 and deltavirrhou!=0:
            self.rs=rs
            self.mvir=mvir
            rvir=(mvir/(4*pi/3*deltavirrhou))**(1.0/3)
            self.deltavirrhou=deltavirrhou
            self.rvir=rvir
            self.c=self.rvir/self.rs
            if rdecrvir==0:
                m1=getmassfromzhao0(alpha,beta,gamma,1.0*mvir.unit/(rs.unit**3),rs,rvir)
                self.rho0=mvir/m1*mvir.unit/(rs.unit**3)
            else:
                rhof1=lambda r,rhos: self.rhofunc(r*rs.unit,self.alpha,self.beta,self.gamma,rhos*(mvir.unit/(rs.unit**3)),self.rs,self.rvir,rdecrvir=rdecrvir)
                m1=lambda rho0: getmassfromprofile(rhof1,rvir,rho0)
                x=minimize_scalar(lambda y: abs(log10(m1(y)/mvir)))['x']
                self.rho0=x
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif c!=0 and mvir!=0 and deltavirrhou!=0:
            self.c=c
            self.mvir=mvir
            self.deltavirrhou=deltavirrhou
            self.rvir=(mvir/(4*pi/3*deltavirrhou))**(1.0/3)
            self.rs=self.rvir/copy.copy(c)
            if rdecrvir==0:
                m1=getmassfromzhao0(alpha,beta,gamma,1*self.mvir.unit/(self.rs.unit**3),self.rs,self.rvir)
                self.rho0=self.mvir/m1*self.mvir.unit/(self.rs.unit**3)
            else:
                rhof1=lambda r,rhos: self.rhofunc(r*self.rs.unit,self.alpha,self.beta,self.gamma,rhos*(self.mvir.unit/(self.rs.unit**3)),self.rs,self.rvir,rdecrvir=rdecrvir)
                m1=lambda rho0: getmassfromprofile(rhof1,self.rvir,self.rho0)
                x=minimize_scalar(lambda y: abs(log10(m1(y)/self.mvir)))['x']
                self.rho0=x*self.mvir.unit/(self.rs.unit**3)
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif rmax!=0 and vmax!=0 and mvir!=0 and rdecrvir==0:

            self.mvir=mvir
            self.rs=rmax/getxmaxzhao0(self.alpha,self.beta,self.gamma)
            mwithinrmax=(vmax**2*rmax/GN).to(mvir.unit)
            m2=getmassfromzhao0(self.alpha,self.beta,self.gamma,1*mvir.unit/(rmax.unit**3),self.rs,rmax)
            self.rho0=(mwithinrmax/m2*(mvir.unit/rmax.unit**3)).to(self.rhounit)
            m1=lambda rvira: getmassfromzhao0(self.alpha,self.beta,self.gamma,self.rho0,self.rs,rvira*units.kpc)

            rvirlo,rvirhi=self.bracket_rvir_m_zhao(self.alpha, self.beta,
                self.gamma, self.rs, self.rho0, mvir)
            rvirlo=rvirlo.to(rmax.unit)
            rvirhi=rvirhi.to(rmax.unit)

            x=minimize_scalar(lambda y: abs(log10((m1(y)/mvir).decompose().value)),bounds=(rvirlo.value,rvirhi.value),method='bounded')
            self.rvir=x['x']*rmax.unit
            self.c=self.rvir/self.rs
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif rmax!=0 and vmax!=0 and rvir!=0 and rdecrvir==0:
            self.rvir=rvir
            self.rs=rmax/getxmaxzhao0(self.alpha,self.beta,self.gamma)
            mwithinrmax=vmax**2*rmax/GN
            m2=getmassfromzhao0(self.alpha,self.beta,self.gamma,1*units.M_sun/(rmax.unit**3),self.rs,rmax)
            self.rho0=(mwithinrmax/m2*(units.M_sun/rmax.unit**3)).to(self.rhounit)
            self.mvir=getmassfromzhao0(self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir).to(self.munit)
            self.c=self.rvir/self.rs
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif vmax!=0 and mvir!=0 and deltavirrhou!=0 and rdecrvir==0:
            self.mvir=mvir
            self.deltavirrhou=deltavirrhou
            self.rvir=(mvir/(4*pi/3*deltavirrhou))**(1.0/3)
            mmax=lambda params: (np.log10(getmassfromzhao0(self.alpha,self.beta,self.gamma,10**params[0]*units.M_sun/units.kpc**3,params[1]*units.kpc,self.rvir).value/mvir.value))**2+(np.log10(getmassfromzhao0(self.alpha,self.beta,self.gamma,10**params[0]*units.M_sun/units.kpc**3,params[1]*units.kpc,params[1]*units.kpc*getxmaxzhao0(self.alpha,self.beta,self.gamma)).value/(vmax**2*params[1]*units.kpc*getxmaxzhao0(self.alpha,self.beta,self.gamma)/GN).to(units.M_sun).value))**2
            x=minimize(mmax,[6,.1*self.rvir.value],bounds=[[-2,30],[.01,1000]])
            xwithunit=[10**x['x'][0]*mvir.unit/(units.kpc**3),x['x'][1]*units.kpc]
            self.rho0=xwithunit[0]
            self.rs=xwithunit[1]
            self.c=self.rvir/self.rs
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif rmax!=0 and vmax!=0 and deltavirrhou!=0 and rdecrvir==0:

            self.rs=rmax/getxmaxzhao0(self.alpha,self.beta,self.gamma)

            mwithinrmax=(vmax**2*rmax/GN).to(units.M_sun)
            m2=getmassfromzhao0(self.alpha,self.beta,self.gamma,1*units.M_sun/(rmax.unit**3),self.rs,rmax)
            self.rho0=(mwithinrmax/m2*(units.M_sun/rmax.unit**3)).to(self.rhounit)
            self.deltavirrhou=deltavirrhou
            if rdecrvir==0:
                m1=lambda rvira: getmassfromzhao0(alpha,beta,gamma,self.rho0,self.rs,rvira)
            else:
                rhof1=lambda r,rvira: self.rhofunc(r*rs.unit,self.alpha,self.beta,self.gamma,self.rho0,self.rs,rvira*rs.unit,rdecrvir=rdecrvir)
                m1=lambda rvira: getmassfromprofile(rhof1,rvira,rvira)
            rvirlo, rvirhi = self.bracket_rvir_zhao(self.alpha,
                self.beta, self.gamma, self.rs, self.rho0, deltavirrhou)
            x=minimize_scalar(lambda y: abs((m1(y*rmax.unit)/(4*pi/3*(y*rmax.unit)**3)).value-(deltavirrhou.to(units.M_sun/rmax.unit**3).value)),bounds=(rvirlo.value,rvirhi.value),method='bounded')['x']
            # print('rv', minimize_scalar(lambda y: abs((m1(y*rmax.unit)/(4*pi/3*(y*rmax.unit)**3)).value-(deltavirrhou.to(units.M_sun/rmax.unit**3).value)))['x'])
            self.rvir=x*rmax.unit
            self.mvir=(4*pi/3*(x*rmax.unit)**3*deltavirrhou).to(self.munit)
            self.c=self.rvir/self.rs
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif vatr!=0 and rforv!=0 and mvir!=0:
            self.mvir=mvir
            rhof1=lambda r,haloparams: self.rhofunc(r*rforv.unit,self.alpha,self.beta,self.gamma,haloparams[0]*(mvir.unit/(rforv.unit**3)),haloparams[1]*rforv.unit,haloparams[2]*rforv.unit,rdecrvir=rdecrvir)
            m1=lambda haloparams: getmassfromprofile(rhof1,rforv,haloparams)
            m2=lambda haloparams: getmassfromprofile(rhof1,haloparams[2],haloparams)

            x=minimize(lambda params: abs((GN*m1([params[0]*mvir.unit/(rforv.unit**3),params[1]*rforv.unit,params[2]*rforv.unit])/rforv).to(vatr.unit**2)-vatr**2).value,[(20*mvir/(4*pi/3*rforv**3)).value,rforv.value,5*rforv.value],constraints={'type':'eq','fun': lambda params: log10(m2([params[0]*mvir.unit/(rforv.unit**3),params[1]*rforv.unit,params[2]*rforv.unit])/mvir)})

            xwithunit=[x['x'][0]*mvir.unit/(rforv.unit**3),x['x'][1]*rforv.unit,x['x'][2]*rforv.unit]
            self.rho0=xwithunit[0]
            self.rs=xwithunit[1]
            self.rvir=xwithunit[2]
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif vatr!=0 and rforv!=0 and rvir!=0:
            self.rvir=rvir
            rhof1=lambda r,haloparams: self.rhofunc(r*rforv.unit,self.alpha,self.beta,self.gamma,haloparams[0],haloparams[1],rvir,rdecrvir=rdecrvir)
            m1=lambda haloparams: getmassfromprofile(rhof1,rforv,haloparams)

            mguess=(vatr**2*rforv/GN).to(units.M_sun)

            x=minimize(lambda params: abs(((GN*m1([params[0]*units.M_sun/(rforv.unit**3),params[1]*rforv.unit])/rforv).to(vatr.unit**2)-vatr**2).value),[(mguess/rforv**3).value,3*rforv.value])
            xwithunit=[x['x'][0]*units.M_sun/(rforv.unit**3),x['x'][1]*rforv.unit]
            self.rho0=xwithunit[0]
            self.rs=xwithunit[1]
            self.mvir=getmassfromprofile(rhof1,rvir,xwithunit)
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)


    def rhofunc(self,r,alpha=nan,beta=nan,gamma=nan,rho0=nan,rs=nan,rvir=nan,rdecrvir=.1):

        if ~isfinite(alpha):
            alpha=self.alpha
        if ~isfinite(beta):
            beta=self.beta
        if ~isfinite(gamma):
            gamma=self.gamma
        if ~isfinite(rho0):
            rho0=self.rho0
        if ~isfinite(rs):
            rs=self.rs
        if ~isfinite(rvir):
            rvir=self.rvir

        if len(shape(r))>0:

            if rdecrvir==0:
                top=rho0
                roverrs=(r/rs).decompose().value
                ba=(roverrs)**gamma
                bb=(1+(roverrs)**alpha)
                bc=bb**((beta-gamma)/alpha)

                return top/(ba*bc)

            else:

                rho=zeros(len(r))
                w0=where(r==0)
                rho[w0]=inf
                win=where(r<=rvir)
                wout=where(r>rvir)

                top=rho0
                ba=(r[win]/rs)**gamma
                bb=(1+(r[win]/rs)**alpha)
                bc=bb**((beta-gamma)/alpha)

                rho[win]=top/(ba*bc)

                rdec=rdecrvir*rvir

                c=rvir/rs
                epsilon=(-gamma-beta*c**alpha)/(1+c**alpha)+1.0/rdecrvir

                ca=rho0/(c**gamma*(1+c**alpha)**((beta-gamma)/alpha))
                cb=(r[wout]/rvir)**epsilon
                cd=exp(-(r[wout]-rvir)/rdec)

                rho[wout]=ca*cb*cd

                return rho

        else:

            if r<=rvir or rdecrvir==0:
                if r==0 and gamma>0:
                    return inf
                top=rho0
                roverrs=(r/rs).decompose().value
                ba=(roverrs)**gamma
                bb=(1+(roverrs)**alpha)
                bc=bb**((beta-gamma)/alpha)

                return top/(ba*bc)

            else:

                rdec=rdecrvir*rvir

                c=rvir/rs
                epsilon=(-gamma-beta*c**alpha)/(1+c**alpha)+1.0/rdecrvir

                ca=rho0/(c**gamma*(1+c**alpha)**((beta-gamma)/alpha))
                cb=(r/rvir)**epsilon
                cd=exp(-(r-rvir)/rdec)

                return ca*cb*cd

    def get_mass(self,r):
        if self.rdecrvir==0:
            return getmassfromzhao0(self.alpha,self.beta,self.gamma,self.rho0,self.rs,r).to(self.munit)
        else:
            return super(Zhao,self).get_mass(r)

    def get_mtot(self):
        return (4*np.pi*self.rho0*self.rs**3*(gammafunc((self.beta-3.0)/self.alpha)*gammafunc((3.0-self.gamma)/self.alpha))/(self.alpha*gammafunc(1.0*(self.beta-self.gamma)/self.alpha))).to(self.munit)
        
    def rmax_ftomin(self,xm):
        print(('m',xm))
        if xm<=0:
            return np.inf
        else:

            a=1.0*(3-self.gamma)/self.alpha
            b=1.0*(self.beta-self.gamma)/self.alpha
            c=1.0*(self.alpha-self.gamma+3)/self.alpha



            top=lambda xt: xt**(3-self.gamma)/(3-self.gamma)*hyp2f1(a,b,c,-xt**self.alpha)
            bot1=lambda xt: xt**(2-self.gamma)*hyp2f1(a,b,c,-xt**self.alpha)
            bot2=lambda xt: xt**(3-self.gamma)/(3-self.gamma)*(-self.alpha*xt**(self.alpha-1))*a*b/c*hyp2f1(a+1,b+1,c+1,-xt**self.alpha)

            print((top(xm),bot1(xm),bot2(xm)))
            print((abs(log10(top(xm))-log10(xm*(bot1(xm)+bot2(xm))))))
            #return abs(log10(top(xm))-log10(xm*(bot1(xm)+bot2(xm))))
            return abs(top(xm)/(xm*(bot1(xm)+bot2(xm))))


    def get_rmax(self):
        if self.xmax==-1:
            a=float(self.alpha)
            b=float(self.beta)
            g=float(self.gamma)

#            x=fsolve(lambda x: (-1.*(3 - g)*x**(1 - g)*((1 + x**a)**(-(b - g)/a) - hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a)))/(-3. + g) - (1.*(2 - g)*x**(1 - g)*hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a))/(-3. + g),1,fprime=lambda x:(x**(2 - g)*(((3 - g)*(-((b - g)*x**(-1 + a)*(1 + x**a)**(-1 - (b - g)/a)) - ((3 - g)*((1 + x**a)**(-(b - g)/a) - hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a)))/x))/x - ((3 - g)*((1 + x**a)**(-(b - g)/a) - hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a)))/x**2))/(3 - g) + (2*(2 - g)*((1 + x**a)**(-(b - g)/a) - hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a)))/x**g + ((1 - g)*(2 - g)*hyp2f1((3 - g)/a,(b - g)/a,(3 + a - g)/a,-x**a))/((3 - g)*x**g))

            x=minimize_scalar(lambda x: -getmassfromzhao0(a,b,g,1,1,10**x)/10**x)


            #a=1.0*(3-self.gamma)/self.alpha
            #b=1.0*(self.beta-self.gamma)/self.alpha
            #c=1.0*(self.alpha-self.gamma+3)/self.alpha


#            print(a,b,c)
#            top=lambda xm: xm**(3-self.gamma)/(3-self.gamma)*hyp2f1(a,b,c,-xm**self.alpha)
#            bot1=lambda xm: xm**(2-self.gamma)*hyp2f1(a,b,c,-xm**self.alpha)
#            bot2=lambda xm: xm**(3-self.gamma)/(3-self.gamma)*(-self.alpha*xm**(self.alpha-1))*a*b/c*hyp2f1(a+1,b+1,c+1,-xm**self.alpha)

            #minfunc=lambda xm:abs(log10(top(xm))-log10(xm*(bot1(xm)+bot2(xm))))
            #x=minimize_scalar(lambda xm: self.rmax_ftomin(xm),[.5,1,100])
            self.xmax=10**x['x']
            return self.xmax*self.rs
        else:
            return self.xmax*self.rs

    def get_vmax(self):
        rmax=self.get_rmax()
        return self.get_vcirc(rmax)

    def get_sigma_0(self):
        if self.beta==4 and self.alpha==1:
            return 3.0*(1-self.gamma)/(4*(3.0-2*self.gamma)*(5-2*self.gamma))*self.GN/self.rs

    def get_phi(self,r):
        return zhaopotential(r,self.rho0,self.rs,self.alpha,self.beta,self.gamma)

    def get_projected(self,r):

        if r==0:
            if self.gamma==1:
                return np.inf
            else:
                return 2*self.rs*self.rho0*abs(gammafunc(1-self.gamma)*gammafunc(self.gamma-1+1.0*self.beta/self.alpha-1.0*self.gamma/self.alpha)/gammafunc(1.0*self.beta/self.alpha-self.gamma/self.alpha))

        modbeta=self.alpha*(self.beta-3)
        if isinstance(self.alpha,int) and self.alpha>0 and isinstance(modbeta,int) and modbeta>0:
            ui=0
            for i in range(0,self.alpha*(2-self.gamma)):
                print(i)
                print((self.alpha,self.beta,self.gamma))
                print((zhaofi(i,self.alpha,self.beta,self.gamma)))
                print((zhaou(self.alpha*(self.beta-2)+i,r/self.rs,self.alpha)))
                ui=ui+zhaofi(i,self.alpha,self.beta,self.gamma)*zhaou(self.alpha*(self.beta-2)+i,r/self.rs,self.alpha)

            return ui/(r/self.rs)


    def get_dispersion(self,r):
        x=r/self.rs
        drhodroverrho=-self.gamma/r-(self.beta-self.gamma)/self.alpha**2*x**(1.0/self.alpha-1)/self.rs/(1+x)
        return (-G*self.get_mass(r)/r**2/drhodroverrho).to((units.km/units.s)**2)


    def bracket_rvir_zhao(self,alpha,beta,gamma,rs,rho0,deltavirrhou):
        """
        This method takes the Zhao profile spectral parameters (alpha,
        beta, gamma), scale radius rs, scale density rho0, and the mean
        virial density of halos deltavirrhou. It returns two radii
        (rlo, rhi) satisfying rlo < rvir < rhi where rvir is the halo
        virial radius. This works for Zhao profiles where 0<=gamma<beta,
        gamma<3, and alpha>0.
        """
        dentest = 3*getmassfromzhao0(alpha, beta, gamma, rho0, rs, rs) / \
            (4 * np.pi * rs**3)

        if gamma == 0.:
            # Cored profile with central density rho0. Use a profile f(r)
            # with f(0)=rho0 and matches the inner mean density at rs. We
            # also want its slope to be less steep than the inner mean
            # density at rs and all radii larger than rs.
            # Consider f(r) = rho0 / (r/C+1)**B
            #   where B=3*(1-rho(rs)/rhobar(rs)) with rho(r) the density
            #   profile and rhobar(r) the inner mean density profile, and
            #   C = 1 / ( (rho0/rhobar(rs)) ^ (1/B) - 1 )
            dens = rho0 / 2**(beta/alpha)
            expon = 1.0 / 3.0 / (1.0 - (dens/dentest).decompose().value)
            rtest = rs * (((rho0 / deltavirrhou).decompose().value)**expon - 1.0) / \
                (((rho0 / dentest).decompose().value)**expon - 1.0)
        else:
            # Use a profile f(r) that matches the inner mean density at rs,
            # but whose slope is less steep everywhere.
            # Consider f(r) = const / r**gamma
            try:
                rtest = rs * ((dentest / deltavirrhou).decompose().value)**(1/gamma)
            except OverflowError:
                dens = rho0 / 2**(beta/alpha)
                expon = 1.0 / 3.0 / (1.0 - (dens/dentest).decompose().value)
                rtest = rs * (((rho0 / deltavirrhou).decompose().value)**expon - 1.0) / \
                    (((rho0 / dentest).decompose().value)**expon - 1.0)


        if rs < rtest:
            return rs, rtest
        else:
            return rtest, rs

    def bracket_rvir_m_zhao(self,alpha,beta,gamma,rs,rho0,mvir):
        """
        This method takes the Zhao profile spectral parameters (alpha,
        beta, gamma), scale radius rs, scale density rho0, and the
        virial mass mvir. It returns two radii (rlo, rhi) satisfying
        rlo < rvir < rhi where rvir is the halo virial radius. This
        works for Zhao profiles where 0<=gamma<beta, gamma<3, and
        alpha>0.
        """
        # Use a test mass profile Mtest(r) = M(rs)*(r/rs)**3.
        ms = getmassfromzhao0(alpha, beta, gamma, rho0, rs, rs)
        rtest = rs * (mvir / ms)**(1./3.)
        if rs < rtest:
            return rs, rtest
        else:
            return rtest, rs


class NFW(RhoProfile):

    rho0=0
    rs=0
    rvir=0
    mvir=0
    deltavirrhou=0
    c=0
    xmax=2.163
    munit=0
    rhounit=0
    runit=0

    def __init__(self,rho0=0,rs=0,rvir=0,mvir=0,deltavirrhou=0,c=0,vatr=0,rforv=0,vmax=0,rmax=0):

        self.rhounit=units.M_sun/units.kpc**3
        self.runit=units.kpc
        self.munit=units.M_sun


        if rs==0 and c!=0 and rvir!=0:
            rs=rvir/c
        if rvir==0 and c!=0 and rs!=0:
            rvir=rs*c

        if vmax!=0 and rmax!=0:
            vatr=vmax
            rforv=rmax
        if rho0!=0 and rs!=0 and rvir!=0:
            self.rho0=rho0
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            rhof=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhof)
            self.mvir=self.get_mass(rvir)
            self.deltavirrhou=self.mvir/(4*pi/3*rvir**3)
        elif mvir!=0 and rs!=0 and rvir!=0:
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            self.deltavirrhou=mvir/(4*pi/3*rvir**3)
            self.mvir=mvir
            m1=getmassfromnfw(1*mvir.unit/(rs.unit)**3,rs,rvir)
            self.rho0=mvir/m1*(mvir.unit/rs.unit**3)
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)
        elif deltavirrhou!=0 and rs!=0 and rvir!=0:
            mvir=4.0*pi/3*deltavirrhou*rvir**3
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            self.mvir=mvir
            self.deltavirrhou=deltavirrhou
            m1=getmassfromnfw(1*mvir.unit/(rs.unit)**3,rs,rvir)
            self.rho0=mvir/m1*mvir.unit/(rs.unit**3)
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)
        elif rho0!=0 and rvir!=0 and mvir!=0:
            self.rho0=rho0
            self.rvir=rvir
            self.mvir=mvir
            self.deltavirrhou=mvir/(4*pi/3*rvir**3)
            m1=lambda rsa: getmassfromnfw(rho0,rsa,rvir)
            x=minimize_scalar(lambda y: abs(log10(m1(y)/mvir)))['x']
            self.rs=x*rvir.unit
            self.c=self.rvir/self.rs
            rhofn=lambda r: self.rhofunc(r*rvir.unit,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)
        elif rho0!=0 and rs!=0 and mvir!=0:
            self.rho0=rho0
            self.mvir=mvir
            self.rs=rs
            m1=lambda rvira: getmassfromnfw(rho0,rs,rvira)
            x=minimize_scalar(lambda y: abs(log10(m1(x)/mvir)))['x']
            self.rvir=x*rs.unit
            self.c=self.rvir/self.rs
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif rho0!=0 and rs!=0 and deltavirrhou!=0:
            self.rho0=rho0
            self.rs=rs
            self.deltavirrhou=deltavirrhou
            self.c=self.rvir/self.rs
            m1=lambda rvira: getmassfromnfw(rho0,rs,rvira)
            x=minimize_scalar(lambda y: abs(m1(y)/(4*pi/3*y**3)-deltavirrhou))['x']
            self.rvir=x*rs.unit
            self.mvir=4*pi/3*(x*rs.unit)**3*deltavirrhou
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)
        elif rho0!=0 and rvir!=0 and deltavirrhou!=0:
            self.rho0=rho0
            self.rvir=rvir
            self.deltavirrhou=deltavirrhou
            mvir=4*pi/3*rvir**3
            self.mvir=mvir
            m1=lambda rsa: getmassfromnfw(rho0,rsa,rvir)
            x=minimize_scalar(lambda y: abs(log10(m1(y)/mvir)))['x']
            self.rs=x*rvir.unit
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif rho0!=0 and mvir!=0 and deltavirrhou!=0:
            self.rho0=rho0
            self.mvir=mvir
            self.deltavirrhou=deltavirrhou
            rvir=(mvir/(4*pi/3*deltavirrhou))**(1.0/3)
            self.rvir=rvir
            self.c=self.rvir/self.rs
            m1=lambda rsa: getmassfromnfw(rho0,rsa,rvir)
            x=minimize_scalar(lambda y: abs(log10(m1(y)/mvir)))['x']
            self.rs=x*rvir.unit
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif rs!=0 and mvir!=0 and deltavirrhou!=0:
            self.rs=rs
            self.mvir=mvir
            rvir=(mvir/(4*pi/3*deltavirrhou))**(1.0/3)
            self.deltavirrhou=deltavirrhou
            self.rvir=rvir
            self.c=self.rvir/self.rs
            m1=getmassfromnfw(1*mvir.unit/(rs.unit**3),rs,rvir)
            self.rho0=mvir/m1*(mvir.unit/(rs.unit**3))
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif c!=0 and mvir!=0 and deltavirrhou!=0:
            self.c=c
            self.mvir=mvir
            self.deltavirrhou=deltavirrhou
            self.rvir=(mvir/(4*pi/3*deltavirrhou))**(1.0/3)
            self.rs=self.rvir/copy.copy(c)
            m1=getmassfromnfw(1*self.mvir.unit/(self.rs.unit**3),self.rs,self.rvir)
            self.rho0=mvir/m1*self.mvir.unit/(self.rs.unit**3)
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif vmax!=0 and rmax!=0 and mvir!=0:
            self.mvir=mvir
            self.rs=rmax/2.163
            mwithinrmax=vmax**2*rmax/GN
            m2=getmassfromnfw(1*mvir.unit/(rmax.unit**3),self.rs,rmax)
            self.rho0=mwithinrmax/m2*(mvir.unit/rmax.unit**3)
            m1=lambda rvira: getmassfromnfw(rho0,rs,rvira)
            x=minimize_scalar(lambda y: abs(log10(m1(x)/mvir)))['x']
            self.rvir=x*rmax.unit
            self.c=self.rvir/self.rs
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif vmax!=0 and rmax!=0 and rvir!=0:
            self.rvir=rvir
            self.rs=rmax/2.163
            mwithinrmax=vmax**2*rmax/GN
            m2=getmassfromnfw(1*self.munit/(rmax.unit**3),self.rs,rmax)
            self.rho0=mwithinrmax/m2*(self.munit/rmax.unit**3)
            self.mvir=getmassfromnfw(self.rho0,self.rs,self.rvir)
            self.c=self.rvir/self.rs
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif vmax!=0 and mvir!=0 and rvir!=0:
            self.rvir=rvir
            self.mvir=mvir
            vvir2=(GN*mvir/rvir)
            conc=10**minimize_scalar(lambda con: abs(vmax**2-.216*vvir2*10**con/nfwfx(10**con)),bounds=[-1,3],method='bounded')['x']

            self.rs=rvir/conc
            self.c=conc
            m1=getmassfromnfw(1*self.mvir.unit/(self.rs.unit**3),self.rs,self.rvir)
            self.rho0=mvir/m1*(self.mvir.unit/(self.rs.unit**3))
            self.c=self.rvir/self.rs
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif vatr!=0 and rforv!=0 and mvir!=0:
            self.mvir=mvir
            rhof1=lambda r,haloparams: self.rhofunc(r*rforv.unit,self.alpha,self.beta,self.gamma,haloparams[0]*(mvir.unit/(rforv.unit**3)),haloparams[1]*rforv.unit,haloparams[2]*rforv.unit,rdecrvir=rdecrvir)
            m1=lambda haloparams: getmassfromprofile(rhof1,rforv,haloparams)
            m2=lambda haloparams: getmassfromprofile(rhof1,haloparams[2],haloparams)

            x=minimize(lambda params: abs((GN*m1([params[0]*mvir.unit/(rforv.unit**3),params[1]*rforv.unit,params[2]*rforv.unit])/rforv).to(vatr.unit**2)-vatr**2).value,[(20*mvir/(4*pi/3*rforv**3)).value,rforv.value,5*rforv.value],constraints={'type':'eq','fun': lambda params: log10(m2([params[0]*mvir.unit/(rforv.unit**3),params[1]*rforv.unit,params[2]*rforv.unit])/mvir)})

            xwithunit=[x['x'][0]*mvir.unit/(rforv.unit**3),x['x'][1]*rforv.unit,x['x'][2]*rforv.unit]
            self.rho0=xwithunit[0]
            self.rs=xwithunit[1]
            self.rvir=xwithunit[2]
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.alpha,self.beta,self.gamma,self.rho0,self.rs,self.rvir,rdecrvir=rdecrvir)
            RhoProfile.__init__(self,rhofn)

        elif vatr!=0 and rforv!=0 and rvir!=0:
            self.rvir=rvir
            rhof1=lambda r,haloparams: self.rhofunc(r,haloparams[0],haloparams[1],rvir)
            m1=lambda haloparams: getmassfromprofile(rhof1,rforv,haloparams)
            m2=lambda haloparams: getmassfromprofile(rhof1,rvir,haloparams)

            x=minimize(lambda params: abs((GN*m1([params[0]*units.M_sun/(rforv.unit**3),params[1]*rforv.unit])/rforv).to(vatr.unit**2)-vatr**2).value,[4000,rforv.value])

            xwithunit=[x['x'][0]*units.M_sun/(rforv.unit**3),x['x'][1]*rforv.unit]
            self.rho0=xwithunit[0]
            self.rs=xwithunit[1]
            self.mvir=getmassfromprofile(rhof1,rvir,xwithunit)
            self.deltavirrhou=self.mvir/(4*pi/3*self.rvir**3)
            rhofn=lambda r: self.rhofunc(r,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

    def rhofunc(self,r,rho0=nan,rs=nan,rvir=nan):

        if ~isfinite(rho0):
            rho0=self.rho0
        if ~isfinite(rs):
            rs=self.rs
        if ~isfinite(rvir):
            rvir=self.rvir

        return rho0/((r/rs)*(1+r/rs)**2)

    def nfwfx(self,x):
        return log(1+x)-x/(1+x)

    def get_mass(self,r,rho0=nan,rs=nan):

        if ~isfinite(rho0):
            rho0=self.rho0
        if ~isfinite(rs):
            rs=self.rs

        return 4*pi*rs**3*rho0*self.nfwfx(r/rs)

    def get_phi(self,r,rho0=nan,rs=nan):

        if ~isfinite(rho0):
            rho0=self.rho0
        if ~isfinite(rs):
            rs=self.rs

        return -4*GN*pi*rho0*rs**3/r*log(1+r/rs)

    def get_vcirc(self,r):

        x=r/self.rs
        return (sqrt(GN*self.mvir/self.rvir*self.c/x*self.nfwfx(x)/self.nfwfx(self.c))).to(self.vunit)

    def get_vmax(self):
        return self.get_vcirc(2.16258*self.rs)

    def get_rmax(self):
        return 2.16258*self.rs

    def get_projected(self,r):

        x=(r/self.rs).decompose().value
        front=2*self.rho0*self.rs/(x**2-1)

        if x>1:
            return front*(1-2.0/(np.sqrt(x**2-1))*np.arctan(np.sqrt((x-1)/(x+1))))
        elif x<1:
            return front*(1-2.0/(np.sqrt(1-x**2))*np.arctanh(np.sqrt((1-x)/(x+1))))
        else:
            return front*1.0/3

class Einasto(RhoProfile):

    rs=0
    rho0=0
    rvir=0
    mvir=0
    deltavirrhou=0
    c=0
    alpha=0
    rhounit=0
    runit=0
    munit=0

    def __init__(self,alpha,rho0=0,c=0,rs=0,deltavirrhou=0,mvir=0,rvir=0):

        self.alpha=alpha
        self.rhounit=units.M_sun/units.kpc**3
        self.runit=units.kpc
        self.munit=units.M_sun

        if c!=0 and rvir!=0:
            rs=rvir/c

        if c!=0 and rs!=0:
            rvir=c*rs


        if rs!=0 and mvir!=0 and deltavirrhou!=0:
            self.rs=rs
            self.mvir=mvir
            rvir=(mvir/(4*pi/3*deltavirrhou))**(1.0/3)
            self.deltavirrhou=deltavirrhou
            self.rvir=rvir
            self.c=self.rvir/self.rs
            m1=getmasseinasto(1*mvir.unit/(rs.unit**3),rs,rvirs,self.alpha)
            self.rho0=mvir/m1*(mvir.unit/(rs.unit**3))
            rhofn=lambda r: self.rhofunc(r,alpha,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif mvir!=0 and rs!=0 and rvir!=0:
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            self.deltavirrhou=mvir/(4*pi/3*rvir**3)
            self.mvir=mvir
            m1=getmasseinasto(1*mvir.unit/(rs.unit)**3,rs,rvir,self.alpha)
            self.rho0=mvir/m1*(mvir.unit/rs.unit**3)
            rhofn=lambda r: self.rhofunc(r,alpha,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif rho0!=0 and rs!=0 and rvir!=0:
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            self.rho0=rho0
            mvir=getmasseinasto(rho0,rs,rvir,self.alpha)
            self.mvir=mvir
            self.deltavirrhou=mvir/(4*pi/3*rvir**3)
            rhofn=lambda r: self.rhofunc(r,alpha,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif rs!=0 and rho0!=0 and deltavirrhou!=0:
            self.rho0=rho0
            self.rs=rs
            self.deltavirrhou=deltavirrhou
            self.c=self.rvir/self.rs
            m1=lambda rvira: getmasseinasto(rho0,rs,rvira*rs.unit,self.alpha)
            x=minimize_scalar(lambda y: abs(m1(y)/(4*pi/3*(y*rs.unit)**3)-deltavirrhou))['x']
            self.rvir=x*rs.unit
            self.mvir=4*pi/3*(x*rs.unit)**3*deltavirrhou
            rhofn=lambda r: self.rhofunc(r,alpha,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)



    def rhofunc(self,r,alpha=nan,rho0=nan,rs=nan,rvir=nan):

        if ~isfinite(alpha):
            alpha=self.alpha
        if ~isfinite(rho0):
            rho0=self.rho0
        if ~isfinite(rs):
            rs=self.rs
        if ~isfinite(rvir):
            rvir=self.rvir

        h=rs/getdneinasto(alpha)
        return rho0*exp(-(r/h)**(1.0/alpha))


    def get_mass(self,r):
        return getmasseinasto(self.rho0,self.rs,r,self.alpha)

class Plummer(RhoProfile):

    m=0
    a=0

    def __init__(self,m,a):

        self.a=a
        self.m=m

    def get_rho(self,r):

        first=3*self.m/4/np.pi/self.a**3
        second=(1+(r/self.a)**2)**(-5.0/2)

        return first*second

    def get_potential(self,r):
        return -GN*self.m/(np.sqrt(r**2+a**2))

    def get_mass(self,r):

        return self.m*(r**3/(r**2+self.a**2)**1.5)

    def get_projected(self,r):

        left=2*3*self.m/4/np.pi/self.a**3
        top=2*self.a**5
        bottom=3.0*(self.a**2+r**2)**2
        return left*top/bottom

class Einasto(RhoProfile):

    rs=0
    rho0=0
    rvir=0
    mvir=0
    deltavirrhou=0
    c=0
    alpha=0
    rhounit=0
    runit=0
    munit=0

    def __init__(self,alpha,rho0=0,c=0,rs=0,deltavirrhou=0,mvir=0,rvir=0):

        self.alpha=alpha
        self.rhounit=units.M_sun/units.kpc**3
        self.runit=units.kpc
        self.munit=units.M_sun

        if c!=0 and rvir!=0:
            rs=rvir/c

        if c!=0 and rs!=0:
            rvir=c*rs


        if rs!=0 and mvir!=0 and deltavirrhou!=0:
            self.rs=rs
            self.mvir=mvir
            rvir=(mvir/(4*pi/3*deltavirrhou))**(1.0/3)
            self.deltavirrhou=deltavirrhou
            self.rvir=rvir
            self.c=self.rvir/self.rs
            m1=getmasseinasto(1*mvir.unit/(rs.unit**3),rs,rvirs,self.alpha)
            self.rho0=mvir/m1*(mvir.unit/(rs.unit**3))
            rhofn=lambda r: self.rhofunc(r,alpha,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif mvir!=0 and rs!=0 and rvir!=0:
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            self.deltavirrhou=mvir/(4*pi/3*rvir**3)
            self.mvir=mvir
            m1=getmasseinasto(1*mvir.unit/(rs.unit)**3,rs,rvir,self.alpha)
            self.rho0=mvir/m1*(mvir.unit/rs.unit**3)
            rhofn=lambda r: self.rhofunc(r,alpha,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif rho0!=0 and rs!=0 and rvir!=0:
            self.rs=rs
            self.rvir=rvir
            self.c=rvir/rs
            self.rho0=rho0
            mvir=getmasseinasto(rho0,rs,rvir,self.alpha)
            self.mvir=mvir
            self.deltavirrhou=mvir/(4*pi/3*rvir**3)
            rhofn=lambda r: self.rhofunc(r,alpha,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)

        elif rs!=0 and rho0!=0 and deltavirrhou!=0:
            self.rho0=rho0
            self.rs=rs
            self.deltavirrhou=deltavirrhou
            self.c=self.rvir/self.rs
            m1=lambda rvira: getmasseinasto(rho0,rs,rvira*rs.unit,self.alpha)
            x=minimize_scalar(lambda y: abs(m1(y)/(4*pi/3*(y*rs.unit)**3)-deltavirrhou))['x']
            self.rvir=x*rs.unit
            self.mvir=4*pi/3*(x*rs.unit)**3*deltavirrhou
            rhofn=lambda r: self.rhofunc(r,alpha,self.rho0,self.rs,self.rvir)
            RhoProfile.__init__(self,rhofn)



    def rhofunc(self,r,alpha=nan,rho0=nan,rs=nan,rvir=nan):

        if ~isfinite(alpha):
            alpha=self.alpha
        if ~isfinite(rho0):
            rho0=self.rho0
        if ~isfinite(rs):
            rs=self.rs
        if ~isfinite(rvir):
            rvir=self.rvir

        h=rs/getdneinasto(alpha)
        return rho0*exp(-(r/h)**(1.0/alpha))


    def get_mass(self,r):
        return getmasseinasto(self.rho0,self.rs,r,self.alpha)


class LimaNeto(RhoProfile):
#Lima Neto 1999
    mtot=0
    rs=0
    n=0
    p=0
    rho0=0
    re=0
    nu=0

    def __init__(self,n,rho0=0,re=0,p=0,rs=0,mtot=0):

        if mtot!=0 and re!=0:
            
            nu=1.0/n
            p= 1.0-0.6097*nu+0.05463*nu**2

            lnreovera=(0.6959-np.log(nu))/nu-0.1789
            a=re/np.exp(lnreovera)

            sigma0=mtot/a**2/(2*np.pi*gammafunc(2.0/nu)/nu)
            
            rho0=sigma0*gammafunc(2.0/nu)/(2*a*gammafunc((3-p)/nu))

            self.rs=a
            self.mtot=mtot
            self.rho0=rho0
            self.n=n
            self.re=re
            self.p=p
            self.nu=nu
    
            rhofn=lambda r: self.rhofunc(r,rho0,a,p,nu)
            RhoProfile.__init__(self,rhofn)

    def rhofunc(self,r,rho0=np.nan,rs=np.nan,p=np.nan,nu=np.nan):
        if ~np.isfinite(rho0):
            rho0=self.rho0
        if ~np.isfinite(rs):
            rs=self.rs
        if ~np.isfinite(p):
            p=self.p
        if ~np.isfinite(nu):
            nu=self.nu

        x=(r/rs).decompose().value
        return rho0*(x)**(-p)*np.exp(-(x)**nu)
    
    def get_mass(self,r):

        front=4*np.pi*self.rho0*self.rs**3/self.nu
        x=(r/self.rs).decompose().value
        second=gammainc((3-self.p)/self.nu,x**self.nu)
        return front*second
