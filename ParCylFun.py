import numpy as np
#from scipy.special import pbdv
from scipy.special import gamma

def ParCylFun(v,x):
    L = len(x)
    dv = np.zeros((1,L),dtype='cdouble')
    dp = np.copy(dv)
    pdf = np.copy(dv)
    pdd = np.copy(dv)
    x = x.flatten()
    xa = np.abs(x)
    factor = (-np.sign(x) * x * x/4)

    I1 = (xa <=5.8).nonzero()[0]
    I2 = np.intersect1d((xa >5.8).nonzero(),(x<=1000).nonzero())

    J1 = np.intersect1d((x <=0).nonzero(),(xa<=5.8).nonzero())
    J2 = np.intersect1d((x <=0).nonzero(),(xa>5.8).nonzero())
    J3 = np.intersect1d((x >0).nonzero(),(x<=2).nonzero())
    J4 = np.intersect1d((x >2).nonzero(),(x<=5.8).nonzero())
    J5 = np.intersect1d((x >5.8).nonzero(),(x<=1000).nonzero())

    K1 = (x>1000).nonzero()

    if I1.size !=0 and v>=0:
        pdf[I1],pdd[I1] = pbdv(v,x[I1])
    if I2.size != 0 and v>=0:
        pdf[I2],pdd[I2] = pbdv(v,x[I2])
    if J1.size != 0 and v <0:
        ans = pbdv(v,x[J1])
        pdf[:,J1] = ans[0]
        pdd[:,J1] = ans[1]
        #pdf[J1],pdd[J1] =pbdv(v,x[J1])
    if J2.size != 0 and v < 0:
        pdf[J2], pdd[J2] = pbdv(v, x[J2])
    if J3.size != 0 and v < 0:
        pdf[J3], pdd[J3] = pbdv(v, x[J3])
    if J4.size != 0 and v < 0:
        pdf[J4], pdd[J4] = pbdv(v, x[J4])
    if J5.size != 0 and v < 0:
        pdf[J5], pdd[J5] = pbdv(v, x[J5])
    if pdf[0] == 0:
        pdf[0] = 1.e-70
        pdd[0] = 1.e-70
    #print(pdf,pdd)
    return pdf,pdd

def ndarray_insert(arr,index,value):
    """emulates the behavior of MATLAB"""
    for v in range(0,index):
        arr = np.append(arr,0)

    arr = np.append(arr,value)
    return arr

def pbdv(v,x):
    """I have no mouth but I must scream"""
    v1=[]
    pd1 = []
    pd0 = []
    v2 = []
    f1 = []
    f0 = []

    pdf = []
    pdd = []

    dv = []
    x1 = x[0]
    xa=abs(x)
    xa1 = xa[0]
    vh = v
    I1 = (x >=0).nonzero()
    I2 = (x <0).nonzero()
    if v==0:
        pdf[I1] = np.ones((0,len(I1)))
        pdf[I2] = np.exp(-(x**2)/2)
        pdd = 0.5*x*pdf
        return pdf,pdd

    v = v+(np.abs(1)*np.sign(v))
    nv = np.fix(v)
    v0 = v-nv
    na = np.abs(nv)
    #ep = np.ndarray((10000,10000))
    if I2:
        ep = np.exp(-.5e0*x[I2]*x[I2])
    else:
        ep = 1 # TODO -fix this
    if na >= 1:
        ja = 1
    if x1 <=1500:
        if v >= 0.0:
            if v0 == 0.0:
                pd0 = ep
                pd1 = x*ep
            else:
                for l in range(0,ja+1):
                    vl = v0+l
                    if xa1 <=5.8:
                        pd1 = dvsa(v1,x)
                    if xa1 > 5.8:
                        pd1 = dvla(v1,x)
                    if l == 0:
                        pd0 = pd1
            dv = pd0
            dv = np.append(dv, pd1)
            for k  in range(2,na):
                pdf = x*pd1 - (k+v0-1.0)*pd0
                dv[k,:] = pdf
                pd0 = pd1
                pd1 = pdf
        else:
            if x1 <= 0.0:
                if xa1 <= 5.8e0:
                    pd0 = dvsa(v0,x)
                    v1 = v0 - 1.00e0
                    pd1 = dvsa(v1,x)
                else:
                    pd0 = dvla(v0,x)
                    v1 = v0 - 1.0e0
                    pd1 = dvla(v1,x)
                dv= pd0
                dv = np.append(dv,pd1)
                for k in range(2,int(na)+1):
                    pd = (-x*pd1+pd0)/(k-1.0e0-v0)
                    dv = np.append(dv,pd)
                    pd0 = pd1
                    pd1 = pdf
            else:
                if x1 <= 0.0:
                    if xa1 <= 5.800:
                        pd0 = dvsa(v0,x)
                        v1 = v0 - 1.000
                        pd1 = dvsa(v1,x)
                    else:
                        pd0 = dvla(v0,x)
                        v1 = v0 - 1.000
                        pd1 = dvla(v1,x)
                    dv[0,:] = pd0
                    dv[1,:] = pd1
                    for k in range(2,na+1):
                        pd = (-x*pd1+pd0)/(k-1.000-v0)
                        dv[k,:] =pd
                        pd0 = pd1
                        pd1 = pd
                elif x1 <=2.0:
                    v2 = nv+v0
                    if nv ==0:
                        v2 = v2-1.000
                    nk = np.fix(-v2)
                    f1 = dvsa(v2,x)
                    v1 = v2+1.000
                    f0 = dvsa(v1,x)
                    dv[nk,:] = f1
                    dv[nk-1,:] = f0
                    for k in range(nk-2,0+1,-1):
                        f = x*f0+(k-v0+1.000)*f1
                        dv[k,:] = f
                        f1 = f0
                        f0 = f
                else:
                    if xa1 <=5.8:
                        pd0 = dvsa(v0,x)
                    if xa1 > 5.8:
                        pd0 = dvla(v0,x)
                    dv = np.ndarray(shape=(1, 0))
                    dv = np.insert(dv, 0, pd0)
                    m = 100+na
                    f1 = 0.0e0
                    f0 = 1.0e-30
                    for k in range(int(m),-1,-1):
                        f = x*f0+(k-v0+1.000)*f1
                        if k <= na:
                            if len(dv) < na:
                                dv = ndarray_insert(dv,k-1,f)
                            else:
                                dv[k] = f
                        f1 =f0
                        f0 =f
                    s0 = pd0/f
                    for k in range(0,int(na)+1):
                        dv[k] = s0*dv[k]
        """
        v1 = np.abs(v0)
        if v >= 0.0000:
            dp = 0.500 * x * dv[0] - dv[0 + 1]
        else:
            dp = -0.500 * x * dv[0] - v1 * dv[0 + 1]
        """
        dp = np.empty(shape=(0,0))
        for k in range(0,int(na)-1):
            v1 = np.abs(v0) + k

            if v >= 0.0000:
                dp = np.append(dp, 0.500*x*dv[k] - dv[k+1])
            else:
               dp  = np.append(dp,-0.5*x*dv[k] - v1*dv[k+1])
        pdf = dv[int(na)-1]
        pdd = dp[int(na)-2]
        v = vh
    else:
        a = -v-0.5
        mult = 1-(a+0.5)*(a+3/2)/(2*x**2) + (a+0.5)*(a+3/2)*(a+5/2)*(a+7/2)/(2*4*x**4)
        pdf = (x**(-a-0.5))*mult*ep
        pdd = -0.5*x*pdf

    return pdf,pdd


def dvsa(va,x):
    """
        for small argument
       Input:   x  --- Argument
                va --- Order
       Output:  PD --- Dv(x)
       Routine called: GAMMA for computing â(x)
    ===================================================
    """
    va0 = []
    ga0 = []
    g1 = []
    vt = []
    g0 = []
    vm = []
    gm = []
    eps = 1.0e-15
    pi = 3.141592653589793
    sq2 = np.sqrt(2.0)

    I1 = (x>=0).nonzero()
    I2 = (x<0).nonzero()
    """
    ep = []
    ep[I1] = np.ones((0,len(I1)))
    ep[I2] = np.exp[-.5*x[I2]*x[I2]]
    """
    ep = np.exp(-.5*x[I2]*x[I2])
    pd = np.zeros((1,len(x)))
    r = np.copy(pd)
    va0 = 0.500*(1.00-va)

    I1 = (x==0).nonzero()[0]
    I2 = (x!=0).nonzero()[0]
    a0 = np.ndarray(shape=(1,0))
    if va == 0.0:
        pd = ep
    else:
        if I1.size !=0:
            if va0 <= 0.0 and va0 == np.fix(va0):
                pd[I1] = 0.000
            else:
                ga0 = gamma(va0)
                pd[I1] = np.sqrt(pi)/(2.00**(-.500*va)*ga0)
        if I2.size !=0:
            g1 = gamma(-va)
            a0 = np.append(a0, 2.00**(-0.5*va-1.00)*ep/g1)
            vt = -.5*va
            g0 = gamma(vt)
            pd =np.insert(pd,I2,g0)
            r = np.insert(r,I2,1.0e0)
            rvlag = 1
            if np.round(va) == va and va > -2:
                rvlag = 0
            if rvlag:
                gamodd = gamma(-0.5*(1+va))
                gameven = gamma(-0.5*va)
            I2r = I2
            r1 = []
            for m in range(1,251):
                vm = .5*(m-va)
                if rvlag:
                    if np.floor(m/2) == m/2:
                        gameven = gameven*(vm-1)
                        gm = gameven
                    else:
                        gamodd = gamodd*(vm-1)
                        gm = gamodd
                else:
                    gm = gamma(vm)
                r[I2r] = -r[I2r]*sq2*x[I2r]/m
                #r1[I2r]=gm*r[I2r]
                r1 = np.insert(r1,I2r,gm*r[I2r])
                pd[I2r] = pd[I2r]+r1[I2r]
                index = (np.abs(r1[I2r]) >= eps*np.abs(pd[I2r])).nonzero()[0]
                if index.size ==0:
                    break
                else:
                    I2r = I2r[index]
            pd[I2] = a0[I2]*pd[I2][0]

    return pd[:-1]


def dvla(va, x):

    x1 = []
    v1 = []
    gl = []
    pi = 3.141592653589793e0;
    eps = 1.0e-12

    I1 = (x>=0).nonzero()
    I2 = (x <0).nonzero()
    #ep = np.ndarray(dtype="double",shape=(I1, I2))
    #ep[I1] = np.ones(0,len(I1))
    #ep[I2] = np.exp(-.5e0*x[I2]*x[I2])
    ep = [1]
    a0 = np.abs(x)**va*ep
    pd = np.ones((1,len(x)))[0]
    r = np.copy(pd)
    I = (x<0).nonzero()[0]
    Ir = np.arange(len(x))
    for k in range(1,17):
        r[Ir] = -0.5e0*r[Ir]*(2.0*k-va-1.0)*(2.0*k-va-2.0)/(k*x[Ir]*x[Ir])
        pd[Ir] = pd[Ir] + r[Ir]
        index = (np.abs(r[Ir]) >= eps*np.abs(pd[Ir])).nonzero()[0]
        if index.size == 0:
            break
        else:
            Ir = Ir[index.astype('int')]
    pd = a0*pd
    if I.size != 0:
        x1[I] = -x[I]
        v1[I] = vvla(va,x1[I])
        gl = gamma(-va)
        pd[I] = pi*v1[I]/gl+np.cos(pi*va)*pd[I]
    return pd

def vvla(va,x):
    """
    VVLA computes parabolic cylinder function Vv(x) for large arguments.

  Licensing:
    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However,
    they give permission to incorporate this routine into a user program
    provided that the copyright is acknowledged.

  Modified:
    06 April 2012

  Author:
    Shanjie Zhang, Jianming Jin

  Reference:
    Shanjie Zhang, Jianming Jin,
    Computation of Special Functions,
    Wiley, 1996,
    ISBN: 0-471-11963-6,
    LC: QA351.C45.

  Parameters:

    Input, double precision X, the argument.

    Input, double precision VA, the order nu.

    Output, double precision PV, the value of V(nu,x).
    """
    x1 = np.ndarray(dtype="double")
    pd1 = np.ndarray(dtype="double")
    g1 = np.ndarray(dtype="double")
    pi = 3.141592653589793e0
    eps = 1.0e-12

    I1 = (x >=0).nonzero()
    I2 = (x <0).nonzero()
    qe = np.ndarray(dtype="double",size=(I1,I2))
    qe[I1] = np.ones(0,len(I1))
    qe[I2] = np.exp(-.5e0*x[I2]*x[I2])
    a0 = np.abs(x)**(-va-1.0e0)*np.sqrt(2.0e0/pi)*qe
    pv = np.ones(0,len(x))
    r = pv
    I = (x<0).nonzero()
    Ir = range(0,len(x)+1)
    for k in range(1,19):
        r[Ir] = 0.5e0*r[Ir]*(2.0*k+va-1.0)*(2*k+va)/(k*x[Ir]*x[Ir])
        pv[Ir] = pv[Ir] + r[Ir]
        index = (np.abs(r[Ir])>=eps*np.abs(pv[Ir]))
        if index.size == 0:
            break
        else:
            Ir = Ir[index]
    pv = a0*pv
    if I.size != 0:
        x1[I] = -x[I]
        pd1[I] = dvla(va,x1[I])
        g1 = gamma(-va)
        ds1 = np.sin(pi*va)*np.sin(pi*va)
        pv[I] = ds1*g1/pi*pd1[I]-np.cos(pi*va)*pv[I]
    return pv
