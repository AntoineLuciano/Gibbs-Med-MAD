import numpy as np
import scipy

from Gibbs_Med_MAD import *


# --- INITIALIZATION ---

def X_init_odd(N,m,s,k=None,delta=0):
    if N%2==0:
        print("N is not odd")
        return None
    n=N//2
    if k==None:k=int(np.ceil(n/2))
    if delta:xmad=m+s
    else:xmad=m-s
    X=[m,xmad]
    X=X+[m-3*s/2]*(n-k+delta)+[m-s/2]*(k-1)+[m+s/2]*(n-k)+[m+3*s/2]*(k-delta)
    return X

# --- TOOLS FUNCTIONS ---

def print_zones_odd(X,m=None,s=None):
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    unique,counts=np.unique([zone_odd(xi,m=0,s=1) for xi in X],return_counts=True)
    print(dict(zip(unique, counts)))
    
    
def medMAD(X): return (np.median(X),scipy.stats.median_abs_deviation(X))

def sym(m,x): return 2*m-x 

def zone_odd(xi,m,s):
    if xi==m: return "m"
    elif np.round(xi,10)==np.round(m+s,10): return "m+s"
    elif np.round(xi,10)==np.round(m-s,10): return "m-s"
    elif xi<m-s: return 1
    elif xi<m: return 2
    elif xi<m+s: return 3
    else : return 4
    
def zone_odd_ab(xi,m,s):
    if xi==m: return m,m
    elif np.round(xi,10)==np.round(m+s,10): return np.round(m+s,10),np.round(m+s,10)
    elif np.round(xi,10)==np.round(m-s,10): return np.round(m-s,10),np.round(m-s,10)
    elif xi<m-s: return -np.inf,np.round(m-s,10)
    elif xi<m: return np.round(m-s,10),np.round(m,10)
    elif xi<m+s: return m,np.round(m+s,10) 
    else : return np.round(m+s,10),np.inf
    
def zone_odd_C_ab(xi,m,s):
    if xi==m: return m,m
    elif np.round(xi,10)==np.round(m-s,10): return np.round(m+s,10),np.round(m+s,10)
    elif np.round(xi,10)==np.round(m+s,10): return np.round(m-s,10),np.round(m-s,10)
    elif xi<m-s: return m,np.round(m+s,10)
    elif xi<m: return np.round(m+s,10),np.inf
    elif xi<m+s: return -np.inf,np.round(m-s,10) 
    else : return np.round(m-s,10),np.round(m,10)
    
def zone_odd_S_ab(xi,m,s):
    if xi==m: return m,m
    elif np.round(xi,10)==np.round(m+s,10): return np.round(m+s,10),np.round(m+s,10)
    elif np.round(xi,10)==np.round(m-s,10): return np.round(m-s,10),np.round(m-s,10)
    elif xi>m+s: return -np.inf,np.round(m-s,10)
    elif xi>m: return np.round(m-s,10),np.round(m,10)
    elif xi>m-s: return m,np.round(m+s,10)
    else : return np.round(m+s,10),np.inf
    
def move2odd(X,mean=None,std=None,verbose=False,index=None,m=None,s=None,par=[]):
    if (mean,std)==(None,None): mean,std=np.mean(X),np.std(X)
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    n=len(X)//2
    if index==None:
        index=np.random.choice(len(X),2,replace=False)
        
    xij=np.round(X[index],10)
    xij=X[index]
    xi,xj=xij[0],xij[1]
    
    if len(par)==0:
        if np.round(m+s,10) in X: 
            xmad=np.round(m+s,10)
        elif np.round(m-s,10) in X:
            xmad=np.round(m-s,10)
        else:
            print("pas de mad ???")
        i_MAD=np.where(X==xmad)[0][0]
        par=[i_MAD,xmad]
    [i_MAD,xmad]=par
    if m in xij and xmad in xij: 
        case="1"
        xnew1,xnew2=xi,xj
    elif m in xij:
        case="2"
        if xi==m:xother=xj
        elif xj==m:xother=xi
        else: print("Probleme m")
        a,b=zone_odd_ab(xother,m,s)
        xnew1,xnew2= scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0],m
    elif xmad in xij:
        if xi==xmad: xother=xj
        elif xj==xmad: xother=xi
        else : print("Probleme xmad")
        if (xmad-m)*(xother-m)>0:
            case="3a"
            a,b=zone_odd_ab(xother,m,s)
            xnew1,xnew2= scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0],xmad
        else:
            if np.abs(xother-m)>s:
                xnew1=truncated_norm_2inter(mean,std,-np.inf,m-s,m+s,np.inf)
                case="3b"
            else:
                case="3c"
                a,b=m-s,m+s
                xnew1= scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
            if xnew1 >m: xmad=m-s
            else: xmad=m+s  
            xnew2=xmad    
        i_MAD=index[1]
    else: 
        if sorted([zone_odd(xi,m,s),zone_odd(xj,m,s)]) in [[1,3],[2,4]]:
            case="4a"
            xnew1=np.random.normal(mean,std,1)[0]
            a,b=zone_odd_C_ab(xnew1,m,s)
            xnew2=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
        else:          
            case="4b"
            a1,b1=zone_odd_ab(xi,m,s)
            a2,b2=zone_odd_ab(xj,m,s)
            xnew1,xnew2= scipy.stats.truncnorm.rvs(a=(a1-mean)/std,b=(b1-mean)/std,size=1,loc=mean,scale=std)[0],scipy.stats.truncnorm.rvs(a=(a2-mean)/std,b=(b2-mean)/std,size=1,loc=mean,scale=std)[0]

    X[index]=np.round(np.array([xnew1,xnew2]),10).reshape(-1)

    return X,[i_MAD,xmad],case





def zone_vect_odd(X_arg,m=0,s=0,par=[]):
    if len(par)==0:
        if np.round(m+s,10) in X: 
            xmad=np.round(m+s,10)
        elif np.round(m-s,10) in X:
                xmad=np.round(m-s,10)
        else: print("PROBLEME XMAD DANS move ZONE RESAMPLE")
        par=[xmad]
    xmad=par[0]   
    X=np.copy(X_arg)
    X=np.repeat(X,2).reshape(-1,2)
    r = np.where(X<m-s, (-np.inf,m-s),(m+s,np.inf))
    r = np.where(np.logical_and(m-s<X,X<m),(m-s,m),r)
    r = np.where(np.logical_and(m<X,X<m+s),(m,m+s),r)
    r = np.where(X==xmad,(xmad,xmad),r)
    r = np.where(X==m,(m,m),r)
    
    return r


def move_resample_zone_odd(X,mean=None,std=None,m=None,s=None,par=[]):
    if (mean,std)==(None,None): mean,std=np.mean(X),np.std(X)
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    if len(par)==0:
        if np.round(m+s,10) in X: 
            xmad=np.round(m+s,10)
        elif np.round(m-s,10) in X:
            xmad=np.round(m-s,10)
        else:
            print("pas de mad ???")
        i_MAD=np.where(X==xmad)[0][0]
        par=[i_MAD,xmad]
    [i_MAD,xmad]=par
    
    if xmad==np.round(m+s,10): delta=1
    elif xmad==np.round(m-s,10): delta=0
    else: print("PROBLEME XMAD RESAMPLE ZONE")
    n=len(X)//2
    
    k=max(np.sum(np.where(X>m+s,1,0)),1)    
    a,b=np.repeat([-np.inf,m-s,m,m+s],[n-k+delta,k-1,n-k,k-delta]),np.repeat([m-s,m,m+s,np.inf],[n-k+delta,k-1,n-k,k-delta])
    sample=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,loc=mean,scale=std)
    res=np.append(sample,[xmad,m])
    res=np.round(res,10)
    par=[len(X)-2,xmad]
    return res,par



def move_k_odd(X,mean=None,std=None,m=None,s=None,par=[]):
    #X=np.round(X,8)
    if (mean,std)==(None,None): mean,std=np.mean(X),np.std(X)
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    i=np.random.choice(len(X),1)
    xi=X[i]
    
    if len(par)==0:
        if np.round(m+s,10) in X: 
            xmad=np.round(m+s,10)
        elif np.round(m-s,10) in X:
            xmad=np.round(m-s,10)
        else:
            print("pas de mad ???")
        i_MAD=np.where(X==xmad)[0][0]
        par=[i_MAD,xmad]
    [i_MAD,xmad]=par
    
    xi=m
    while xi in np.round([m+s,m-s,m],10):
        i=np.random.choice(len(X),1)
        xi=X[i]
    
    a1,b1 = zone_odd_C_ab(xi,m,s)
    index_C=np.where(np.logical_and(X>a1,X<b1))[0]

    while len(index_C)==0:
        a,b = zone_odd_S_ab(xi,m,s)
        i=np.random.choice(np.where(np.logical_and(X>a,X<b))[0],1)[0]
        xi=X[i]
        a1,b1 = zone_odd_C_ab(xi,m,s)
        index_C=np.where(np.logical_and(X>a1,X<b1))[0]
        

    j=np.random.choice(index_C,1)[0]
    xj=X[j]
    while i==j or type(zone_odd(xj,m,s))!=int:
        print("probleme move k odd")
        j=np.random.choice(len(X),1)
        xj=X[j]
    a1,b1=zone_odd_S_ab(xi,m,s)
    xnew1=scipy.stats.truncnorm.rvs(a=(a1-mean)/std,b=(b1-mean)/std,size=1,loc=mean,scale=std)[0]
    a2,b2=zone_odd_C_ab(xnew1,m,s)
    xnew2=scipy.stats.truncnorm.rvs(a=(a2-mean)/std,b=(b2-mean)/std,size=1,loc=mean,scale=std)[0]

    X[i]=np.round(xnew1,10)
    X[j]=np.round(xnew2,10)
    
    return X,par
    
def move_Xmad_odd(X,mean=None,std=None,m=None,s=None,par=[]):
    
    if (mean,std)==(None,None): mean,std=np.mean(X),np.std(X)
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    if len(np.where(np.logical_or(X==np.round(m+s,8),X==np.round(m-s,8)))[0])==0: print(medMAD(X))
    if len(par)==0:
        if np.round(m+s,10) in X: 
            xmad=np.round(m+s,10)
        elif np.round(m-s,10) in X:
            xmad=np.round(m-s,10)
        else:
            print("pas de mad ???")
        i_MAD=np.where(X==xmad)[0][0]
        par=[i_MAD,xmad]
    [i_MAD,xmad]=par
        
    
    xother=m
    while xother in np.round([m+s,m-s,m],10):
        if xmad>m:i_other=np.random.choice(np.where(X<m)[0],1)[0]
        else: i_other=np.random.choice(np.where(X>m)[0],1)[0]
        xother=X[i_other]
    
    
    a,b=zone_odd_S_ab(xother,m,s)
    if(a>=b):print("PROBLEME MAD",a,b, xother,i_other,i_MAD,xmad,sorted(X))
    xnew1= scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
    xnew2=np.round(sym(m,xmad),10)
    
    X[i_other]=np.round(xnew1,10)
    X[i_MAD]=np.round(xnew2,10)
    return X,[i_MAD,xnew2]