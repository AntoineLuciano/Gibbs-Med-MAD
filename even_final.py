import numpy as np
import scipy
from Gibbs_Med_MAD import *




# --- INITIALIZATION ---

def X_init_even(N,m,s,k=None,d1=0,d2=0,delta_i=0,delta_e=0):
    if(d1,d2)==(0,0):
        d1=s/10
        d2=s/10
    n=N//2
    if k==None: k=np.ceil(n/2)
    
    if N%2!=0: 
        print("N is not even")
        return None
    elif N<=4: 
        print("N should be >= 6")
        return None
    
    if delta_i: xmad1=m+s-d2
    else: xmad1=m-s+d2
    if delta_e: xmad2=m+s+d2
    else: xmad2=m-s-d2
        
    X=[m-d1,m+d1,xmad1,xmad2]

    if N==6: 
        X=X+([m-3*s/2]*int(delta_i+delta_e))+int(k-delta_e-delta_i)*[m+3*s/2] 
    else:
        X=X+([m-3*s/2]*int(n-k+delta_e-1))+([m-s/2]*int(k-2+delta_i))+([m+s/2]*int(n-k-1-delta_i))+int(k-delta_e)*[m+3*s/2]
    return X

# --- TOOLS FUNCTIONS ---
def print_zones_even(X,m=None,s=None):
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    unique,counts=np.unique([zone_even(xi,m=0,s=1) for xi in X],return_counts=True)
    print(dict(zip(unique, counts)))
    
def truncated_norm_2inter(mean,std,a,b,c,d):
    if (a>=b)or(c>=d): print("a = {}, b = {}, c = {},d = {}".format(a,b,c,d))
    if np.random.uniform(0,1,1)<(scipy.stats.norm.cdf((b-mean)/std)-scipy.stats.norm.cdf((a-mean)/std))/(scipy.stats.norm.cdf((b-mean)/std)-scipy.stats.norm.cdf((a-mean)/std)+scipy.stats.norm.cdf((d-mean)/std)-scipy.stats.norm.cdf((c-mean)/std)): 
        return scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
    else: 
        return scipy.stats.truncnorm.rvs(a=(c-mean)/std,b=(d-mean)/std,size=1,loc=mean,scale=std)[0]
    
def sym(m,x): return 2*m-x 

def medMAD(X): return (np.median(X),scipy.stats.median_abs_deviation(X))

def zone_even(xi,X,m=None,s=None,par=[]):
    X=np.array(X)
    n=len(X)//2  
    if (m,s)==(None,None): 
        m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    
    if len(par)==0:
        n=len(X)//2
        X_s=np.sort(X)
        m1=X_s[n-1]
        m2=X_s[n]
        S=np.abs(X-m)
        S_s= np.sort(S)
        s1,s2= S_s[n-1],S_s[n]
        
        [i_MAD1,i_MAD2]=np.argsort(S)[n-1:n+1]
        Xmad1,Xmad2=X[[i_MAD1,i_MAD2]]
        par=[s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]
    par=np.round(par,10)
    [s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]=par
    
    if xi==m1: return "m1"
    elif xi==m2: return "m2"
    elif xi==np.round(m+s1,10): return "m+s1"
    elif xi==np.round(m-s1,10): return "m-s1"
    elif xi==np.round(m+s2,10): return "m+s2"
    elif xi==np.round(m-s2,10): return "m-s2"
    elif xi<m-s2: return 1
    elif m-s1<xi<m1: return 2
    elif m2<xi<m+s1: return 3
    elif xi>m+s2: return 4
    
def zone_even_ab(xi,m1,m2,s1,s2):
    m=(m1+m2)/2
    if xi<m-s2: return -np.inf,m-s2
    elif xi<m-s1: return m-s2,m-s1
    elif xi<m1: return m-s1,m1
    elif xi<m2: return m1,m2
    elif xi<m+s1: return m2,m+s1
    elif xi<m+s2: return m+s1,m+s2
    else: return m+s2,np.inf
    
def zone_even_E_ab(xi,m1,m2,s1,s2):
    m,s=(m1+m2)/2,(s1+s2)/2
    if xi<m-s: return -np.inf,m-s
    elif xi<m1: return m-s,m1
    elif xi<m2: return m1,m2
    elif xi<m+s: return m2,m+s
    else: return m+s,np.inf
    
def zone_even_C_ab(xi,m1,m2,s1,s2):
    m,s=(m1+m2)/2,(s1+s2)/2
    if xi<m-s2: return m2,m+s1
    elif xi<m-s1: return m-s2,m-s1
    elif xi<m1: return m+s2,np.inf
    elif xi<m2: return m1,m2
    elif xi<m+s1: return -np.inf,m-s2
    elif xi<m+s2: return m+s1,m+s2
    else: return m-s1,m1
    
def zone_even_S_ab(xi,m1,m2,s1,s2):
    m,s=(m1+m2)/2,(s1+s2)/2
    if xi<m-s2: return m+s2,np.inf
    elif xi<m-s1: return m+s1,m+s2
    elif xi<m1: return m2,m+s1
    elif xi<m2: return m1,m2
    elif xi<m+s1: return m-s1,m1
    elif xi<m+s2: return m-s2,m-s1
    else: return -np.inf,m-s2

def zone_vect_even(X_arg,m=0,s=0,par=[]):
    if len(par)==0:
        n=len(X)//2
        X_s=np.sort(X)
        m1=X_s[n-1]
        m2=X_s[n]
        S=np.abs(X-m)
        S_s= np.sort(S)
        s1,s2= S_s[n-1],S_s[n]
        [i_MAD1,i_MAD2]=np.argsort(S)[n-1:n+1]
        Xmad1,Xmad2=X[[i_MAD1,i_MAD2]]
        par=[s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]
    else: [s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]=par
    
    X=np.repeat(X_arg,2).reshape(-1,2)
    r = np.where(X<m-s2, (-np.inf,m-s2),(m+s2,np.inf))
    r = np.where(np.logical_and(m-s1<X,X<m1),(m-s1,m1),r)
    r = np.where(np.logical_and(m2<X,X<m+s1),(m2,m+s1),r)
    r = np.where(X==Xmad1,(Xmad1,Xmad1),r)
    r = np.where(X==Xmad2,(Xmad2,Xmad2),r)
    r = np.where(X==m1,(m1,m1),r)
    r = np.where(X==m2,(m2,m2),r)
    return r

# --- PERTURBATIONS ---

def move2even(X,mean=None,std=None,verbose=False,index=None,m=None,s=None,par=[]):
    if (mean,std)==(None,None): mean,std=np.mean(X),np.std(X)
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    if index==None:
        index=np.random.choice(len(X),2,replace=False)
    X=np.array(X)
    xij=X[index]
    xi,xj=xij[0],xij[1]
    n=len(X)//2
    if len(par)==0:
        n=len(X)//2
        X_s=np.sort(X)
        m1=X_s[n-1]
        m2=X_s[n]
        S=np.abs(X-m)
        S_s= np.sort(S)
        s1,s2= S_s[n-1],S_s[n]
        [i_MAD1,i_MAD2]=np.argsort(S)[n-1:n+1]
        Xmad1,Xmad2=X[[i_MAD1,i_MAD2]]
        par=[s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]
        
    par=np.round(par,10)
    
    [s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]=par
    
    change_m=False
    change_s=False

    if sorted(xij)==[m1,m2]:
        case="1"
        s3= np.sort(np.abs(X-m))[2]
        
        a,b=m-s3,m+s3
        xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
        xnew2=sym(m,xnew1)

        change_m=True
    elif sorted(xij)==sorted([Xmad1,Xmad2]):
        S=np.sort(np.abs(X-m))
        epsilon=np.minimum(s1-S[n-2],S[n+1]-s2)
        if xi<m and xj<m: 
            case="2b"
            a,b= m-s2-epsilon,m-s1+epsilon
            if a>=b: 
                print("BIZARRE 2B")
                xnew1,xnew2=xi,xj
            else:
                xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
                xnew2=sym(m-s,xnew1)
            
        elif xi>m and xj>m: 
            case="2a"
            a,b= m+s1-epsilon,m+s2+epsilon
            if a>=b: 
                xnew1,xnew2=xi,xj
                print("BIZARRE 2A")
            else:
                xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
                xnew2=sym(m+s,xnew1)
        else:
            case="2c"
            a1,b1,a2,b2=m-s2-epsilon,m-s1+epsilon,m+s1-epsilon,m+s2+epsilon
            if a1==b1 or a2==b2:
                print("BIZARRE 2C")
                xnew1,xnew2=xi,xj
            else: 
                xnew1=truncated_norm_2inter(mean,std,a1,b1,a2,b2)
                if xnew1>m: xnew2=sym(m-s,sym(m,xnew1))
                else: xnew2=sym(m+s,sym(m,xnew1))
        change_s=True
        
    elif (m1 in xij or m2 in xij) and (Xmad1 in xij or Xmad2 in xij):
        case="3"
        xnew1,xnew2=xi,xj    
    elif m1 in xij or m2 in xij : 
        if xi in [m1,m2]: 
            xm,xother=xi,xj
        elif xj in [m1,m2]: 
            xm,xother=xj,xi
        else: print("probleme 4")
        if xm==m1 and m+s1>xother>m:
            case = "4a"
            a,b=m,m+s1
            xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
            if xnew1<m2: 
                xnew2=sym(m,xnew1)
                change_m=True
            else: xnew2=xm
        elif xm==m2 and m-s1<xother<m:
            case = "4b"
            a,b=m-s1,m
            xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
            if xnew1>m1: 
                xnew2=sym(m,xnew1)
                change_m=True
            else: xnew2=xm
        else: 
            case="4c"
            a,b=zone_even_ab(xother,m1,m2,s1,s2)
            xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
            xnew2=xm
    elif (Xmad1 in xij or Xmad2 in xij):
        if xi in [Xmad1,Xmad2]:xmad,xother=xi,xj
        elif xj in [Xmad1,Xmad2]:xmad,xother=xj,xi
        else: print("PROBLEME 5")
        if (xmad-m)*(xother-m)>0 and (np.abs(xmad-m)-s)*(np.abs(xother-m)-s)>0:
            case="5b "
            a,b=zone_even_ab(xother,m1,m2,s1,s2)
            xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
            xnew2=xmad
        elif (xmad-m)*(xother-m)>0 and (np.abs(xmad-m)-s)*(np.abs(xother-m)-s)<0:
            case="5a"
            a,b=zone_even_E_ab(xother,m1,m2,s1,s2)
            xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
            if m-s2<=xnew1<=m-s1: 
                xnew2=sym(m-s,xnew1)
                change_s=True
            elif m+s1<=xnew1<=m+s2: 
                xnew2=sym(m+s,xnew1)
                change_s=True
            else: xnew2=xmad
        elif (xmad-m)*(xother-m)<0 and (np.abs(xmad-m)-s)*(np.abs(xother-m)-s)>0:
            case="5c "
            a1,b1=zone_even_ab(xother,m1,m2,s1,s2)
            a2,b2=zone_even_S_ab(xother,m1,m2,s1,s2)
            xnew1=truncated_norm_2inter(mean,std,a1,b1,a2,b2)
            if a2<=xnew1<=b2:
                xnew2=sym(m,xmad)
                if xmad==Xmad1: Xmad1=sym(m,Xmad1)
                elif xmad==Xmad2: Xmad2=sym(m,Xmad2)
                else: print("\n\PROBLEME 5C\n\n")
            else: 
                xnew2=xmad
            
        else:
            case="5d "
            a,b=zone_even_E_ab(xother,m1,m2,s1,s2)
            xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
            if m-s2<=xnew1<=m-s1: 
                xnew2=sym(m+s,sym(m,xnew1))
                change_s=True
            elif m+s1<=xnew1<=m+s2: 
                xnew2=sym(m-s,sym(m,xnew1))
                change_s=True
            else: xnew2=xmad    

    else: 
        l_zone=[zone_even(xi,X,m=m,s=s,par=par),zone_even(xj,X,m=m,s=s,par=par)]
        sort_zone=sorted(l_zone)
        if sort_zone in [[1,2],[3,4]]:
            case="6a "
            if xi<m: a,b=-np.inf,m1
            else: a,b=m2,np.inf
            xnew1=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
            if m-s2<=xnew1<=m-s1: 
                xnew2=sym(m-s,xnew1)
                change_s=True
            elif m+s1<=xnew1<=m+s2: 
                xnew2=sym(m+s,xnew1)
                change_s=True
            elif xnew1<m-s2: 
                xnew2=scipy.stats.truncnorm.rvs(a=(m-s1-mean)/std,b=(m1-mean)/std,size=1,loc=mean,scale=std)[0]
            elif xnew1>m+s2: 
                xnew2=scipy.stats.truncnorm.rvs(a=(m2-mean)/std,b=(m+s1-mean)/std,size=1,loc=mean,scale=std)[0]
            elif m1>xnew1>m-s1: 
                xnew2=scipy.stats.truncnorm.rvs(a=-np.inf,b=(m-s2-mean)/std,size=1,loc=mean,scale=std)[0]
            else: 
                xnew2=scipy.stats.truncnorm.rvs(a=(m+s2-mean)/std,b=np.inf,size=1,loc=mean,scale=std)[0]
        elif sort_zone==[2,3]:
            case="6b "
            xnew1=scipy.stats.truncnorm.rvs(a=(m-s1-mean)/std,b=(m+s1-mean)/std,size=1,loc=mean,scale=std)[0]
            if m1<xnew1<m2: 
                xnew2=sym(m,xnew1)
                change_m=True
            elif xnew1<m1: 
                xnew2=scipy.stats.truncnorm.rvs(a=(m2-mean)/std,b=(m+s1-mean)/std,size=1,loc=mean,scale=std)[0]
            else: 
                xnew2=scipy.stats.truncnorm.rvs(a=(m-s1-mean)/std,b=(m1-mean)/std,size=1,loc=mean,scale=std)[0]
        elif sort_zone in [[1,3],[2,4]]:
            case="6c "
            xnew1=truncated_norm_2inter(mean,std,-np.inf,m1,m2,np.inf)
            if m-s2<=xnew1<=m-s1: 
                xnew2=sym(m+s,sym(m,xnew1))
                change_s=True
            elif m+s1<=xnew1<=m+s2: 
                xnew2=sym(m-s,sym(m,xnew1))
                change_s=True
    
            else: 
                a,b=zone_even_C_ab(xnew1,m1,m2,s1,s2)
                xnew2=scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
        else:          
            case="6d"
            a1,b1=zone_even_ab(xi,m1,m2,s1,s2)
            xnew1=scipy.stats.truncnorm.rvs(a=(a1-mean)/std,b=(b1-mean)/std,size=1,loc=mean,scale=std)[0]
            a2,b2=zone_even_ab(xj,m1,m2,s1,s2)
            xnew2=scipy.stats.truncnorm.rvs(a=(a2-mean)/std,b=(b2-mean)/std,size=1,loc=mean,scale=std)[0]
    
    [xnew1,xnew2]=np.round([xnew1,xnew2],10)
    X[index]=np.array([xnew1,xnew2]).reshape(-1) 

    if change_m: 
        [m1,m2]=sorted([xnew1,xnew2])
    
    if change_s:
        S_s=np.sort([np.abs(xnew1-m),np.abs(xnew2-m)])
        [s1,s2]=S_s
        [Xmad1,Xmad2]=np.array([xnew1,xnew2])[np.argsort([np.abs(xnew1-m),np.abs(xnew1-m)])]
        [i_MAD1,i_MAD2]=np.array(index)[np.argsort([np.abs(xnew1-m),np.abs(xnew1-m)])]
    

    par=np.round([s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2],10)
    
    return X,par,case

    


def move_resample_zone_even(X,mean=None,std=None,m=None,s=None,par=[]):
    if len(X)==6:
        print("For resampling N should be >=8 !")
        return None
    if (mean,std)==(None,None): mean,std=np.mean(X),np.std(X)
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    if len(par)==0:
        n=len(X)//2
        X_s=np.sort(X)
        m1=X_s[n-1]
        m2=X_s[n]
        S=np.abs(X-m)
        S_s= np.sort(S)
        s1,s2= S_s[n-1],S_s[n]
        
        [i_MAD1,i_MAD2]=np.argsort(S)[n-1:n+1]
        Xmad1,Xmad2=X[[i_MAD1,i_MAD2]]
        par=[s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]
    
    [s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]=np.round(par,10)
    [Xmad1,Xmad2]=np.array([Xmad1,Xmad2])[np.argsort(np.abs(np.array([Xmad1,Xmad2])-m))]
    if Xmad1==np.round(m+s1,10): delta_i=1
    elif Xmad1==np.round(m-s1,10): delta_i=0
    else: 
        print("PROBLEME XMAD1 RESAMPLE ZONE")
        print(Xmad1,np.round(m+s1,10),np.round(m-s1,10))
    
    if Xmad2==np.round(m+s2,10): delta_e=1
    elif Xmad2==np.round(m-s2,10): delta_e=0
    else: 
        print("PROBLEME XMAD2 RESAMPLE ZONE")
        print(Xmad2,np.round(m+s2,10),np.round(m-s2,10))
    i_MAD1,i_MAD2=len(X)-4,len(X)-3
    par=np.round([s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2],10)
    n=len(X)//2
    k=max(np.sum(np.where(X>m+s2,1,0)),2)
    a,b=np.repeat([-np.inf,m-s1,m2,m+s2],[n-k+delta_e-1,k+delta_i-2,n-k-delta_i-1,k-delta_e]),np.repeat([m-s2,m1,m+s1,np.inf],[n-k+delta_e-1,k+delta_i-2,n-k-delta_i-1,k-delta_e])
    return np.round(np.append(scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,loc=mean,scale=std),[Xmad1,Xmad2,m1,m2]),10),par
    
    
def move_k_even(X,mean=None,std=None,m=None,s=None,par=[]):
    if (mean,std)==(None,None): mean,std=np.mean(X),np.std(X)
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
    
    if len(par)==0:
        n=len(X)//2
        X_s=np.sort(X)
        m1=X_s[n-1]
        m2=X_s[n]
        S=np.abs(X-m)
        S_s= np.sort(S)
        s1,s2= S_s[n-1],S_s[n]
        
        [i_MAD1,i_MAD2]=np.argsort(S)[n-1:n+1]
        Xmad1,Xmad2=X[[i_MAD1,i_MAD2]]
        par=[s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]
        
    [s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]=np.round(par,10)
    
    xi=np.round(m1,10)
    while xi in np.round([Xmad1,Xmad2,m1,m2],10):
        i=np.random.choice(len(X),1)
        xi=X[i]
        
    a1,b1 = zone_even_C_ab(xi,m1,m2,s1,s2)
    index_C=np.where(np.logical_and(X>a1,X<b1))[0]
    if len(index_C)==0:
        a,b = zone_even_S_ab(xi,m1,m2,s1,s2)
        i=np.random.choice(np.where(np.logical_and(X>a,X<b))[0],1)[0]
        xi=X[i]
        a1,b1 = zone_even_C_ab(xi,m1,m2,s1,s2)
        index_C=np.where(np.logical_and(X>a1,X<b1))[0]
        
    j=np.random.choice(index_C,1)[0]
    xj=X[j]
    while a1>xj or xj>b1 or i==j or xi in [Xmad1,Xmad2,m1,m2]:
        print(a1<xj or xj>b1 or i==j or xi in [Xmad1,Xmad2,m1,m2],a1>xj, xj>b1 , i==j , xi in [Xmad1,Xmad2,m1,m2])
        print("Probleme dans move k even",a1,b1,xj,[Xmad1,Xmad2,m1,m2])
        
        j=np.random.choice(np.where(np.logical_and(X>a1,X<b1))[0],1)[0]
        xj=X[j]
        
    
    xnew1=scipy.stats.truncnorm.rvs(a=(a1-mean)/std,b=(b1-mean)/std,size=1,loc=mean,scale=std)[0]
    a2,b2 = zone_even_C_ab(xnew1,m1,m2,s1,s2)
    xnew2=scipy.stats.truncnorm.rvs(a=(a2-mean)/std,b=(b2-mean)/std,size=1,loc=mean,scale=std)[0]
    X[i]=np.round(xnew1,10)
    X[j]=np.round(xnew2,10)
    par=np.round([s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2],10)

    return X,par

def move_Xmad_even(X,mean=None,std=None,m=None,s=None,par=[]):
    if (mean,std)==(None,None): mean,std=np.mean(X),np.std(X)
    if (m,s)==(None,None):m,s=np.median(X),scipy.stats.median_abs_deviation(X)
   
    if len(par)==0:
        n=len(X)//2
        X_s=np.sort(X)
        m1=X_s[n-1]
        m2=X_s[n]
        S=np.abs(X-m)
        S_s= np.sort(S)
        s1,s2= S_s[n-1],S_s[n]
        
        [i_MAD1,i_MAD2]=np.argsort(S)[n-1:n+1]
        Xmad1,Xmad2=X[[i_MAD1,i_MAD2]]
        par=[s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]
        
    
    [s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2]=np.round(par,10)

    xother=np.round(Xmad1,10)
    
    xmad=np.random.choice([Xmad1,Xmad2])
    while xother in np.round([Xmad1,Xmad2,m1,m2],10):
        if xmad>m:i_other=np.random.choice(np.where(X<m)[0],1)[0]
        else: i_other=np.random.choice(np.where(X>m)[0],1)[0]
        xother=X[i_other]
        
    
        
    a,b=zone_even_S_ab(xother,m1,m2,s1,s2)
    if(a>=b):print("PROBLEME MAD",a,b, xother,i_other,i_MAD,xmad,sorted(X))
    
    
    
    xnew1= scipy.stats.truncnorm.rvs(a=(a-mean)/std,b=(b-mean)/std,size=1,loc=mean,scale=std)[0]
    xnew2=np.round(sym(m,xmad),10)
    
    if xmad==Xmad1: 
        i_MAD=i_MAD1
        Xmad1=xnew2
    elif xmad==Xmad2:
        i_MAD=i_MAD2
        Xmad2=xnew2
    else: print("probleme xmad even")

    X[i_other]=np.round(xnew1,10)
    X[int(i_MAD)]=xnew2
    
    par=np.round([s1,s2,Xmad1,Xmad2,i_MAD1,i_MAD2,m1,m2],10)

    return X,par
