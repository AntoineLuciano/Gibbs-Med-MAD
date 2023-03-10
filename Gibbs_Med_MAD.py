import numpy as np
import scipy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from multiESS import multiESS

from even_final import *
from odd_final import *


# --- TOOLS FUNCTIONS ---

def medMAD(X): return (np.median(X),scipy.stats.median_abs_deviation(X))
    



def display_post(dico,burnin=0):
    n=len(dico["X"])
    
    f,ax=plt.subplots(2,2,figsize=(30,10))
    m,s=np.median(dico["X"]),scipy.stats.median_abs_deviation(dico["X"])
    
    sns.kdeplot(dico["chains"][0][burnin:],ax=ax[0,0],label="KDE")
    ax[0,0].axvline(np.median(dico["X"]),color="red",linestyle="dashed",label="Median")
    ax[0,0].set_xlabel("$\mu$",fontsize=15)
    ax[0,0].set_ylabel("$\pi(\mu|\sigma^2,m,s)$",fontsize=15)
    ax[0,0].legend(fontsize="x-large",prop={"size":20})
    
    sns.kdeplot(dico["chains"][1][burnin:],ax=ax[0,1],label="KDE")
    ax[0,1].axvline((s*1.4826)**2,color="red",linestyle="dashed",label="$(1.4826 \cdot s)^2$ = {}".format(np.round((1.4826*s)**2,2)))
    ax[0,1].set_ylabel("$\pi(\sigma|\mu,m,s)$",fontsize=15)
    ax[0,1].set_xlabel("$\sigma$",fontsize=15)
    ax[0,1].legend(fontsize="x-large",prop={"size":20})
     
    ax[1,0].plot(dico["chains"][0][burnin:])
    ax[1,0].axhline(m,color="red",linestyle="dashed",label="Median")
    ax[1,0].set_xlabel("Time",fontsize=15)
    ax[1,0].set_ylabel("$\mu$",fontsize=15)
    
    ax[1,1].plot(dico["chains"][1][burnin:])
    ax[1,1].axhline((s*1.4826)**2,color="red",linestyle="dashed",label="$(1.4826 \cdot s)^2$ = {}".format(np.round((1.4826*s)**2,2)))
    ax[1,1].set_xlabel("Time",fontsize=15)
    ax[1,1].set_ylabel("$\sigma^2$",fontsize=15)
    
    #f.suptitle("Posteriors of $\mu$ and $\sigma$ given median and MAD for n = {} m = {} s = {} mESS = {}".format(n,m,s,int(multiESS(dico["chains"][:,burnin:].T))),fontsize=26)
    f.tight_layout()


# --- POSTERIOR FUNCTIONS ---

def post_NG(X,par_prior):
    n=len(X)
    [mu_0,nu,alpha,beta]=par_prior
    tau=np.random.gamma(shape=alpha+n/2,scale=1/(beta+np.sum((X-np.mean(X))**2)/2+n*nu*(np.mean(X)-mu_0)**2/(2*(nu+n))),size=1)[0]
    mu=np.random.normal(loc=(nu*mu_0+np.sum(X))/(nu+n),scale=1/np.sqrt((nu+n)*tau),size=1)[0]
    return [mu,tau]

def post_NIG(X,par_prior):
    n=len(X)
    [mu_0,nu,alpha,beta]=par_prior
    sigma2=scipy.stats.invgamma(a=alpha+n/2,scale=(beta+np.sum((X-np.mean(X))**2)/2+n*nu*(np.mean(X)-mu_0)**2/(2*(nu+n)))).rvs(1)[0]
    mu=np.random.normal(loc=(nu*mu_0+np.sum(X))/(nu+n),scale=np.sqrt(sigma2/(nu+n)),size=1)[0]
    return [mu,sigma2]

# --- INITIALIZATION ----

def X_init(N,m,s):
    if N%2==0: return X_init_even(N,m,s)
    return X_init_odd(N,m,s)

# --- PERTURBATIONS ----

def move2coords(X,mean=None,std=None,verbose=False,index=None,m=None,s=None,par=[]):
    if len(X)%2==0:
        return move2even(X=X,mean=mean,std=std,verbose=verbose,index=index,m=m,s=s,par=par)
    return move2odd(X=X,mean=mean,std=std,verbose=verbose,index=index,m=m,s=s,par=par)


def move_resample_zone(X,mean=None,std=None,m=None,s=None,par=[]):
    if len(X)%2==0: return move_resample_zone_even(X=X,mean=mean,std=std,m=m,s=s,par=par)
    return move_resample_zone_odd(X=X,mean=mean,std=std,m=m,s=s,par=par)


def move_k(X,mean=None,std=None,m=None,s=None,par=[]):
    if len(X)%2==0: return move_k_even(X=X,mean=mean,std=std,m=m,s=s,par=par)
    return move_k_odd(X=X,mean=mean,std=std,m=m,s=s,par=par)

def move_xMAD(X,mean=None,std=None,m=None,s=None,par=[]):
    if len(X)%2==0: return move_Xmad_even(X=X,mean=mean,std=std,m=m,s=s,par=par)
    return move_Xmad_odd(X=X,mean=mean,std=std,m=m,s=s,par=par)


# --- GIBBS SAMPLER ---

def Gibbs_Med_MAD(T : int,N : int,m:float,s:float,par_prior =[0,1,1,1] ,n_shuffle=1,simple_peturb=True, k_perturb=False,MAD_perturb=False,freq_resample=0,random=True,verbose=True,n_resample_begin=0):
    """Gibbs Sampler 

    Args:
        T (int): Number of iterations
        N (int): Sample size of X
        m (float): Median of X
        s (float): MAD of X
        par_prior (list, optional): Hyperparameters for the NormalGamma prior/posterior.  Defaults to [0,1,1,1].
        n_shuffle (int, optional): Number of perturbations at each iteration. Defaults to 1.
        simple_perturb (bool, optional): If True we apply a simple perturbation of 2 coordinates. Defaults to True.
        random (bool, optional): If True the indexes of the 2 coordinates are chosen randomly, else we pick them in ascending order. Defaults to True
        k_perturb (bool, optional): If True we apply a perturbation that change the repartition k of X. Defaults to False.
        MAD_perturb (bool, optional): If True we apply a perturbation that switch Xmad's side wrt m. Defaults to False.Defaults to False.
        verbose (bool, optional): If True we display algorithm progression. Defaults to False.
        freq_resample (int, optional): 1/Frequence of full resampling of the vector. Defaults to 0.
        verbose (bool, optional): Defaults to False.
        n_resample_begin (int, optional): Number of resampling iterations. Defaults to 0.

    Returns:
        A dictionary we the following keys :
        chains : Markov Chains of mu and sigma2, 
        X : the final vector X, 
        label : Algorithm label,
        time : Computation time,
        L_X : List of the vector X at each iteration, 
        MEAN : Evolution of the empiric mean of X, 
        VAR : Evolution of the empiric variance of X, 
        burnin : Recommanded burnin size
    """
    tps=time.time()
    if n_shuffle<1: n_shuffle=int(n_shuffle*N)
    lab=""
    if simple_peturb: lab+="Simple"
    if k_perturb: lab+=" K"
    if MAD_perturb: lab+=" MAD"
    lab+=" pertubation"
    if n_shuffle>1:lab+=" ("+str(n_shuffle)+"times/it)"
    if freq_resample>0:lab+=" + Resample at freq = 1/"+str(freq_resample)
    if N%2==0: lab+=" even case"
    else: lab+=" odd case"
    if n_resample_begin>0: 
        lab+=" + Resample "+str(n_resample_begin)+" time at the beginning"
        burnin=n_resample_begin+1
    else: burnin=5*N
    
    X=X_init(N, m, s) # Initialization
    
    X1=np.copy(X)
    X1=np.round(X1,10)
    
    L_X=[list(X1)]

    mu,tau=post_NG(X1,par_prior)
    
    SIM=[[mu,1/tau]]
    MEAN=[np.mean(X1)]
    VAR=[np.var(X1)]
    par=[]
    
    if verbose: print(lab,end=" : ")
    for i in tqdm(range(T),disable=not(verbose)):
        if i<n_resample_begin:
            X1,par=move_resample_zone(X=X1,mean=SIM[-1][0],std=np.sqrt(SIM[-1][1]),m=m,s=s,par=par)
        else : 
            for j in range(n_shuffle):
                if simple_peturb:
                    if random:
                        X1,par,case=move2coords(X=X1,mean=SIM[-1][0],std=np.sqrt(SIM[-1][1]),m=m,s=s,par=par,verbose=verbose)
                    else:
                        X1,par,case=move2coords(X=X1,mean=SIM[-1][0],std=np.sqrt(SIM[-1][1]),m=m,s=s,par=par,verbose=verbose,index=[i%N,(i+1)%N])
                        if i%N==0: np.random.shuffle(X1)
                    
                elif k_perturb: 
                    X1,par=move_k(X=X1,mean=SIM[-1][0],std=np.sqrt(SIM[-1][1]),m=m,s=s,par=par) 
                    
                elif MAD_perturb:
                    X1,par=move_xMAD(X=X1,mean=SIM[-1][0],std=np.sqrt(SIM[-1][1]),m=m,s=s,par=par)

            if freq_resample>0:
                if i%freq_resample==0: X1,par=move_resample_zone(X=X1,mean=SIM[-1][0],std=np.sqrt(SIM[-1][1]),m=m,s=s,par=par)
        
  
        mu,tau=post_NG(X1,par_prior)
        sim=[mu,1/tau]

        if (np.round(medMAD(X1),10)!=[m,s]).any(): print("i=",i,"PROBLEME MED OU MAD",medMAD(X1))
        SIM.append(sim)
        L_X.append(list(X1))
        MEAN.append(np.mean(X1))
        VAR.append(np.var(X1))

    if (np.round(medMAD(X1),10)!=[m,s]).any(): 
        print("PROBLEME MED OU MAD",medMAD(X1))
    return {"chains":np.array(SIM).squeeze().T,"X":X1,"label":lab,"time":time.time()-tps,"L_X":np.array(L_X),"mean":MEAN,"var":VAR,"burnin":burnin}

