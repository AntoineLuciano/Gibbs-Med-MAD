# Gibbs-Med-MAD

Functions : 
- Gibbs_Med_MAD : Gibbs Sampler of the posterior of the mean and the variance of a Gaussian vector observing only its median and its MAD. 

    Args:
        - T (int): Number of iterations
        - N (int): Sample size of X
        - m (float): Median of X
        - s (float): MAD of X
        - par_prior (list, optional): Hyperparameters for the NormalGamma prior/posterior.  Defaults to [0,1,1,1].
        - n_shuffle (int, optional): Number of perturbations at each iteration. Defaults to 1.
        - simple_perturb (bool, optional): If True we apply a simple perturbation of 2 coordinates. Defaults to True.
        - random (bool, optional): If True the indexes of the 2 coordinates are chosen randomly, else we pick them in ascending order. Defaults to True
        - k_perturb (bool, optional): If True we apply a perturbation that change the repartition k of X. Defaults to False.
        - MAD_perturb (bool, optional): If True we apply a perturbation that switch Xmad's side wrt m. Defaults to False.Defaults to False.
        - verbose (bool, optional): If True we display algorithm progression. Defaults to False.
        - freq_resample (int, optional): 1/Frequence of full resampling of the vector. Defaults to 0.
        - verbose (bool, optional): Defaults to False.
        - n_resample_begin (int, optional): Number of resampling iterations. Defaults to 0.

    Returns: A dictionary we the following keys :
        - chains : Markov Chains of mu and sigma2, 
        - X : the final vector X, 
        - label : Algorithm label,
        - time : Computation time,
        - L_X : List of the vector X at each iteration, 
        - MEAN : Evolution of the empiric mean of X, 
        - VAR : Evolution of the empiric variance of X, 
        - burnin : Recommanded burnin size

- display_post : Display the trace and a KDE of the two chains. 
    Args : 
         - dico : Gibbs Sampler output
         - burnin : Burn-in period
 
    
