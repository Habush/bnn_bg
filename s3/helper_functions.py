# Helper functions to generate synthetic data and choose hyperparameters #

import numpy as np
from scipy.stats import logistic, binom
import jax.random as random

# Generate synthetic dataset
def synthetic_data(seed, n, p, s0, error_std=1, type='linear', scale=True, signal='constant',
                   random_index=False):
    true_beta = np.zeros(shape=[p])

    rng = random.PRNGKey(seed)
    idx_key, beta_key, err_key = random.split(rng, 3)

    if random_index:
        true_beta_idx = random.choice(idx_key, np.arange(p), shape=(s0,), replace=False)
    else:
        true_beta_idx = np.arange(s0)
    s0 = min(p, s0)
    if s0>0:
        if signal=='constant':
            true_beta[true_beta_idx] = 2
        elif signal=='decay':
            true_beta[true_beta_idx] = 2**(-(np.arange(s0)+1-9)/4)
        else:
            return("Error: input parameter signal must be 'constant' or 'decay'")

    X = np.array(random.normal(beta_key, shape=(n,p)))
    if scale:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_truebeta = X@true_beta

    if type=='linear':
        error_terms = np.array(random.normal(err_key, shape=(n,))*error_std)
        y = X_truebeta + error_terms
    elif type=='probit':
        true_aug_y = np.array(X_truebeta + random.normal(err_key, shape=(n,)))
        y = np.where(true_aug_y > 0, 1, 0)
    elif type=='logistic':
        true_aug_y = logistic.rvs(loc=X_truebeta)
        y = np.where(true_aug_y > 0, 1, 0)
    else:
        return("Error: input parameter 'type' must be 'linear' or 'logistic'")
    return({'X':X,'y':y,'true_beta':true_beta, 'true_beta_idx':true_beta_idx})

def spike_slab_params(n,p,type='linear'):
    K = max(10,np.log(n))
    q_seq = np.arange(1/p,(1-1/p),1/p)
    probs = abs(binom.cdf(k=K,n=p,p=q_seq)-0.9)
    q = min(q_seq[probs == np.min(probs)])
    tau0 = 1.0/(n**0.5)
    tau1 = 1.0
    # tau1 <- sqrt(max(1, p^(2.1)/(100*n))) # Alternative choice for tau1
    a0 = 1.0
    b0 = 1.0
    if type=='linear':
        return({'q':q,'tau0':tau0,'tau1':tau1,'a0':a0,'b0':b0})
    elif type=='probit':
        return({'q': q,'tau0':tau0,'tau1':tau1})
    elif type=='logistic':
        return({'q': q,'tau0':tau0,'tau1':tau1})
    else:
        return ("Error: input parameter 'type' must be 'linear' or 'logistic'")


