# Author Abdulrahman S. Omar <xabush@singularitynet.io>
import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
import haiku as hk
import optax
from core.sgmcmc import *
from tqdm import tqdm
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
import operator

tfd = tfp.distributions

def get_log_prob_first(tau0, tau1, x, y, mlp_fn,
                      binary=False):

    def log_prob_fn(params, state):
        z = state["z"]
        sigma2 = state["sigma2"]
        # sigma2 = 1./jax.nn.softplus(sigma2)
        sigma2 = jax.nn.softplus(sigma2)

        y_pred = mlp_fn(params, x).ravel()

        if binary:
            log_prob1 = jnp.sum(-optax.sigmoid_binary_cross_entropy(labels=y, logits=y_pred))
        else:
            # pred_dist = tfd.MultivariateNormalDiag(y_pred, scale_diag=jnp.ones_like(y_pred)*sigma2)
            pred_dist = tfd.MultivariateNormalDiag(y_pred, scale_diag=jnp.ones_like(y_pred))
            log_prob1 = jnp.sum(pred_dist.log_prob(y))

        slab_lp_fn = lambda t: tfd.Normal(0, tau1*sigma2).log_prob(t)
        spike_lp_fn = lambda t: tfd.Normal(0, tau0*sigma2).log_prob(t)

        def lp_fn(z, t):
            a = slab_lp_fn(t)
            b = spike_lp_fn(t)
            return z*a + (1-z)*b

        log_prob2 = jnp.sum(jax.vmap(lp_fn)(z, params))
        return log_prob1 + log_prob2

    return log_prob_fn

def get_log_prob_rest(tau0, tau1, x, y, mlp_fn,
                      binary=False):

    def log_prob_fn(params, state):
        z = state["z"]
        sigma2 = state["sigma2"]
        # sigma2 = 1./jax.nn.softplus(sigma2)
        sigma2 = jax.nn.softplus(sigma2)

        y_pred = mlp_fn(params, x).ravel()

        if binary:
            log_prob1 = jnp.sum(-optax.sigmoid_binary_cross_entropy(labels=y, logits=y_pred))
        else:
            # pred_dist = tfd.MultivariateNormalDiag(y_pred, scale_diag=jnp.ones_like(y_pred)*sigma2)
            pred_dist = tfd.MultivariateNormalDiag(y_pred, scale_diag=jnp.ones_like(y_pred))
            log_prob1 = jnp.sum(pred_dist.log_prob(y))

        slab_lp_fn = lambda t: tfd.Normal(0, tau1*sigma2).log_prob(t)
        spike_lp_fn = lambda t: tfd.Normal(0, tau0*sigma2).log_prob(t)

        def lp_fn(z, t):
            a = slab_lp_fn(t)
            b = spike_lp_fn(t)
            return z*a + (1-z)*b

        log_prob2 = jnp.sum(jax.tree_util)
        return log_prob1 + log_prob2

    return log_prob_fn
