# Author Abdulrahman S. Omar <xabush@singularitynet.io>
import math
import jax
import jax.numpy as jnp
from utils import tree_utils
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

"""
Partly adapted from https://github.com/google-research/google-research/blob/master/bnn_hmc/utils/losses.py
"""

def make_xent_log_likelihood(temperature):

  def xent_log_likelihood(net_apply, params, net_state, batch, is_training):
    """Computes the negative log-likelihood."""
    _, y = batch
    logits, net_state = net_apply(params, net_state, None, batch, is_training)
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(y, num_classes)
    softmax_xent = jnp.sum(labels * jax.nn.log_softmax(logits)) / temperature

    return softmax_xent, net_state

  return xent_log_likelihood


def make_gaussian_log_prior(weight_decay, temperature):
  """Returns the Gaussian log-density and delta given weight decay (precision)."""

  def log_prior(params):
    """Computes the Gaussian prior log-density."""
    # ToDo izmailovpavel: make temperature treatment the same as in gaussian
    # likelihood function.

    n_params = sum([p.size for p in jax.tree_leaves(params)])
    # jax.debug.print("n_params: {}", n_params)
    log_prob = -(0.5 * tree_utils.tree_dot(params, params) * weight_decay +
                 0.5 * n_params * jnp.log(weight_decay / (2 * jnp.pi)))
    return log_prob / temperature

  return log_prior


def make_spike_slap_log_prior(slab_log_prob_fn, spike_log_prob_fn, prior_log_prob_fn,
                            temperature):
    """Returns a spike-and-slap prior log-density on the first layer and Gaussian prior on the rest"""

    # log_prior_fn = make_log_prior_fn(prior_log_prob_fn)
    # gaussian_log_prior = make_gaussian_log_prior(1., temperature)
    @jax.jit
    def log_prob_fn(params, gamma):
        beta = params["dropout"]["w"]
        gamma_first = gamma["dropout"]["w"]
        rest = {k: params[k] for k in params if k != "dropout"}

        def fn(z, t):
            a = slab_log_prob_fn(t)
            b = spike_log_prob_fn(t)
            return z * a + (1 - z) * b

        # log_prob of the first layer weights
        log_prob1 = jnp.sum(jax.vmap(fn)(gamma_first, beta))

        # log_prob of the rest of the weights
        log_prob2 = 0
        for p in jax.tree_leaves(rest):
            log_prob2 += jnp.sum(prior_log_prob_fn(p))
        # log_prob2 = gaussian_log_prior(rest)
        return (log_prob1 + log_prob2) / temperature

    return log_prob_fn


def preprocess_network_outputs_gaussian(predictions):
  """Apply softplus to std output if available.

  Returns predictive mean and standard deviation.
  """
  predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
  predictions_std = jax.nn.softplus(predictions_std)
  return jnp.concatenate([predictions_mean, predictions_std], axis=-1)


def make_gaussian_likelihood(temperature):

  def gaussian_log_likelihood(net_apply, params, net_state, batch, is_training):
    """Computes the negative log-likelihood.

    The outputs of the network should be two-dimensional.
    The first output is treated as predictive mean. The second output is treated
    as inverse-softplus of the predictive standard deviation.
    """
    _, y = batch
    # x = x @ jnp.diag(gamma)
    predictions, net_state = net_apply(params, net_state, None, batch,
                                       is_training)

    predictions = preprocess_network_outputs_gaussian(predictions)
    predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
    predictions_mean, predictions_std = predictions_mean.ravel(), predictions_std.ravel()
    # predictions_mean, predictions_std = predictions, 1.0
    tempered_std = predictions_std * jnp.sqrt(temperature)

    se = (predictions_mean - y)**2
    log_likelihood = (-0.5 * se / tempered_std**2 -
                      0.5 * jnp.log(tempered_std**2 * 2 * math.pi))
    log_likelihood = jnp.sum(log_likelihood)
    return log_likelihood, net_state

  return gaussian_log_likelihood

# def make_bin_log_prior(J, eta, mu):
#     """Returns the ising prior log-density given the coupling matrix J"""
#     def log_prob_fn(gamma):
#         return eta*(gamma.T @ J @ gamma) - mu*jnp.sum(gamma)
#
#     return log_prob_fn
#
# def make_bin_log_likelihood(tau0, tau1, weight_decay, temperature):
#     """Returns the posterior conditional of the latent binary variables"""
#
#     # sigma2 = 1./weight_decay
#     def log_prob_fn(gamma, params):
#         beta = params["linear_0"]["w"] #TODO include the bias params as well??
#         # jax.debug.print("bin ll - beta : {}", beta.shape)
#
#         slab_lp_fn = lambda t: tfd.MultivariateNormalDiag(0, jnp.ones_like(t)*tau1/weight_decay).log_prob(t)
#         spike_lp_fn = lambda t: tfd.MultivariateNormalDiag(0, jnp.ones_like(t)*tau0/weight_decay).log_prob(t)
#
#         def fn(z, t):
#             a = slab_lp_fn(t)
#             b = spike_lp_fn(t)
#             return z * a + (1 - z) * b
#
#         log_prob = jnp.sum(jax.vmap(fn)(gamma, beta))
#         # jax.debug.print("bin ll - log_prob : {}", log_prob)
#         return log_prob / temperature
#
#     return log_prob_fn

def make_bin_log_prior(J, eta, mu):
    """Returns the ising prior log-density given the coupling matrix J"""
    def log_prob_fn(gamma):
        # return eta*(gamma.T @ J @ gamma) - mu*jnp.sum(gamma)
        return 0.

    return log_prob_fn

EPS = 1e-8

def make_bin_log_likelihood(slab_log_prob_fn, spike_log_prob_fn, q, weight_decay, temperature):
    """Returns the posterior conditional of the latent binary variables"""

    sigma2 = 1./weight_decay
    def log_prob_fn(gamma, params):
        beta = params["dropout"]["w"]
        gamma_first = gamma["dropout"]["w"]

        slab_lp_fn = lambda t: jnp.log(q) + slab_log_prob_fn(t)
        spike_lp_fn = lambda t: jnp.log(1-q) + spike_log_prob_fn(t)

        def fn(z, t):
            a = slab_lp_fn(t)
            b = spike_lp_fn(t)
            return z*a + (1-z)*b

        log_prob = jnp.sum(jax.vmap(fn)(gamma_first, beta))

        return jnp.sum(log_prob) / temperature

    return log_prob_fn


def make_base_dist_log_prob(dist_name, loc, scale):
    """Returns the log-probability of the base distribution"""
    dist = None
    if dist_name == "normal":
        dist = tfd.Normal(loc, scale)
    elif dist_name == "laplace":
        dist = tfd.Laplace(loc, scale)
    elif dist_name == "student_t":
        dist = tfd.StudentT(loc, scale, 1)
    elif dist_name == "mvn" or dist_name == "multivariate_normal":
        dist = tfd.MultivariateNormalDiag(loc, scale)
    else:
        raise ValueError(f"Unknown distribution {dist_name}, choose from "
                         f"['normal', 'laplace', 'student_t', 'mvn']")

    def log_prob_fn(x):
        return dist.log_prob(x)

    return log_prob_fn