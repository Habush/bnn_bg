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
tfb = tfp.bijectors

EPS = 1e-8
def get_log_prob_beta(tau0, tau1, x, y, mlp_fn,
                      binary=False):

    def log_prob_fn(beta, state):
        z = state["z"]
        sigma2 = state["sigma2"]
        # sigma2 = 1./jax.nn.softplus(sigma2)
        sigma2 = jax.nn.softplus(sigma2)

        y_pred = mlp_fn(x, beta).ravel()

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

        log_prob2 = jnp.sum(jax.vmap(lp_fn)(z, beta))
        return log_prob1 + log_prob2

    return log_prob_fn

def get_log_prob_z(tau0, tau1, eta, mu, J, temp=1.0):

    def ising_prior(x):
        return eta*(x.T @ J @ x) - mu*jnp.sum(x)

    def log_prob_fn(z, state):
        beta = state["beta"]
        sigma2 = state["sigma2"]
        # sigma2 = 1./jax.nn.softplus(sigma2)
        sigma2 = jax.nn.softplus(sigma2)

        log_prob1 = ising_prior(z)

        slab_lp_fn = lambda t: tfd.Normal(0, tau1*sigma2).log_prob(t)
        spike_lp_fn = lambda t: tfd.Normal(0, tau0*sigma2).log_prob(t)

        def lp_fn(z, t):
            a = slab_lp_fn(t)
            b = spike_lp_fn(t)
            return z * a + (1 - z) * b

        log_prob2 = jnp.sum(jax.vmap(lp_fn)(z, beta))

        log_prob = log_prob1 + log_prob2

        return jnp.sum(log_prob) / temp

    return log_prob_fn


def get_log_prob_sigma2(a0, b0,
                    tau0, tau1, x, y, mlp_fn, prior_scale=1.0):

    def log_prob_fn(sigma2, state):

        # log_prob1 = beta_log_prob_fn(state["beta"], state)
        sigma2 = jax.nn.softplus(sigma2)
        beta = state["beta"]
        z = state["z"]

        y_pred = mlp_fn(x, beta).ravel()
        pred_dist = tfd.MultivariateNormalDiag(y_pred, scale_diag=jnp.ones_like(y_pred)*(sigma2**2))
        pred_log_prob = jnp.sum(pred_dist.log_prob(y))

        slab_lp_fn = lambda t: tfd.Normal(0, tau1*sigma2).log_prob(t)
        spike_lp_fn = lambda t: tfd.Normal(0, tau0*sigma2).log_prob(t)
        def lp_fn(z, t):
            a = slab_lp_fn(t)
            b = spike_lp_fn(t)
            return z*a + (1-z)*b

        beta_log_prob = jnp.sum(jax.vmap(lp_fn)(z, beta))

        log_prob1 = pred_log_prob + beta_log_prob
        log_prob2 = tfd.HalfCauchy(a0, b0).log_prob(sigma2)
        # jax.debug.print("sigma2: {}, log_prob1: {}, log_prob2: {}", sigma2, log_prob1, log_prob2)
        log_prob =  log_prob1 + prior_scale*log_prob2
        return log_prob

    return log_prob_fn

def get_discrete_kernel(seed, step_size_fn, log_prob_fn, optimizer_fn,
                        x0, mh=False,
                        temp=1.0, preconditioner=None,
                        cat=False, dim=None, num_cls=None):
    if cat:
        sampler = optimizer_fn(step_size_fn, seed, dim, num_cls,
                               preconditioner=preconditioner, mh=mh, temp=temp)
    else:
        sampler = optimizer_fn(step_size_fn, seed,
                               preconditioner=preconditioner, mh=mh, temp=temp)
    init_opt_state = sampler.init(x0)

    def step(z, state, opt_state):
        def lp_fn(x):
            return log_prob_fn(x, state)

        z, opt_state, accept_prob = sampler.update(z, lp_fn, opt_state)
        return z, opt_state, {"accept_prob": accept_prob}

    return step, init_opt_state


def get_continuous_kernel(seed, step_size_fn, log_prob_fn, optimizer_fn,
                          x0, mh=False, temp=1.0,
                          preconditioner=get_rmsprop_preconditioner(),
                          momentum=0.9, prior_scale=1.0):
    sampler = optimizer_fn(step_size_fn, seed, preconditioner=preconditioner, momentum_decay=momentum)
    init_opt_state = sampler.init(x0)

    def proposal_dist(x_new, x_prev, state, step_size):
        log_prob, grad = jax.value_and_grad(log_prob_fn)(x_prev, state)
        theta = x_new - x_prev - step_size*grad
        theta_dot = jnp.linalg.norm(theta)**2
        # theta_dot = jax.tree_util.tree_reduce(
        #     operator.add, jax.tree_util.tree_map(lambda x: jnp.sum(x * x), theta))

        return -0.25*(1.0 / step_size) * theta_dot, log_prob

    # def safe_energy_diff(initial_energy: float, new_energy: float) -> float:
    #     delta_energy = initial_energy - new_energy
    #     delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
    #     return delta_energy

    def step(x, state, opt_state):
        grad = jax.grad(log_prob_fn)(x, state)
        updates, opt_state = sampler.update(grad, opt_state)
        x_new = optax.apply_updates(x, updates)
        accept_prob = 1.0 #default in the unadjusted case where mh=False
        accepted = True
        if mh:
            step_size = step_size_fn(opt_state.count - 1) #minus 1 b/c the count has been incremented in the update
            q_forward, log_prob_x = proposal_dist(x_new, x, state, step_size)
            q_reverse, log_prob_x_new = proposal_dist(x, x_new, state, step_size)

            m = (log_prob_x_new - log_prob_x) + (q_reverse - q_forward)
            delta = jnp.exp(m)
            delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
            accept_prob = jnp.clip(delta, a_max=1.0)
            u = jax.random.uniform(opt_state.rng_key)
            accepted = accept_prob > u

            # x_new = x_new if accepted else x #accept/reject
            x_new = jnp.where(accepted, x_new, x)
        return x_new, opt_state, {"accepted": accepted, "accept_prob": accept_prob}

    return step, init_opt_state

# TODO Fix sigma2 inference
def spike_slab_inference(*, seed, step_sizes, tau0, tau1, a0, b0, q,
                         J, eta, mu, x, y, mlp_fn, binary=False, mh=False,
                         n_iters=1000, burn_in=500, show_progress=False, const_schedule=False):

    n, p = x.shape
    rng = random.PRNGKey(seed)
    init_key, key = random.split(rng, 2)

    z_key, beta_key, sigma2_key = random.split(init_key, 3)

    # init
    z0 = jax.random.bernoulli(z_key, q, shape=(p,))*1.0
    sigma20 = jax.random.uniform(key, minval=0.1, maxval=1.)
    beta0 = jax.random.normal(beta_key, shape=(p,))*(z0*tau1+(1-z0)*tau0)*jnp.sqrt(sigma20)
    #log_probs
    z_log_prob_fn = get_log_prob_z(tau0, tau1, eta, mu, J)
    beta_log_prob_fn = get_log_prob_beta(tau0, tau1, x, y, mlp_fn, binary=binary)

    z_schedule = optax.constant_schedule(step_sizes["z"])

    if const_schedule:
        beta_schedule = optax.constant_schedule(step_sizes["beta"])
    else:
        beta_schedule = optax.exponential_decay(step_sizes["beta"], n_iters, 0.9, end_value=1e-5)

    z_kernel, z_opt_state = get_discrete_kernel(seed, z_schedule, z_log_prob_fn,
                                                disc_bin_sgld_gradient_update, z0,
                                                cat=False, mh=mh)

    beta_kernel, beta_opt_state = get_continuous_kernel(seed, beta_schedule,
                                                        beta_log_prob_fn,
                                                      sgld_gradient_update, beta0, mh=mh)


    # jit versions
    z_kernel = jax.jit(z_kernel)
    beta_kernel = jax.jit(beta_kernel)

    num_samples = n_iters - burn_in

    if not binary:
        sigma2_log_prob_fn = get_log_prob_sigma2(a0, b0, tau0, tau1, x, y, mlp_fn)
        sigma2_schedule = optax.constant_schedule(step_sizes["sigma2"])
        sigma2_kernel, sigma2_opt_state = get_continuous_kernel(seed, sigma2_schedule, sigma2_log_prob_fn,
                                                                sgld_gradient_update, sigma20)
        sigma2_kernel = jax.jit(sigma2_kernel)

    state = {"z": z0, "beta": beta0, "sigma2": sigma20}
    samples = {"z": np.zeros((num_samples, p)),
               "beta": np.zeros((num_samples, p)),
               "sigma2": np.zeros(num_samples)}

    accept_probs = {"z": np.zeros((n_iters)),
               "beta": np.zeros((n_iters)),
               "sigma2": np.zeros(n_iters)}

    if show_progress:
        steps = tqdm(range(n_iters))
    else:
        steps = range(n_iters)

    for t in steps:
        z, z_opt_state, z_mh_info = z_kernel(state["z"], state, z_opt_state)
        # if z_mh_info["accepted"]:
        #     z_opt_state = z_opt_new_state
        beta, beta_opt_new_state, beta_mh_info = beta_kernel(state["beta"], state, beta_opt_state)
        if beta_mh_info["accepted"]:
            beta_opt_state = beta_opt_new_state
        if not binary:
            sigma2, sigma2_opt_new_state, sigma_mh_info = sigma2_kernel(state["sigma2"], state, sigma2_opt_state)
            if sigma_mh_info["accepted"]:
                sigma2_opt_state = sigma2_opt_new_state
            state = {"z": z, "beta": beta, "sigma2": sigma2}
        else:
            state = {"z": z, "beta": beta, "sigma2": sigma20}
        if mh:
            accept_probs["beta"][t] = beta_mh_info["accept_prob"]
            accept_probs["z"][t] = z_mh_info["accept_prob"]
            if not binary:
                accept_probs["sigma2"][t] = sigma_mh_info["accept_prob"]

        if t >= burn_in:
            for k in state.keys():
                samples[k][t-burn_in] = state[k]

    return samples, accept_probs


def fpr(z_samples, p, idx):
    mask = np.ones(p, dtype=bool)
    mask[idx] = False
    emp_prob = np.mean(z_samples, axis=0)
    on_vars = emp_prob[mask] > 0.5
    return np.sum(on_vars) / (p - len(idx))
def tpr(z_samples, idx):
    emp_prob = np.mean(z_samples, axis=0)
    on_vars = emp_prob[idx] > 0.5
    return np.sum(on_vars) / len(idx)