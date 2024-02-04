import pickle
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

PRNGKey = Any

def make_constant_lr_schedule_with_cosine_burnin(init_lr, final_lr,
                                                 burnin_steps):
    """Cosine LR schedule with burn-in for SG-MCMC."""

    def schedule(step):
        t = jnp.minimum(step / burnin_steps, 1.)
        coef = (1 + jnp.cos(t * np.pi)) * 0.5
        return coef * init_lr + (1 - coef) * final_lr

    return schedule

def make_cyclical_cosine_lr_schedule(init_lr, total_steps, cycle_length):
    """Cosine LR schedule with burn-in for SG-MCMC."""

    def schedule(step):
        k = total_steps // cycle_length
        t = (step % k) / k
        coef = (1 + jnp.cos(t * np.pi)) * 0.5
        return coef * init_lr

    return schedule

def make_cyclical_cosine_lr_schedule_with_const_burnin(init_lr, burnin_steps,
                                                       cycle_length):

    def schedule(step):
        t = jnp.maximum(step - burnin_steps - 1, 0.)
        t = (t % cycle_length) / cycle_length
        return 0.5 * init_lr * (1 + jnp.cos(t * jnp.pi))

    return schedule

def make_step_size_fn(init_lr, schedule, alpha, n_samples,
                      cycle_len):
    if schedule == "constant":
        return lambda _: init_lr

    if schedule == "exponential":
        return optax.exponential_decay(init_lr, decay_rate=alpha,
                                       transition_begin=0, transition_steps=n_samples)

    if schedule == "cyclical":
        if cycle_len is None or cycle_len < 0:
            cycle_len = 10

        return make_cyclical_cosine_lr_schedule(init_lr, n_samples, cycle_len)
    
def get_prior(prior_name, scale):
    if prior_name.lower() == "laplace":
        dist = tfd.Laplace(0, scale)
        return dist
    if prior_name.lower() == "normal":
        dist = tfd.Normal(0, scale)
        return dist
    if prior_name.lower() == "cauchy":
        dist = tfd.Cauchy(0, scale)
        return dist

    if prior_name.lower() == "student_t":
        dist = tfd.StudentT(df=2, loc=0, scale=scale)
        return dist
    raise ValueError(f"Unsupported prior {prior_name}")

def build_network(X, net_intr, net_intr_rev):
    p = X.shape[1]
    J = np.zeros((p, p))
    cols = X.columns
    intrs = []
    intrs_rev = []
    for i, g1 in enumerate(cols):
        try:
            g_intrs = net_intr.loc[g1]
            if isinstance(g_intrs, int):
                g_intrs = [g_intrs]
            else:
                g_intrs = list(g_intrs)
            for g2 in g_intrs:
                if (g2, g1) not in intrs_rev: # check if we haven't encountered the reverse interaction
                    j = cols.get_loc(g2)
                    J[i, j] = 1.0
                    J[j, i] = 1.0
                    intrs.append((g1, g2))
        except KeyError:
            continue

        # Check the reverse direction
        try:
            g_intrs_rev = net_intr_rev.loc[g1]
            if isinstance(g_intrs_rev, int):
                g_intrs_rev = [g_intrs_rev]
            else:
                g_intrs_rev = list(g_intrs_rev)
            for g2 in g_intrs_rev:
                if (g1, g2) not in intrs:
                    j = cols.get_loc(g2)
                    J[i, j] = 1.0
                    J[j, i] = 1.0
                    intrs_rev.append((g2, g1))

        except KeyError:
            continue


    return J

def build_network_string(gene_names, string_ppi):

    net_intr = pd.Series(string_ppi["symbolA"].values, index=string_ppi["symbolB"])
    net_intr_rev = pd.Series(string_ppi["symbolB"].values, index=string_ppi["symbolA"]) 

    p = len(gene_names)
    J = np.zeros((p, p))
    intrs = []
    intrs_rev = []
    for i, g1 in enumerate(gene_names):
        try:
            g_intrs = net_intr.loc[g1]
            if isinstance(g_intrs, int):
                g_intrs = [g_intrs]
            else:
                g_intrs = list(g_intrs)
            for g2 in g_intrs:
                if (g2, g1) not in intrs_rev: # check if we haven't encountered the reverse interaction
                    if g2 in gene_names:
                        j = gene_names.index(g2)
                        weight = string_ppi[(string_ppi["symbolA"] == g1) & (string_ppi["symbolB"] == g2)]["weight"].values[0]
                        J[i, j] = weight
                        J[j, i] = weight
                        intrs.append((g1, g2))
        except KeyError:
            continue

        # Check the reverse direction
        try:
            g_intrs_rev = net_intr_rev.loc[g1]
            if isinstance(g_intrs_rev, int):
                g_intrs_rev = [g_intrs_rev]
            else:
                g_intrs_rev = list(g_intrs_rev)
            for g2 in g_intrs_rev:
                if (g1, g2) not in intrs:
                    if g2 in gene_names:
                        j = gene_names.index(g2)
                        weight = string_ppi[(string_ppi["symbolB"] == g1) & (string_ppi["symbolA"] == g2)]["weight"].values[0]
                        J[i, j] = weight
                        J[j, i] = weight
                        intrs_rev.append((g2, g1))

        except KeyError:
            continue


    return J
def save_model(save_dir, seed, params, gammas, bg, net_state=None):
    if bg:
        params_path = f"{save_dir}/bg_bnn_params_s_{seed}.npy"
        gammas_path = f"{save_dir}/bg_bnn_gammas_s_{seed}.npy"
        tree_path = f"{save_dir}/bg_bnn_params_s_{seed}.pkl"
        if net_state is not None:
            net_state_path = f"{save_dir}/bg_bnn_net_tree_s_{seed}.pkl"
    else:
        params_path = f"{save_dir}/bnn_params_s_{seed}.npy"
        gammas_path = f"{save_dir}/bnn_gammas_s_{seed}.npy"
        tree_path = f"{save_dir}/bnn_params_s_{seed}.pkl"
        if net_state is not None:
            net_state_path = f"{save_dir}/bnn_net_state_s_{seed}.pkl"

    np.save(gammas_path, gammas)
    with open(params_path, "wb") as fp:
        for x in jax.tree_util.tree_leaves(params):
            np.save(fp, x, allow_pickle=False)

    params_tree_struct = jax.tree_util.tree_map(lambda t: 0, params)
    with open(tree_path, "wb") as fp:
        pickle.dump(params_tree_struct, fp)

    if net_state is not None:
        with open(net_state_path, "wb") as fp:
            pickle.dump(net_state, fp)

def load_model(save_dir, seed, bg, net_state=False):
    if bg:
        params_path = f"{save_dir}/bg_bnn_params_s_{seed}.npy"
        gammas_path = f"{save_dir}/bg_bnn_gammas_s_{seed}.npy"
        tree_path = f"{save_dir}/bg_bnn_params_s_{seed}.pkl"
        if net_state:
            net_state_path = f"{save_dir}/bg_bnn_net_tree_s_{seed}.pkl"
    else:
        params_path = f"{save_dir}/bnn_params_s_{seed}.npy"
        gammas_path = f"{save_dir}/bnn_gammas_s_{seed}.npy"
        tree_path = f"{save_dir}/bnn_params_s_{seed}.pkl"
        if net_state:
            net_state_path = f"{save_dir}/bnn_net_state_s_{seed}.pkl"

    with open(tree_path, "rb") as fp:
        tree_struct = pickle.load(fp)

    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(params_path, "rb") as fp:
        flat_state = [np.load(fp) for _ in leaves]

    params = jax.tree_util.tree_unflatten(treedef, flat_state)
    gammas = np.load(gammas_path)

    if net_state:
        with open(net_state_path, "rb") as fp:
            net_state = pickle.load(fp)
        return params, gammas, net_state

    return params, gammas


