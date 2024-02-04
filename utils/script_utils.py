# Author Abdulrahman S. Omar <xabush@singularitynet.io>
import jax
import jax.numpy as jnp
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils.losses_ext as losses_ext
import utils.losses as losses
import utils.train_utils as train_utils
import utils.tree_utils as tree_utils
import core.optim as optim
import core.models as models
import core.sgmcmc as sgmcmc
import core.sgmcmc_ext as sgmcmc_ext
from scipy.stats import binom
from utils import metrics
from utils import nn_util
import mlflow
import haiku as hk
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

plt.style.use("ggplot")

#Source https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html#Implementing-a-Logging-Callback:
def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)
    counter = study.user_attrs.get("counter", 0)
    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        study.set_user_attr("counter", 0)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
    else:
        if frozen_trial.value: #Can be None if the predictions are nan
            improvement_percent = ((winner - frozen_trial.value) / study.best_value) * 100
            if improvement_percent < 0.1:
                counter += 1
                if counter == 20:
                    print(f"No improvement greater than 0.1%. Terminating at trial {frozen_trial.number + 1}")
                    study.stop()
                study.set_user_attr("counter", counter)

def plot_feature_importance(feat_importances):
    """
    Plots feature importance for an XGBoost model.

    Args:
    - model: A trained XGBoost model

    Returns:
    - fig: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    indices = np.arange(len(feat_importances))
    ax.bar(indices, feat_importances, color="r")

    return fig


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def calculate_bnn_metrics(test_preds, y_test):
    metrics_fns = {
        "mse": lambda preds, y: metrics.mse(preds, y),
        "rmse": lambda preds, y: metrics.rmse(preds, y),
        "pearson_corr": lambda preds, y: stats.pearsonr(preds[:, 0], y)[0],
        "pearson_pval": lambda preds, y: stats.pearsonr(preds[:, 0], y)[1],
    }
    metrics_val = {"mse": [], "rmse": [], "pearson_corr": [], "pearson_pval": []}
    for pred in test_preds:
        for k in metrics_val:
            if np.isnan(pred).any():
                metrics_val[k].append(np.nan)
            else:
                metrics_val[k].append(metrics_fns[k](pred, y_test))

    res = {}
    for k in metrics_val:
        res[k] = np.mean(metrics_val[k])
    return res


def calculate_sklearn_model_metric(preds, y_test):
    metrics_fns = {
        "mse": lambda preds, y: np.mean((preds - y) ** 2),
        "rmse": lambda preds, y: np.sqrt(np.mean((preds - y) ** 2)),
        "pearson_corr": lambda preds, y: stats.pearsonr(preds, y)[0],
        "pearson_pval": lambda preds, y: stats.pearsonr(preds, y)[1],
    }
    metrics_val = {"mse": 0., "rmse": 0., "pearson_corr": 0., "pearson_pval": 0.}
    for k in metrics_val:
        if np.isnan(preds).any():
            metrics_val[k] = np.nan
        else:
            metrics_val[k] = metrics_fns[k](preds, y_test)
    return metrics_val
def get_prior_q(n, p):
    K = max(10, np.log(n))
    q_seq = np.arange(1 / p, (1 - 1 / p), 1 / p)
    probs = abs(stats.binom.cdf(k=K, n=p, p=q_seq) - 0.9)
    q = min(q_seq[probs == np.min(probs)])
    return q

def init_gamma(key, tree, q):
    num_leaves = len(jax.tree_util.tree_leaves(tree))
    keys = list(jax.random.split(key, num_leaves))
    treedef = jax.tree_util.tree_structure(tree)
    keys = jax.tree_util.tree_unflatten(treedef, keys)
    return jax.tree_map(lambda k, x: tfd.Bernoulli(probs=jnp.ones_like(x)*q).sample(seed=k)*1.0,keys, tree)

def resample_params(key, params, std=0.005):
    num_leaves = len(jax.tree_util.tree_leaves(params))
    normal_keys = list(jax.random.split(key, num_leaves))
    treedef = jax.tree_util.tree_structure(params)
    normal_keys = jax.tree_util.tree_unflatten(treedef, normal_keys)
    params = jax.tree_util.tree_map(lambda p, k: jax.random.normal(k, p.shape) * std,
                                    params, normal_keys)
    return params
def train_bnn_first_ss_model(rng, train_set, test_set, J,
                             params, predict_fn=train_utils.get_regression_gaussian_predictions):

    n_iters, burnin_len = params["n_iters"], params["burnin"]
    n_batches, save_freq = params["n_batches"], params["save_freq"]
    temperature = params["temperature"]
    tau0, tau1 = params["tau0"], params["tau1"]
    base_dist = params["base_dist"]
    slab_log_prob_fn = losses.make_base_dist_log_prob(base_dist, 0, tau1)
    spike_log_prob_fn = losses.make_base_dist_log_prob(base_dist, 0, tau0)
    prior_log_prob_fn = losses.make_base_dist_log_prob(base_dist, 0, params["scale"])
    log_prior_fn = losses.make_spike_slap_log_prior(slab_log_prob_fn, spike_log_prob_fn,
                                                    prior_log_prob_fn, temperature)
    log_likelihood_fn = losses.make_gaussian_likelihood(temperature)

    eta, mu = params["eta"], params["mu"]
    bin_log_prior_fn = losses.make_bin_log_prior(J, eta, mu)
    q = params["q"]
    bin_log_likelihood_fn = losses.make_bin_log_likelihood(slab_log_prob_fn, spike_log_prob_fn, q, 1.0,
                                                           temperature)

    layer_dims = [params["layer_dim"]] * params["n_layers"]
    net_fn = models.make_bnn_model(layer_dims=layer_dims, invsp_noise_std=params["invsp_noise_std"],
                                   dropout_layer=True)
    net = hk.transform_with_state(net_fn)
    net_apply, net_init = net.apply, net.init

    lr = params["lr"]
    bin_lr = params["bin_lr"]
    mom = 0.95
    rmsprop_precond = sgmcmc_ext.get_rmsprop_preconditioner()
    rmsprop_precond_bin = sgmcmc_ext.get_rmsprop_preconditioner()
    lr_schedule = nn_util.make_constant_lr_schedule_with_cosine_burnin(lr, 1e-6, n_iters)
    bin_lr_schedule = nn_util.make_constant_lr_schedule_with_cosine_burnin(bin_lr, 1e-6, n_iters)

    optimizer = sgmcmc_ext.sgld_gradient_update(lr_schedule, momentum_decay=mom, seed=rng,
                                              preconditioner=rmsprop_precond)
    bin_optimizer = sgmcmc_ext.disc_bin_sgld_gradient_update(bin_lr_schedule,
                                                         seed=rng, preconditioner=rmsprop_precond_bin)

    init_rng, key = jax.random.split(rng)

    params, net_state = net.init(init_rng, (train_set[0][0], None), True)

    first_layer = {k: v for k, v in params.items() if k == "dropout"}
    gamma = init_gamma(init_rng, first_layer, q)
    params = resample_params(init_rng, params, std=0.05)
    # params["dropout"]["w"] = jnp.zeros_like(gamma["dropout"]["w"])

    opt_state = optimizer.init(params)
    bin_opt_state = bin_optimizer.init(gamma)
    sgmcmc_train_epoch_mixed = train_utils.make_sgd_train_epoch_mixed(net_apply, log_likelihood_fn,
                                                                      log_prior_fn, optimizer, bin_log_likelihood_fn,
                                                                      bin_log_prior_fn, bin_optimizer, num_batches=n_batches)
    save_freq = 100

    param_samples = []
    gamma_samples = []
    all_test_preds = []

    for iteration in range(n_iters):
        params, gamma, net_state, opt_state, bin_opt_state, logprob_avg, bin_loss_avg, key = sgmcmc_train_epoch_mixed(
            params, gamma, net_state, opt_state, bin_opt_state, train_set, key)

        if iteration > burnin_len and iteration % save_freq == 0:
            param_samples.append(params)
            gamma_samples.append(gamma)
            _, test_predictions = predict_fn(net_apply, params, net_state, test_set)
            all_test_preds.append(np.asarray(test_predictions))

    return param_samples, gamma_samples, all_test_preds

def train_bnn_all_ss_model(rng, train_set, test_set, J, params,
                           predict_fn=train_utils.get_regression_gaussian_predictions):

    n_iters, burnin_len = params["n_iters"], params["burnin"]
    n_batches, save_freq = params["n_batches"], params["save_freq"]
    temperature = params["temperature"]
    tau0, tau1 = params["tau0"], params["tau1"]
    base_dist = params["base_dist"]
    slab_log_prob_fn = losses_ext.make_base_dist_log_prob(base_dist, 0, tau1)
    spike_log_prob_fn = losses_ext.make_base_dist_log_prob(base_dist, 0, tau0)
    log_prior_fn = losses_ext.make_spike_slap_log_prior(slab_log_prob_fn, spike_log_prob_fn, temperature)
    log_likelihood_fn = losses_ext.make_gaussian_likelihood(temperature)

    eta, mu = params["eta"], params["mu"]
    bin_log_prior_fn = losses_ext.make_bin_log_prior(J, eta, mu)
    q_first, q_rest = params["q_first"], params["q_rest"]
    bin_log_likelihood_fn = losses_ext.make_bin_log_likelihood(slab_log_prob_fn, spike_log_prob_fn, (q_first, q_rest),
                                                           temperature)

    layer_dims = [params["layer_dim"]] * params["n_layers"]
    net_fn = models.make_bnn_model(layer_dims=layer_dims, invsp_noise_std=params["invsp_noise_std"],
                                   dropout_layer=True)
    net = hk.transform_with_state(net_fn)
    net_apply, net_init = net.apply, net.init

    lr = params["lr"]
    bin_lr = params["bin_lr"]
    mom = 0.95
    rmsprop_precond = sgmcmc_ext.get_rmsprop_preconditioner()
    rmsprop_precond_bin = sgmcmc_ext.get_rmsprop_preconditioner()
    lr_schedule = nn_util.make_constant_lr_schedule_with_cosine_burnin(lr, 1e-6, n_iters)
    bin_lr_schedule = nn_util.make_constant_lr_schedule_with_cosine_burnin(bin_lr, 1e-6, n_iters)

    optimizer = sgmcmc_ext.sgld_gradient_update(lr_schedule, momentum_decay=mom, seed=rng,
                                              preconditioner=rmsprop_precond)
    bin_optimizer = sgmcmc_ext.disc_bin_sgld_gradient_update(bin_lr_schedule,
                                                         seed=rng, preconditioner=rmsprop_precond_bin)

    init_rng, key = jax.random.split(rng)

    params, net_state = net.init(init_rng, (train_set[0][0], None), True)

    first_layer = {k: v for k, v in params.items() if k == "dropout"}
    rest_layers = {k: v for k, v in params.items() if k != "dropout"}

    first_layer_gamma = init_gamma(init_rng, first_layer, q_first)
    rest_layers_gamma = init_gamma(init_rng, rest_layers, q_rest)
    gamma = {**first_layer_gamma, **rest_layers_gamma}
    params = resample_params(init_rng, params, std=0.05)
    # params["dropout"]["w"] = jnp.zeros_like(gamma["dropout"]["w"])

    opt_state = optimizer.init(params)
    bin_opt_state = bin_optimizer.init(gamma)
    sgmcmc_train_epoch_mixed = train_utils.make_sgd_train_epoch_mixed(net_apply, log_likelihood_fn,
                                                                      log_prior_fn, optimizer, bin_log_likelihood_fn,
                                                                      bin_log_prior_fn, bin_optimizer, num_batches=n_batches)
    param_samples = []
    gamma_samples = []
    all_test_preds = []

    for iteration in range(n_iters):
        params, gamma, net_state, opt_state, bin_opt_state, logprob_avg, bin_loss_avg, key = sgmcmc_train_epoch_mixed(
            params, gamma, net_state, opt_state, bin_opt_state, train_set, key)

        if iteration > burnin_len and iteration % save_freq == 0:
            param_samples.append(params)
            gamma_samples.append(gamma)
            _, test_predictions = predict_fn(net_apply, params, net_state, test_set)
            all_test_preds.append(np.asarray(test_predictions))

    return param_samples, gamma_samples, all_test_preds