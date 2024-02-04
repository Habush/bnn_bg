# Author Abdulrahman S. Omar <xabush@singularitynet.io>
import jax
import optax
import jax.numpy as jnp
import numpy as onp
import functools
from utils import ensemble_utils, data_utils, metrics, losses, tree_utils

"""numpy implementations of prediction quality metrics.

Partly adapted from
https://github.com/google-research/google-research/blob/master/bnn_hmc/utils/train_utils.py
"""

def get_task_specific_fns(task, data_info):
  if task == data_utils.Task.CLASSIFICATION:
    likelihood_fn = losses.make_xent_log_likelihood
    ensemble_fn = (
        ensemble_utils.compute_updated_ensemble_predictions_classification)
    predict_fn = get_softmax_predictions
    metrics_fns = {
        "accuracy": metrics.accuracy,
        "nll": metrics.nll,
        "ece": lambda preds, y: metrics.calibration_curve(preds, y)["ece"]
    }
    tabulate_metrics = [
        "train/accuracy", "test/accuracy", "test/nll", "test/ens_accuracy",
        "test/ens_nll", "test/ens_ece"
    ]
  elif task == data_utils.Task.REGRESSION:
    likelihood_fn = losses.make_gaussian_likelihood
    ensemble_fn = ensemble_utils.compute_updated_ensemble_predictions_regression
    predict_fn = get_regression_gaussian_predictions

    data_scale = data_info["y_scale"]
    metrics_fns = {
        "scaled_nll": metrics.regression_nll,
        "scaled_mse": metrics.mse,
        "scaled_rmse": metrics.rmse,
        "nll": lambda preds, y: metrics.regression_nll(preds, y, data_scale),
        "mse": lambda preds, y: metrics.mse(preds, y, data_scale),
        "rmse": lambda preds, y: metrics.rmse(preds, y, data_scale),
    }
    tabulate_metrics = [
        "train/rmse", "train/nll", "test/rmse", "test/nll", "test/ens_rmse",
        "test/ens_nll"
    ]
  return likelihood_fn, predict_fn, ensemble_fn, metrics_fns, tabulate_metrics

def make_likelihood_prior_grad_fns(net_apply, log_likelihood_fn,
                                   log_prior_fn):
    """Make functions for training and evaluation.

    Functions return likelihood, prior and gradients separately. These values
    can be combined differently for full-batch and mini-batch methods.
    """

    def likelihood_prior_and_grads_fn(params, net_state, batch):
        loss_val_grad = jax.value_and_grad(
            log_likelihood_fn, has_aux=True, argnums=1)
        (likelihood,
         net_state), likelihood_grad = loss_val_grad(net_apply, params, net_state,
                                                     batch, True)
        prior, prior_grad = jax.value_and_grad(log_prior_fn)(params)
        return likelihood, likelihood_grad, prior, prior_grad, net_state

    return likelihood_prior_and_grads_fn

def make_likelihood_prior_grad_mixed_fns(net_apply, log_likelihood_fn,
                                   log_prior_fn):
    """Make functions for training and evaluation.

    Functions return likelihood, prior and gradients separately. These values
    can be combined differently for full-batch and mini-batch methods.
    """

    def likelihood_prior_and_grads_fn(params, gamma, net_state, batch):
        loss_val_grad = jax.value_and_grad(
            log_likelihood_fn, has_aux=True, argnums=1)
        (likelihood,
         net_state), likelihood_grad = loss_val_grad(net_apply, params, net_state,
                                                     batch, True)
        prior, prior_grad = jax.value_and_grad(log_prior_fn)(params, gamma)
        return likelihood, likelihood_grad, prior, prior_grad, net_state

    return likelihood_prior_and_grads_fn


def make_bin_likelihood_prior_grad_fns(log_likelihood_fn,
                                       log_prior_fn):
    """Make functions for training and evaluation for binary latent variables."""

    def likelihood_prior_and_grads_fn(gamma, params):
        likelihood, likelihood_grad = jax.value_and_grad(log_likelihood_fn)(gamma, params)
        prior, prior_grad = jax.value_and_grad(log_prior_fn)(gamma)

        return likelihood, likelihood_grad, prior, prior_grad

    return likelihood_prior_and_grads_fn

def make_minibatch_log_prob_and_grad(
        likelihood_prior_and_grads_fn, num_batches):
    """Make log-prob and grad function for mini-batch methods."""
    @jax.jit
    def log_prob_and_grad(dataset, params, net_state):
        likelihood, likelihood_grad, prior, prior_grad, net_state = (
            likelihood_prior_and_grads_fn(params, net_state, dataset))

        log_prob = likelihood * num_batches + prior
        grad = jax.tree_map(lambda gl, gp: gl * num_batches + gp,
                            likelihood_grad, prior_grad)
        return log_prob, grad, net_state

    return log_prob_and_grad

def make_minibatch_log_prob_and_grad_mixed(
        likelihood_prior_and_grads_fn, num_batches):
    """Make log-prob and grad function for mini-batch methods."""
    @jax.jit
    def log_prob_and_grad(dataset, params, gamma, net_state):
        likelihood, likelihood_grad, prior, prior_grad, net_state = (
            likelihood_prior_and_grads_fn(params, gamma, net_state, dataset))

        log_prob = likelihood * num_batches + prior
        grad = jax.tree_map(lambda gl, gp: gl * num_batches + gp,
                            likelihood_grad, prior_grad)
        return log_prob, grad, net_state

    return log_prob_and_grad


def make_bin_log_prob_and_grad(
        likelihood_prior_and_grads_fn, num_batches):
    """Make log-prob and grad function for binary latent variables"""

    def log_prob_and_grad(gamma, params):
        likelihood, likelihood_grad, prior, prior_grad, = (
            likelihood_prior_and_grads_fn(gamma, params))

        log_prob = likelihood*num_batches + prior
        grad = jax.tree_map(lambda gl, gp: gl + gp,
                            likelihood_grad, prior_grad)
        return log_prob, grad

    return log_prob_and_grad


def evaluate_metrics(preds, targets, metrics_fns):
    """Evaluate performance metrics on predictions."""
    stats = {}
    for metric_name, metric_fn in metrics_fns.items():
        stats[metric_name] = metric_fn(preds, targets)
    return stats


def make_sgd_train_epoch(net_apply, log_likelihood_fn, log_prior_fn, optimizer,
                         num_batches):
    """Make a training epoch function for SGD-like optimizers.
    """
    likelihood_prior_and_grads_fn = (
        make_likelihood_prior_grad_fns(net_apply, log_likelihood_fn,
                                       log_prior_fn))

    log_prob_and_grad = make_minibatch_log_prob_and_grad(
        likelihood_prior_and_grads_fn, num_batches)

    @jax.jit
    def sgd_train_epoch_fn(params, net_state, opt_state, train_set, key):
        n_data = train_set[0].shape[0]
        batch_size = n_data // num_batches
        indices = jax.random.permutation(key, jnp.arange(n_data))
        indices = jax.tree_map(lambda x: x.reshape((num_batches, batch_size)),
                               indices)

        def train_step(carry, batch_indices):
            batch = jax.tree_map(lambda x: x[batch_indices], train_set)
            params_, net_state_, opt_state_ = carry
            loss, grad, net_state_ = log_prob_and_grad(
                batch, params_, net_state_)

            updates, opt_state_ = optimizer.update(grad, opt_state_)
            params_ = optax.apply_updates(params_, updates)
            return (params_, net_state_, opt_state_), loss

        (params, net_state,
         opt_state), losses = jax.lax.scan(train_step,
                                           (params, net_state, opt_state), indices)

        new_key, = jax.random.split(key, 1)
        return losses, params, net_state, opt_state, new_key

    def sgd_train_epoch(params, net_state, opt_state, train_set, key):
        losses, params, net_state, opt_state, new_key = (
            sgd_train_epoch_fn(params, net_state, opt_state, train_set, key))

        # params, opt_state = map(tree_utils.get_first_elem_in_sharded_tree,
        #                         [params, opt_state])
        loss_avg = jnp.mean(losses)
        return params, net_state, opt_state, loss_avg, new_key

    return sgd_train_epoch


def make_sgd_train_epoch_mixed(net_apply, log_likelihood_fn, log_prior_fn, optimizer,
                               bin_loglikelihood_fn, bin_logprior_fn, bin_optimizer, num_batches):
    """Make a training epoch function for SGD-like optimizers. Used for training a mixed support posterior
    """
    likelihood_prior_and_grads_fn = (
        make_likelihood_prior_grad_mixed_fns(net_apply, log_likelihood_fn,
                                       log_prior_fn))

    bin_likelihood_prior_and_grads_fn = make_bin_likelihood_prior_grad_fns(bin_loglikelihood_fn,
                                                                           bin_logprior_fn)

    log_prob_and_grad = make_minibatch_log_prob_and_grad_mixed(
        likelihood_prior_and_grads_fn, num_batches)

    bin_log_prob_and_grad = make_bin_log_prob_and_grad(bin_likelihood_prior_and_grads_fn, num_batches)

    @jax.jit
    def sgd_train_epoch_fn(params, gamma, net_state, opt_state, bin_opt_state,
                           train_set, key):
        n_data = train_set[0].shape[0]
        batch_size = n_data // num_batches
        indices = jax.random.permutation(key, jnp.arange(n_data))
        if n_data % batch_size != 0:
            indices = indices[:batch_size*num_batches] # drop remainder
        indices = jax.tree_map(lambda x: x.reshape((num_batches, batch_size)),
                               indices)

        def train_step(carry, batch_indices):
            batch = jax.tree_map(lambda x: x[batch_indices], train_set)
            params_, gamma_, net_state_, opt_state_, bin_opt_state_ = carry
            # update the continuous weights
            loss, grad, net_state_ = log_prob_and_grad(
                batch, params_, gamma_, net_state_)

            updates, opt_state_ = optimizer.update(grad, opt_state_)
            params_ = optax.apply_updates(params_, updates)
            # update the binary latent variables
            bin_log_prob, bin_grad = bin_log_prob_and_grad(gamma_, params_)
            bin_updates, bin_opt_state_ = bin_optimizer.update(gamma_, bin_grad, bin_opt_state_)
            # gamma_ = (1 - gamma_) * bin_updates + gamma_ * (1 - bin_updates)
            gamma_ = jax.tree_map(lambda g, b: (1 - g) * b + g * (1 - b), gamma_, bin_updates)
            return (params_, gamma_, net_state_, opt_state_, bin_opt_state_), (loss, bin_log_prob)

        (params, gamma, net_state,
         opt_state, bin_opt_state), losses = jax.lax.scan(train_step,
                                                          (params, gamma, net_state,
                                                           opt_state, bin_opt_state), indices)

        new_key, = jax.random.split(key, 1)
        return (losses, params, gamma, net_state,
                opt_state, bin_opt_state, new_key)

    @jax.jit
    def sgd_train_epoch(params, gamma, net_state, opt_state, bin_opt_state,
                        train_set, key):
        (losses, params, gamma, net_state,
         opt_state, bin_opt_state, new_key) = (
            sgd_train_epoch_fn(params, gamma, net_state, opt_state, bin_opt_state, train_set, key))
        # params, opt_state = map(tree_utils.get_first_elem_in_sharded_tree,
        #                         [params, opt_state])
        # gamma, bin_opt_state = map(tree_utils.get_first_elem_in_sharded_tree,
        #                            [gamma, bin_opt_state])
        loss_avg = jnp.mean(losses[0])
        bin_loss_avg = jnp.mean(losses[1])
        return params, gamma, net_state, opt_state, bin_opt_state, loss_avg, bin_loss_avg, new_key

    return sgd_train_epoch


def make_get_predictions(activation_fn, num_batches=1, is_training=False):
    """Make a function for getting predictions from a network."""

    def get_predictions(net_apply, params, net_state, dataset):
        batch_size = dataset[0].shape[0] // num_batches
        dataset = jax.tree_map(
            lambda x: x.reshape((num_batches, batch_size, *x.shape[1:])), dataset)

        def get_batch_predictions(current_net_state, x):
            y, current_net_state = net_apply(params, current_net_state, None, x,
                                             is_training)
            batch_predictions = activation_fn(y)
            return current_net_state, batch_predictions

        net_state, predictions = jax.lax.scan(get_batch_predictions, net_state,
                                              dataset)
        predictions = predictions.reshape(
            (num_batches * batch_size, *predictions.shape[2:]))
        return net_state, predictions

    return get_predictions


get_softmax_predictions = make_get_predictions(jax.nn.softmax)
get_regression_gaussian_predictions = make_get_predictions(
    losses.preprocess_network_outputs_gaussian)
