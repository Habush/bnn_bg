# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optax implementations of SGMCMC optimizers."""

import jax
from optax import Params
from jax import numpy as jnp
import optax
from optax import GradientTransformation
from typing import Any, NamedTuple
import tree_utils
import haiku as hk

class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState

class ResnetState(NamedTuple):
    params: hk.Params
    opt_state: hk.Params
    net_state: hk.State

Momentum = Any  # An arbitrary pytree of `jnp.ndarrays`
GradMomentEstimates = optax.Params  # Same type as parameters
PreconditionerState = NamedTuple  # State of a preconditioner


class OptaxSGLDState(NamedTuple):
    """Optax state for the SGLD optimizer"""
    count: jnp.ndarray
    momentum: Momentum
    preconditioner_state: PreconditionerState

class BNNState(NamedTuple):
    params: hk.Params

def sgd_gradient_update(step_size_fn,
                        momentum_decay=0.,
                        preconditioner=None):
    """Optax implementation of the SGD optimizer.
    """

    if preconditioner is None:
        preconditioner = get_identity_preconditioner()

    def init_fn(params):
        return OptaxSGLDState(
            count=jnp.zeros([], jnp.int32),
            momentum=jax.tree_map(jnp.zeros_like, params),
            preconditioner_state=preconditioner.init(params))

    def update_fn(gradient, state):
        lr = step_size_fn(state.count)
        lr_sqrt = jnp.sqrt(lr)

        preconditioner_state = preconditioner.update_preconditioner(
            gradient, state.preconditioner_state)

        def update_momentum(m, g):
            return momentum_decay * m + g * lr_sqrt



        momentum = jax.tree_map(update_momentum, state.momentum, gradient)
        updates = preconditioner.multiply_by_m_inv(momentum, preconditioner_state)
        updates = jax.tree_map(lambda m: m * lr_sqrt, updates)
        return updates, OptaxSGLDState(
            count=state.count + 1,
            momentum=momentum,
            preconditioner_state=preconditioner_state)

    return GradientTransformation(init_fn, update_fn)


def sgld_gradient_update(step_size_fn,
                         momentum_decay=0.,
                         preconditioner=None):
    """Optax implementation of the SGLD optimizer.

    If momentum_decay is set to zero, we get the SGLD method [1]. Otherwise,
    we get the underdamped SGLD (SGHMC) method [2].

    Args:
      step_size_fn: a function taking training step as input and prodng the
        step size as output.
      momentum_decay: float, momentum decay parameter (default: 0).
      preconditioner: Preconditioner, an object representing the preconditioner
        or None; if None, identity preconditioner is used (default: None).  [1]
          "Bayesian Learning via Stochastic Gradient Langevin Dynamics" Max
          Welling, Yee Whye Teh; ICML 2011  [2] "Stochastic Gradient Hamiltonian
          Monte Carlo" Tianqi Chen, Emily B. Fox, Carlos Guestrin; ICML 2014
    """

    if preconditioner is None:
        preconditioner = get_identity_preconditioner()

    def init_fn(params):
        return OptaxSGLDState(
            count=jnp.zeros([], jnp.int32),
            momentum=jax.tree_map(jnp.zeros_like, params),
            preconditioner_state=preconditioner.init(params))

    def update_fn(gradient, state, key):
        lr = step_size_fn(state.count)
        lr_sqrt = jnp.sqrt(lr)
        noise_std = jnp.sqrt(2 * (1 - momentum_decay))

        preconditioner_state = preconditioner.update_preconditioner(
            gradient, state.preconditioner_state)

        noise_std = jnp.sqrt(2 * (1 - momentum_decay))
        noise, _ = tree_utils.normal_like_tree(gradient, key)
        noise = preconditioner.multiply_by_m_sqrt(noise, preconditioner_state)

        def update_momentum(m, g, n):
            return momentum_decay * m + g * lr_sqrt + n * noise_std



        momentum = jax.tree_map(update_momentum, state.momentum, gradient, noise)
        updates = preconditioner.multiply_by_m_inv(momentum, preconditioner_state)
        updates = jax.tree_map(lambda m: m * lr_sqrt, updates)
        return updates, OptaxSGLDState(
            count=state.count + 1,
            momentum=momentum,
            preconditioner_state=preconditioner_state)

    return GradientTransformation(init_fn, update_fn)


def disc_sgd_gradient_update(step_size_fn,
                             momentum_decay=0.,
                             preconditioner=None):
    """Optax implementation of the SGD optimizer.
    """

    if preconditioner is None:
        preconditioner = get_identity_preconditioner()

    def init_fn(gamma):
        return OptaxSGLDState(
            count=jnp.zeros([], jnp.int32),
            momentum=jax.tree_map(jnp.zeros_like, gamma),
            preconditioner_state=preconditioner.init(gamma))

    def update_fn(key, gamma, gradient, state):
        lr = step_size_fn(state.count)
        lr_sqrt = jnp.sqrt(lr)

        preconditioner_state = preconditioner.update_preconditioner(
            gradient, state.preconditioner_state)

        def update_momentum(m, g):
            return momentum_decay * m + g * lr_sqrt

        def proposal(theta, g, step_size):
            diff = (0.5*g*-(2*theta - 1)) - (1./(2*step_size))
            prob = jax.nn.sigmoid(diff)
            prob_inv = 1 - prob
            prob = prob[...,None]
            prob_inv = prob_inv[...,None]
            delta = jnp.argmax(jnp.concatenate([prob, prob_inv], axis=1), axis=-1)

            theta_delta = (1 - theta)*delta + theta*(1 - delta)
            return theta_delta*1.

        momentum = jax.tree_map(update_momentum, state.momentum, gradient)
        updates = preconditioner.multiply_by_m_inv(momentum, preconditioner_state)
        updates = jax.tree_map(lambda m: m * lr_sqrt, updates)
        gamma = proposal(gamma, updates, lr)

        return gamma, OptaxSGLDState(
            count=state.count + 1,
            momentum=momentum,
            preconditioner_state=preconditioner_state)

    return GradientTransformation(init_fn, update_fn)


def disc_sgld_gradient_update(step_size_fn,
                              momentum_decay=0.,
                              preconditioner=None):
    """Optax implementation of the  DULA [1] with momentum.

    [1] - "A Langevin-like Sampler for Discrete Distributions" by Ruqi Zhang et.al 2022
    """

    if preconditioner is None:
        preconditioner = get_identity_preconditioner()

    def init_fn(gamma):
        return OptaxSGLDState(
            count=jnp.zeros([], jnp.int32),
            momentum=jax.tree_map(jnp.zeros_like, gamma),
            preconditioner_state=preconditioner.init(gamma))

    def update_fn(key, gamma, gradient, state):
        lr = step_size_fn(state.count)
        lr_sqrt = jnp.sqrt(lr)

        preconditioner_state = preconditioner.update_preconditioner(
            gradient, state.preconditioner_state)

        def update_momentum(m, g):
            return momentum_decay * m + g * lr_sqrt

        def proposal(key, theta, g, step_size):
            diff = (-0.5*g*(2*theta - 1)) - (1./(2*step_size))
            delta = jax.random.bernoulli(key, jax.nn.sigmoid(diff))
            theta_delta = (1 - theta)*delta + theta*(1 - delta)
            return theta_delta*1.



        momentum = jax.tree_map(update_momentum, state.momentum, gradient)
        updates = preconditioner.multiply_by_m_inv(momentum, preconditioner_state)
        updates = jax.tree_map(lambda m: m * lr_sqrt, updates)
        gamma = proposal(key, gamma, updates, lr)


        return gamma, OptaxSGLDState(
            count=state.count + 1,
            momentum=momentum,
            preconditioner_state=preconditioner_state)

    return GradientTransformation(init_fn, update_fn)


class Preconditioner(NamedTuple):
    """Preconditioner transformation"""
    init: Any  # TODO @izmailovpavel: fix
    update_preconditioner: Any
    multiply_by_m_sqrt: Any
    multiply_by_m_inv: Any
    multiply_by_m_sqrt_inv: Any


class RMSPropPreconditionerState(PreconditionerState):
    grad_moment_estimates: GradMomentEstimates


def get_rmsprop_preconditioner(running_average_factor=0.99, eps=1.e-7):

    def init_fn(params):
        return RMSPropPreconditionerState(
            grad_moment_estimates=jax.tree_map(jnp.zeros_like, params))

    def update_preconditioner_fn(gradient, preconditioner_state):
        grad_moment_estimates = jax.tree_map(
            lambda e, g: e * running_average_factor + \
                         g**2 * (1 - running_average_factor),
            preconditioner_state.grad_moment_estimates, gradient)
        return RMSPropPreconditionerState(
            grad_moment_estimates=grad_moment_estimates)

    def multiply_by_m_inv_fn(vec, preconditioner_state):
        return jax.tree_map(lambda e, v: v / (eps + jnp.sqrt(e)),
                            preconditioner_state.grad_moment_estimates, vec)

    def multiply_by_m_sqrt_fn(vec, preconditioner_state):
        return jax.tree_map(lambda e, v: v * jnp.sqrt(eps + jnp.sqrt(e)),
                            preconditioner_state.grad_moment_estimates, vec)

    def multiply_by_m_sqrt_inv_fn(vec, preconditioner_state):
        return jax.tree_map(lambda e, v: v / jnp.sqrt(eps + jnp.sqrt(e)),
                            preconditioner_state.grad_moment_estimates, vec)

    return Preconditioner(
        init=init_fn,
        update_preconditioner=update_preconditioner_fn,
        multiply_by_m_inv=multiply_by_m_inv_fn,
        multiply_by_m_sqrt=multiply_by_m_sqrt_fn,
        multiply_by_m_sqrt_inv=multiply_by_m_sqrt_inv_fn)


class IdentityPreconditionerState(PreconditionerState):
    """Identity preconditioner is stateless."""


def get_identity_preconditioner():

    def init_fn(_):
        return IdentityPreconditionerState()

    def update_preconditioner_fn(*args, **kwargs):
        return IdentityPreconditionerState()

    def multiply_by_m_inv_fn(vec, _):
        return vec

    def multiply_by_m_sqrt_fn(vec, _):
        return vec

    def multiply_by_m_sqrt_inv_fn(vec, _):
        return vec

    return Preconditioner(
        init=init_fn,
        update_preconditioner=update_preconditioner_fn,
        multiply_by_m_inv=multiply_by_m_inv_fn,
        multiply_by_m_sqrt=multiply_by_m_sqrt_fn,
        multiply_by_m_sqrt_inv=multiply_by_m_sqrt_inv_fn)

def make_cyclical_lr_fn(lr_0, total, num_cycles):
    k = total // num_cycles
    def schedule_fn(step):
        rk = (step % k)
        cos_inner = jnp.pi * rk
        cos_inner /= k
        cos_out = jnp.cos(cos_inner) + 1
        lr = 0.5*cos_out*lr_0

        return lr

    return schedule_fn
