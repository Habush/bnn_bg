import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from optim import *
from typing import NamedTuple
from jax.random import PRNGKey

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

def spike_slab_log_prob(w, prior_name, sigma_1, sigma_2, gamma):
    """compute the log probability of the spike and slab prior"""
    eps = 1e-12
    dist_1, dist_2 = get_prior(prior_name, sigma_1), get_prior(prior_name, sigma_2)
    mask_1, mask_2 = jnp.diag(1 - gamma), jnp.diag(gamma)
    prob1, prob2 = jnp.exp(dist_1.log_prob(w)), jnp.exp(dist_2.log_prob(w))
    out = jnp.log((mask_1@prob1 + mask_2@prob2) + eps)
    return jnp.sum(out)
class DropoutLayer(hk.Module):
    def __init__(self, in_dim, out_dim, init_fn, name=None):
        super().__init__(name)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_fn = init_fn

    def __call__(self, x, gamma):
        w = hk.get_parameter("w", [self.in_dim, self.out_dim], jnp.float32, self.init_fn)
        out = jnp.dot(x, w)
        b = hk.get_parameter("b", [self.out_dim], jnp.float32, self.init_fn)
        b = jnp.broadcast_to(b, out.shape)
        out = out + b
        return out


class TrainingState(NamedTuple):
    params: hk.Params
    gamma: jnp.ndarray
    opt_state: optax.OptState
    disc_opt_state: optax.OptState
class BayesNN:
    def __init__(self, sgd_optim: optax.GradientTransformation, sgld_optim: optax.GradientTransformation,
                 disc_sgld_optim: optax.GradientTransformation,
                 temperature, sigma_1, sigma_2, data_size, hidden_sizes,
                 J, eta, mu, act_fn, init_fn, weight_prior="Laplace",
                 classification=False):
        """
        Initialise the Bayesian neural network
        There are two optimisers for each of the continuous and discrete components, one is the Stochastic Gradient
        Descent (SGD) and the other is the Stochastic Gradient Langevin Dynamics (SGLD) algorithm. We use the SGD in the
        optimisation step and the SGLD in the sampling step. See https://arxiv.org/abs/1902.03932 for more details.
        :param sgd_optim: SGD optimiser for the continuous weights, i.e neural network weights
        :param sgld_optim: SGLD optimiser for the continuous weights, i.e neural network weights
        :param disc_sgld_optim: SGLD optimiser for the discrete weights, i.e latent binary variables / feature selectors
        :param temperature: The posterior temperature. It is shown that cold posterior (i.e. low temperature) leads to better performance. See https://arxiv.org/abs/2106.06596 for more details.
        :param sigma_1: The standard deviation of the spike distribution in the spike and slab prior
        :param sigma_2: The standard deviation of the slab distribution in the spike and slab prior (sigma_1 << sigma_2)
        :param data_size: The size of the training data
        :param hidden_sizes: The number of hidden units in each layer.
        :param J: The background graph. It could an adjacency matrix or a Laplacian matrix.
        :param eta: Controls the strength of the effect of the background graph in the Ising. This is a hyperparameter.
        :param mu: Controls the sparsity in the Ising prior. This is a hyperparameter.
        :param act_fn: The activation function
        :param init_fn: The initialisation function for the weights
        :param weight_prior: Prior distribution over the weights. Currently supported are "Laplace", "Gaussian" and "StudentT"
        :param classification: Whether the model is used for classification or regression
        """
        self.hidden_sizes = hidden_sizes
        self.act_fn = act_fn
        self.sgd_optim = sgd_optim
        self.sgld_optim = sgld_optim
        self.optimiser = sgd_optim

        # self.disc_sgld_optim = disc_sgld_optim
        self.disc_optimiser = disc_sgld_optim

        self._forward = hk.transform(self._forward_fn)
        self.log_prob = jax.jit(self.contin_log_prob)
        self.disc_log_prob = jax.jit(self.disc_log_prob)
        self.update = jax.jit(self.update)

        self.temperature = temperature
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.data_size = data_size
        self.add_noise = False
        self.J = J
        self.eta = eta
        self.mu = mu
        self.init_fn = init_fn

        self.weight_prior = weight_prior
        self.dropout_layer = DropoutLayer
        self.classification = classification

        if classification:
            self.log_likelihood = self.clf_log_likelihood
        else:
            self.log_likelihood = self.reg_log_likelihood


    def init(self, rng: PRNGKey, x: jnp.ndarray) -> TrainingState:
        gamma = tfd.Bernoulli(0.5*jnp.ones(x.shape[-1])).sample(seed=rng)*1.
        params = self._forward.init(rng, x, gamma, True)
        opt_state = self.optimiser.init(params)
        disc_opt_state = self.disc_optimiser.init(gamma)
        return TrainingState(params, gamma, opt_state, disc_opt_state)

    def apply(self, params: hk.Params, gamma: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return self._forward.apply(params, None, x, gamma).ravel()

    def contin_log_prob(self, params: hk.Params, gamma: jnp.ndarray,
                        x: jnp.ndarray, y: jnp.ndarray) -> jnp.float32:
        logprob_prior = self.log_prior(params, gamma)
        logprob_likelihood = self.log_likelihood(params, gamma, x, y)
        return (logprob_likelihood + logprob_prior)/self.temperature

    def disc_log_prob(self, gamma: jnp.ndarray, params: hk.Params) -> jnp.float32:
        prior_logprob = self.ising_prior(gamma)
        spike_slab_prior = spike_slab_log_prob(params["dropout"]["w"], self.weight_prior,
                                            self.sigma_1, self.sigma_2, gamma)
        return (prior_logprob + spike_slab_prior)/self.temperature

    def update(self, key: jax.random.PRNGKey, train_state: TrainingState,
               x: jnp.ndarray, y: jnp.ndarray) -> TrainingState:

        params, gamma, opt_state, disc_opt_state = train_state

        if self.add_noise:
            self.optimiser = self.sgld_optim
        else:
            self.optimiser = self.sgd_optim

        grads = jax.grad(self.log_prob)(params, gamma, x, y)
        if self.add_noise:
            updates, opt_state = self.optimiser.update(grads, opt_state, key)
        else:
            updates, opt_state = self.optimiser.update(grads, opt_state)

        params = optax.apply_updates(params, updates)

        disc_grads = jax.grad(self.disc_log_prob)(gamma, params)
        gamma, disc_opt_state = self.disc_optimiser.update(key, gamma, disc_grads, disc_opt_state)

        return TrainingState(params, gamma, opt_state, disc_opt_state)

    def _forward_fn(self, x: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
        z = self.dropout_layer(x.shape[-1], self.hidden_sizes[0], self.init_fn, "dropout")(x, gamma)
        z = self.act_fn(z)
        for i, hd in enumerate(self.hidden_sizes[1:]):
            z = hk.Linear(hd, name=f"linear_{i+1}", w_init=self.init_fn, b_init=self.init_fn)(z)
            z = self.act_fn(z)

        z = hk.Linear(1, name="out_layer", w_init=self.init_fn, b_init=self.init_fn)(z)
        return z

    def log_prior(self, params, gamma):
        """Computes the prior log-probability."""
        dropout_log_prob = spike_slab_log_prob(params["dropout"]["w"],
                                               self.weight_prior,
                                               self.sigma_1, self.sigma_2, gamma)
        dist = get_prior(self.weight_prior, self.sigma_2)
        p = {x: params[x] for x in params if (x != "dropout" and x != "~")}
        logprob_tree = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: jnp.sum(dist.log_prob(x.reshape(-1))),
                                                                        p))
        return dropout_log_prob + sum(logprob_tree)

    def reg_log_likelihood(self, params: hk.Params, gamma: jnp.ndarray,
                           x: jnp.ndarray, y: jnp.ndarray):
        preds = self.apply(params, gamma, x).ravel()
        log_prob = jnp.sum(tfd.Normal(preds, 1.0).log_prob(y))
        batch_size = x.shape[0]
        log_prob = (self.data_size/batch_size)*log_prob
        return log_prob

    def clf_log_likelihood(self, params, gamma, x, y):
        logits = self.apply(params, gamma, x).ravel()
        log_p = jax.nn.log_sigmoid(logits)
        log_not_p = jax.nn.log_sigmoid(-logits)
        log_prob = jnp.sum(y*log_p + (1 - y)*log_not_p)
        batch_size = x.shape[0]
        log_prob = (self.data_size/batch_size)*log_prob
        return log_prob

    def ising_prior(self, gamma: jnp.ndarray) -> jnp.float32:
        """Log probability of the Ising model - prior over the discrete variables"""
        return 0.5*self.eta*(gamma.T @ self.J @ gamma) - self.mu*jnp.sum(gamma)
