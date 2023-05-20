import jax
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from bnn_models_old import *
from bnn_models import *
from nn_util import roc_auc_score
from resnet_models import *
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from horseshoe_bnn.data_handling.dataset import Dataset
from horseshoe_bnn.models import HorseshoeBNN
from horseshoe_bnn.parameters import HorseshoeHyperparameters
from torch import optim
import yaml
import datetime
import torch
import tensorflow as tf
import tensorflow_datasets as tfds
from data_utils import *

def get_act_fn(name):
    if name == "relu":
        return jax.nn.relu
    if name == "swish":
        return jax.nn.swish
    if name == "tanh":
        return jax.nn.tanh
    if name == "sigmoid":
        return jax.nn.sigmoid
    if name == "celu":
        return jax.nn.celu
    if name == "relu6":
        return jax.nn.relu6
    if name == "glu":
        return jax.nn.glu
    if name == "elu":
        return jax.nn.elu
    if name == "leaky_relu":
        return jax.nn.leaky_relu
    if name == "log_sigmoid":
        return jax.nn.log_sigmoid

    return ValueError(f"Unknown activation function: {name}")

def init_bg_bnn_model(seed, train_loader, epochs, lr_0, disc_lr_0, num_cycles, temp, sigma_1, sigma_2,
                         hidden_sizes, J, eta, mu, act_fn, prior_dist, dropout_version,
                         init_fn=None, classifier=False):
    torch.manual_seed(seed)
    num_batches = len(train_loader)
    data_size = train_loader.dataset.data.shape[0]
    # data_size = next(iter(train_loader))[0].shape[0]
    total_steps = num_batches*epochs
    step_size_fn = make_cyclical_lr_fn(lr_0, total_steps, num_cycles)
    disc_step_size_fn = make_cyclical_lr_fn(disc_lr_0, total_steps, num_cycles)

    sgd_optim = sgd_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    sgld_optim = sgld_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    # sgd_optim = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(step_size_fn))
    # sgld_optim = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(step_size_fn))
    #TODO change this

    disc_sgd_optim = disc_sgld_gradient_update(disc_step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    disc_sgld_optim = disc_sgld_gradient_update(disc_step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())

    if init_fn is None:
        init_fn = hk.initializers.VarianceScaling()

    if classifier:
        model = BgBayesClassifier(sgd_optim, sgld_optim, disc_sgd_optim, disc_sgld_optim,
                      temp, sigma_1, sigma_2, data_size, hidden_sizes,
                      J, eta, mu, act_fn, init_fn, prior_dist, dropout_version)
    else:
        model = BgBayesNN2(sgd_optim, sgld_optim, disc_sgd_optim, disc_sgld_optim,
                          temp, sigma_1, sigma_2, data_size, hidden_sizes,
                          J, eta, mu, act_fn, init_fn, prior_dist, dropout_version)

    return model


def train_bg_bnn_model(seed, train_loader, epochs, num_cycles, beta, m, lr_0, disc_lr_0,
                       hidden_sizes, temp, sigma_1, sigma_2, eta, mu, J, act_fn_name,
                       show_pgbar=True, prior_dist="laplace", classifier=False,
                          dropout_version=1):

    rng_key = jax.random.PRNGKey(seed)
    act_fn = get_act_fn(act_fn_name)
    init_fn = hk.initializers.VarianceScaling()

    model = init_bg_bnn_model(seed, train_loader, epochs, lr_0, disc_lr_0, num_cycles,
                              temp, sigma_1, sigma_2, hidden_sizes, J, eta, mu,
                              act_fn, prior_dist, dropout_version, init_fn, classifier)

    num_batches = len(train_loader)
    M = (epochs*num_batches) // num_cycles
    cycle_len = epochs // num_cycles
    init_params, init_gamma, init_opt_state, init_disc_opt_state = model.init(rng_key, next(iter(train_loader))[0])
    # train_state = model.init(rng_key, next(iter(train_loader))[0])
    states = []
    disc_states = []

    params, gamma, opt_state, disc_opt_state = init_params, init_gamma, init_opt_state, init_disc_opt_state
    step = 0
    key = rng_key

    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)

    model.add_noise = True
    do_sample = False
    for _ in pgbar:
        for batch_x, batch_y in train_loader:
            _, key = jax.random.split(key, 2)
            rk = (step % M) / M
            if rk > beta:
                do_sample = True
            else:
                do_sample = False

            model.add_noise = do_sample
            params, gamma, opt_state, disc_opt_state = model.update(key, params, gamma, opt_state, disc_opt_state,
                                                                                            batch_x, batch_y)
            step += 1

        # y_val_pred = model.apply(params, gamma, x_val, False).ravel()
        # val_loss = jnp.mean((y_val_pred - y_val)**2)
        # val_losses.append(val_loss)
        #
        # y_test_pred = model.apply(params, gamma, x_test, False).ravel()
        # test_loss = jnp.mean((y_test_pred - y_test)**2)
        # test_losses.append(test_loss)
            if (step % M) + 1 > (M - m):
                states.append(params)
                disc_states.append(gamma)


    return model, states, disc_states


def train_mlp_model(seed, train_loader, epochs, lr_0, hidden_sizes, act_fn_name, show_pgbar=True):
    rng_key = jax.random.PRNGKey(seed)
    act_fn = get_act_fn(act_fn_name)
    init_fn = hk.initializers.VarianceScaling()
    optim = optax.chain(optax.scale_by_adam(), optax.scale(-lr_0)) # since we're minimizing a loss

    model = MLP(optim, hidden_sizes, init_fn, act_fn)

    init_params, init_opt_state = model.init(rng_key, next(iter(train_loader))[0])
    params, opt_state = init_params, init_opt_state
    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)
    for _ in pgbar:
        for x, y in train_loader:
            params, opt_state = model.update(params, opt_state, x, y)

    return model, params

def apply_bnn_model(rng, model, X, y, params, gammas, is_training, classifier=False,
                       has_state=False, net_state=None):

    y_preds = np.zeros((len(params), len(y)))
    for i, (param, gamma) in enumerate(zip(params, gammas)):
        if has_state:
            preds, _ = model.apply(param, net_state, rng, gamma, X, is_training)
            preds = preds.ravel()
        else:
            preds = model.apply(param, gamma, X, is_training).ravel()
        if classifier:
            y_preds[i] = jax.nn.sigmoid(preds)
        else:
            y_preds[i] = preds

    return y_preds

def score_bg_bnn_model(rng, model, X, y, params, gammas, is_training, classifier=False,
                       has_state=False, net_state=None, y_mean=0.0, y_std=1.0):
    y_preds = apply_bnn_model(rng, model, X, y, params, gammas, is_training, classifier,
                              has_state, net_state)
    if classifier:
        y_preds = np.mean(y_preds, axis=0)
        score = roc_auc_score(y, y_preds)
        acc = accuracy_score(y, y_preds > 0.5)
        return score, acc
    else:
        y_preds = y_preds * y_std + y_mean
        y_preds = np.mean(y_preds, axis=0)
        score = jnp.sqrt(jnp.mean((y - y_preds)**2))
        if np.isfinite(y_preds).all():
            r2 = r2_score(y, y_preds)
        else:
            r2 = np.nan
        return score, r2


def score_bg_bnn_model_batched(rng, model, X, y, params, gammas, batch_size, is_training,
                               classifier=False, has_state=False, net_state=None, y_mean=0.0, y_std=1.0):

    data = tf.data.Dataset.from_tensor_slices((X, y))
    data_loader = tfds.as_numpy(data.batch(batch_size).prefetch(1))

    y_preds = []
    for batch_x, batch_y in data_loader:
        batch_preds = apply_bnn_model(rng, model, batch_x, batch_y, params, gammas, is_training, classifier,
                                        has_state, net_state)
        y_preds.append(batch_preds)

    y_preds = np.concatenate(y_preds, axis=1)
    y_preds = np.mean(y_preds, axis=0)
    if classifier:
        auc = roc_auc_score(y, y_preds)
        acc = accuracy_score(y, y_preds > 0.5)
        return auc, acc
    else:
        y_preds = y_preds * y_std + y_mean
        score = jnp.sqrt(jnp.mean((y - y_preds)**2))
        if np.isfinite(y_preds).all():
            r2 = r2_score(y, y_preds)
        else:
            r2 = np.nan
        return score, r2



def score_bg_bnn_model_v2(model, X, y, params, gammas, is_training, classifier=False):
    preds = jax.vmap(lambda p, g: model.apply(p, g, X, is_training).ravel())(params, gammas)
    preds = preds.squeeze()
    y_preds = np.mean(preds, axis=0)
    if classifier:
        score = roc_auc_score(y, y_preds)
        acc = accuracy_score(y, y_preds > 0.5)
        return score, acc
    else:
        score = jnp.sqrt(jnp.mean((y - y_preds)**2))
        if np.isfinite(y_preds).all():
            r2 = r2_score(y, y_preds)
        else:
            r2 = np.nan
        return score, r2


def train_nn_model(rng_key, data_loader, epochs, num_cycles, lr_0,
                   block_type, num_blocks, hidden_sizes, init_fn, weight_decay,
                   act_fn_name, dropout_rate ,show_pgbar=True):


    act_fn = get_act_fn(act_fn_name)
    total_steps = len(data_loader)*epochs

    schedule_fn = make_cyclical_lr_fn(lr_0, total_steps, num_cycles)
    optim = optax.chain(optax.scale_by_adam(), optax.add_decayed_weights(weight_decay),
                        optax.scale_by_schedule(schedule_fn), optax.scale(-1.0))
    if block_type == "ResNet":
        block_class = ResNetBlock
    else:
        block_class = PreActResNetBlock

    model = ResNet(block_class, num_blocks, hidden_sizes, optim, init_fn, act_fn, dropout_rate)

    cycle_len = epochs // num_cycles
    init_state = model.init(rng_key, next(iter(data_loader))[0])

    state = init_state

    # print(f"Total iterations: {epochs*num_batches}, Num Batches: {num_batches}, Cycle Len: {M}")
    states = []
    val_losses = []
    step = 0
    key = rng_key

    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)

    for epoch in pgbar:
        for batch_x, batch_y in data_loader:
            _, key = jax.random.split(key, 2)
            state = model.update(key, state, batch_x, batch_y)
            step += 1

        if epoch != 0 and ((epoch + 1) % cycle_len == 0): # take snapshot
            states.append(state)


    return model, states, val_losses


def init_resnet_model(train_loader, epochs, lr_0, disc_lr_0, num_cycles, temp, sigma_1, sigma_2,
                         weight_decay, block_class, num_blocks, hidden_sizes,
                         J, eta, mu, act_fn,
                         prior_dist, dropout_version, dropout_rate,
                         init_fn=None):

    num_batches = len(train_loader)
    data_size = train_loader.dataset.data.shape[0]
    total_steps = num_batches*epochs
    step_size_fn = make_cyclical_lr_fn(lr_0, total_steps, num_cycles)
    disc_step_size_fn = make_cyclical_lr_fn(disc_lr_0, total_steps, num_cycles)

    sgd_optim = sgd_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    sgld_optim = sgld_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    # sgd_optim = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(step_size_fn))
    # sgld_optim = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(step_size_fn))
    #TODO change this
    disc_sgd_optim = disc_sgld_gradient_update(disc_step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    disc_sgld_optim = disc_sgld_gradient_update(disc_step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())

    if init_fn is None:
        init_fn = hk.initializers.VarianceScaling()

    if block_class == "ResNet":
        block_type = ResNetBlock

    elif block_class == "PreActResNet":
        block_type = PreActResNetBlock
    else:
        raise ValueError(f"Unknown Resnet architecture: {block_class}")

    model = BgResNet(sgd_optim, sgld_optim, disc_sgd_optim, disc_sgld_optim,
                      temp, sigma_1, sigma_2, weight_decay, data_size, block_type, num_blocks, hidden_sizes,
                      J, eta, mu, act_fn, init_fn, prior_dist, dropout_version, dropout_rate=dropout_rate)

    return model

def train_resnet_bg_model(rng_key, train_loader, epochs, num_cycles, beta, m, lr_0, disc_lr_0,
                   block_class, num_blocks, hidden_sizes, temp, sigma_1, sigma_2, eta, mu, J,
                   act_fn_name, show_pgbar=True, prior_dist="laplace", classifier=False,
                   dropout_version=1, dropout_rate=0.0, weight_decay=1e-4):



    act_fn = get_act_fn(act_fn_name)
    init_fn = hk.initializers.VarianceScaling()

    model = init_resnet_model(train_loader, epochs, lr_0, disc_lr_0, num_cycles, temp, sigma_1, sigma_2,
                                 weight_decay, block_class, num_blocks, hidden_sizes,
                                 J, eta, mu, act_fn,
                                 prior_dist, dropout_version, dropout_rate, init_fn)

    num_batches = len(train_loader)
    M = (epochs * num_batches) // num_cycles
    cycle_len = epochs // num_cycles
    init_params, init_gamma, init_opt_state, \
        init_disc_opt_state, init_net_state = model.init(rng_key, next(iter(train_loader))[0])
    # train_state = model.init(rng_key, next(iter(train_loader))[0])
    states = []
    disc_states = []

    params, gamma, opt_state, disc_opt_state, net_state = init_params, init_gamma, init_opt_state, \
        init_disc_opt_state, init_net_state
    step = 0
    key = rng_key

    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)

    for epoch in pgbar:
        for batch_x, batch_y in train_loader:
            _, key = jax.random.split(key, 2)
            rk = (step % M) / M
            params, gamma, opt_state, disc_opt_state, net_state = model.update(key, params, gamma, opt_state,
                                                                            disc_opt_state, net_state,
                                                                            batch_x, batch_y)
            # train_state = model.update(key, train_state, batch_x, batch_y)
            if rk > beta:
                model.add_noise = True
            else:
                model.add_noise = False

            step += 1
        if epoch != 0 and (epoch % cycle_len) + 1 > (cycle_len - m):
            states.append(params)
            disc_states.append(gamma)

    return model, states, disc_states, net_state


def eval_nn_model(key, model, x, y, states):

    if isinstance(states, list):
        y_preds = np.zeros((len(states), len(y)))
        for i, state in enumerate(states):
            preds, _ = model.apply(state, x, key, False)
            y_preds[i] = preds.squeeze()

        y_preds = np.mean(y_preds, axis=0)
        rmse = jnp.sqrt(jnp.mean((y - y_preds)**2))
    else:
        y_preds, _ = model.apply(states, x, key, False)
        rmse = jnp.sqrt(jnp.mean((y - y_preds.squeeze())**2))

    return rmse

def eval_mlp_model(model, x, y, state):
    y_preds = model.apply(state, x)
    return jnp.sqrt(jnp.mean((y - y_preds)**2))

def eval_resnet_bg_model(key, model, x, y, states, is_training=False, eval_r2=False):

    if isinstance(states, list):
        y_preds = np.zeros((len(states), len(y)))
        for i, state in enumerate(states):
            preds, _ = model.apply(state, x, key, is_training)
            y_preds[i] = preds.squeeze()

        y_preds = np.mean(y_preds, axis=0)
        rmse = jnp.sqrt(jnp.mean((y - y_preds)**2))
    else:
        y_preds, _ = model.apply(states, x, key, is_training)
        rmse = jnp.sqrt(jnp.mean((y - y_preds.squeeze())**2))


    if eval_r2:
        r2 = r2_score(y, y_preds)
        return rmse, r2
    return rmse


def train_rf_model(seed, X, y, train_idxs, val_idxs, classifier=False):

    # cv = KFold(n_splits=5)
    if train_idxs is not None and val_idxs is not None:
        cv = [(train_idxs, val_idxs) for _ in range(5)]
    else:
        cv = 5
    param_grid = {
        'max_depth': [80, 100, 120],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 500, 1000]
    }

    if classifier:
        rf_model = RandomForestClassifier(random_state=seed, max_samples=1.0)
        scoring = "roc_auc"
    else:
        rf_model = RandomForestRegressor(random_state=seed, max_samples=1.0)
        scoring = "neg_root_mean_squared_error"
    grid_cv = GridSearchCV(estimator = rf_model, param_grid = param_grid,
                           cv = cv, n_jobs = -1, verbose = 0, scoring=scoring).fit(X, y)

    if classifier:
        rf_model = RandomForestClassifier(random_state=seed, max_samples=1.0, **grid_cv.best_params_)
    else:
        rf_model = RandomForestRegressor(random_state=seed, max_samples=1.0, **grid_cv.best_params_)

    rf_model.fit(X, y)

    return rf_model

def run_horsehoe_bnn_model(seed, config_file, X_train, X_test, y_train, y_test,
                           epochs, batch_size, hyperparam_config, data_name,
                           classification=False, show_pgbar=False):
    with open(config_file, "r") as c:
        horseshoe_bnn_config = yaml.load(c, yaml.FullLoader)
        horseshoe_bnn_config['n_features'] = X_train.shape[-1]
        horseshoe_bnn_config['timestamp'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        horseshoe_bnn_config['dataset_name'] = data_name

    horseshoe_bnn_config["batch_size"] = batch_size
    horseshoe_bnn_config["n_hidden_units"] = hyperparam_config["n_hidden"]
    horseshoe_bnn_config["classification"] = classification

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda"
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    hyperparams = HorseshoeHyperparameters(**horseshoe_bnn_config)
    hbnn_model = HorseshoeBNN(device, hyperparams)
    optimizer = optim.Adam(hbnn_model.parameters(), lr=hyperparams.learning_rate)

    mean_y_train, std_y_train = 0.0, 1.0

    dataset_train = Dataset(X_train, y_train, data_name)
    dataset_test = Dataset(X_test, y_test, data_name)

    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)

    for epoch in pgbar:
        hbnn_model.train_model(dataset_train, epoch, optimizer, False)

    _, _, _, _, ensemble_output = hbnn_model.predict(dataset_test, mean_y_train=mean_y_train, std_y_train=std_y_train)

    if classification: #weird if statement to make work with the horse-shoe bnn code. In original code, only if it is
        # regression task the output will be converted to numpy array
        ensemble_output = ensemble_output.cpu().numpy()

    # torch.set_default_tensor_type(torch.FloatTensor)
    if classification:
        y_preds = jnp.mean(jax.nn.sigmoid(ensemble_output), axis=1)
        score = roc_auc_score(y_test, y_preds)
        acc = np.mean(y_test == (y_preds > 0.5))
        return hbnn_model, score, acc
    else:
        score = jnp.sqrt(jnp.mean((y_test - jnp.mean(ensemble_output, axis=1))**2))
        r2 = r2_score(y_test, jnp.mean(ensemble_output, axis=1))
        return hbnn_model, score, r2

def eval_sklearn_model(model, X, y, y_mean=0, y_std=1, classifier=False):
    if classifier:
        y_preds = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_preds)
        acc = accuracy_score(y, y_preds > 0.5)
        return auc, acc
    else:
        y_preds = model.predict(X)
        y_preds = y_preds * y_std + y_mean
        rmse = jnp.sqrt(jnp.mean((y - y_preds)**2))
        r2 = r2_score(y, y_preds)
        return rmse, r2

def zero_out_score(model, X, y, states, disc_states, m, lst, is_training):
    feat_idxs = lst[:m]
    mask = np.zeros(X.shape[1])
    mask[feat_idxs] = 1.0
    X_mask = X @ np.diag(mask)
    rmse, _ = score_bg_bnn_model(model, X_mask, y, states, disc_states, is_training)
    return rmse

def zero_out_score_v2(model, X, y, states, disc_states, m, lst, is_training):
    feat_idxs = lst[:m]
    mask = np.zeros(X.shape[1])
    mask[feat_idxs] = 1.0
    X_mask = X @ np.diag(mask)
    rmse, _ = score_bg_bnn_model_v2(model, X_mask, y, states, disc_states, is_training)
    return rmse