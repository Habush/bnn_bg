import datetime
import yaml
from horseshoe_bnn.data_handling.dataset import Dataset
from horseshoe_bnn.models import HorseshoeBNN
from horseshoe_bnn.parameters import HorseshoeHyperparameters
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from torch import optim
from tqdm import tqdm

from core.bnn_model import *
from core.optim import *
from utils.data_utils import *


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

def init_bnn_model(seed, train_loader, epochs, lr_0, disc_lr_0, num_cycles, temp, sigma_1, sigma_2,
                         hidden_sizes, J, eta, mu, act_fn, prior_dist,
                         init_fn=None, classifier=False):

    torch.manual_seed(seed)
    num_batches = len(train_loader)
    data_size = train_loader.dataset.data.shape[0]
    total_steps = num_batches*epochs
    step_size_fn = make_cyclical_lr_fn(lr_0, total_steps, num_cycles)
    disc_step_size_fn = make_cyclical_lr_fn(disc_lr_0, total_steps, num_cycles)

    sgd_optim = sgd_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())
    sgld_optim = sgld_gradient_update(step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())


    disc_sgld_optim = disc_sgld_gradient_update(disc_step_size_fn, momentum_decay=0.9, preconditioner=get_rmsprop_preconditioner())

    if init_fn is None:
        init_fn = hk.initializers.VarianceScaling()

    model = BayesNN(sgd_optim, sgld_optim, disc_sgld_optim,
          temp, sigma_1, sigma_2, data_size, hidden_sizes,
          J, eta, mu, act_fn, init_fn, prior_dist, classification=classifier)


    return model


def train_bnn_model(seed, train_loader, epochs, num_cycles, beta, m, lr_0, disc_lr_0,
                       hidden_sizes, temp, sigma_1, sigma_2, eta, mu, J, act_fn_name,
                       show_pgbar=True, prior_dist="laplace", classifier=False):

    rng_key = jax.random.PRNGKey(seed)
    act_fn = get_act_fn(act_fn_name)
    init_fn = hk.initializers.VarianceScaling()

    model = init_bnn_model(seed, train_loader, epochs, lr_0, disc_lr_0, num_cycles,
                              temp, sigma_1, sigma_2, hidden_sizes, J, eta, mu,
                              act_fn, prior_dist, init_fn, classifier)

    num_batches = len(train_loader)
    M = (epochs*num_batches) // num_cycles
    train_state = model.init(rng_key, next(iter(train_loader))[0])
    states = []
    step = 0
    key = rng_key

    if show_pgbar:
        pgbar = tqdm(range(epochs))
    else:
        pgbar = range(epochs)

    model.add_noise = True
    for _ in pgbar:
        for batch_x, batch_y in train_loader:
            _, key = jax.random.split(key, 2)
            rk = (step % M) / M
            if rk > beta:
                model.add_noise = True
            else:
                model.add_noise = False
            train_state = model.update(key, train_state, batch_x, batch_y)
            step += 1

            if (step % M) + 1 > (M - m):
                states.append(train_state)


    return model, states

def apply_bnn_model(model, X, y, states, classifier=False):

    y_preds = np.zeros((len(states), len(y)))
    for i, state in enumerate(states):
        param, gamma = state.params, state.gamma
        preds = model.apply(param, gamma, X).ravel()
        if classifier:
            y_preds[i] = jax.nn.sigmoid(preds)
        else:
            y_preds[i] = preds

    return y_preds

def score_bnn_model(model, X, y, states, classifier=False,
                       y_mean=0.0, y_std=1.0):
    y_preds = apply_bnn_model(model, X, y, states, classifier)
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


def score_bnn_model_batched(model, X, y, states, batch_size,
                               classifier=False, y_mean=0.0, y_std=1.0):

    data_loader = NumpyLoader(NumpyData(X, y), batch_size=batch_size, shuffle=False)

    y_preds = []
    for batch_x, batch_y in data_loader:
        batch_preds = apply_bnn_model(model, batch_x, batch_y, states, classifier)
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

def zero_out_score(model, X, y, params, gammas, m, lst):
    params, gammas = tree_utils.tree_unstack(params), tree_utils.tree_unstack(gammas)
    states = []
    for param, gamma in zip(params, gammas):
        states.append(TrainingState(param, gamma, None, None))
    feat_idxs = lst[:m]
    mask = np.zeros(X.shape[1])
    mask[feat_idxs] = 1.0
    X_mask = X @ np.diag(mask)
    rmse, _ = score_bnn_model(model, X_mask, y, states)
    return rmse
