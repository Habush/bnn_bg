from data_utils import NumpyLoader, NumpyData
from train_utils import *
import optuna
import torch
import pandas as pd
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor, XGBClassifier

# def objective_bg_bnn(trial, seed, x_train, x_val, y_train, y_val,
#                      J, epochs, beta, act_fn, hidden_size, prior_dist,
#                     dropout_version=2, classifier=False,
#                      bg=True,):
#
#     # timeout used 180 seconds
#
#     lr_0, disc_lr_0 = 1e-3, 0.5
#     temp = [0.01, 1.0]
#     batch_size = 16 if x_train.shape[0] < 200 else 32
#     m = 1
#
#     sigma_1, sigma_2 = 0.01, 1.0
#     num_cycles = 50
#     eta, mu = 1.0, 1.0
#
#     # num_hidden = trial.suggest_categorical("num_hidden", [64, 128, 256, 512])
#     num_layers = trial.suggest_int("num_layers", 2, 6)
#     hidden_sizes = [hidden_size]*num_layers
#     torch.manual_seed(seed)
#     data_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size, shuffle=True,
#                               drop_last=True)
#
#     model, states, disc_states = train_bg_bnn_model(seed, data_loader, epochs, num_cycles, beta, m, lr_0,
#                                                            disc_lr_0, hidden_sizes, temp, sigma_1, sigma_2, eta, mu,
#                                                            J, act_fn, show_pgbar=False, prior_dist=prior_dist,
#                                                            dropout_version=dropout_version, classifier=classifier)
#
#     if classifier:
#         score, _ = score_bg_bnn_model(jax.random.PRNGKey(seed), model, x_val, y_val,
#                                       states, disc_states, False,
#                                       classifier=classifier)
#         return 1 - score
#     else:
#         rmse, _ = score_bg_bnn_model(jax.random.PRNGKey(seed), model, x_val, y_val,
#                                      states, disc_states, False,
#                                      classifier=classifier)
#
#         return rmse


def objective_bg_bnn(trial, seed, x_train, x_val, y_train, y_val,
                     J, epochs, beta, act_fn, hidden_size, prior_dist,
                    dropout_version=2, classifier=False,
                     bg=True,):

    lr_0, disc_lr_0 = 1e-3, 0.5
    temp = trial.suggest_categorical("temp", [0.1, 1.0])
    batch_size = 16 if x_train.shape[0] < 200 else 32
    m = 1

    sigma_1, sigma_2 = 0.01, 1.0
    num_cycles = 50
    # if bg:
    #     eta = trial.suggest_float("eta", 1.0, 1e3, log=True)
    # else:
    #     eta = 1.0
    # mu = trial.suggest_float("mu", 1.0, 1e3, log=True)
    eta, mu = 1.0, 1.0

    # num_hidden = trial.suggest_categorical("num_hidden", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    hidden_sizes = [hidden_size]*num_layers
    torch.manual_seed(seed)
    data_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size, shuffle=True,
                              drop_last=True)

    model, states, disc_states = train_bg_bnn_model(seed, data_loader, epochs, num_cycles, beta, m, lr_0,
                                                           disc_lr_0, hidden_sizes, temp, sigma_1, sigma_2, eta, mu,
                                                           J, act_fn, show_pgbar=False, prior_dist=prior_dist,
                                                           dropout_version=dropout_version, classifier=classifier)

    if classifier:
        score, _ = score_bg_bnn_model(jax.random.PRNGKey(seed), model, x_val, y_val,
                                      states, disc_states, False,
                                      classifier=classifier)
        return 1 - score
    else:
        rmse, _ = score_bg_bnn_model(jax.random.PRNGKey(seed), model, x_val, y_val,
                                     states, disc_states, False,
                                     classifier=classifier)

        return rmse


def objective_bg_bnn_v2(trial, seed, x_train, x_val, y_train, y_val, J, epochs, beta, hidden_sizes,
                        act_fn, bg=True, classifier=False, prior_dist="laplace",
                        dropout_version=2, laplacian=False, gpu_id="/gpu:0"):
    lr_0, disc_lr_0 = 1e-3, 0.5
    # lr_0 =  trial.suggest_categorical("lr_0", [1e-3, 5e-3, 1e-2])
    # disc_lr_0 = trial.suggest_categorical("disc_lr_0", [0.01, 0.1, 0.5])
    num_cycles = trial.suggest_categorical("num_cycles", [2, 4, 6, 8, 10])
    # num_cycles = 10
    temp = trial.suggest_categorical("temp", [1e-3, 1e-2, 1e-1, 1.])
    # temp = 1.0
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    if x_train.shape[0] < 200:
        batch_size = 16
    else:
        batch_size = 32
    m = 1
    # sigma_1 = trial.suggest_categorical("sigma_1", [1e-3, 1e-2, 1e-1])
    sigma_1 = 0.01
    sigma_2 = 1.0
    if bg:
        if laplacian:
            eta = trial.suggest_float("eta", -1e2, -1.0)
        else:
            eta = trial.suggest_float("eta", 1.0, 1e2)

    else:
        eta = 1.0

    mu = trial.suggest_float("mu", 1.0, 1e2)

    with tf.device(gpu_id):
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        data_loader = tfds.as_numpy(train_data.shuffle(1000, seed=seed).batch(batch_size,
                                                                              drop_remainder=True).prefetch(1))

        bg_bnn_model, states, disc_states = train_bg_bnn_model(seed, data_loader, epochs, num_cycles, beta, m, lr_0,
                                                               disc_lr_0, hidden_sizes, temp, sigma_1, sigma_2, eta, mu,
                                                               J, act_fn,
                                                               show_pgbar=False, prior_dist=prior_dist,
                                                               dropout_version=dropout_version, classifier=classifier)

        if classifier:
            score = eval_bg_bnn_model(bg_bnn_model, x_val, y_val, states, disc_states, True, True)
            return 1 - score
        else:
            rmse = eval_bg_bnn_model(bg_bnn_model, x_val, y_val, states, disc_states, True)

            return rmse


def objective_bg_bnn_v3(trial, seed, x_train, x_val, y_train, y_val, J, epochs, beta, bg=True,
                        classifier=False,
                        device="cpu", dropout_version=2, laplacian=False):
    lr_0 = trial.suggest_categorical("lr_0", [1e-3, 5e-3, 1e-2])
    disc_lr_0 = trial.suggest_categorical("disc_lr_0", [0.01, 0.1, 0.5])
    # lr_0 = 1e-3
    # disc_lr_0 = 0.5
    num_cycles = trial.suggest_categorical("num_cycles", [4, 6, 8, 10])
    temp = trial.suggest_categorical("temp", [1e-3, 1e-2, 1e-1, 1.])
    if x_train.shape[0] < 200:
        batch_size = 16
    else:
        batch_size = 32
    m = 1
    sigma_1 = trial.suggest_categorical("sigma_1", [1e-2, 5e-2, 1e-1])
    sigma_2 = trial.suggest_categorical("sigma_2", [1.0, 5.0, 10.0])
    if bg:
        if laplacian:
            eta = trial.suggest_float("eta", -1e2, -1.0)
        else:
            eta = trial.suggest_float("eta", 1, 1e2)

    else:
        eta = 1.0

    mu = trial.suggest_float("mu", 1.0, 1e2)

    num_layers = trial.suggest_int("num_layers", 1, 5)
    num_hidden = trial.suggest_categorical("num_hidden", [256, 512, 1024])

    hidden_sizes = []
    for i in range(num_layers):
        hidden_sizes.append(num_hidden)

    prior_dist = trial.suggest_categorical("prior_dist", ["normal", "laplace", "student_t", "cauchy"])
    act_fn = trial.suggest_categorical("act_fn", ["relu", "swish"])

    torch.manual_seed(seed)
    train_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True,
                               device=device)
    bg_bnn_model, states, disc_states = train_bg_bnn_model(seed, train_loader, epochs, num_cycles, beta, m, lr_0,
                                                           disc_lr_0,
                                                           hidden_sizes, temp, sigma_1, sigma_2, eta, mu, J, act_fn,
                                                           show_pgbar=False, prior_dist=prior_dist,
                                                           dropout_version=dropout_version, classifier=classifier)

    if classifier:
        score = eval_bg_bnn_model(bg_bnn_model, x_val, y_val, states, disc_states, True, True)
        return 1 - score
    else:
        rmse = eval_bg_bnn_model(bg_bnn_model, x_val, y_val, states, disc_states, True)

        return rmse

def objective_benchmark(trial, seed, x_train, x_val, y_train, y_val, J, epochs, batch_size, beta, hidden_sizes,
                        act_fn, bg=True, classifier=False, prior_dist="laplace",
                        dropout_version=2, laplacian=False, gpu_id="/gpu:0", y_mean=0., y_std=1.):

    lr_0, disc_lr_0 = 1e-3, 0.5
    num_cycles = 50
    temp = trial.suggest_categorical("temp", [1e-3, 1e-2, 1e-1, 1.])
    m = 1
    sigma_1 = 0.01
    sigma_2 = 1.0
    if bg:
        if laplacian:
            eta = trial.suggest_float("eta", -1e2, -1.0)
        else:
            eta = trial.suggest_float("eta", 1.0, 1e3, log=True)

    else:
        eta = 1.0

    mu = trial.suggest_float("mu", 1.0, 1e3, log=True)


    data_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)

    model, states, disc_states = train_bg_bnn_model(seed, data_loader, epochs, num_cycles, beta, m, lr_0,
                                                               disc_lr_0, hidden_sizes, temp, sigma_1, sigma_2, eta, mu,
                                                               J, act_fn, show_pgbar=False, prior_dist=prior_dist,
                                                               dropout_version=dropout_version, classifier=classifier)

    if len(y_val) > 10000:
        rmse, _ = score_bg_bnn_model_batched(jax.random.PRNGKey(seed), model, x_val, y_val,
                                              states, disc_states, 2000, False,
                                              y_mean=y_mean, y_std=y_std,
                                              classifier=classifier)
    else:
        rmse, _ = score_bg_bnn_model(jax.random.PRNGKey(seed), model, x_val, y_val,
                                      states, disc_states, False,
                                      y_mean=y_mean, y_std=y_std, classifier=classifier)
    return rmse
def objective_horseshoe_bnn(trial, seed, config_file, x_train, x_val, y_train, y_val, epochs, num_hidden,
                            data_name,classifier=False):

    lr_0 = 1e-3
    if x_train.shape[0] < 200:
        batch_size = 16
    else:
        batch_size = 32

    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4, 5, 6])
    hidden_sizes = [num_hidden] * num_layers

    horseshoe_config = {"lr": lr_0, "batch_size": batch_size, "n_hidden": hidden_sizes}

    _, score, _ = run_horsehoe_bnn_model(seed, config_file, x_train, x_val, y_train, y_val, epochs,
                                      batch_size, horseshoe_config,
                                      data_name, classifier)

    if classifier:
        return 1 - score

    return score



def objective_resnet_bg(trial, seed, x_train, x_val, y_train, y_val, J, epochs, beta,
                        act_fn, bg=True, prior_dist="laplace",
                        dropout_version=2, laplacian=False):
    lr_0, disc_lr_0 = 1e-3, 0.5

    # weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    # dropout_rate = trial.suggest_categorical("dropout_rate", [0., 0.1, 0.2, 0.3])
    dropout_rate = 0.0
    num_cycles = 10
    block_type = trial.suggest_categorical("block_type", ["ResNet", "PreActResNet"])
    if x_train.shape[0] < 200:
        batch_size = 16
    else:
        batch_size = 32

    num_layers = trial.suggest_int("num_layers", 1, 5)
    num_blocks = trial.suggest_categorical("num_blocks", [1, 2])
    layer_size = trial.suggest_categorical("block_size", [32, 64, 128, 256])

    blocks = [num_blocks] * num_layers
    hidden_sizes = [layer_size] * num_layers

    m = 1

    temp = 1.0
    sigma_1 = 0.1
    sigma_2 = 1.0

    if bg:
        if laplacian:
            eta = trial.suggest_float("eta", -1e2, -1.0)
        else:
            eta = trial.suggest_float("eta", 1.0, 1e2)

    else:
        eta = 1.0

    mu = trial.suggest_float("mu", 1.0, 1e2)
    rng = jax.random.PRNGKey(seed)
    torch.manual_seed(seed)
    train_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size,
                               shuffle=True, drop_last=True, device="cpu")

    bnn_model, bnn_states, bnn_disc_states, net_state = train_resnet_bg_model(rng, train_loader, epochs,
                                                                              num_cycles,
                                                                              beta, m,
                                                                              lr_0, disc_lr_0, block_type,
                                                                              blocks, hidden_sizes, temp, sigma_1,
                                                                              sigma_2,
                                                                              eta, mu, J, act_fn, show_pgbar=False,
                                                                              prior_dist=prior_dist,
                                                                              dropout_version=dropout_version,
                                                                              dropout_rate=dropout_rate,
                                                                              weight_decay=0.0)

    rmse, _ = score_bg_bnn_model(rng, bnn_model, x_val, y_val,
                                 bnn_states, bnn_disc_states, is_training=False,
                                 has_state=True,
                                 net_state=net_state)

    return rmse


def objective_rf(trial, seed, x_train, x_val, y_train, y_val, classifier=False):
    params = {
        "n_estimators" : trial.suggest_int("n_estimators", 100, 1000),
        "max_depth" : trial.suggest_int("max_depth", 10, 100),
        "max_features" : trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
        "min_samples_split" : trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 5),
        "bootstrap" : trial.suggest_categorical("bootstrap", [True, False])
    }

    if classifier:
        model = RandomForestClassifier(**params, random_state=seed)
        model.fit(x_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
        return 1 - auc
    else:
        model = RandomForestRegressor(**params, random_state=seed)
        model.fit(x_train, y_train)
        rmse = np.sqrt(np.mean((y_val - model.predict(x_val))**2))
        return rmse


def objective_esnet(trial, seed, x_train, x_val, y_train, y_val):
    alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.1)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed)

    model.fit(x_train, y_train)
    rmse = np.sqrt(np.mean((y_val - model.predict(x_val)) ** 2))
    return rmse

def objective_xgb(trial, seed, x_train, x_val, y_train, y_val, classifier=False):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }

    if classifier:
        model = XGBClassifier(**params, random_state=seed)
        model.fit(x_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
        return 1 - auc
    else:
        model = XGBRegressor(**params, random_state=seed)
        model.fit(x_train, y_train)
        rmse = np.sqrt(np.mean((y_val - model.predict(x_val))**2))
        return rmse

def trials_to_df(study, calc_num_params=False, num_params=None,
                 num_features=None, bias=True):
    """
    Convert optuna study to pandas dataframe
    :param study: The optuna study
    :param calc_num_params: Whether to calculate the number of parameters in the model (should be set to True for
    when the model architecture is not fixed)
    :return:
    """
    assert calc_num_params or num_params is not None, "If calc_num_params is False, num_params must be provided"
    trials = study.trials
    params_dicts = {"trial": [i for i in range(len(trials))], "score": [], "num_params": []}
    param_keys = study.trials[0].params.keys()
    for key in param_keys:
        params_dicts[key] = []

    for trial in trials:
        for key in param_keys:
            params_dicts[key].append(trial.params[key])

        params_dicts["score"].append(trial.value)

        if calc_num_params:
            assert num_features is not None, "num_features must be provided if calc_num_params is True"
            num_params = 0
            hidden_sizes = []
            for i in range(trial.params["num_layers"]):
                hidden_sizes.append(trial.params["num_hidden"])

            if bias:
                num_params += (num_features * hidden_sizes[0]) + hidden_sizes[0]
            for i, num_hidden in enumerate(hidden_sizes[1:]):
                if bias:
                    num_params += (hidden_sizes[i] * num_hidden) + num_hidden
                else:
                    num_params += hidden_sizes[i] * num_hidden

            if bias:
                num_params += 2 * hidden_sizes[-1]
            else:
                num_params += hidden_sizes[-1]

        params_dicts["num_params"].append(num_params)

    df = pd.DataFrame(params_dicts)

    return df
