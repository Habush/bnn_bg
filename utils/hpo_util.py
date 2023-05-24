from utils.train_utils import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def objective_bg_bnn(trial, seed, x_train, x_val, y_train, y_val,
                     J, epochs, beta, act_fn, hidden_size, prior_dist, classifier=False):

    lr_0, disc_lr_0 = 1e-3, 0.5
    temp = trial.suggest_categorical("temp", [0.1, 1.0])
    batch_size = 16 if x_train.shape[0] < 200 else 32
    m = 1

    sigma_1, sigma_2 = 0.01, 1.0
    num_cycles = 50
    eta, mu = 1.0, 1.0
    num_layers = trial.suggest_int("num_layers", 2, 6)
    hidden_sizes = [hidden_size]*num_layers
    torch.manual_seed(seed)
    data_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size, shuffle=True,
                              drop_last=True)

    model, states = train_bnn_model(seed, data_loader, epochs, num_cycles, beta, m, lr_0,
                                                           disc_lr_0, hidden_sizes, temp, sigma_1, sigma_2, eta, mu,
                                                 J, act_fn, show_pgbar=False, prior_dist=prior_dist,classifier=classifier)

    if classifier:
        score, _ = score_bnn_model(model, x_val, y_val,
                                      states,
                                      classifier=classifier)
        return 1 - score
    else:
        rmse, _ = score_bnn_model(model, x_val, y_val,
                                     states,
                                     classifier=classifier)

        return rmse


def objective_benchmark(trial, seed, x_train, x_val, y_train, y_val, J, epochs, batch_size, beta, hidden_sizes,
                        act_fn, bg=True, classifier=False, prior_dist="laplace",
                        y_mean=0., y_std=1.):

    lr_0, disc_lr_0 = 1e-3, 0.5
    num_cycles = 50
    temp = trial.suggest_categorical("temp", [1e-3, 1e-2, 1e-1, 1.])
    m = 1
    sigma_1 = 0.01
    sigma_2 = 1.0
    if bg:
        eta = trial.suggest_float("eta", 1.0, 1e3, log=True)

    else:
        eta = 1.0

    mu = trial.suggest_float("mu", 1.0, 1e3, log=True)


    data_loader = NumpyLoader(NumpyData(x_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)

    model, states = train_bnn_model(seed, data_loader, epochs, num_cycles, beta, m, lr_0,
                                                               disc_lr_0, hidden_sizes, temp, sigma_1, sigma_2, eta, mu,
                                                               J, act_fn, show_pgbar=False, prior_dist=prior_dist,
                                                               classifier=classifier)

    if len(y_val) > 10000:
        rmse, _ = score_bnn_model_batched(model, x_val, y_val,
                                              states, 2000,
                                              y_mean=y_mean, y_std=y_std,
                                              classifier=classifier)
    else:
        rmse, _ = score_bnn_model(model, x_val, y_val,
                                      states, y_mean=y_mean, y_std=y_std, classifier=classifier)
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

