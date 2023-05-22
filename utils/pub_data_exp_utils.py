import pickle

from utils.hpo_util import *
from utils.train_utils import *
import optuna
from utils.drug_exp_utils import get_result_df

def get_benchmark_res_df(seeds, save_dir, version):
    res_dfs = []
    for seed in seeds:
        df = pd.read_csv(f"{save_dir}/results/bnn_rf_bg_s_{seed}_v{version}.csv")
        res_dfs.append(df)
    bnn_rf_df = pd.concat(res_dfs, ignore_index=True, axis=0)
    return bnn_rf_df


def get_summary_benchmark_results(seeds, data_names, exp_dir, version, models,
                                  hbnn_version=None):
    data_res_all = []
    num_models = len(models)
    for data in data_names:
        save_dir = f"{exp_dir}/{data}"
        data_res_df = get_result_df(seeds, save_dir, version, hbnn_version=hbnn_version)
        data_name_col = [data for _ in range(num_models * len(seeds))]
        data_res_df.insert(0, column="dataset", value=data_name_col)
        data_res_all.append(data_res_df)

    data_res_all = pd.concat(data_res_all, axis=0)
    perf_df = data_res_all.groupby(["dataset", "model"]).agg(
        {"score": ["mean", "std"]})

    perf = perf_df["score"]
    perf["summary"] = perf.apply(lambda row: f"{round(row['mean'], 3)}  ± {round(row['std'], 3)}",
                                 axis=1).values
    perf_table = pd.pivot_table(perf[["summary"]], index=["dataset"], columns=["model"], aggfunc="first")

    return perf_table

def load_bikeshare_data(data_dir):
    bikeshare_data = pd.read_csv(f"{data_dir}/bikeshare/hour.csv", index_col="instant")
    train_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                  'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    label = 'cnt'

    X = bikeshare_data[train_cols]
    y = bikeshare_data[label]

    return X, y, False, None, 256


def load_wine_data(data_dir):
    wine_data = pd.read_table(f"{data_dir}/wine/winequality-white.csv", sep=";")

    X = wine_data.drop(["quality"], axis=1)
    y = wine_data["quality"]

    return X, y, False, None, 128

def load_credit_data(data_dir):
    df = pd.read_csv(f"{data_dir}/credit/creditcard.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y, True, None, 2048


def load_support2_data(data_dir):
    df = pd.read_csv(f"{data_dir}/support2/support2.csv")

    cat_cols = ['sex', 'dzclass', 'race', 'ca', 'income']
    target_variables = ['hospdead']
    remove_features = ['death', 'slos', 'd.time', 'dzgroup', 'charges', 'totcst',
                       'totmcst', 'aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m',
                       'dnr', 'dnrday', 'avtisst', 'sfdm2']

    df = df.drop(remove_features, axis=1)

    rest_colmns = [c for c in df.columns if c not in (cat_cols + target_variables)]
    # Impute the missing values for 0.
    df[rest_colmns] = df[rest_colmns].fillna(0.)

    income_df = df.loc[:, ['income']]
    income_df[income_df['income'].isna()] = 'NaN'
    income_df[income_df['income'] == 'under $11k'] = ' <$11k'

    race_df = df.loc[:, ['race']]
    race_df[race_df['race'].isna()] = 'NaN'

    X = df.drop(target_variables, axis=1)
    y = df[target_variables[0]]

    return X, y, True, cat_cols, 256


def load_churn_data(data_dir):
    df = pd.read_csv(f"{data_dir}/churn/churn.csv")

    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(drop_cols, axis=1)
    cols = ["Age", "Balance", "CreditScore", "EstimatedSalary",
            "Gender", "Geography", "HasCrCard", "IsActiveMember",
            "NumOfProducts", "Tenure"]

    cat_cols = ["IsActiveMember", "Gender", "Geography", "HasCrCard"]

    X, y = df[cols], df["Exited"]

    return X, y, True, cat_cols, 256

def load_data(data_name, data_dir):
    if data_name == "wine":
        return load_wine_data(data_dir)
    if data_name == "bikeshare":
        return load_bikeshare_data(data_dir)
    if data_name == "credit":
        return load_credit_data(data_dir)
    if data_name == "support2":
        return load_support2_data(data_dir)
    if data_name == "churn":
        return load_churn_data(data_dir)
    else:
        raise ValueError(f"Data {data_name} not supported")


def run_bnn_model(seed, save_dir, version, X_train_outer, X_test, y_train_outer, y_test,
                  epochs, batch_size, J, hyperparam_config, n_trials, classification=False,
                  bg=True, show_pgbar=False):
    torch.manual_seed(seed)
    hidden_size = [hyperparam_config["n_hidden"]] * hyperparam_config["num_layers"]
    act_fn = hyperparam_config["act_fn"]
    prior_dist = hyperparam_config["prior_dist"]


    X_train, X_val, y_train, y_val = train_test_split(X_train_outer, y_train_outer, test_size=0.2, random_state=seed,
                                                      shuffle=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: objective_benchmark(trial, seed, X_train, X_val, y_train, y_val, J, epochs,
                                                     batch_size, 0.25, hidden_size, act_fn, bg=bg,
                                                     classifier=classification,
                                                     prior_dist=prior_dist),
                   n_trials=n_trials)

    bnn_config = study.best_params

    if bg:
        config_path = f"{save_dir}/configs/bg_bnn_config_s_{seed}_v{version}.csv"
        study_path = f"{save_dir}/optuna/study_bg_bnn_s_{seed}_v{version}.csv"
    else:
        config_path = f"{save_dir}/configs/bnn_config_s_{seed}_v{version}.csv"
        study_path = f"{save_dir}/optuna/study_bnn_s_{seed}_v{version}.csv"

    with open(config_path, "wb") as fp:
        pickle.dump(bnn_config, fp)
        fp.flush()

    with open(study_path, "wb") as fp:
        pickle.dump(study, fp)
        fp.flush()

    temp = bnn_config["temp"]
    mu = bnn_config["mu"]
    if bg:
        eta = bnn_config["eta"]
    else:
        eta = 1.0

    sigma_1, sigma_2 = 0.01, 1.0
    num_cycles = 50
    lr_0, disc_lr_0 = 0.001, 0.5

    data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size, shuffle=True,
                              drop_last=True, device="cpu")

    bnn_model, states = train_bnn_model(seed, data_loader, epochs, num_cycles,
                                                                0.25, 1, lr_0, disc_lr_0, hidden_size, temp,
                                                                sigma_1, sigma_2, eta, mu,
                                                                J, act_fn, show_pgbar=show_pgbar, prior_dist=prior_dist,
                                                                classifier=classification)

    if len(y_test) > 10000:
        rmse, r2 = score_bg_bnn_model_batched(bnn_model, X_test, y_test,
                                              states, 2000,
                                              classifier=classification)
    else:
        rmse, r2 = score_bg_bnn_model(bnn_model, X_test, y_test,
                                      states, classifier=classification)

    bnn_states, bnn_disc_states = [],[]
    for state in states:
        bnn_states.append(state.params)
        bnn_disc_states.append(state.gamma)

    return bnn_states, bnn_disc_states, rmse, r2


def run_rf_model(seed, save_dir, version, X_train_outer, X_test, y_train_outer, y_test, n_trials, classification=False):
    X_train, X_val, y_train, y_val = train_test_split(X_train_outer, y_train_outer, test_size=0.2, random_state=seed,
                                                      shuffle=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: objective_rf(trial, seed, X_train, X_val, y_train, y_val), n_trials=n_trials)

    rf_config = study.best_params

    rf_config_path = f"{save_dir}/configs/rf_config_s_{seed}_v{version}.csv"
    study_path = f"{save_dir}/optuna/study_rf_s_{seed}_v{version}.csv"

    with open(rf_config_path, "wb") as fp:
        pickle.dump(rf_config, fp)
        fp.flush()

    with open(study_path, "wb") as fp:
        pickle.dump(study, fp)
        fp.flush()

    if classification:
        rf_model = RandomForestClassifier(**rf_config, random_state=seed)
    else:
        rf_model = RandomForestRegressor(**rf_config, random_state=seed)

    rf_model.fit(X_train_outer, y_train_outer)
    y_preds = rf_model.predict(X_test)
    if classification:
        rmse = roc_auc_score(y_test, y_preds)
    else:
        rmse = jnp.sqrt(jnp.mean((y_test - y_preds) ** 2))

    return rf_model, rmse


