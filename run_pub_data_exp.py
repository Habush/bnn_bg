#!/usr/local/bin/python3
import functools
import os
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from category_encoders import LeaveOneOutEncoder
from hpo_util import *
from multiprocessing import Pool
from train_utils import *
import argparse
from exp_utils import get_result_df

data_dir = "/home/abdu/bio_ai/moses-incons-pen-xp/data"
uci_data_dir = f"{data_dir}/uci"


def load_bikeshare_data():
    bikeshare_data = pd.read_csv(f"{uci_data_dir}/bikeshare/hour.csv", index_col="instant")
    train_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                  'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    label = 'cnt'

    X = bikeshare_data[train_cols]
    y = bikeshare_data[label]

    return X, y, False, None, 256


def load_wine_data():
    wine_data = pd.read_table(f"{uci_data_dir}/wine/winequality-white.csv", sep=";")

    X = wine_data.drop(["quality"], axis=1)
    y = wine_data["quality"]

    return X, y, False, None, 128


def load_year_data():
    year_msd_data = pd.read_csv(f"{uci_data_dir}/year/YearPredictionMSD.csv")

    X, y = year_msd_data.iloc[:, 1:], year_msd_data.iloc[:, 0]
    return X, y, False, None, 4096  # train_size// 5


def load_credit_data():
    df = pd.read_csv(f"{uci_data_dir}/credit/creditcard.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y, True, None, 2048  # train_size // 5


def load_support2_data():
    df = pd.read_csv(f"{uci_data_dir}/support2/support2.csv")

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


def load_churn_data():
    df = pd.read_csv(f"{uci_data_dir}/churn/churn.csv")

    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(drop_cols, axis=1)
    cols = ["Age", "Balance", "CreditScore", "EstimatedSalary",
            "Gender", "Geography", "HasCrCard", "IsActiveMember",
            "NumOfProducts", "Tenure"]

    cat_cols = ["IsActiveMember", "Gender", "Geography", "HasCrCard"]

    X, y = df[cols], df["Exited"]

    return X, y, True, cat_cols, 256


def load_housing_data():
    df = pd.read_csv(f"{uci_data_dir}/housing/housing.csv")

    X, y = df.drop(["median_house_value"], axis=1), df["median_house_value"]
    X.fillna(0.0, inplace=True)
    X = pd.get_dummies(X, columns=["ocean_proximity"])

    return X, y, False, None, 256


def load_data(data_name):
    if data_name == "wine":
        return load_wine_data()
    if data_name == "bikeshare":
        return load_bikeshare_data()
    if data_name == "year":
        return load_year_data()
    if data_name == "credit":
        return load_credit_data()
    if data_name == "support2":
        return load_support2_data()
    if data_name == "churn":
        return load_churn_data()
    if data_name == "housing":
        return load_housing_data()
    else:
        raise ValueError(f"Data {data_name} not supported")


def run_bnn_model(seed, save_dir, version, X_train_outer, X_test, y_train_outer, y_test,
                  epochs, batch_size, J, hyperparam_config, n_trials, classification=False,
                  dropout_version=2, bg=True, show_pgbar=False):
    torch.manual_seed(seed)
    hidden_size = [hyperparam_config["n_hidden"]] * hyperparam_config["num_layers"]
    act_fn = hyperparam_config["act_fn"]
    prior_dist = hyperparam_config["prior_dist"]

    if not classification:
        # mean_y_train_outer, std_y_train_outer = np.mean(y_train_outer), np.std(y_train_outer)
        # y_train_outer = (y_train_outer - mean_y_train_outer) / std_y_train_outer
        # y_test = (y_test - mean_y_train_outer) / std_y_train_outer
        mean_y_train_outer, std_y_train_outer = 0.0, 1.0
    else:
        mean_y_train_outer, std_y_train_outer = 0.0, 1.0

    X_train, X_val, y_train, y_val = train_test_split(X_train_outer, y_train_outer, test_size=0.2, random_state=seed,
                                                      shuffle=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: objective_benchmark(trial, seed, X_train, X_val, y_train, y_val, J, epochs,
                                                     batch_size, 0.25, hidden_size, act_fn, bg=bg,
                                                     classifier=classification,
                                                     prior_dist=prior_dist, dropout_version=dropout_version),
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

    bnn_model, bnn_states, bnn_disc_states = train_bg_bnn_model(seed, data_loader, epochs, num_cycles,
                                                                0.25, 1, lr_0, disc_lr_0, hidden_size, temp,
                                                                sigma_1, sigma_2, eta, mu,
                                                                J, act_fn, show_pgbar=show_pgbar, prior_dist=prior_dist,
                                                                classifier=classification,
                                                                dropout_version=dropout_version)

    # params_bnn = tree_utils.tree_stack(bnn_states)
    # gammas_bnn = tree_utils.tree_stack(bnn_disc_states)
    if len(y_test) > 10000:
        rmse, r2 = score_bg_bnn_model_batched(jax.random.PRNGKey(seed), bnn_model, X_test, y_test,
                                              bnn_states, bnn_disc_states, 2000, False,
                                              y_mean=mean_y_train_outer, y_std=std_y_train_outer,
                                              classifier=classification)
    else:
        rmse, r2 = score_bg_bnn_model(jax.random.PRNGKey(seed), bnn_model, X_test, y_test,
                                      bnn_states, bnn_disc_states, False,
                                      y_mean=mean_y_train_outer, y_std=std_y_train_outer, classifier=classification)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments on GDSC drug sensitivity data")
    parser.add_argument("--exp_dir", type=str, default="/home/abdu/bio_ai/moses-incons-pen-xp/data/exp_data_5/uci",
                        help="Path to the directory where the experiment data will be saved")
    parser.add_argument("--seeds", type=str, help="Path to the file containing the seeds")
    parser.add_argument("--data_names", type=str, default="bikeshare,wine,support2,churn",
                        help="Comma separated list of dataset names")
    parser.add_argument("--version", type=str, default="1a",
                        help="Version of the current experiment - useful for tracking experiments")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--num_hidden", type=int, default=64, help="Number of hidden units in each layer")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--prior_dist", default="laplace", const="laplace", nargs="?",
                        choices=["laplace", "normal", "student_t"]
                        , help="Prior distribution for the weights. Options: laplace, normal, student_t")
    parser.add_argument("--act_fn", default="swish", const="swish", nargs="?",
                        choices=["swish", "relu", "tanh", "sigmoid"],
                        help="Activation function for the hidden layers. Options: swish, relu, tanh, sigmoid")
    parser.add_argument("--dropout_version", default='2', const='2', nargs='?', choices=['1', '2'],
                        help="1: Feature dropout and spike-slab prior on the first layer, 2: spike-slab prior on "
                             "the first layer")
    # parser.add_argument("--num_folds", type=int, default=20, help="Number of folds for cross validation")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout for hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for hyperparameter optimization")
    parser.add_argument("--use_laplacian", default='0', const='0', nargs='?', choices=['0', '1'],
                        help="Whether to use laplacian regularization or not")
    parser.add_argument("--num_folds", type=int, default=20, help="Number of folds for cross validation")
    parser.add_argument("--scale_output", default='1', const='1', nargs='?', choices=['0', '1'],
                        help="Whether to scale the output or not")
    parser.add_argument("--saved_config", default='0', const='0', nargs='?', choices=['0', '1'],
                        help="Whether to scale the output or not")

    parser.add_argument("--horseshoe_bnn", default='0', const='0', nargs='?', choices=['0', '1'],
                        help="Whether to use horseshoe BNN or not")

    return parser.parse_args()

def run_benchmark_exps(data_name, *, seeds, hbnn_config_file, model_configs,
                       epochs, exp_dir, version, n_trials, use_horseshoe_bnn=False):

    years_test_size = 51630
    print(f"Running experiments on {data_name} dataset")
    save_dir = f"{exp_dir}/{data_name}/"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/optuna").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/configs").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/ft_importance").mkdir(parents=True, exist_ok=True)

    X, y, classification, cat_cols, batch_size = load_data(data_name)
    for seed in tqdm(seeds):
        models_score = {"seed": [], "model": [], "score": []}

        if classification:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y,
                                                                shuffle=True)
        else:
            if data_name == "year":
                X_train, X_test, y_train, y_test = X[:-years_test_size], X[-years_test_size:], y[:-years_test_size], y[
                                                                                                                     -years_test_size:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed,
                                                                    shuffle=True)

        if cat_cols is not None:
            cat_encoder = LeaveOneOutEncoder(cols=cat_cols).fit(X_train, y_train)
            X_train = cat_encoder.transform(X_train)
            X_test = cat_encoder.transform(X_test)

        X_train, X_test = X_train.values.astype(np.float32), X_test.values.astype(np.float32)
        y_train, y_test = y_train.values.astype(np.float32), y_test.values.astype(np.float32)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if use_horseshoe_bnn:

            hidden_sizes = [model_configs["horseshoe_bnn"]["n_hidden"]] * model_configs["horseshoe_bnn"]["num_layers"]
            config = {"n_hidden": hidden_sizes, "classification": classification}
            hbnn_model, hbnn_score, _,  = run_horsehoe_bnn_model(seed, hbnn_config_file, X_train, X_test,
                                                                             y_train, y_test, epochs, batch_size,
                                                                             config, data_name,
                                                                             classification)

            models_score["seed"].append(seed)
            models_score["model"].append("Horseshoe BNN")
            models_score["score"].append(hbnn_score)

            # Feature Weights
            beta_weights = hbnn_model.l1.beta.sample(model_configs["horseshoe_bnn"]["n_samples_testing"])
            w_norm_hs_bnn = torch.mean(beta_weights, dim=0)
            w_mean_hs_bnn = torch.linalg.vector_norm(w_norm_hs_bnn, dim=0).detach().cpu().numpy()
            np.save(f"{save_dir}/ft_importance/horseshoe_bnn_s_{seed}_v{version}.npy" ,w_mean_hs_bnn)

            with open(f"{save_dir}/results/horseshoe_bnn_s_{seed}_v{version}.csv", "w") as fp:
                models_score_df = pd.DataFrame(models_score)
                models_score_df.to_csv(fp, index=False)
                fp.flush()

        else:
            J = np.corrcoef(X_train, rowvar=False)
            np.fill_diagonal(J, 0.0)
            J[np.isnan(J)] = 0.0
            J_zeros = np.zeros_like(J)

            bnn_states, bnn_disc_states, bnn_rmse, bnn_r2 = run_bnn_model(seed, save_dir, version, X_train, X_test,
                                                                          y_train, y_test,
                                                                          epochs, batch_size, J_zeros, model_configs["bnn"],
                                                                          n_trials,
                                                                          classification, dropout_version=2, bg=False)

            bnn_bg_states, bnn_bg_disc_states, bnn_bg_rmse, bnn_bg_r2 = run_bnn_model(seed, save_dir, version, X_train,
                                                                                      X_test,
                                                                                      y_train, y_test,
                                                                                      epochs, batch_size, J,
                                                                                      model_configs["bnn + bg"], n_trials,
                                                                                      classification, dropout_version=2,
                                                                                      bg=True)

            params_bnn = tree_utils.tree_stack(bnn_states)
            # gammas_bnn = tree_utils.tree_stack(bnn_disc_states)

            params_bnn_bg = tree_utils.tree_stack(bnn_bg_states)
            # gammas_bnn_bg = tree_utils.tree_stack(bnn_bg_disc_states)

            rf_model, rf_rmse = run_rf_model(seed, save_dir, version, X_train, X_test, y_train, y_test, n_trials,
                                             classification)

            models_score["seed"].append(seed)
            models_score["model"].append("BNN w/o BG")
            models_score["score"].append(bnn_rmse)

            models_score["seed"].append(seed)
            models_score["model"].append("BNN + BG")
            models_score["score"].append(bnn_bg_rmse)

            models_score["seed"].append(seed)
            models_score["model"].append("RF")
            models_score["score"].append(rf_rmse)


            w_norm_bnn = jnp.mean(params_bnn["dropout"]["w"], axis=0)
            w_mean_bnn = jax.device_get(jax.vmap(lambda x: jnp.linalg.norm(x))(w_norm_bnn))
            np.save(f"{save_dir}/ft_importance/bnn_ft_importance_s_{seed}_v{version}.npy",w_mean_bnn)

            w_norm_bnn_bg = jnp.mean(params_bnn_bg["dropout"]["w"], axis=0)
            w_mean_bnn_bg = jax.device_get(jax.vmap(lambda x: jnp.linalg.norm(x))(w_norm_bnn_bg))
            np.save(f"{save_dir}/ft_importance/bg_bnn_ft_importance_s_{seed}_v{version}.npy" ,w_mean_bnn_bg)

            w_norm_rf = rf_model.feature_importances_
            np.save(f"{save_dir}/ft_importance/rf_ft_importance_s_{seed}_v{version}.npy", w_norm_rf)

            with open(f"{save_dir}/results/bnn_rf_bg_s_{seed}_v{version}.csv", "w") as fp:
                models_score_df = pd.DataFrame(models_score)
                models_score_df.to_csv(fp, index=False)
                fp.flush()


    print(f"Done for {data_name}!")


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
        {"score": ["mean", "std"]}
    )
    # perf_df = perf_df.rename(columns={"BNN w/o BG": "BNN"})
    perf = perf_df["score"]
    perf["summary"] = perf.apply(lambda row: f"{round(row['mean'], 3)}  ± {round(row['std'], 3)}",
                                 axis=1).values
    perf_table = pd.pivot_table(perf[["summary"]], index=["dataset"], columns=["model"], aggfunc="first")

    def highlight_min(s, props=''):
        return np.where(s == np.nanmin(s.values), props, '')

    # perf_table["summary"].style.apply(highlight_min, props='font-weight:bold', axis=1)
    return perf_table


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = parse_args()
    seeds = []
    with open(args.seeds, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))

    scale_output = int(args.scale_output) == 1
    use_horseshoe_bnn = int(args.horseshoe_bnn) == 1
    saved_config = int(args.saved_config) == 1

    model_configs = {"bnn": {"n_hidden": args.num_hidden, "act_fn": args.act_fn, "num_layers": args.num_layers,
                             "prior_dist": args.prior_dist},
                     "bnn + bg": {"n_hidden": args.num_hidden, "act_fn": args.act_fn, "num_layers": args.num_layers,
                             "prior_dist": args.prior_dist},
                     "horseshoe_bnn": {"lr": 0.001,  "n_hidden": args.num_hidden, "num_layers": args.num_layers, "n_samples_testing": 100}}

    exp_dir = args.exp_dir
    hbnn_config_file = f"{uci_data_dir}/horseshoeBNN_config.yaml"

    exp_fn = functools.partial(run_benchmark_exps, seeds=seeds[:args.num_folds], hbnn_config_file=hbnn_config_file,
                               model_configs=model_configs, epochs=args.num_epochs, exp_dir=exp_dir,
                               version=args.version,
                               n_trials=args.n_trials, use_horseshoe_bnn=use_horseshoe_bnn)

    data_names = [data_name for data_name in args.data_names.split(",")]

    pool = Pool(len(data_names))
    pool.map(exp_fn, data_names)
    pool.close()
    pool.join()
