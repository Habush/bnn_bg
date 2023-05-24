#!/usr/bin/python3
import functools

from sklearn.preprocessing import StandardScaler
from category_encoders import LeaveOneOutEncoder
import argparse
from utils.pub_data_exp_utils import *
from multiprocessing import Pool

def parse_args():

    parser = argparse.ArgumentParser(description="Run experiments on public data")
    parser.add_argument("--data_dir", type=str, default="./data/pub_data",
                        help="Path to the directory containing the data")
    parser.add_argument("--exp_dir", type=str, default="./data/pub_data/exps",
                        help="Path to the directory where the experiment data will be saved")
    parser.add_argument("--seeds", type=str, help="Path to the file containing the seeds. Each seed should be in a separate line",
                        default="./data/seeds.txt")
    parser.add_argument("--data_names", type=str, default="bikeshare,wine,support2,churn",
                        help="Comma separated list of dataset names")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--num_hidden", type=int, default=64, help="Number of hidden units in each layer")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--prior_dist", default="laplace", const="laplace", nargs="?",
                        choices=["laplace", "normal", "student_t"]
                        , help="Prior distribution for the weights. Options: laplace, normal, student_t")
    parser.add_argument("--act_fn", default="swish", const="swish", nargs="?",
                        choices=["swish", "relu"],
                        help="Activation function for the hidden layers. Options: swish, relu, tanh, sigmoid")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout for hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of trials for hyperparameter optimization")
    parser.add_argument("--saved_config", default='0', const='0', nargs='?', choices=['0', '1'],
                        help="Whether to scale the output or not")

    parser.add_argument("--horseshoe_bnn", default='0', const='0', nargs='?', choices=['0', '1'],
                        help="Whether to use horseshoe BNN or not")

    return parser.parse_args()

def run_benchmark_exps(data_name, *, data_dir, seeds, hbnn_config_file, model_configs,
                       epochs, exp_dir, n_trials, use_horseshoe_bnn=False):

    print(f"Running experiments on {data_name} dataset")
    save_dir = f"{exp_dir}/{data_name}/"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/optuna").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/configs").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/ft_importance").mkdir(parents=True, exist_ok=True)

    X, y, classification, cat_cols, batch_size = load_data(data_name, data_dir)
    for seed in tqdm(seeds):
        models_score = {"seed": [], "model": [], "score": []}

        if classification:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y,
                                                                shuffle=True)
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
            np.save(f"{save_dir}/ft_importance/horseshoe_bnn_s_{seed}.npy" ,w_mean_hs_bnn)

            with open(f"{save_dir}/results/horseshoe_bnn_s_{seed}.csv", "w") as fp:
                models_score_df = pd.DataFrame(models_score)
                models_score_df.to_csv(fp, index=False)
                fp.flush()

        else:
            J = np.corrcoef(X_train, rowvar=False)
            np.fill_diagonal(J, 0.0)
            J[np.isnan(J)] = 0.0
            J_zeros = np.zeros_like(J)

            bnn_states, bnn_disc_states, bnn_rmse, bnn_r2 = run_bnn_model(seed, save_dir, X_train, X_test,
                                                                          y_train, y_test,
                                                                          epochs, batch_size, J_zeros, model_configs["bnn"],
                                                                          n_trials,
                                                                          classification, bg=False)

            bnn_bg_states, bnn_bg_disc_states, bnn_bg_rmse, bnn_bg_r2 = run_bnn_model(seed, save_dir, X_train,
                                                                                      X_test,
                                                                                      y_train, y_test,
                                                                                      epochs, batch_size, J,
                                                                                      model_configs["bnn + bg"], n_trials,
                                                                                      classification,
                                                                                      bg=True)

            params_bnn = tree_utils.tree_stack(bnn_states)
            # gammas_bnn = tree_utils.tree_stack(bnn_disc_states)

            params_bnn_bg = tree_utils.tree_stack(bnn_bg_states)
            # gammas_bnn_bg = tree_utils.tree_stack(bnn_bg_disc_states)

            rf_model, rf_rmse = run_rf_model(seed, save_dir, X_train, X_test, y_train, y_test, n_trials,
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
            np.save(f"{save_dir}/ft_importance/bnn_ft_importance_s_{seed}.npy",w_mean_bnn)

            w_norm_bnn_bg = jnp.mean(params_bnn_bg["dropout"]["w"], axis=0)
            w_mean_bnn_bg = jax.device_get(jax.vmap(lambda x: jnp.linalg.norm(x))(w_norm_bnn_bg))
            np.save(f"{save_dir}/ft_importance/bg_bnn_ft_importance_s_{seed}.npy" ,w_mean_bnn_bg)

            w_norm_rf = rf_model.feature_importances_
            np.save(f"{save_dir}/ft_importance/rf_ft_importance_s_{seed}.npy", w_norm_rf)

            with open(f"{save_dir}/results/bnn_rf_bg_s_{seed}.csv", "w") as fp:
                models_score_df = pd.DataFrame(models_score)
                models_score_df.to_csv(fp, index=False)
                fp.flush()


    print(f"Done for {data_name}!")



if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = parse_args()
    seeds = []
    with open(args.seeds, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))

    use_horseshoe_bnn = int(args.horseshoe_bnn) == 1
    saved_config = int(args.saved_config) == 1

    model_configs = {"bnn": {"n_hidden": args.num_hidden, "act_fn": args.act_fn, "num_layers": args.num_layers,
                             "prior_dist": args.prior_dist},
                     "bnn + bg": {"n_hidden": args.num_hidden, "act_fn": args.act_fn, "num_layers": args.num_layers,
                             "prior_dist": args.prior_dist},
                     "horseshoe_bnn": {"lr": 0.001,  "n_hidden": args.num_hidden, "num_layers": args.num_layers, "n_samples_testing": 100}}

    exp_dir = args.exp_dir
    data_dir = args.data_dir
    hbnn_config_file = f"{data_dir}/horseshoeBNN_config.yaml"

    exp_fn = functools.partial(run_benchmark_exps, data_dir=data_dir, seeds=seeds,
                               hbnn_config_file=hbnn_config_file,
                               model_configs=model_configs, epochs=args.num_epochs, exp_dir=exp_dir,
                               n_trials=args.n_trials, use_horseshoe_bnn=use_horseshoe_bnn)

    data_names = [data_name for data_name in args.data_names.split(",")]

    pool = Pool(len(data_names))
    pool.map(exp_fn, data_names)
    pool.close()
    pool.join()
