import jax
import jax.numpy as jnp
import typing
import haiku as hk
import os
import pathlib
import mlflow
import optuna
import utils.losses_ext as losses_ext
import utils.losses as losses
import utils.train_utils as train_utils
import utils.tree_utils as tree_utils
import core.optim as optim
import core.models as models
import core.sgmcmc as sgmcmc
import core.sgmcmc_ext as sgmcmc_ext
from scipy.stats import binom
from utils import metrics
from scipy import stats
import numpy as np
import xgboost as xgb
from sklearn.linear_model import ElasticNet
import typer
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import utils.nn_util as nn_util
import yaml
from tqdm import tqdm
from utils import script_utils


app = typer.Typer()


data_scale = 1.0
optuna.logging.set_verbosity(optuna.logging.ERROR)


def run_xgb_model(exp_id, save_dir, seed, x_train, x_val, x_test,
                    y_train, y_val, y_test, n_trials):
    def objective_xgb(trial):
        with mlflow.start_run(nested=True):
            # Define hyperparameters
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            }

            if params["booster"] == "gbtree" or params["booster"] == "dart":
                params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
                params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                params["grow_policy"] = trial.suggest_categorical(
                    "grow_policy", ["depthwise", "lossguide"]
                )

            model = xgb.XGBRegressor(**params, random_state=seed)
            preds = model.fit(x_train, y_train).predict(x_val)
            metrics_dict = script_utils.calculate_sklearn_model_metric(preds, y_val)

            # Log to MLflow
            mlflow.log_params(params)
            for m in metrics_dict:
                mlflow.log_metric(m, metrics_dict[m])

        return metrics_dict["rmse"]

    print("Starting XGBoost Run")
    with mlflow.start_run(experiment_id=exp_id, run_name=f"xgboost_seed_{seed}",
                          nested=True):
        tpe_sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=tpe_sampler)
        study.optimize(objective_xgb, n_trials=n_trials,
                       callbacks=[script_utils.champion_callback])

        # Log to MLflow
        mlflow.log_params(study.best_params)
        model = xgb.XGBRegressor(**study.best_params, random_state=seed)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics_dict = script_utils.calculate_sklearn_model_metric(preds, y_test)
        for m in metrics_dict:
            mlflow.log_metric(m, metrics_dict[m])

        xgb_feat_importances = model.feature_importances_
        xgb_feat_ranking = np.argsort(xgb_feat_importances)[::-1]

        mlflow.log_figure(figure=script_utils.plot_feature_importance(xgb_feat_importances),
                          artifact_file="feature_importance.png")

        p = os.path.join(save_dir, f"xgb_feat_ranking_s_{seed}.npy")
        np.save(p, xgb_feat_ranking)
        mlflow.log_artifact(p)

        print("XGBoost Run Complete")


def run_elastic_net_model(exp_id, save_dir, seed, x_train, x_val, x_test,
                    y_train, y_val, y_test, n_trials):

    def objective_elastic_net(trial):
        with mlflow.start_run(nested=True):
            # Define hyperparameters
            params = {
                "alpha": trial.suggest_float("alpha", 0.0, 1.0),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }

            model = ElasticNet(**params, random_state=seed)
            preds = model.fit(x_train, y_train).predict(x_val)
            metrics_dict = script_utils.calculate_sklearn_model_metric(preds, y_val)

            # Log to MLflow
            mlflow.log_params(params)
            for m in metrics_dict:
                mlflow.log_metric(m, metrics_dict[m])

        return metrics_dict["rmse"]

    print("Starting ElasticNet Run")
    with mlflow.start_run(experiment_id=exp_id, run_name=f"elastic_net_seed_{seed}",
                          nested=True):
        tpe_sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=tpe_sampler)
        study.optimize(objective_elastic_net, n_trials=n_trials,
                       callbacks=[script_utils.champion_callback])

        # Log to MLflow
        mlflow.log_params(study.best_params)
        enet_model = ElasticNet(**study.best_params, random_state=seed)
        enet_model.fit(x_train, y_train)
        preds = enet_model.predict(x_test)
        enet_metric = script_utils.calculate_sklearn_model_metric(preds, y_test)
        for m in enet_metric:
            mlflow.log_metric(m, enet_metric[m])

        enet_feat_importances = np.abs(enet_model.coef_)
        mlflow.log_figure(figure=script_utils.plot_feature_importance(enet_feat_importances),
                            artifact_file="feature_importance.png")
        enet_feat_ranking = np.argsort(enet_feat_importances)[::-1]
        p = os.path.join(save_dir, f"enet_feat_ranking_s_{seed}.npy")
        np.save(p, enet_feat_ranking)
        mlflow.log_artifact(p)

def run_bnn_model_first_layer_ss(exp_id, save_dir, seed, x_train, x_val, x_test,
                    y_train, y_val, y_test, n_trials):

    p = x_train.shape[1]
    J = jnp.zeros((p, p))
    train_set = (x_train, y_train)
    val_set = (x_val, y_val)
    test_set = (x_test, y_test)
    rng = jax.random.PRNGKey(seed)

    print("Starting BNN Run")
    def objective_bnn(trial):
        with mlflow.start_run(nested=True):
            params = {
                "tau0": trial.suggest_float("tau0", 0.0, 1e-2),
                "tau1": trial.suggest_float("tau1", 0.1, 1.0),
                "scale": trial.suggest_float("scale", 1e-2, 1.0),
                "base_dist": trial.suggest_categorical("base_dist", ["laplace", "normal"]),
                "layer_dim": trial.suggest_categorical("layer_dim", [32, 64, 128]),
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "lr": trial.suggest_float("lr", 1e-6, 1e-1, log=True),
                "bin_lr": trial.suggest_float("bin_lr", 1e-4, 0.5, log=True),
                "invsp_noise_std": trial.suggest_float("invsp_noise_std", 1e-3, 1e-1, log=True),
                "q": trial.suggest_float("q", 1e-3, 0.5),
                "n_iters": 3000,
                "burnin": 2000,
                "n_batches" : 2,
                "save_freq" : 100,
                "temperature" : 1.0,
                "eta": 1.0,
                "mu": 1.0
            }

            _, _, val_preds = script_utils.train_bnn_first_ss_model(rng, train_set, val_set, J,
                                                                        params)

            bnn_metrics = script_utils.calculate_bnn_metrics(val_preds, y_val)
            mlflow.log_params(params)
            for m in bnn_metrics:
                mlflow.log_metric(m, bnn_metrics[m])

            return bnn_metrics["rmse"]

    with mlflow.start_run(experiment_id=exp_id, run_name=f"bnn_first_layer_ss_seed_{seed}",
                          nested=True):
        tpe_sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=tpe_sampler)
        study.optimize(objective_bnn, n_trials=n_trials,
                       callbacks=[script_utils.champion_callback])

        # Log to MLflow
        best_params = {
                "tau0": study.best_params["tau0"],
                "tau1": study.best_params["tau1"],
                "scale": study.best_params["scale"],
                "base_dist": study.best_params["base_dist"],
                "layer_dim": study.best_params["layer_dim"],
                "n_layers": study.best_params["n_layers"],
                "lr": study.best_params["lr"],
                "bin_lr": study.best_params["bin_lr"],
                "invsp_noise_std": study.best_params["invsp_noise_std"],
                "q": study.best_params["q"],
                "n_iters": 3000,
                "burnin": 2000,
                "n_batches" : 2,
                "save_freq" : 100,
                "temperature" : 1.0,
                "eta": 1.0,
                "mu": 1.0
            }

        mlflow.log_params(best_params)
        nn_params, _, test_preds = script_utils.train_bnn_first_ss_model(rng, train_set, test_set, J,
                                                                        best_params)

        bnn_metrics = script_utils.calculate_bnn_metrics(test_preds, y_test)
        for m in bnn_metrics:
            mlflow.log_metric(m, bnn_metrics[m])

        nn_params = tree_utils.tree_stack(nn_params)
        feat_importances = np.mean(np.abs(nn_params["dropout"]["w"]), axis=0)
        mlflow.log_figure(figure=script_utils.plot_feature_importance(feat_importances),
                            artifact_file="feature_importance.png")
        feat_ranking = np.argsort(feat_importances)[::-1]
        p = os.path.join(save_dir, f"bnn_feat_ranking_s_{seed}.npy")
        np.save(p, feat_ranking)
        mlflow.log_artifact(p)

        print("BNN Run Complete")
def run_bnn_model_all_ss(exp_id, save_dir, seed, x_train, x_val, x_test,
                    y_train, y_val, y_test, n_trials):

    p = x_train.shape[1]
    J = jnp.zeros((p, p))
    train_set = (x_train, y_train)
    val_set = (x_val, y_val)
    test_set = (x_test, y_test)
    rng = jax.random.PRNGKey(seed)

    print("Starting BNN Run")
    def objective_bnn(trial):
        with mlflow.start_run(nested=True):
            params = {
                "tau0": trial.suggest_float("tau0", 0.0, 1e-2),
                "tau1": trial.suggest_float("tau1", 0.1, 1.0),
                "base_dist": trial.suggest_categorical("base_dist", ["laplace", "normal"]),
                "layer_dim": trial.suggest_categorical("layer_dim", [32, 64, 128]),
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "lr": trial.suggest_float("lr", 1e-6, 1e-1, log=True),
                "bin_lr": trial.suggest_float("bin_lr", 1e-4, 0.5, log=True),
                "invsp_noise_std": trial.suggest_float("invsp_noise_std", 1e-3, 1e-1, log=True),
                "q_first": trial.suggest_float("q_first", 1e-3, 0.5),
                "q_rest": trial.suggest_float("q_rest", 1e-3, 0.5),
                "n_iters": 3000,
                "burnin": 2000,
                "n_batches" : 2,
                "save_freq" : 100,
                "temperature" : 1.0,
                "eta": 1.0,
                "mu": 1.0
            }

            _, _, val_preds = script_utils.train_bnn_all_ss_model(rng, train_set, val_set, J,
                                                                    params)

            bnn_metrics = script_utils.calculate_bnn_metrics(val_preds, y_val)
            mlflow.log_params(params)
            for m in bnn_metrics:
                mlflow.log_metric(m, bnn_metrics[m])

            return bnn_metrics["rmse"]

    with mlflow.start_run(experiment_id=exp_id, run_name=f"bnn_all_layers_ss_seed_{seed}",
                          nested=True):
        tpe_sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=tpe_sampler)
        study.optimize(objective_bnn, n_trials=n_trials,
                       callbacks=[script_utils.champion_callback])

        # Log to MLflow
        best_params = {
                "tau0": study.best_params["tau0"],
                "tau1": study.best_params["tau1"],
                "base_dist": study.best_params["base_dist"],
                "layer_dim": study.best_params["layer_dim"],
                "n_layers": study.best_params["n_layers"],
                "lr": study.best_params["lr"],
                "bin_lr": study.best_params["bin_lr"],
                "invsp_noise_std": study.best_params["invsp_noise_std"],
                "q_first": study.best_params["q_first"],
                "q_rest": study.best_params["q_rest"],
                "n_iters": 3000,
                "burnin": 2000,
                "n_batches" : 2,
                "save_freq" : 100,
                "temperature" : 1.0,
                "eta": 1.0,
                "mu": 1.0
        }

        mlflow.log_params(best_params)
        nn_params, _, test_preds = script_utils.train_bnn_all_ss_model(rng, train_set, test_set, J,
                                                                            best_params)
        bnn_metrics = script_utils.calculate_bnn_metrics(test_preds, y_test)
        for m in bnn_metrics:
            mlflow.log_metric(m, bnn_metrics[m])

        nn_params = tree_utils.tree_stack(nn_params)
        feat_importances = np.mean(np.abs(nn_params["dropout"]["w"]), axis=0)
        mlflow.log_figure(figure=script_utils.plot_feature_importance(feat_importances),
                          artifact_file="feature_importance.png")
        feat_ranking = np.argsort(feat_importances)[::-1]
        p = os.path.join(save_dir, f"bnn_feat_ranking_s_{seed}.npy")
        np.save(p, feat_ranking)
        mlflow.log_artifact(p)

        print("BNN Run Complete")
@app.command()
def train_bnn_model(exp_name: str=None,
                    mlflow_host: str="localhost",
                    mlflow_port: int=8899,
                    input_file: str=None,
                    output_dir: str=None,
                    seeds: str=None,
                    n_trials: int=100):
    seeds = np.loadtxt(seeds, dtype=int)
    riboflavin = pd.read_csv(input_file)
    riboflavin_y = riboflavin['y'].to_numpy()
    riboflavin_x = riboflavin.drop(['y', 'Unnamed: 0'], axis=1).to_numpy()

    mlflow.set_tracking_uri(f"http://{mlflow_host}:{mlflow_port}")
    exp_id = script_utils.get_or_create_experiment(exp_name)
    mlflow.set_experiment(experiment_id=exp_id)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for seed in tqdm(seeds):
        print(f"Running seed {seed}")
        x_train, x_test, y_train, y_test = train_test_split(riboflavin_x, riboflavin_y, test_size=0.1,
                                                            random_state=seed)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1,
                                                        random_state=seed)
        run_bnn_model_all_ss(exp_id, output_dir, seed, x_train, x_val, x_test,
                            y_train, y_val, y_test, n_trials)
        run_bnn_model_first_layer_ss(exp_id, output_dir, seed, x_train, x_val, x_test,
                                    y_train, y_val, y_test, n_trials)
        run_xgb_model(exp_id, output_dir, seed, x_train, x_val, x_test,
                    y_train, y_val, y_test, n_trials)
        run_elastic_net_model(exp_id, output_dir, seed, x_train, x_val, x_test,
                            y_train, y_val, y_test, n_trials)


if __name__ == "__main__":
    app()