import os
import os.path
import pathlib

import optuna
from netZooPy.panda import Panda
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.data_utils import NumpyLoader, NumpyData, preprocess_data
from utils.hpo_util import *
from utils.nn_util import *


def get_result_df(seeds, save_dir, zero_out=False,
                  include_horseshoe=True, include_gnn=True):


    res_dfs = []
    for seed in seeds:
        if zero_out:
            df = pd.read_csv(f"{save_dir}/results/feat_zero_out_comp_bnn_bg_rf_s_{seed}.csv")
        else:
            df = pd.read_csv(f"{save_dir}/results/bnn_rf_bg_s_{seed}.csv")

            if include_horseshoe:
                hbnn_df = pd.read_csv(f"{save_dir}/results/horseshoe_bnn_s_{seed}.csv")
                df = pd.concat([df, hbnn_df], axis=0, ignore_index=True)

            if include_gnn:
                gnn_df = pd.read_csv(f"{save_dir}/results/bg_gnn_s_{seed}.csv")
                df = pd.concat([df, gnn_df], axis=0, ignore_index=True)

        res_dfs.append(df)

    bnn_rf_df = pd.concat(res_dfs, ignore_index=True, axis=0)
    return bnn_rf_df


def get_summary_results(seeds, drug_names, exp_dir,
                       min_highlight=False):
    drug_res_all = []
    num_models = 5
    for drug in drug_names:
        save_dir = f"{exp_dir}/{drug}"
        drug_res_df = get_result_df(seeds, save_dir)
        drug_name_col = [drug for _ in range(num_models * len(seeds))]
        drug_res_df.insert(0, column="drug", value=drug_name_col)
        drug_res_all.append(drug_res_df)

    drug_res_all = pd.concat(drug_res_all, axis=0)
    perf_df = drug_res_all.groupby(["drug", "model"]).agg(
        {"test_rmse": ["mean", "std"]}
    )
    perf_df = perf_df.rename(columns={"BNN w/o BG": "BNN"})
    perf = perf_df["test_rmse"]
    perf["summary"] = perf.apply(lambda row: f"{round(row['mean'], 3)}  ± {round(row['std'], 3)}",
                                 axis=1).values
    perf_table = pd.pivot_table(perf[["summary"]], index=["drug"], columns=["model"], aggfunc="first")

    if min_highlight:
        return perf_table["summary"].style.highlight_min(props='font-weight:bold', axis=1)
    return perf_table


def get_feature_ranking_summary(seeds, drug_names, exp_dir, k=50):
    drug_res_all = []
    num_models = 3
    for drug in drug_names:
        save_dir = f"{exp_dir}/{drug}"
        drug_res_df = get_result_df(seeds, save_dir, zero_out=True, include_horseshoe=False, include_gnn=False)
        drug_res_df = drug_res_df[drug_res_df["num_feats"] == k]
        drug_name_col = [drug for _ in range(num_models * len(seeds))]
        drug_res_df.insert(0, column="drug", value=drug_name_col)
        drug_res_all.append(drug_res_df)

    drug_res_all = pd.concat(drug_res_all, axis=0)
    perf_df = drug_res_all.groupby(["drug", "model"]).agg(
        {"test_rmse_score": ["mean", "std"]})

    perf = perf_df["test_rmse_score"]
    perf["summary"] = perf.apply(lambda row: f"{round(row['mean'], 3)}  ± {round(row['std'], 3)}",
                                 axis=1).values
    perf_table = pd.pivot_table(perf[["summary"]], index=["drug"], columns=["model"], aggfunc="last")

    return perf_table

def run_bnn_model(seed, save_dir, X_train_outer, X_train, X_val, X_test,
                  y_train_outer, y_train, y_val, y_test, J,
                  hyperparam_config, timeout, saved_config=False, bg=True):

    torch.manual_seed(seed)
    hidden_size = hyperparam_config["num_hidden"]
    act_fn = hyperparam_config["act_fn"]
    prior_dist = hyperparam_config["prior_dist"]
    beta, M = hyperparam_config["beta"], hyperparam_config["num_models"]
    epochs = hyperparam_config["epochs"]

    if bg:
        config_path = f"{save_dir}/configs/bg_bnn_config_s_{seed}_optuna.pkl"
        study_path = f"{save_dir}/optuna/study_bg_bnn_s_{seed}.pkl"
    else:
        config_path = f"{save_dir}/configs/bnn_config_s_{seed}_optuna.pkl"
        study_path = f"{save_dir}/optuna/study_bnn_s_{seed}_optuna.pkl"

    if saved_config and os.path.exists(config_path):
        bnn_config = pickle.load(open(config_path, "rb"))
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler)
        study.optimize(lambda trial: objective_bg_bnn(trial, seed, X_train, X_val, y_train, y_val, J, epochs,
                                                      beta, act_fn, hidden_size, prior_dist), timeout=timeout)

        with open(study_path, "wb") as fp:
            pickle.dump(study, fp)
            fp.flush()

        bnn_config = study.best_params
        with open(config_path, "wb") as fp:
            pickle.dump(bnn_config, fp)
            fp.flush()

    num_cycles = 50
    batch_size = 16 if X_train.shape[0] < 200 else 32
    lr_0, disc_lr_0 = 1e-3, 0.5
    sigma_1, sigma_2 = 0.01, 1.0
    temp = bnn_config["temp"]
    eta, mu = 1.0, 1.0

    hidden_sizes = [256]*bnn_config["num_layers"]

    torch.manual_seed(seed)
    data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size, shuffle=True,
                              drop_last=True)

    bnn_model, states = train_bnn_model(seed, data_loader, epochs,
                                                                         num_cycles,
                                                                         beta, M, lr_0, disc_lr_0, hidden_sizes,temp,
                                                                         sigma_1, sigma_2, eta, mu, J, act_fn,
                                                                         show_pgbar=False,
                                                                         prior_dist=prior_dist)

    rmse, r2 = score_bnn_model(bnn_model, X_test, y_test, states)

    bnn_states, bnn_disc_states = [], []
    for state in states:
        bnn_states.append(state.params)
        bnn_disc_states.append(state.gamma)

    return bnn_states, bnn_disc_states, rmse, r2

def cross_val_runs(seeds, X, y, tissue_motif_data, string_ppi, hgnc_map,
                      save_dir, model_save_dir, saved_config=False, timeout=180,
                      n_trials=30, **configs):


    optuna.logging.set_verbosity(optuna.logging.WARNING)
    gene_list = X.columns.to_list()

    for seed in tqdm(seeds):
        bnn_rf_bg_dict = {"seed": [], "model": [], "test_rmse": [], "test_r2_score": []}
        transformer = StandardScaler()
        X_train_outer, X_train, X_val, X_test, \
            y_train_outer, y_train, y_val, y_test, (train_indices, val_indices, _) = preprocess_data(seed, X, y, None,
                                                                                                     transformer,
                                                                                                     val_size=0.2,
                                                                                                     test_size=0.2)

        graph_path, col_idx_path = f"{save_dir}/pandas/pandas_net_s_{seed}.npy", f"{save_dir}/pandas/pandas_col_idxs_s_{seed}.npy"
        if os.path.exists(graph_path) and os.path.exists(col_idx_path):
            J = np.load(graph_path)
            col_idxs = np.load(col_idx_path)

        else:
            J, col_idxs = get_inferred_network(X_train_outer, tissue_motif_data, string_ppi, hgnc_map, gene_list)
            np.save(graph_path, J)
            np.save(col_idx_path, col_idxs)

        J_zeros = np.zeros_like(J)

        X_train_outer, X_train, X_val, X_test = X_train_outer[:,col_idxs], X_train[:, col_idxs], \
                                                X_val[:,col_idxs], X_test[:,col_idxs]

        #BNN w/ BG
        bnn_bg_states, bnn_bg_disc_states, bnn_bg_rmse, bnn_bg_r2 = run_bnn_model(seed, save_dir,
                                                                                  X_train_outer, X_train, X_val,
                                                                                  X_test, y_train_outer, y_train,
                                                                                  y_val, y_test, J, configs, timeout,
                                                                                  saved_config, bg=True)
        params_bg_bnn = tree_utils.tree_stack(bnn_bg_states)
        gammas_bg_bnn = tree_utils.tree_stack((bnn_bg_disc_states))
        save_model(model_save_dir, seed, params_bg_bnn, gammas_bg_bnn, True)

        #BNN w/o BG
        bnn_states, bnn_disc_states, bnn_rmse, bnn_r2 = run_bnn_model(seed, save_dir,
                                                                          X_train_outer, X_train, X_val,
                                                                          X_test, y_train_outer, y_train,
                                                                          y_val, y_test, J_zeros, configs, n_trials,
                                                                          saved_config, bg=False)
        params_bnn = tree_utils.tree_stack(bnn_states)
        gammas_bnn = tree_utils.tree_stack(bnn_disc_states)
        save_model(model_save_dir, seed, params_bnn, gammas_bnn, False)

        ## RF
        rf_model_path = f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl"
        rf_config_path = f"{save_dir}/configs/rf_config_s_{seed}_optuna.pkl"
        if saved_config and os.path.exists(rf_model_path):  # standard scaler
            rf_model = pickle.load(open(rf_model_path, "rb"))
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(sampler=sampler)
            study.optimize(lambda trial: objective_rf(trial, seed, X_train, X_val, y_train, y_val), timeout=timeout)

            with open(f"{save_dir}/optuna/study_rf_s_{seed}.csv", "wb") as fp:
                pickle.dump(study, fp)
                fp.flush()

            rf_config = study.best_params
            with open(rf_config_path, "wb") as fp:
                pickle.dump(rf_config, fp)
                fp.flush()
            rf_model = RandomForestRegressor(**rf_config, random_state=seed)
            rf_model.fit(X_train_outer, y_train_outer)
            pickle.dump(rf_model, open(rf_model_path, "wb"))


        rmse_test_rf, r2_test_rf = eval_sklearn_model(rf_model, X_test, y_test)

        bnn_rf_bg_dict["seed"].append(seed)
        bnn_rf_bg_dict["model"].append("RF")
        bnn_rf_bg_dict["test_rmse"].append(rmse_test_rf)
        bnn_rf_bg_dict["test_r2_score"].append(r2_test_rf)

        bnn_rf_bg_dict["seed"].append(seed)
        bnn_rf_bg_dict["model"].append("BNN w/o BG")
        bnn_rf_bg_dict["test_rmse"].append(bnn_rmse)
        bnn_rf_bg_dict["test_r2_score"].append(bnn_r2)

        bnn_rf_bg_dict["seed"].append(seed)
        bnn_rf_bg_dict["model"].append("BNN + BG")
        bnn_rf_bg_dict["test_rmse"].append(bnn_bg_rmse)
        bnn_rf_bg_dict["test_r2_score"].append(bnn_bg_r2)


        print(f"RF scores - rmse: {rmse_test_rf}, r2: {r2_test_rf}")
        print(f"BNN w/o scores - rmse: {bnn_rmse}, r2: {bnn_r2}")
        print(f"BNN + BG scores - rmse: {bnn_bg_rmse}, r2: {bnn_bg_r2}")

        with open(f"{save_dir}/results/bnn_rf_bg_s_{seed}.csv", "w") as fp:
            pd.DataFrame(bnn_rf_bg_dict).to_csv(fp, index=False)
            fp.flush()

    return print("Done")



def cross_val_horseshoe_bnn(seeds, X, y, save_dir, model_save_dir,
                            saved_config=False, n_trial=7, **configs):

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    horseshoe_config_file = configs["horseshoe_config_file"]
    drug_name = configs["drug_name"]
    num_hidden = configs["num_hidden"]
    epochs = configs["epochs"]

    for seed in tqdm(seeds):
        horseshoe_res_df_dict = {"seed": [], "model": [], "test_rmse": [], "test_r2_score": []}
        transformer = StandardScaler()
        X_train_outer, X_train, X_val, X_test, \
            y_train_outer, y_train, y_val, y_test, (train_indices, val_indices, _) = preprocess_data(seed, X, y, None,
                                                                                                     transformer,
                                                                                                     val_size=0.2,
                                                                                                     test_size=0.2)
        col_idx_path = f"{save_dir}/pandas/pandas_col_idxs_s_{seed}.npy"
        col_idxs = np.load(col_idx_path)

        X_train_outer, X_train, X_val, X_test = X_train_outer[:, col_idxs], X_train[:, col_idxs], \
            X_val[:, col_idxs], X_test[:, col_idxs]

        ## Horsehoe BNN

        horseshoe_config_path = f"{save_dir}/configs/horseshoe_bnn_config_s_{seed}_optuna.pkl"
        if os.path.exists(horseshoe_config_path) and saved_config:
            with open(horseshoe_config_path, "rb") as fp:
                horseshoe_config = pickle.load(fp)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(sampler=sampler)
            study.optimize(lambda trial: objective_horseshoe_bnn(trial, seed, horseshoe_config_file, X_train, X_val,
                                                                 y_train, y_val, epochs, num_hidden, drug_name),
                           n_trials=n_trial)

            horseshoe_config = study.best_params

            with open(f"{save_dir}/optuna/study_horseshoe_bnn_s_{seed}.pkl", "wb") as fp:
                pickle.dump(study, fp)
                fp.flush()


        batch_size = 16 if X_train.shape[0] < 200 else 32
        hidden_sizes = [num_hidden]*horseshoe_config["num_layers"]
        config = {"n_hidden": hidden_sizes, "classification": False}
        horseshoe_model, horseshoe_rmse, horseshoe_r2 =  run_horsehoe_bnn_model(seed, horseshoe_config_file, X_train_outer, X_test,
                                                                                y_train_outer, y_test, epochs,
                                                                                batch_size, config,
                                                                                drug_name, False)

        with open(f"{save_dir}/configs/horseshoe_bnn_config_s_{seed}_optuna.pkl", "wb") as fp:
                pickle.dump(horseshoe_config, fp)
                fp.flush()

        horseshoe_res_df_dict["seed"].append(seed)
        horseshoe_res_df_dict["model"].append("Horseshoe BNN")
        horseshoe_res_df_dict["test_rmse"].append(horseshoe_rmse)
        horseshoe_res_df_dict["test_r2_score"].append(horseshoe_r2)

        torch.save(horseshoe_model.state_dict(), f"{model_save_dir}/horseshoe_bnn_s_{seed}.pt")

        print(f"Horseshoe BNN scores - rmse: {horseshoe_rmse}, r2: {horseshoe_r2}")

        with open(f"{save_dir}/results/horseshoe_bnn_s_{seed}.csv", "w") as fp:
            pd.DataFrame(horseshoe_res_df_dict).to_csv(fp, index=False)
            fp.flush()

    return print("Done")

def zero_out_ranking(seeds, X, y, save_dir, model_save_dir, num_feats, **configs):
    epochs = configs["epochs"]
    act_fn = configs["act_fn"]
    prior_dist = configs["prior_dist"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pathlib.Path(f"{save_dir}/ft_importance").mkdir(parents=True, exist_ok=True)

    i = 0
    for seed in tqdm(seeds):
        bg_bnn_rf_res_dict = {"seed": [], "model": [], "num_feats": [], "test_rmse_score": []}

        transformer = StandardScaler()
        X_train_outer, X_train, X_val, X_test, \
            y_train_outer, y_train, y_val, y_test, (train_indices, val_indices, _) = preprocess_data(seed, X, y, None,
                                                                                                     transformer,
                                                                                                     val_size=0.2,
                                                                                                     test_size=0.2)

        J = np.load(f"{save_dir}/pandas/pandas_net_s_{seed}.npy")
        col_idxs = np.load(f"{save_dir}/pandas/pandas_col_idxs_s_{seed}.npy")

        X_train_outer, X_train, X_val, X_test = X_train_outer[:, col_idxs], X_train[:, col_idxs], \
            X_val[:, col_idxs], X_test[:, col_idxs]
        p = X_train_outer.shape[-1]

        J_zeros = np.zeros_like(J)
        ### BNN + BG

        bnn_bg_config = pickle.load(open(f"{save_dir}/configs/bg_bnn_config_s_{seed}_optuna.pkl", "rb"))

        num_cycles = 50
        batch_size = 16 if X_train.shape[0] < 200 else 32
        lr_0, disc_lr_0 = 1e-3, 0.5
        temp = bnn_bg_config["temp"]
        sigma_1, sigma_2 = 0.01, 1.0
        eta, mu = 1.0, 1.0
        hidden_sizes = [256]*bnn_bg_config["num_layers"]

        torch.manual_seed(seed)
        outer_data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size, shuffle=True,
                                        drop_last=True)

        bg_bnn_model = init_bnn_model(seed, outer_data_loader, epochs, lr_0, disc_lr_0,
                                            num_cycles, temp, sigma_1, sigma_2,
                                            hidden_sizes, J, eta, mu, get_act_fn(act_fn), prior_dist)

        params_bg_bnn, gammas_bg_bnn = load_model(model_save_dir, seed, True)

        w_mean_bg = jnp.mean(params_bg_bnn["dropout"]["w"], axis=0)
        w_norm_bg = jax.device_get(jax.vmap(lambda x: jnp.linalg.norm(x))(w_mean_bg))
        bg_bnn_feat_idx = np.argsort(w_norm_bg)[::-1]
        np.save(f"{save_dir}/ft_importance/bg_bnn_ft_importance_s_{seed}.npy", w_norm_bg)

        ##### BNN w/o BG
        bnn_config = pickle.load(open(f"{save_dir}/configs/bnn_config_s_{seed}_optuna.pkl", "rb"))

        temp = bnn_config["temp"]
        hidden_sizes = [256]*bnn_config["num_layers"]

        torch.manual_seed(seed)
        outer_data_loader = NumpyLoader(NumpyData(X_train_outer, y_train_outer), batch_size=batch_size, shuffle=True,
                                        drop_last=True)
        bnn_model = init_bnn_model(seed, outer_data_loader, epochs, lr_0, disc_lr_0,
                                         num_cycles, temp, sigma_1, sigma_2,
                                         hidden_sizes, J_zeros, eta, mu, get_act_fn(act_fn), prior_dist)

        params_bnn, gammas_bnn = load_model(model_save_dir, seed, False)

        w_mean_bnn = jnp.mean(params_bnn["dropout"]["w"], axis=0)
        w_norm_bnn = jax.device_get(jax.vmap(lambda x: jnp.linalg.norm(x))(w_mean_bnn))
        bnn_feat_idx = np.argsort(w_norm_bnn)[::-1]
        np.save(f"{save_dir}/ft_importance/bnn_ft_importance_s_{seed}.npy", w_norm_bnn)

        rf_model_path = f"{save_dir}/checkpoints/rf_model_s_{seed}.pkl"

        rf_model = pickle.load(open(rf_model_path, "rb"))
        rf_feat_idx = np.argsort(rf_model.feature_importances_)[::-1]
        np.save(f"{save_dir}/ft_importance/rf_ft_importance_s_{seed}.npy", rf_model.feature_importances_)

        for num_feat in num_feats:
            ### BNN + BG
            rmse_bg_bnn = zero_out_score(bg_bnn_model, X_test, y_test, params_bg_bnn, gammas_bg_bnn,
                                            num_feat, bg_bnn_feat_idx)

            ## BNN w/o BG
            rmse_bnn = zero_out_score(bnn_model, X_test, y_test, params_bnn, gammas_bnn,
                                         num_feat, bnn_feat_idx)

            ## RF
            rf_mask = np.zeros(p)
            rf_mask[rf_feat_idx[:num_feat]] = 1.0
            X_test_rf_m = X_test @ np.diag(rf_mask)
            rmse_rf, _ = eval_sklearn_model(rf_model, X_test_rf_m, y_test)

            bg_bnn_rf_res_dict["seed"].append(seed)
            bg_bnn_rf_res_dict["model"].append("RF")
            bg_bnn_rf_res_dict["num_feats"].append(num_feat)
            bg_bnn_rf_res_dict["test_rmse_score"].append(rmse_rf)

            bg_bnn_rf_res_dict["seed"].append(seed)
            bg_bnn_rf_res_dict["model"].append("BNN w/o BG")
            bg_bnn_rf_res_dict["num_feats"].append(num_feat)
            bg_bnn_rf_res_dict["test_rmse_score"].append(rmse_bnn)

            bg_bnn_rf_res_dict["seed"].append(seed)
            bg_bnn_rf_res_dict["model"].append("BNN + BG")
            bg_bnn_rf_res_dict["num_feats"].append(num_feat)
            bg_bnn_rf_res_dict["test_rmse_score"].append(rmse_bg_bnn)

        pd.DataFrame(bg_bnn_rf_res_dict).to_csv(
            f"{save_dir}/results/feat_zero_out_comp_bnn_bg_rf_s_{seed}.csv", index=False)

        i += 1


def get_feature_importance(seeds, save_dir):
    res_dict = {"BNN w/o BG": [], "BNN + BG": [], "RF": []}
    for seed in seeds:
        res_dict["BNN + BG"].append(np.load(f"{save_dir}/ft_importance/bg_bnn_ft_importance_s_{seed}.npy"))
        res_dict["BNN w/o BG"].append(np.load(f"{save_dir}/ft_importance/bnn_ft_importance_s_{seed}.npy"))
        res_dict["RF"].append(np.load(f"{save_dir}/ft_importance/rf_ft_importance_s_{seed}.npy"))

    for k in res_dict:
        res_dict[k] = np.mean(res_dict[k], axis=0)

    return res_dict


def save_gene_rank_files(seeds, gene_symbols, save_dir, drug_name,
                         normalize=False):

    ft_importance_dict = get_feature_importance(seeds, save_dir)
    gene_symbols = np.array(gene_symbols)
    rnk_path = f"{save_dir}/rnk"
    pathlib.Path(rnk_path).mkdir(exist_ok=True, parents=True)
    for model in ft_importance_dict:
        ft_weights = ft_importance_dict[model]
        if normalize:
            ft_weights /= sum(ft_weights)  # normalize the weights
        ranked_ft_idx = np.argsort(ft_weights)[::-1]
        ranked_ft_weights = ft_weights[ranked_ft_idx]
        ranked_gene_syms = gene_symbols[ranked_ft_idx]
        res_dict = {"gene": [], "score": []}

        for gene, score in zip(ranked_gene_syms, ranked_ft_weights):
            res_dict["gene"].append(gene)
            res_dict["score"].append(score)

        res_df = pd.DataFrame(res_dict)
        if model == "BNN w/o BG":
            res_df.to_csv(f"{rnk_path}/bnn_ranked_genes_{drug_name}.rnk",
                          index=False, header=None, sep="\t")

        elif model == "BNN + BG":
            res_df.to_csv(f"{rnk_path}/bnn_bg_ranked_genes_{drug_name}.rnk",
                          index=False, header=None, sep="\t")

        elif model == "RF":
            res_df.to_csv(f"{rnk_path}/rf_ranked_genes_{drug_name}.rnk",
                          index=False, header=None, sep="\t")
        else:
            raise ValueError(f"Unsupported model {model}")


def get_inferred_network(X_train_outer,
                         tissue_motif_data, string_ppi, hgnc_map,
                         gene_list, laplacian=False):
    ens_syms = []
    for i, hgnc_sym in enumerate(gene_list):
        try:
            ens_syms.append(hgnc_map[hgnc_sym])
        except KeyError:
            continue

    X_ensmbl = pd.DataFrame(X_train_outer, columns=ens_syms)
    X_ensmbl = X_ensmbl.T
    panda_obj = Panda(X_ensmbl, tissue_motif_data, string_ppi,
                      save_memory=True,
                      computing="gpu", modeProcess="legacy", remove_missing=True)

    J = panda_obj.correlation_matrix
    np.fill_diagonal(J, 0.)
    if laplacian:
        J = sparse.csgraph.laplacian(J)

    X_ensmbl = pd.DataFrame(X_train_outer, columns=ens_syms)
    col_idxs = []
    bg_cols = panda_obj.panda_network.columns.to_list()
    for col in bg_cols:
        col_idxs.append(X_ensmbl.columns.get_loc(col))

    return J, col_idxs