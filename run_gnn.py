#!/usr/bin/python3
import argparse
import functools
import pickle
from typing import Any, Dict, List, Callable, Tuple, Sequence

import optax
import haiku as hk
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from core import gnn_models
from utils.data_utils import *
from utils.drug_exp_utils import get_inferred_network
import jraph

def get_edges_from_adj_matrix(J):
    p = J.shape[0]
    senders = []
    receivers = []
    edges = []
    for i in range(p):
        for j in range(p):
            if i != j:
                w = J[i, j]
                if w != 0:
                    senders.append(i)
                    receivers.append(j)
                    edges.append(w)

    senders = jnp.asarray(senders)
    receivers = jnp.asarray(receivers)
    edges = jnp.asarray(edges)[:, None]
    return senders, receivers, edges


def build_cell_line_graph(X, y, senders, receivers, edges):
    assert X.shape[0] == y.shape[0], "X and y must have the same sample sizes"
    graphs = []
    n, p = X.shape
    n_nodes, n_edges = p, len(edges)
    for i in range(n):
        row, target = X[i], y[i]
        nodes = jnp.array(row[:, None])
        graph = jraph.GraphsTuple(
            n_node=jnp.asarray([n_nodes]),
            n_edge=jnp.asarray([n_edges]),
            nodes=nodes, edges=edges,
            senders=senders, receivers=receivers,
            globals=jnp.array([[0.]])  # set it to zero initially,
        )
        graphs.append({"input_graph": graph, "target": target})

    return graphs


# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py

def make_edge_update_fn(hidden_size: List[int]) -> Callable:
    """Make edge update function for graph net."""

    @jraph.concatenated_args
    def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
        """Edge update function for graph net"""
        for layer_size in hidden_size[:-1]:
            feats = hk.Linear(layer_size)(feats)
            feats = jax.nn.relu(feats)
        feats = hk.Linear(hidden_size[-1])(feats)
        return feats

    return edge_update_fn


def make_node_update_fn(hidden_size: List[int]) -> Callable:
    """Make node update function for graph net."""

    @jraph.concatenated_args
    def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
        """Node update function for graph net"""
        for layer_size in hidden_size[:-1]:
            feats = hk.Linear(layer_size)(feats)
            feats = jax.nn.relu(feats)
        feats = hk.Linear(hidden_size[-1])(feats)
        return feats

    return node_update_fn


def make_global_update_fn(hidden_size: List[int]) -> Callable:
    @jraph.concatenated_args
    def global_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
        """Global update function for graph net."""
        # It is a regression task so a single output
        # for layer_size in hidden_size[:-1]:
        #     feats = hk.Linear(layer_size)(feats)
        #     feats = jax.nn.relu(feats)
        #
        # feats = hk.Linear(hidden_size[-1])(feats)
        net = hk.Sequential(
            [hk.Linear(hidden_size[0]), jax.nn.relu,
             hk.Linear(1)])
        return net(feats)

    return global_update_fn


def graph_fn(graph: jraph.GraphsTuple, hidden_size: List[int],
             message_passing_steps: int, dropout_rate: float,
             skip_connections: bool = True, layer_norm: bool = True,
             num_classes: int = 1,
             net_type: str = "gat") -> jraph.GraphsTuple:
    if net_type == "gat":
        graph_net = gnn_models.GraphAttNet(
            latent_size=hidden_size[0],
            num_mlp_layers=len(hidden_size),
            message_passing_steps=message_passing_steps,
            output_globals_size=num_classes,
            dropout_rate=dropout_rate,
            skip_connections=skip_connections,
            layer_norm=layer_norm)
    elif net_type == "gcn":
        graph_net = gnn_models.GraphConvNet(
            latent_size=hidden_size[0],
            num_mlp_layers=len(hidden_size),
            message_passing_steps=message_passing_steps,
            output_globals_size=num_classes,
            dropout_rate=dropout_rate,
            skip_connections=skip_connections,
            layer_norm=layer_norm)

    elif net_type == "gn":
        graph_net = gnn_models.GraphNet(
            latent_size=hidden_size[0],
            num_mlp_layers=len(hidden_size),
            message_passing_steps=message_passing_steps,
            output_globals_size=num_classes,
            dropout_rate=dropout_rate,
            skip_connections=skip_connections,
            layer_norm=layer_norm)

    else:
        raise ValueError(f"Invalid net_type: {net_type}. Valid options are 'gat', 'gcn', 'gn'")

    return graph_net(graph)


def compute_loss(params: hk.Params, rng: jax.random.PRNGKey, graph: jraph.GraphsTuple,
                 target: jnp.ndarray, net: hk.Transformed) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """compute loss and mae"""
    pred_graph = net.apply(params, rng, graph)
    preds = pred_graph.globals.squeeze()
    mae = jnp.abs(target - preds)
    return mae, mae


# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def train(rng: jax.random.PRNGKey, dataset: List[Dict[str, Any]],
          hidden_size: Sequence[int], num_train_steps: int,
          lr: float, message_passing_steps: int, skip_connections: bool = True,
          layer_norm: bool = True, num_class: int = 1, net_type: str = "gat",
          dropout: float = 0.0, patience: int = 20, show_pgbar: bool = True, ) -> Tuple[hk.Transformed, hk.Params]:
    """Training loop."""

    # Transform impure `net_fn` to pure functions with hk.transform.
    net_fn = functools.partial(graph_fn, hidden_size=hidden_size,
                               message_passing_steps=message_passing_steps, dropout_rate=dropout,
                               skip_connections=skip_connections, layer_norm=layer_norm,
                               num_classes=num_class, net_type=net_type)
    net = hk.transform(net_fn)
    # Get a candidate graph and label to initialize the network.
    graph = dataset[0]['input_graph']

    # Initialize the network.
    params = net.init(rng, graph)
    # Initialize the optimizer.

    opt_init, opt_update = optax.adam(lr)
    opt_state = opt_init(params)

    compute_loss_fn = functools.partial(compute_loss, net=net)
    # We jit the computation of our loss, since this is the main computation.
    # Using jax.jit means that we will use a single accelerator. If you want
    # to use more than 1 accelerator, use jax.pmap. More information can be
    # found in the jax documentation.
    compute_loss_fn = jax.jit(jax.value_and_grad(
        compute_loss_fn, has_aux=True))

    best_params = params
    best_loss = 1e6

    if show_pgbar:
        pgbar = tqdm(range(num_train_steps))

    else:
        pgbar = range(num_train_steps)

    k = 0
    prev_loss = best_loss
    key = rng
    for idx in pgbar:
        _, key = jax.random.split(key)
        graph = dataset[idx % len(dataset)]['input_graph']
        target = dataset[idx % len(dataset)]['target']

        (loss, mae), grad = compute_loss_fn(params, key, graph, target)
        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        if idx % 100 == 0:
            if loss < best_loss:
                best_loss = loss
                best_params = params
            if show_pgbar:
                pgbar.set_description(f'step: {idx}, loss: {loss}, lst_mae: {best_loss}')

            if prev_loss < loss:
                k += 1
            prev_loss = loss

            if k > patience:
                break

    return net, best_params


def evaluate(rng: jax.random.PRNGKey, dataset: List[Dict[str, Any]],
             net: hk.Transformed,
             params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluation Script."""

    # Get a candidate graph and label to initialize the network.
    targets = []
    preds = []
    for idx in range(len(dataset)):
        graph = dataset[idx]['input_graph']
        target = dataset[idx]['target']
        targets.append(target)
        pred_graph = net.apply(params, rng, graph)
        pred = pred_graph.globals.squeeze()
        preds.append(pred)
    targets = jnp.array(targets)
    preds = jnp.array(preds)
    rmse = jnp.sqrt(jnp.mean((targets - preds) ** 2))
    return rmse

def run_gnn(seeds, tissue_motif_data, string_ppi, hgnc_map,
            save_dir, model_save_dir, version,
            X, y, **model_configs):
    """Run GNN on the dataset"""

    hidden_size = model_configs["hidden_size"]
    num_train_steps = model_configs["epochs"]
    lr = model_configs["lr"]
    message_passing_steps = model_configs["message_passing_steps"]
    skip_connections = model_configs["skip_connections"]
    layer_norm = model_configs["layer_norm"]
    net_types = model_configs["net_type"]

    gene_list = X.columns.to_list()

    for seed in tqdm(seeds):
        rng = jax.random.PRNGKey(seed)
        res_dict = {"seed": [], "model": [], "test_rmse": []}
        transformer = StandardScaler()
        X_train_outer, _, _, X_test, \
            y_train_outer, _, _, y_test, (_, _, _) = preprocess_data(seed, X, y, None, transformer,
                                                                     val_size=0.2, test_size=0.2)

        graph_path, col_idx_path = f"{save_dir}/pandas/pandas_net_s_{seed}.npy", f"{save_dir}/pandas/pandas_col_idxs_s_{seed}.npy"
        if os.path.exists(graph_path) and os.path.exists(col_idx_path):
            J = np.load(graph_path)
            col_idxs = np.load(col_idx_path)

        else:
            J, col_idxs = get_inferred_network(X_train_outer, tissue_motif_data, string_ppi, hgnc_map, gene_list)
            np.save(graph_path, J)
            np.save(col_idx_path, col_idxs)

        X_train_outer, X_test = X_train_outer[:, col_idxs], X_test[:, col_idxs]

        senders, receivers, edges = get_edges_from_adj_matrix(J)
        graphs_train = build_cell_line_graph(X_train_outer, y_train_outer, senders, receivers, edges)
        graphs_test = build_cell_line_graph(X_test, y_test, senders, receivers, edges)

        for net_type in net_types:
            net, params = train(rng, graphs_train,
                                hidden_size, num_train_steps, lr,
                                message_passing_steps=message_passing_steps, skip_connections=skip_connections,
                                layer_norm=layer_norm, num_class=1, net_type=net_type.lower(), dropout=0.,
                                show_pgbar=False)
            test_rmse = evaluate(rng, graphs_test, net, params)
            res_dict["seed"].append(seed)
            res_dict["model"].append(net_type)
            res_dict["test_rmse"].append(test_rmse)

            with open(f"{model_save_dir}/bg_gnn_s_{seed}_v{version}_{net_type.lower()}.pkl", "wb") as fp:
                pickle.dump(params, fp)
                fp.flush()

        with open(f"{save_dir}/results/bg_gnn_s_{seed}_v{version}.csv", "w") as fp:
            res_df = pd.DataFrame(res_dict)
            res_df.to_csv(fp, index=False)
            fp.flush()


drug_ids = [1007, 1558, 1199, 1191, 1089,
           1168, 1013, 1088, 1085, 1080, 1084]    # Docetaxel, Lapatinib , Tamoxifen
                                                # Bortezomib, Oxaliplatin, Erlotinib, Nilotinib,
                                                # Irinotecan, "Paclitaxel", "Rapamycin"
def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments on GDSC drug sensitivity data")
    parser.add_argument("--data_dir", type=str, default="./data/gdsc",
                        help="Path to the directory where the data is stored. Each seed should be in a separate line")
    parser.add_argument("--exp_dir", type=str, default="./data/gdsc/exps",
                        help="Path to the directory where the experiment data will be saved")
    parser.add_argument("--seeds", type=str, help="Path to the file containing the seeds")
    parser.add_argument("--version", type=str, default="1", help="Version of the current experiment - useful for "
                                                                 "tracking experiments")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--num_hidden", type=int, default=64, help="Number of hidden units in each layer")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--message_passing_steps", type=int, default=3, help="Number of message passing steps")
    parser.add_argument("--skip_connections", default='0', const='0', nargs='?', choices=['0', '1'])
    parser.add_argument("--layer_norm", default='1', const='1', nargs='?', choices=['0', '1'])
    parser.add_argument("--net_type", type=str, default="gat", choices=["gcn", "gat", "gn"])

    return parser.parse_args()

def main(drug_id, config, version):
    tissue_motif_data, string_ppi, hgnc2ens_map, \
        X, target, drug_name, save_dir, model_save_dir = load_gdsc_cancer_data(drug_id, data_dir, exp_dir)

    print(f"Running for drug {drug_name}({drug_id})...")
    run_gnn(seeds, tissue_motif_data, string_ppi, hgnc2ens_map,
            save_dir, model_save_dir, version, X, target, **config)

    print(f"Done for drug {drug_name} ({drug_id})")


if __name__ == "__main__":
    args = parse_args()
    version = args.version
    seeds = []
    with open(args.seeds, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))

    data_dir = args.data_dir
    exp_dir = args.exp_dir
    hidden_size = [args.num_hidden]*args.num_layers

    layer_norm = int(args.layer_norm) == 1
    skip_connections = int(args.skip_connections) == 1
    net_type = args.net_type.upper()

    config = {"hidden_size": hidden_size, "epochs": args.num_epochs, "lr": 1e-3,
              "message_passing_steps": args.message_passing_steps, "skip_connections": skip_connections,
              "layer_norm": layer_norm, "net_type": [net_type]}

    print(f"Running with config: {config}")
    for drug_id in drug_ids:
        main(drug_id, config, version)
