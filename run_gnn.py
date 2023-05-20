#!/home/abdu/miniconda3/bin/python3
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import jraph
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import haiku as hk
from typing import Any, Tuple, Dict, List, Optional, Callable, Sequence
import functools
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from data_utils import *
import gnn_models

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
    edges = jnp.asarray(edges)[:,None]
    return senders, receivers, edges


def build_cell_line_graph(X, y, senders, receivers, edges):
    assert X.shape[0] == y.shape[0], "X and y must have the same sample sizes"
    graphs = []
    n, p = X.shape
    n_nodes, n_edges = p, len(edges)
    for i in range(n):
        row, target = X[i], y[i]
        nodes = jnp.array(row[:,None])
        graph = jraph.GraphsTuple(
            n_node=jnp.asarray([n_nodes]),
            n_edge=jnp.asarray([n_edges]),
            nodes=nodes, edges=edges,
            senders=senders, receivers=receivers,
            globals=jnp.array([[0.]]) #set it to zero initially,
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

# def graph_fn(graph: jraph.GraphsTuple, hidden_size: List[int]) -> jraph.GraphsTuple:
#     # Embed the graph features
#     embedder = jraph.GraphMapFeatures(
#         embed_node_fn=hk.Linear(hidden_size[0]),
#         embed_edge_fn=hk.Linear(hidden_size[0]),
#         embed_global_fn=hk.Linear(hidden_size[0]))
#     net = jraph.GraphNetwork(
#         update_node_fn=make_node_update_fn(hidden_size),
#         update_edge_fn=make_edge_update_fn(hidden_size),
#         update_global_fn=make_global_update_fn(hidden_size))
#     return net(embedder(graph))

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

    """compute loss and rmse"""
    pred_graph = net.apply(params, rng, graph)
    preds = pred_graph.globals.squeeze()
    # MSE loss
    # mse = jnp.mean((target - preds)**2)
    mae = jnp.abs(target - preds)
    # negative b/c optax updates are additive by default
    # rmse = jnp.sqrt(mse)
    return mae, mae


# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def train(rng:jax.random.PRNGKey, dataset: List[Dict[str, Any]],
          hidden_size: Sequence[int], num_train_steps: int,
          lr:float, message_passing_steps: int, skip_connections: bool = True,
          layer_norm: bool = True, num_class: int = 1, net_type: str = "gat",
          dropout: float = 0.0, patience: int = 20, show_pgbar: bool = True,) -> Tuple[hk.Transformed ,hk.Params]:
  """Training loop."""

  # Transform impure `net_fn` to pure functions with hk.transform.
  net_fn = functools.partial(graph_fn, hidden_size=hidden_size,
                             message_passing_steps=message_passing_steps,dropout_rate=dropout,
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

  return net , best_params

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

  # print('Completed evaluation.')
  targets = jnp.array(targets)
  preds = jnp.array(preds)
  rmse = jnp.sqrt(jnp.mean((targets - preds)**2))
  return rmse


def run_gnn(seeds, save_dir, model_save_dir, version,
            X, y, **model_configs):
    """Run GNN on the dataset"""

    hidden_size = model_configs["hidden_size"]
    num_train_steps = model_configs["epochs"]
    lr = model_configs["lr"]
    message_passing_steps = model_configs["message_passing_steps"]
    skip_connections = model_configs["skip_connections"]
    layer_norm = model_configs["layer_norm"]
    net_types = model_configs["net_type"]

    for seed in tqdm(seeds):
        rng = jax.random.PRNGKey(seed)
        res_dict = {"seed": [], "model": [], "test_rmse": []}
        transformer = StandardScaler()
        X_train_outer, _, _, X_test, \
            y_train_outer, _, _, y_test, (_, _, _) = preprocess_data(seed, X, y, None, transformer,
                                                                     val_size=0.2, test_size=0.2)
        J = np.load(f"{save_dir}/pandas/pandas_net_s_{seed}.npy")
        col_idxs = np.load(f"{save_dir}/pandas/pandas_col_idxs_s_{seed}.npy")
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

def main(drug_id, config, version):
    _, _, _, X, target, \
    drug_name, save_dir, model_save_dir = load_gdsc_cancer_data(drug_id, data_dir, exp_dir, log10_scale=False)

    print(f"Running for drug {drug_name}({drug_id})...")
    run_gnn(seeds, save_dir, model_save_dir, version, X, target, **config)

    print(f"Done for drug {drug_name} ({drug_id})")

if __name__ == "__main__":
    seeds = [422,261,968,282,739,573,220,413,745,775,482,442,210,423,760,57,769,920,226,196]
    version = "4d"
    # drug_ids = [1814, 1007, 1558, 1199, 1191, 1089, 1168, 1013, 1088, 1085]
    drug_ids = [1080, 1084] # Paclitaxel, Rapamycin

    data_dir = "/home/abdu/bio_ai/moses-incons-pen-xp/data"
    exp_dir = f"{data_dir}/exp_data_5/cancer/gdsc"

    config = {"hidden_size": [64, 64, 64], "epochs": 1000, "lr": 1e-3,
              "message_passing_steps": 3, "skip_connections": False,
              "layer_norm": True, "net_type": ["GAT"]}

    for drug_id in drug_ids:
        main(drug_id, config, version)