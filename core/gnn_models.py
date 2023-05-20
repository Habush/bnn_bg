
"""Definition of the GNN model."""

from typing import Callable, Sequence, Tuple
import jax
import haiku as hk
import jax.numpy as jnp
import jraph


def add_graphs_tuples(graphs: jraph.GraphsTuple,
                      other_graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(
        nodes=graphs.nodes + other_graphs.nodes,
        edges=graphs.edges + other_graphs.edges,
        globals=graphs.globals + other_graphs.globals)


def attention_logit_fn(sender_attr: jnp.ndarray, receiver_attr: jnp.ndarray,
                       edges: jnp.ndarray) -> jnp.ndarray:
    """Attention logit function for Graph Attention Networks."""
    del edges
    x = jnp.concatenate((sender_attr, receiver_attr), axis=1)
    return hk.Linear(1)(x)


def add_self_edges_fn(receivers: jnp.ndarray, senders: jnp.ndarray,
                      total_num_nodes: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Adds self edges. Assumes self edges are not in the graph yet."""
    receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)), axis=0)
    senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
    return receivers, senders


class MLP(hk.Module):
    """A multi-layer perceptron."""

    def __init__(self,
                 feature_sizes: Sequence[int],
                 dropout_rate: float = 0.,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
                 name: str = None):
        super().__init__(name)
        self.feature_sizes = feature_sizes
        self.dropout_rate = dropout_rate
        self.activation = activation

    def __call__(self, inputs):
        key = hk.next_rng_key()
        x = inputs
        for size in self.feature_sizes:
            x = hk.Linear(size)(x)
            x = self.activation(x)
            x = hk.dropout(key, self.dropout_rate, x)
        return x


class GraphNet(hk.Module):
    """A complete Graph Network model defined with Jraph."""

    def __init__(self,
                latent_size: int,
                num_mlp_layers: int,
                message_passing_steps: int,
                output_globals_size: int,
                dropout_rate: float = 0,
                skip_connections: bool = True,
                use_edge_model: bool = True,
                layer_norm: bool = True,
                name: str = None):

        super().__init__(name)
        self.latent_size = latent_size
        self.num_mlp_layers = num_mlp_layers
        self.message_passing_steps = message_passing_steps
        self.output_globals_size = output_globals_size
        self.dropout_rate = dropout_rate
        self.skip_connections = skip_connections
        self.use_edge_model = use_edge_model
        self.layer_norm = layer_norm
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # We will first linearly project the original features as 'embeddings'.
        embedder = jraph.GraphMapFeatures(
            embed_node_fn=hk.Linear(self.latent_size),
            embed_edge_fn=hk.Linear(self.latent_size),
            embed_global_fn=hk.Linear(self.latent_size))
        processed_graphs = embedder(graphs)

        # Now, we will apply a Graph Network once for each message-passing round.
        mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
        for _ in range(self.message_passing_steps):
            if self.use_edge_model:
                update_edge_fn = jraph.concatenated_args(
                    MLP(mlp_feature_sizes,
                        dropout_rate=self.dropout_rate))
            else:
                update_edge_fn = None

            update_node_fn = jraph.concatenated_args(
                MLP(mlp_feature_sizes,
                    dropout_rate=self.dropout_rate))
            update_global_fn = jraph.concatenated_args(
                MLP(mlp_feature_sizes,
                    dropout_rate=self.dropout_rate))

            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn)

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(
                    graph_net(processed_graphs), processed_graphs)
            else:
                processed_graphs = graph_net(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True,
									   create_offset=True)(processed_graphs.nodes),
                    edges=hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True,
									   create_offset=True)(processed_graphs.edges),
                    globals=hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True,
									   create_offset=True)(processed_graphs.globals))

        # Since our graph-level predictions will be at globals, we will
        # decode to get the required output logits.
        decoder = jraph.GraphMapFeatures(
            embed_global_fn=hk.Linear(self.output_globals_size))
        processed_graphs = decoder(processed_graphs)

        return processed_graphs


class GraphConvNet(hk.Module):
    """A Graph Convolution Network + Pooling model defined with Jraph."""

    def __init__(self,
                 latent_size: int,
                 num_mlp_layers: int,
                 message_passing_steps: int,
                 output_globals_size: int,
                 dropout_rate: float = 0,
                 skip_connections: bool = True,
                 use_edge_model: bool = True,
                 layer_norm: bool = True,
                 pooling_fn: Callable[
                            [jnp.ndarray, jnp.ndarray, jnp.ndarray],  # pytype: disable=annotation-type-mismatch  # jax-ndarray
                            jnp.ndarray] = jraph.segment_mean,
                 name: str = None):

        super().__init__(name)
        self.latent_size = latent_size
        self.num_mlp_layers = num_mlp_layers
        self.message_passing_steps = message_passing_steps
        self.output_globals_size = output_globals_size
        self.dropout_rate = dropout_rate
        self.skip_connections = skip_connections
        self.use_edge_model = use_edge_model
        self.layer_norm = layer_norm
        self.pooling_fn = pooling_fn
    def pool(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Pooling operation, taken from Jraph."""

        # Equivalent to jnp.sum(n_node), but JIT-able.
        sum_n_node = graphs.nodes.shape[0]  # pytype: disable=attribute-error  # jax-ndarray
        # To aggregate nodes from each graph to global features,
        # we first construct tensors that map the node to the corresponding graph.
        # Example: if you have `n_node=[1,2]`, we construct the tensor [0, 1, 1].
        n_graph = graphs.n_node.shape[0]
        node_graph_indices = jnp.repeat(
            jnp.arange(n_graph),
            graphs.n_node,
            axis=0,
            total_repeat_length=sum_n_node)
        # We use the aggregation function to pool the nodes per graph.
        pooled = self.pooling_fn(graphs.nodes, node_graph_indices,
                                 n_graph)  # pytype: disable=wrong-arg-types  # jax-ndarray
        return graphs._replace(globals=pooled)

    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # We will first linearly project the original node features as 'embeddings'.
        embedder = jraph.GraphMapFeatures(
            embed_node_fn=hk.Linear(self.latent_size))
        processed_graphs = embedder(graphs)

        # Now, we will apply the GCN once for each message-passing round.
        for _ in range(self.message_passing_steps):
            mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
            update_node_fn = jraph.concatenated_args(
                MLP(mlp_feature_sizes,
                    dropout_rate=self.dropout_rate))
            graph_conv = jraph.GraphConvolution(
                update_node_fn=update_node_fn, add_self_edges=True)

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(
                    graph_conv(processed_graphs), processed_graphs)
            else:
                processed_graphs = graph_conv(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True,
									   create_offset=True)(processed_graphs.nodes), )

        # We apply the pooling operation to get a 'global' embedding.
        processed_graphs = self.pool(processed_graphs)

        # Now, we decode this to get the required output logits.
        decoder = jraph.GraphMapFeatures(
            embed_global_fn=hk.Linear(self.output_globals_size))
        processed_graphs = decoder(processed_graphs)

        return processed_graphs


class GraphAttNet(hk.Module):
    """A Graph Attention Network + Pooling model defined with Jraph."""
    def __init__(self,
                 latent_size: int,
                 num_mlp_layers: int,
                 message_passing_steps: int,
                 output_globals_size: int,
                 dropout_rate: float = 0,
                 skip_connections: bool = True,
                 use_edge_model: bool = True,
                 layer_norm: bool = True,
                 pooling_fn: Callable[
                            [jnp.ndarray, jnp.ndarray, jnp.ndarray],  # pytype: disable=annotation-type-mismatch  # jax-ndarray
                            jnp.ndarray] = jraph.segment_mean,
                 name: str = None):

        super().__init__(name)
        self.latent_size = latent_size
        self.num_mlp_layers = num_mlp_layers
        self.message_passing_steps = message_passing_steps
        self.output_globals_size = output_globals_size
        self.dropout_rate = dropout_rate
        self.skip_connections = skip_connections
        self.use_edge_model = use_edge_model
        self.layer_norm = layer_norm
        self.pooling_fn = pooling_fn
    def pool(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Pooling operation, taken from Jraph."""

        # Equivalent to jnp.sum(n_node), but JIT-able.
        sum_n_node = graphs.nodes.shape[0]  # pytype: disable=attribute-error  # jax-ndarray
        # To aggregate nodes from each graph to global features,
        # we first construct tensors that map the node to the corresponding graph.
        # Example: if you have `n_node=[1,2]`, we construct the tensor [0, 1, 1].
        n_graph = graphs.n_node.shape[0]
        node_graph_indices = jnp.repeat(
            jnp.arange(n_graph),
            graphs.n_node,
            axis=0,
            total_repeat_length=sum_n_node)
        # We use the aggregation function to pool the nodes per graph.
        pooled = self.pooling_fn(graphs.nodes, node_graph_indices,
                                 n_graph)  # pytype: disable=wrong-arg-types  # jax-ndarray
        return graphs._replace(globals=pooled)

    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # We will first linearly project the original node features as 'embeddings'.

        # # Add self edges to the graph.
        _, _, receivers, senders, _, _, _ = graphs
        total_num_nodes = graphs.nodes.shape[0]
        receivers, senders = add_self_edges_fn(receivers, senders, total_num_nodes)
        graphs = graphs._replace(receivers=receivers, senders=senders)

        # We will first linearly project the original node features as 'embeddings'.
        embedder = jraph.GraphMapFeatures(
            embed_node_fn=hk.Linear(self.latent_size))
        processed_graphs = embedder(graphs)
        # Now, we will apply the GAT once for each message-passing round.
        for _ in range(self.message_passing_steps):
            mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
            update_node_fn = jraph.concatenated_args(
                MLP(mlp_feature_sizes,
                    dropout_rate=self.dropout_rate))

            gat = jraph.GAT(attention_query_fn=update_node_fn,
                            attention_logit_fn=attention_logit_fn)
            if self.skip_connections:
                processed_graphs = add_graphs_tuples(
                    gat(processed_graphs), processed_graphs)
            else:
                processed_graphs = gat(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True,
									   create_offset=True)(processed_graphs.nodes))

        # We apply the pooling operation to get a 'global' embedding.
        processed_graphs = self.pool(processed_graphs)

        # Now, we decode this to get the required output logits.
        decoder = jraph.GraphMapFeatures(
            embed_global_fn=hk.Linear(self.output_globals_size))
        processed_graphs = decoder(processed_graphs)

        return processed_graphs