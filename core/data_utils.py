# Author Abdulrahman S. Omar <xabush@singularitynet.io>
import jax
import numpy as np
import jax.numpy as jnp

def generate_synthetic_data(*, key, num_tf, num_genes,
                            tf_on, num_samples, binary,
                            val_tf=4, random_index=False):

    assert tf_on == len(val_tf), "Number of on TFs should be equal to the lengths of the val_tf array!"
    p = num_tf + (num_tf * num_genes)

    keys = jax.random.split(key, num_tf)

    def generate_tfs(key):
        tf = jax.random.normal(key=key, shape=(num_samples, ))
        return tf

    def generate_genes(key, tf):
        key_rmh = jax.random.split(key, num_genes)

        def generate_single_gene(key):
            gene = 0.7*tf + 0.51*jax.random.normal(key=key, shape=(num_samples,))
            return gene

        genes = jax.vmap(generate_single_gene)(key_rmh)

        return genes

    tfs = jax.vmap(generate_tfs)(keys)
    genes = jax.vmap(generate_genes)(keys, tfs)

    print(f"tfs: {tfs.shape}, genes: {genes.shape}")

    key_tf, key_genes = jax.random.split(key, 2)

    if random_index:
        idx_on = jax.random.choice(key_tf, jnp.arange(num_tf), shape=(tf_on, ), replace=False)
    else:
        idx_on = jnp.arange(tf_on)

    betas = np.zeros(p)

    X = np.zeros((num_samples, p))

    val_gene = val_tf/np.sqrt(10)

    k = num_genes + 1

    for i in range(num_tf):
        X[:,i] = tfs[i]
        for j in range(i+1, i+k):
            X[:,j] = genes[i, j]

    # num_pos_reg = int(num_genes*perc_pos)
    # if perc_pos < 1:
    #     pos_reg_idx = jax.random.choice(key_genes, jnp.arange(num_genes), shape=(num_pos_reg, ))
    # else:


    for i in range(tf_on):
        idx = idx_on[i]*k
        betas[idx] = val_tf[i]
        for j in range(idx+1, idx+k):
            # if j in pos_reg_idx: # positively regulated gene
            #     betas = betas.at[j].set(val_gene)
            #
            # else: # negatively regulated gene
            #     betas = betas.at[j].set(-val_gene)
            betas[j] = val_gene[i]



    y = jnp.dot(X, betas)

    if binary: # return classification data
        p = jax.nn.sigmoid(y)
        y = (jax.vmap(jax.random.bernoulli, in_axes=(None, 0))(key, p))*1.
    else:
        sigma = num_genes / num_tf
        err = sigma*jax.random.normal(key, shape=(num_samples,))
        y = y + err

    return X, y, betas, idx_on

def get_assoc_mat(*, num_tf, num_genes, corr=1.):
    feats = num_tf + (num_tf * num_genes)
    assoc_mat = np.eye(feats, feats)
    m = num_genes + 1
    for t in range(0, m * num_tf, m):
        for g in range(t + 1, t + m):
            assoc_mat[t, g] = corr
            assoc_mat[g, t] = corr
    return assoc_mat