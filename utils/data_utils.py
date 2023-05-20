import jax
import jax.numpy as jnp
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch
import pathlib


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyData(data.Dataset):

    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.size = X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x, y = self.data[idx], self.target[idx]
        return x, y

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, device="cpu"):

        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn,
                                             generator=torch.Generator(device=device))

def fpkm_to_expr(data_path, gene_id_data, idx_col="model_id"):
    rna_seq_data = pd.read_csv(data_path)
    #Calculate log2 transformed values of TPM (Transcripts per Million) form fpkm
    tpm_data = rna_seq_data.groupby([idx_col])["fpkm"].transform(lambda x : np.log2(((x  / x.sum()) * 1e6) + 1)) # add pseudp-count of 1 to avoid taking log2(0)
    rna_seq_data["log2.tpm"] = tpm_data
    rna_seq_data["idx"] = rna_seq_data.groupby(idx_col).cumcount()
    exp_data = rna_seq_data.pivot(index=idx_col ,columns="idx", values="log2.tpm")
    gene_sym_data = pd.read_csv(gene_id_data, index_col="gene_id")
    gene_ids = rna_seq_data["gene_id"].unique()
    gene_syms = gene_sym_data.loc[gene_ids]["hgnc_symbol"]
    exp_data.columns = gene_syms
    return exp_data


def preprocess_data(seed, X, y, cancer_types, transformer=None,
                    val_size=0.2, test_size=0.2, reset_index=False,
                    normalize_output=False):
    X_train_outer_df, X_test_df, y_train_outer_df, y_test_df = train_test_split(X, y, random_state=seed,
                                                                                shuffle=True, test_size=test_size,
                                                                                stratify=cancer_types)

    if transformer is not None:
        train_transformer = transformer.fit(X_train_outer_df)

        train_transformed = train_transformer.transform(X_train_outer_df)
        test_transformed = train_transformer.transform(X_test_df)
        X_train_outer_df = pd.DataFrame(train_transformed, columns=X_train_outer_df.columns)
        X_test_df = pd.DataFrame(test_transformed, columns=X_test_df.columns)

    if reset_index:
       X_train_outer_df, y_train_outer_df = X_train_outer_df.reset_index().drop("index", axis=1), \
                                            y_train_outer_df.reset_index().drop("index", axis=1).squeeze()

    if normalize_output:
        mean_y_train_outer, std_y_train_outer = y_train_outer_df.mean(), y_train_outer_df.std()
        y_train_outer_df = (y_train_outer_df - mean_y_train_outer) / std_y_train_outer
        y_test_df = (y_test_df - mean_y_train_outer) / std_y_train_outer


    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(X_train_outer_df, y_train_outer_df, shuffle=True,
                                                                  random_state=seed, test_size=val_size)
    train_indices, val_indices, test_indices = X_train_df.index.to_list(), X_val_df.index.to_list(), X_test_df.index.to_list()

    X_train_outer, y_train_outer = X_train_outer_df.values, y_train_outer_df.values
    X_train, y_train = X_train_df.values, y_train_df.values
    X_val, y_val = X_val_df.values, y_val_df.values
    X_test, y_test = X_test_df.values, y_test_df.values

    if normalize_output:
        return X_train_outer, X_train, X_val, X_test, y_train_outer, y_train, y_val, y_test, \
            (train_indices, val_indices, test_indices), (mean_y_train_outer, std_y_train_outer)

    return X_train_outer, X_train, X_val, X_test, y_train_outer, \
           y_train, y_val, y_test, (train_indices, val_indices, test_indices)


def load_gdsc_cancer_data(drug_id, data_dir, exp_dir):

    tissue_motif_data = pd.read_table(f"{data_dir}/tissues_motif.txt",
                                      header=None)
    string_ppi = pd.read_csv(f"{data_dir}/string_ppi.csv", header=None, skiprows=1)
    hgnc_data = pd.read_table(f"{data_dir}/hgnc_to_ensemble_map.txt")
    hgnc2ens_map = dict(zip(hgnc_data["Approved symbol"], hgnc_data["Ensembl gene ID"]))
    gdsc_exp_data = pd.read_csv(f"{data_dir}/gdsc_gene_expr.csv", index_col="model_id")
    cols = gdsc_exp_data.columns.to_list()
    print(f"Number of genes in GDSC data: {len(cols)}")
    cancer_driver_genes_df = pd.read_csv(f"{data_dir}/driver_genes_20221018.csv")
    landmark_genes = pd.read_table(f"{data_dir}/cmap_L1000_genes.txt")["Symbol"].to_list()
    driver_syms = cancer_driver_genes_df["symbol"].to_list()
    drug_metabolism_list = []
    with open(f"{data_dir}/drugMetabolismGenes.txt", "r") as fp:
        for line in fp.readlines():
            drug_metabolism_list.append(line.strip())

    all_genes = set(driver_syms) | set(landmark_genes) | set(drug_metabolism_list)
    intr_genes = list(set(all_genes) & set(cols))
    intr_genes = sorted(list(set(intr_genes) & set(hgnc2ens_map.keys())))  # sorting the genes here is very important
    # for model reproducibility
    gdsc_exp_data_sel = gdsc_exp_data[intr_genes]

    gdsc_response_data = pd.read_csv(f"{data_dir}/GDSC2_fitted_dose_response_24Jul22.csv",
                                     index_col="SANGER_MODEL_ID")
    drug_response_data = gdsc_response_data[gdsc_response_data["DRUG_ID"] == drug_id]
    drug_name = drug_response_data["DRUG_NAME"].iloc[0].lower()

    drug_exp_response = pd.merge(gdsc_exp_data_sel, drug_response_data["LN_IC50"], left_index=True, right_index=True)
    print(f"Starting exp for Drug id: {drug_id}/{drug_name}")
    print(f"Total samples for drug {drug_id}/{drug_name}: {drug_exp_response.shape[0]}")

    save_dir = f"{exp_dir}/{drug_name}"
    model_save_dir = f"{exp_dir}/nn_checkpoints/{drug_name}"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/configs").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/checkpoints").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/dropout").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/pandas").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/optuna").mkdir(parents=True, exist_ok=True)

    X, target = drug_exp_response.iloc[:, :-1], drug_exp_response.iloc[:, -1]

    return tissue_motif_data, string_ppi, hgnc2ens_map, X, target, drug_name, save_dir, model_save_dir