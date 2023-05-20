#!/home/abdu/miniconda3/bin/python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from exp_utils import *
import pathlib
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.sparse import csgraph
import warnings
from data_utils import *
import functools
warnings.filterwarnings("ignore")
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments on GDSC drug sensitivity data")
    parser.add_argument("--exp_dir", type=str, default="/home/abdu/bio_ai/moses-incons-pen-xp/data/exp_data_5/cancer/gdsc",
                        help="Path to the directory where the experiment data will be saved")
    parser.add_argument("--data_dir", type=str, default="/home/abdu/bio_ai/moses-incons-pen-xp/data",
                        help="Path to the directory where the data is stored. Each seed should be in a separate line")
    parser.add_argument("--seeds", type=str, help="Path to the file containing the seeds")
    parser.add_argument("--drug_ids", type=str, default="1814,1007,1558,1199,1191,1089",
                        help="Comma separated list of drug ids")
    parser.add_argument("--version", type=str, default="1a",
                        help="Version of the current experiment - useful for tracking experiments")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--num_hidden", type=int, default=256, help="Number of hidden units in each layer")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--prior_dist", default="laplace", const="laplace", nargs="?", choices=["laplace", "normal", "student_t"]
                        ,help="Prior distribution for the weights. Options: laplace, normal, student_t")
    parser.add_argument("--act_fn", default="swish", const="swish", nargs="?", choices=["swish", "relu", "tanh", "sigmoid"],
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


# drug_ids = [1814, 1007, 1558, 1199, 1191, 1089,
#             1168, 1013, 1088, 1085] # Nelarabine, Docetaxel, Lapatinib ,
                                                # Tamoxifen, Bortezomib, Oxaliplatin, Erlotinib, Nilotinib,
                                                # Irinotecan, Soraefnib

#new_drug_ids = [1372, 1014, 1047, 1008, 1373, 1093, 1080, 2172, 1084, 1086] # "Trametinib", "Refametinib",
                                                                         # "Nutlin-3a (-)", "Methotrexate", "Dabrafenib",
                                                                        # "GSK1904529A","Paclitaxel", "JQ1",
                                                                        # "Rapamycin", "BI-2536"

def run_single_drug(drug_id, args, seeds):
    curr_seeds = seeds[:args.num_folds]
    # curr_seeds = seeds
    data_dir = args.data_dir
    exp_dir = args.exp_dir

    scale_output = int(args.scale_output) == 1
    saved_config = int(args.saved_config) == 1
    use_horseshoe_bnn = int(args.horseshoe_bnn) == 1
    tissue_motif_data, string_ppi, hgnc2ens_map, X, target, \
    drug_name, save_dir, model_save_dir = load_gdsc_cancer_data(drug_id, data_dir, exp_dir,
                                                                log10_scale=scale_output)

    hp_configs = {"epochs": args.num_epochs, "act_fn": args.act_fn,
                  "beta": 0.25, "num_hidden": args.num_hidden,
                  "num_models": 1, "prior_dist": args.prior_dist,
                  "horseshoe_config_file": f"{data_dir}/uci/horseshoeBNN_config.yaml",
                  "drug_name": drug_name, "dropout_version": int(args.dropout_version),
                  "use_laplacian": int(args.use_laplacian) == 1,
                  "scale_output": scale_output, "data_dir": data_dir,
                  "saved_config": saved_config, "use_horseshoe_bnn": use_horseshoe_bnn}

    print(f"Configs: {hp_configs}")

    if use_horseshoe_bnn:
        cross_val_runs_v2(curr_seeds, X, target, tissue_motif_data, string_ppi,
                       hgnc2ens_map, args.version,
                       save_dir, model_save_dir,
                       timeout=args.timeout, n_trials=args.n_trials, **hp_configs)
    else:
        cross_val_runs(curr_seeds, X, target, tissue_motif_data, string_ppi,
                          hgnc2ens_map, args.version,
                          save_dir, model_save_dir,
                          timeout=args.timeout, n_trials=args.n_trials, **hp_configs)
    print(f"Done for drug: {drug_id}/{drug_name}")

if __name__ == "__main__":
    args = parse_args()
    seeds = []
    with open(args.seeds, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))
    # device_ids = [0,1,2,3]
    # drug_device_map = [(drug_id, device_ids[i % len(device_ids)]) for i, drug_id in enumerate(args.drug_ids.split(","))]
    drug_ids = [int(drug_id) for drug_id in  args.drug_ids.split(",")]
    exp_fn = functools.partial(run_single_drug, args=args, seeds=seeds)
    pool = Pool(len(drug_ids))
    pool.map(exp_fn, drug_ids)
    pool.close()
    pool.join()