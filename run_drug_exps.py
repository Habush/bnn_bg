#!/usr/bin/python3

from utils.drug_exp_utils import *
from multiprocessing import Pool
import warnings
from utils.data_utils import load_gdsc_cancer_data
import functools
warnings.filterwarnings("ignore")
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments on GDSC drug sensitivity data")
    parser.add_argument("--data_dir", type=str, default="./data/gdsc",
                        help="Path to the directory where the data is stored. Each seed should be in a separate line")
    parser.add_argument("--exp_dir", type=str, default="./data/gdsc/exps",
                        help="Path to the directory where the experiment data will be saved")
    parser.add_argument("--seeds", type=str, help="Path to the file containing the seeds")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--num_hidden", type=int, default=256, help="Number of hidden units in each layer")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--prior_dist", default="laplace", const="laplace", nargs="?", choices=["laplace", "normal", "student_t"]
                        ,help="Prior distribution for the weights. Options: laplace, normal, student_t")
    parser.add_argument("--act_fn", default="swish", const="swish", nargs="?", choices=["swish", "relu"],
                        help="Activation function for the hidden layers. Options: swish, relu")

    parser.add_argument("--timeout", type=int, default=180, help="Timeout for hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for hyperparameter optimization")
    parser.add_argument("--saved_config", default='0', const='0', nargs='?', choices=['0', '1'],
                                    help="Whether to scale the output or not")

    parser.add_argument("--horseshoe_bnn", default='0', const='0', nargs='?', choices=['0', '1'],
                        help="Whether to use horseshoe BNN or not")


    return parser.parse_args()


drug_ids = [1007, 1558, 1199, 1191, 1089,
           1168, 1013, 1088, 1080, 1084]    # Docetaxel, Lapatinib , Tamoxifen
                                                # Bortezomib, Oxaliplatin, Erlotinib, Nilotinib,
                                                # Irinotecan, "Paclitaxel", "Rapamycin"



def run_single_drug(drug_id, args, seeds):

    data_dir = args.data_dir
    exp_dir = args.exp_dir


    saved_config = int(args.saved_config) == 1
    use_horseshoe_bnn = int(args.horseshoe_bnn) == 1
    tissue_motif_data, string_ppi, hgnc2ens_map, X, target, \
    drug_name, save_dir, model_save_dir = load_gdsc_cancer_data(drug_id, data_dir, exp_dir)

    hbnn_config_file = f"{data_dir}/horseshoeBNN_config.yaml"
    hp_configs = {"epochs": args.num_epochs, "act_fn": args.act_fn,
                  "beta": 0.25, "num_hidden": args.num_hidden,
                  "num_models": 1, "prior_dist": args.prior_dist,
                  "horseshoe_config_file": hbnn_config_file,
                  "drug_name": drug_name, "data_dir": data_dir,
                  "saved_config": saved_config, "use_horseshoe_bnn": use_horseshoe_bnn}

    print(f"Configs: {hp_configs}")

    if use_horseshoe_bnn:
        cross_val_horseshoe_bnn(seeds, X, target,
                       save_dir, model_save_dir,
                       timeout=args.timeout, n_trials=args.n_trials, **hp_configs)
    else:
        cross_val_runs(seeds, X, target, tissue_motif_data, string_ppi, hgnc2ens_map,
                              save_dir, model_save_dir,
                              timeout=args.timeout, n_trials=args.n_trials, **hp_configs)
    print(f"Done for drug: {drug_id}/{drug_name}")


if __name__ == "__main__":
    args = parse_args()
    seeds = []
    with open(args.seeds, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))

    for drug_id in drug_ids:
        exp_fn = functools.partial(run_single_drug, args=args, seeds=seeds)
        exp_fn(drug_id)