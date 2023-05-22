#!/usr/bin/python3
import os
import jax
from utils.drug_exp_utils import *
from multiprocessing import Pool
import warnings
import argparse
from utils.data_utils import *
import functools
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments on GDSC drug sensitivity data")
    parser.add_argument("--data_dir", type=str, default="./data/gdsc",
                        help="Path to the directory where the data is stored. Each seed should be in a separate line")
    parser.add_argument("--exp_dir", type=str, default="./data/gdsc/exps",
                        help="Path to the directory where the experiment data will be saved")
    parser.add_argument("--seeds", type=str, help="Path to the file containing the seeds")
    parser.add_argument("--version", type=str, default="1",
                        help="Version of the current experiment - useful for tracking experiments")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--num_hidden", type=int, default=256, help="Number of hidden units in each layer")
    parser.add_argument("--prior_dist", default="laplace", const="laplace", nargs="?", choices=["laplace", "normal", "student_t"]
                        ,help="Prior distribution for the weights. Options: laplace, normal, student_t")
    parser.add_argument("--act_fn", default="swish", const="swish", nargs="?", choices=["swish", "relu"],
                        help="Activation function for the hidden layers. Options: swish, relu")


    return parser.parse_args()
def run_single_drug(drug_id, args, seeds):
    data_dir = args.data_dir
    exp_dir = args.exp_dir

    _, _, _, X, target, \
        drug_name, save_dir, model_save_dir = load_gdsc_cancer_data(drug_id, data_dir, exp_dir)

    hp_configs = {"epochs": args.num_epochs, "act_fn": args.act_fn,
                  "beta": 0.25, "num_hidden": args.num_hidden,
                  "num_models": 1, "prior_dist": args.prior_dist,
                  "horseshoe_config_file": f"{data_dir}/uci/horseshoeBNN_config.yaml",
                  "drug_name": drug_name,
                  "data_dir": data_dir}

    print(hp_configs)
    num_feats = [10, 20, 30, 40, 50]


    zero_out_ranking(seeds, X, target, args.version, save_dir, model_save_dir, num_feats,
                        **hp_configs)
    print(f"Done for drug: {drug_id}/{drug_name}")

drug_ids = [1007, 1558, 1199, 1191, 1089,
           1168, 1013, 1088, 1085, 1080, 1084]    # Docetaxel, Lapatinib , Tamoxifen
                                                # Bortezomib, Oxaliplatin, Erlotinib, Nilotinib,
                                                # Irinotecan, "Paclitaxel", "Rapamycin"

if __name__ == "__main__":
    args = parse_args()
    seeds = []
    with open(args.seeds, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))

    exp_fn = functools.partial(run_single_drug, args=args, seeds=seeds)
    for drug_id in drug_ids:
        exp_fn(drug_id)