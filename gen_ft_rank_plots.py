#!/usr/bin/python3
import argparse
import matplotlib.pyplot as plt
from utils.drug_exp_utils import get_feature_ranking_summary
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate feature ranking plots for the experiments")
    parser.add_argument("--exp_dir", type=str, help="Path to the directory where the experiment data is stored")
    parser.add_argument("--seeds", type=str, help="Path to the file containing the seeds")
    parser.add_argument("--save_dir", type=str, help="Path to the directory where the figure will be saved")
    parser.add_argument("--num_cols", type=int, default=3, help="Number of columns in the figure")

    return parser.parse_args()

def plot_feature_ranking_res(seeds, drug_names, exp_dir, save_dir, cols=3):

    models = ["BNN + BG", "BNN w/o BG", "RF"]
    num_feats = [10, 20, 30, 40, 50]

    res_dict = {}
    for drug_name in drug_names:
        res_dict[drug_name] = {m: [] for m in models}

    for k in num_feats:
        ft_rank_res = get_feature_ranking_summary(seeds, drug_names, exp_dir, k=k)
        for drug in drug_names:
            for model in models:
                res = float(ft_rank_res.loc[drug]["summary"][model].split("±")[0].strip())
                res_dict[drug][model].append(res)

    total_plots = len(drug_names)
    rows = total_plots // cols
    if total_plots % cols != 0:
        rows += 1

    plt.style.use("ggplot")
    positions = range(1, total_plots + 1)
    fig = plt.figure(1, figsize=(12, len(drug_names)*1.5))
    colors = ["tab:red", "tab:blue", "tab:purple"]
    for k, drug in enumerate(drug_names):
        ax = fig.add_subplot(rows, cols, positions[k])
        drug_res = res_dict[drug]
        for c, model in zip(colors, drug_res):
            ax.errorbar(num_feats, drug_res[model], yerr=np.std(drug_res[model]), label=model,
                        marker='o', color=c)

        ax.set_ylabel("Test RMSE", fontweight="bold")
        ax.set_xlabel("Num feats", fontweight="bold")
        ax.set_xticks(num_feats)

        ax.set_title(drug, fontweight="bold")
        ax.grid(True)

        if k == len(drug_names) - 1:
            ax.legend(bbox_to_anchor=(1, 0.8))

    fig.tight_layout()
    plt.savefig(f"{save_dir}/feature_ranking.png")


drug_names = ["docetaxel", "lapatinib", "tamoxifen", "bortezomib", "oxaliplatin",
                "erlotinib", "nilotinib", "irinotecan", "paclitaxel", "rapamycin"]

if __name__ == "__main__":
    args = parse_args()
    seeds = []
    with open(args.seeds, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))

    plot_feature_ranking_res(seeds, drug_names , args.exp_dir, args.save_dir, args.num_cols)

    print(f"Feature ranking plots saved at {args.save_dir}!")