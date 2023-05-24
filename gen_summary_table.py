#!/usr/bin/python3
import argparse
import os
from utils.drug_exp_utils import get_summary_results as get_drug_summary_results
from utils.pub_data_exp_utils import get_summary_pub_results

def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary table for the experiments")
    parser.add_argument("--exp_dir", type=str, help="Path to the directory where the experiment data is stored")
    parser.add_argument("--seeds", type=str, help="Path to the file containing the seeds")
    parser.add_argument("--save_dir", type=str, help="Path to the directory where the summary table will be saved")
    parser.add_argument("--data_type", type=str, default="gdsc", choices=["gdsc", "pub"], help="Type of data, Choices: gdsc, pub")

    return parser.parse_args()


drug_names = ["docetaxel", "lapatinib", "tamoxifen", "bortezomib", "oxaliplatin",
                "erlotinib", "nilotinib", "irinotecan", "paclitaxel", "rapamycin"]
pub_data = ["bikeshare", "wine", "support2", "churn"]

if __name__ == "__main__":
    args = parse_args()
    exp_dir = args.exp_dir
    save_dir = args.save_dir
    data_type = args.data_type

    args = parse_args()
    seeds = []
    with open(args.seeds, "r") as fp:
        for line in fp:
            seeds.append(int(line.strip()))

    if data_type == "gdsc":
        summary_table = get_drug_summary_results(seeds, drug_names, exp_dir)
    elif data_type == "pub":
        summary_table = get_summary_pub_results(seeds, pub_data, exp_dir, ["BNN + BG", "BNN w/o BG",
                                                           "Horseshoe BNN", "RF"])
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    table_path = f"{save_dir}/summary_table.csv"
    summary_table.to_csv(table_path, index=False)

    print(f"Summary table saved at {table_path}!")