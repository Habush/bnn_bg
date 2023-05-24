# Incorporating graph-based prior knowledge into Bayesian Neural Networks
This repository contains the code for the paper "Incorporating graph-based prior knowledge into Bayesian Neural Networks"

### 0. Prerequisites

In order to run the code, you need to install the project dependencies first. Recommend way to do this is to first 
create a conda environment and install the packages in the  `requirements.txt`

```shell
>> conda create --name bnn_bg
>> conda activate bnn_bg
>> pip install -r requirements.txt
```

In addition to the package dependencies listed in `requirements.txt`, the project depends on two other projects:

- **netZooPy**: For running the PANDA algorithm [1] and generating the prior knowledge graph
- **HorseshoeBNN**: For running the Horseshoe BNN model [2], which is one of the baseline models.

We had to make changes to parts of the code in both of these projects. We made the following changes:

- **netZooPy**: We added a functionality to save the updated Protein-Protein Interaction (PPI) and correlation 
  matrices. The original implementation can be found [here](https://github.com/netZoo/netZooPy).
- **HorseshoeBNN**: We added a functionality to support more than one layer. The original implementation (can be found 
  [here](https://github.com/microsoft/horseshoe-bnn)) only supported one layer. 

In order to adhere to the double-blind review policy, we anonymized the changes we made to both projects and 
included a compressed version of the modified projects. To install the modified versions of these projects, run the following commands:

```shell
unzip netZooPy.zip
cd netZooPy
pip install -r requirements.txt
pip install .
```

```shell
unzip horseshoe-bnn.zip
cd horseshoe-bnn
pip install -r requirements.txt
pip install .
```

### 1. Running GDSC Drug experiments

The GDSC drug experiments are run using the `run_drug_exps.py` scripts. To reproduce the results in the paper, you 
should use the random seeds in the `seeds.txt` file. The data for the experiments can be downloaded from [here](https://drive.google.com/file/d/1N9MGhkPspWA-QD8R4Tc0RZxX9VHWDyf3/view?usp=sharing) (in zip format). The 
data should be placed in the `data/gdsc` (the default path) folder.

The following command runs the experiments for the GDSC drugs for the BNN + BG, BNN w/o BG and Random 
Forest models. The results are saved in the `data/gdsc/exps` folder.

```shell
./run_drug_exps.py --seed seeds.txt 
```
To run the experiments for the Horseshoe BNN model, you should set the `horseshoe_bnn` flag to `True`:

```shell
./run_drug_exps.py --seed seeds.txt --horseshoe_bnn 1
```

### 2. Running Graph Attention (GAT) model

The GAT model is run using the `run_gnn.py` script.

```shell
./run_gnn.py --seed seeds.txt
```

### 3. Run the zero-out feature ranking

```shell
./run_ft_rank.py --seed seeds.txt
```

Note: The feature ranking experiment needs to be run after the GDSC drug experiments as it uses the results from 
these experiments.

### 4. Run the Public datasets experiments

The public datasets experiments are run using the `run_public_exps.py` script. The datasets can be downloaded from 
[here](https://drive.google.com/file/d/1zGYH-JVy-9Tzrj0fycI7ikOBKk7SySCO/view?usp=sharing) (in zip format). The data 
should be placed in the `data/pub_data` (the default path) folder. The results in the 
paper are obtained using 10-fold cross-validation. To reproduce the results, you should use the first 10 seeds in 
the `seeds.txt` file.

```shell
head -n 10 seeds.txt > seeds_10.txt
./run_public_exps.py --seed seeds_10.txt
```

The above command runs the experiments for the BNN + BG, BNN w/o BG and Random Forest models. The results are saved 
in the `data/pub_data/exps` folder. To run the experiments for the Horseshoe BNN model, you should set the `horseshoe_bnn` flag to `True`:

```shell
head -n 10 seeds.txt > seeds_10.txt
./run_public_exps.py --seed seeds_10.txt --horseshoe_bnn 1
```

### 5. Run the feature ranking experiments for GDSC drugs

```shell
./run_drug_ft_rank.py --seed seeds.txt --exp_dir path/to/experiment/results 
```

### 6. Generate Summary Tables

The summary tables in the paper are generated using the `gen_summary_table.py` script. To generate the tables, 
run the following command: 

```shell
./gen_summary_table.py --seed seeds.txt --exp_dir path/to/experiment/results --save_dir path/to/save/tables 
--data_type gdsc
```

Use the `--data_type` flag to specify whether to generate the tables for the GDSC or public datasets experiments.

**Note**: You have to run the experiments first before generating the summary tables.

### 7. Generate feature ranking plots

The feature ranking plots in the paper are generated using the `gen_ft_rank_plots.py` script. To generate the plots, 
run the following command: 

```shell
./gen_ft_rank_plots.py --seed seeds.txt --exp_dir path/to/experiment/results --save_dir path/to/save/plots
```

**Note**: You have to run the feature ranking experiments first before generating the plots.