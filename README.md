[<img src="https://img.shields.io/badge/arXiv-2206.01323-b31b1b"></img>](https://arxiv.org/abs/2206.01323)
[<img src="https://img.shields.io/badge/OpenReview|pdf-pp7onaiM4VB-8c1b13"></img>](https://openreview.net/pdf?id=pp7onaiM4VB)
[<img src="https://img.shields.io/badge/OpenReview|forum-pp7onaiM4VB-8c1b13"></img>](https://openreview.net/forum?id=pp7onaiM4VB)

# SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG

This repository contains code and data accompanying the NeurIPS 2022 publication with the title *SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG*. [[preprint](https://arxiv.org/abs/2206.01323)], [[publication](https://img.shields.io/badge/OpenReview|forum-pp7onaiM4VB-8c1b13)].

## Requirements

All dependencies are managed with the `conda` package manager.
Please follow the user guide to [install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) `conda`.

Once the setup is completed, the dependencies can be installed in a new virtual environment:

```setup
conda env create --file environment.yaml --prefix ./venv
```

## Experiments

Currently 5 public EEG BCI datasets are supported: [BNCI2014001](http://bnci-horizon-2020.eu/database/data-sets), [BNCI2015001](http://bnci-horizon-2020.eu/database/data-sets), [Lee2019](https://gigadb.org/dataset/100542), [Stieger2021](https://doi.org/10.6084/m9.figshare.13123148.v1) and [Hinss2021](https://doi.org/10.5281/zenodo.5055046).

The [moabb](https://neurotechx.github.io/moabb/) and [mne](https://mne.tools) packages are used to download and preprocess these datasets. <br>
**Notice**: there is no need to manually download and preprocess the datasets. This is done automatically on the fly; datasets will be downloaded into the directory `~/mne_data`, unless the environment variable `MNE_DATA` is set and pointing to another directory.

To make sure that the correct conda environment is activated and the working directory is set properly, run this command:

```
conda activate ./venv
cd experiments
```

### Training and evaluating a specific configuration
To train and evaluate the proposed model (i.e., SPD domain-specific momentum batch normalization (SPDDSMBN)) in the inter-session TL scenario with a specific dataset, run this command:

```
python main.py dataset=<bnci2014001|bnci2015001|lee2019|stieger2021|hinss2021>
```
For the inter-subject TL scenario run:
```
python main.py evaluation=inter-subject+uda dataset=<bnci2014001|bnci2015001|lee2019|stieger2021|hinss2021>
```

**Note** that the [hydra](https://hydra.cc/) package is used to manage configuration files.
So, hydra's [override CLI syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) can be used to modify the configuration.

### Running all experiments
To run all the experiments with the public EEG datasets, run this command:

```train
./run_experiments.sh
```
This can take quite some time, because the script loops over datasets, models (including SPDDSMBN) and the evaluation scenarios.
**Note** that the computed results will overwrite the pre-computed results, shipped within this package.

## Figures and Tables

To generate the figures and tables of the paper, the distributed or pre-computed models/results can be used.
To re-compute the figures run these scripts

| Figure      | Command |
| ----------- | ----------- |
| Figure 1    | `python figure1.py`   |
| Figure 2    | `python figure2.py`        |
| Figure 3    | `python figure3.py`       |

To list the dataset specific results and summarize the ablation study (Table 1), run:
```
python tables.py
```
