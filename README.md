# Who Gains from Federated Insider-Threat Detection?

Code and materials for a study of federated insider-threat detection across four public datasets: two synthetic, one with human-made attacks and one from a real enterprise.

## Paper

Alriyami, Q., Mohamad, M. M., and Muhammad, Y. I. *Who Gains from Federated Insider-Threat Detection? Collaboration across Synthetic and Real Security Logs.* Submitted to *Applied Soft Computing* (Elsevier).

BibTeX (to be updated on acceptance):

```bibtex
@article{alriyami_fed_itd,
  author  = {Alriyami, Qasim and Mohamad, Mohd Murtadha and Muhammad, Yusuf Idris},
  title   = {Who Gains from Federated Insider-Threat Detection? Collaboration across Synthetic and Real Security Logs},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Submitted}
}
```

## Overview

Four public insider-threat datasets act as four organisations in a simulated federated-learning set-up: CERT r4.2 and CERT r5.2 (synthetic), SPEDIA v1.0 (attacks by people, simulated normal activity) and LANL (real enterprise authentication logs with red-team activity). All four are converted to one user-day format with 42 features. Each dataset is split by time, except LANL, which is split by user. A multilayer perceptron is trained alone, on pooled data, and with several federated methods: FedAvg (size or equal weights), FedProx, FedPer, a private first layer, local fine-tuning and FedNova. Leave-one-out and single-source transfer are also run. The main measure is PR-AUC per client, tested with an exact paired bootstrap over seeds, a paired t-test as a check, and a user-level bootstrap for SPEDIA. Seeds 0–19 are used for the main runs and seeds 5–19 for the confirmatory runs.

## Datasets and access

This repository does not redistribute any data. Please obtain each dataset from its source and accept its terms.

| Dataset | Source | Licence |
|---|---|---|
| CERT Insider Threat Test Dataset r4.2 and r5.2 | https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247 | See source |
| SPEDIA v1.0 | Zenodo: [link to be added] | CC BY 4.0 |
| LANL Comprehensive Multi-Source Cyber-Security Events | https://csr.lanl.gov/data/cyber1/ | CC0 |

See [`data/README.md`](data/README.md) for where to place the files.

## Installation

Python version: to be added.

```bash
git clone https://github.com/alriyami7777-alt/fed-itd-heterogeneous.git
cd fed-itd-heterogeneous
pip install -r requirements.txt
```

## Configuration

Copy the example configuration and edit the paths:

```bash
cp config/paths.example.yaml config/paths.yaml
```

`config/paths.yaml` is ignored by Git.

## Reproducing the results

| Paper table/figure | Script | Output file |
|---|---|---|
| To be added | To be added | To be added |

## Repository structure

```
fed-itd-heterogeneous/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── config/
│   └── paths.example.yaml
├── data/
│   └── README.md
├── src/
│   ├── harmonise/
│   ├── splits/
│   ├── models/
│   ├── federated/
│   ├── evaluation/
│   └── similarity/
├── scripts/
│   └── README.md
├── results/
│   └── README.md
└── docs/
    └── protocol.md
```

## Results

To be added.

## Related work

- Cross-dataset study (IJACSA 2026): [crossdataset-insider-threat](https://github.com/alriyami7777-alt/crossdataset-insider-threat)

## Licence and contact

Code is released under the MIT Licence (see [`LICENSE`](LICENSE)). The datasets keep their own licences.

Corresponding author: Qasim Alriyami, Universiti Teknologi Malaysia — mohamedmuhanna@graduate.utm.my
