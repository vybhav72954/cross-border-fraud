<div align="center">

# CrossBorder

### Multi-Label Fraud Typology Detection for Payment Networks

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-D97706)](https://docs.astral.sh/ruff/)
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()
[![Dataset](https://img.shields.io/badge/dataset-Sparkov%20CC0-0EA5E9)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
[![statsmodels](https://img.shields.io/badge/core-statsmodels-4F46E5)](https://www.statsmodels.org/)
[![PyG](https://img.shields.io/badge/graph-PyTorch%20Geometric-EF4444)](https://pyg.org/)

</div>

---

Most fraud detection systems ask one question: *fraud or not?*

**CrossBorder** asks five simultaneously — and recognises that a single transaction can cross the boundaries of multiple fraud typologies at once. A velocity burst happening inside a coordinated merchant ring is not the same event as either alone. Collapsing both into a single binary label destroys exactly the information a response team needs to act.

The system is built around a **statistical core with neural extensions**: five independent logistic regression classifiers (one per fraud type) sit at the centre, and graph- and sequence-based neural representations are admitted into the design matrix only after passing a likelihood-ratio test. The GLM stays the inference engine throughout.

---

## What "cross-border" means here

The name refers to **typological boundaries**, not geography. A transaction is *cross-border* when it simultaneously exhibits the behavioral signatures of more than one fraud class — it crosses the line between fraud categories. Empirically, these are the highest-risk events: the system detects, quantifies, and characterises them explicitly.

---

## The five fraud labels

Five non-exclusive binary labels are derived from raw transaction features using rule-based heuristics **before any model is fit**. No label is derived from model output.

| Label | Name | Behavioral signature | Gated on confirmed fraud? |
|---|---|---|---|
| `L_V` | Velocity burst | ≥ 3 transactions from the same card in any 30-minute window | Yes |
| `L_G` | Geographic anomaly | Cardholder home → merchant distance > 500 km | No |
| `L_C` | Category anomaly | Merchant category outside cardholder's 30-day modal history | No |
| `L_R` | Ring membership | ≥ 3 distinct fraudulent cards sharing the same merchant in ± 24 h | Yes |
| `L_T` | Temporal anomaly | Transaction hour outside cardholder's historical Tukey 1.5 × IQR | No |

`L_G`, `L_C`, `L_T` are **anomaly-precursor** labels — they fire on legitimate transactions too, surfacing pre-fraud behavioural drift. `L_V` and `L_R` are **confirmed-subtype** labels gated on `is_fraud = 1`.

A transaction is *cross-border* when `|L| ≥ 2`.

---

## Architecture

```
                         ┌─────────────┐
                         │  Raw data   │  Sparkov · 1.2 M transactions
                         └──────┬──────┘
                                │
               ┌────────────────▼────────────────┐
               │     Rule-based label engine      │  src/labels.py
               │  L_V  L_G  L_C  L_R  L_T        │  runs before any model
               └────────────────┬────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
   ┌──────▼──────┐       ┌──────▼──────┐      ┌──────▼──────┐
   │    GNN      │       │    Mamba    │      │  Classical  │
   │ card↔merch  │       │  per-card   │      │  features   │
   │  bipartite  │       │  sequences  │      │  (tabular)  │
   └──────┬──────┘       └──────┬──────┘      └──────┬──────┘
          │  LR test ✓          │  LR test ✓          │
          └─────────────────────▼─────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Binary-relevance GLM │  K = 5 logistic models
                    │     (statsmodels)     │  one per fraud label
                    └───────────┬───────────┘
                                │
                   L̂_V  L̂_G  L̂_C  L̂_R  L̂_T
                   prediction vector ∈ {0,1}⁵
```

Neural extensions are **feature extractors only** — they augment the GLM design matrix and must pass a per-label likelihood-ratio test before admission. The logistic models remain the classifiers and produce interpretable odds ratios, Wald tests, and calibration diagnostics for every label.

---

## Statistical analysis layer

Beyond the production classifier, the project carries a complete categorical inference layer:

| Analysis | Technique |
|---|---|
| Feature–label association | Pearson χ², Cramér's V across all feature × label pairs |
| Odds ratios | 2×2 tables with Haldane–Anscombe correction and 95% CIs |
| Confounding control | Mantel–Haenszel stratified OR (e.g. `L_V × L_R` by `category`) |
| Label co-occurrence | Pairwise χ² and OR matrix across all 10 label pairs |
| Feature × label 3-way | Log-linear Poisson GLM; tests conditional independence |
| **Label joint distribution** | Log-linear on 2⁵ = 32-cell table; recovers the label association graph |
| Calibration | Hosmer–Lemeshow test per binary logit |
| Severity | Proportional-odds model on `|L|` ∈ {0 … 5} |
| Velocity counts | Poisson / Negative-Binomial regression on transaction rate |

---

## Tech stack

| Layer | Library |
|---|---|
| Statistical inference | `statsmodels` |
| Data wrangling | `pandas` · `numpy` |
| Statistical tests | `scipy.stats` |
| Graph construction | `networkx` |
| Graph neural network | `torch-geometric` — GraphSAGE / GAT |
| Sequence model | `torch` DiscretizedSSM (CPU) · `mamba-ssm` (GPU) |
| Tabular benchmarks | `xgboost` · `lightgbm` · `tabpfn` |
| Multi-label metrics | `scikit-learn` |
| Visualisation | `matplotlib` · `seaborn` |
| Notebooks | `jupyterlab` |

---

## Dataset

**Primary:** [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — Sparkov-generated, **CC0 public domain**
1.3 M train · 556 K test · 983 cardholders · 693 merchants · 14 categories · Jan 2019 – Jun 2020

**Secondary (benchmark only):** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — one external validation fold, not used in label engineering.

> Sparkov is simulator-generated. Fraud patterns are more learnable than real production data; results should be interpreted with that in mind.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/vybhav72954/cross-border-fraud.git
cd cross-border-credit

pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn \
            jupyterlab networkx xgboost lightgbm tabpfn

# PyTorch — CPU build
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# GPU only (requires CUDA 12+)
# pip install mamba-ssm causal-conv1d
```

### 2. Download the dataset

```bash
# Requires ~/.kaggle/kaggle.json  (Kaggle → Account → Create API Token)
pip install kaggle
kaggle datasets download -d kartik2112/fraud-detection \
  -p data/raw --unzip
```

Produces `data/raw/fraudTrain.csv` and `data/raw/fraudTest.csv` (~250 MB total).

### 3. Run notebooks in order

```bash
jupyter lab
```

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_eda` | Schema validation, distributions, imbalance audit |
| 02 | `02_label_engineering` | Derives all 5 labels · saves processed parquets ⚠️ run first |
| 03 | `03_cda_contingency` | χ², Cramér's V, odds ratios, Mantel–Haenszel, label co-occurrence |
| 04 | `04_glm_binary_relevance` | Production model · OR tables · calibration · multi-label metrics |
| 05 | `05_log_linear` | Log-linear on feature tables + 32-cell label joint distribution |
| 06 | `06_gnn` | Graph construction · GraphSAGE · LR-test admission |
| 07 | `07_mamba` | Per-card SSM · temporal features · LR-test admission |
| 08 | `08_benchmark` | Full leaderboard: GLM vs XGBoost vs TabPFN |

---

## Project structure

```
cross-border-credit/
├── data/
│   ├── raw/                    # fraudTrain.csv, fraudTest.csv  (gitignored)
│   └── processed/              # labeled parquets, GNN/SSM features
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_label_engineering.ipynb
│   ├── 03_cda_contingency.ipynb
│   ├── 04_glm_binary_relevance.ipynb
│   ├── 05_log_linear.ipynb
│   ├── 06_gnn.ipynb
│   ├── 07_mamba.ipynb
│   └── 08_benchmark.ipynb
├── src/
│   ├── labels.py               # derive_labels() — rule-based, pre-model
│   ├── features.py             # GLM design matrix construction
│   ├── evaluation.py           # multi-label metrics · LR test · H-L calibration
│   └── models/
│       ├── glm.py              # BinaryRelevanceGLM + companion models
│       ├── gnn.py              # CardMerchantSAGE + bipartite graph builder
│       └── ssm.py              # MambaExtractor + DiscretizedSSM (CPU fallback)
├── Technical_Design_Document.md
├── CLAUDE.md
├── requirements.txt
└── pyproject.toml
```

---

## Benchmark

*Fill in after running `08_benchmark.ipynb`.*

| System | Hamming loss ↓ | Subset accuracy ↑ | Label ranking AP ↑ | Mean AUC ↑ |
|---|---|---|---|---|
| GLM — classical features | — | — | — | — |
| GLM + GNN + Mamba | — | — | — | — |
| XGBoost (binary relevance) | — | — | — | — |
| TabPFN (subsampled) | — | — | — | — |

---

## Key design invariants

**Labels before models.** All label thresholds are fixed constants in `src/labels.py` and declared before any training run. No model output informs a label assignment.

**GLM is always the classifier.** GNN and Mamba change what goes into `X`; they never replace the logit. Odds ratios, Wald tests, and Hosmer–Lemeshow calibration are reported for every label regardless of which extensions are admitted.

**LR test is the gate, per label.** An extension admitted for `L_R` (ring) is not automatically admitted for `L_G` (geography). Each label gets its own nested test.

**Binary relevance, not power-set.** Five independent logits are the production model. The power-set multinomial over top-*M* label combinations is a companion analysis only.

---

## References

- Agresti, A. *Categorical Data Analysis*, 3rd ed. Wiley (2013)
- Hamilton, Ying & Leskovec. *Inductive Representation Learning on Large Graphs.* NeurIPS 2017
- Veličković et al. *Graph Attention Networks.* ICLR 2018
- Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv 2312.00752 (2023)
- Gorishniy et al. *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 2021
- Hollmann et al. *Accurate predictions on small data with a tabular foundation model.* Nature 637 (2025)
- Read et al. *Classifier Chains for Multi-Label Classification.* Machine Learning 85 (2011)
- Zhang & Zhou. *A Review on Multi-Label Learning Algorithms.* IEEE TKDE 26(8) (2014)

---

<div align="center">
<sub>Dataset: Sparkov CC0 &nbsp;·&nbsp; Statistical core: statsmodels &nbsp;·&nbsp; Graph: PyTorch Geometric &nbsp;·&nbsp; Sequences: Mamba / SSM</sub>
</div>
