<div align="center">

# CrossBorder

### A Controlled Benchmark for Multi-Typology Card-Fraud Detection

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Code style](https://img.shields.io/badge/code%20style-ruff-D97706)](https://docs.astral.sh/ruff/)
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()
[![Dataset](https://img.shields.io/badge/dataset-Sparkov%20CC0-0EA5E9)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
[![statsmodels](https://img.shields.io/badge/core-statsmodels-4F46E5)](https://www.statsmodels.org/)
[![PyG](https://img.shields.io/badge/graph-PyTorch%20Geometric-EF4444)](https://pyg.org/)

</div>

---

Most fraud detection asks one question: *fraud or not?* **CrossBorder** asks a sharper one:

> **Which data representation recovers which kind of fraud — and what does it cost to *learn* that structure instead of hand-crafting it?**

A velocity burst, a coordinated merchant ring, and an out-of-profile late-night transaction are different events with different *shapes*. Some of those shapes are visible in a single table row; others only exist in a **graph** (a ring is a merchant fan-in) or a **sequence** (a temporal anomaly is rare *for that card*). Collapsing them into one binary label throws away exactly the structure a response team needs — and exactly the structure that decides whether a graph or sequence model is worth its complexity.

CrossBorder answers this with a **controlled benchmark**: it plants fraud with *known* typology signatures into a legitimate-traffic background, then measures, per typology, whether a tabular GLM, a graph network, or a state-space sequence model is the representation that recovers it.

---

## The pivot: don't detect downloaded fraud — plant it

The project began as plain Sparkov fraud detection and pivoted once the downloaded labels proved structurally empty. On the raw `is_fraud` labels the geographic signal was **fraud-uncorrelated** (lift ≈ 1.0) and the category signal saturated — there was no usable ring / geographic / sequence structure for a graph or sequence model to recover. Any neural extension would have "worked" only by overfitting noise.

So instead of *detecting* unknown fraud, CrossBorder **injects** fraud with ground-truth signatures (`src/inject.py`) and scores each representation against the answer key. Every neural method is handed data *designed to reward it* — which turns each model into a falsifiable claim ("a graph recovers a ring; tabular features can't") rather than a leaderboard number with no mechanism behind it.

### Five injected typologies → the representation that should catch each

| typology   | injected signature                                                   | detector slot       |
|------------|----------------------------------------------------------------------|---------------------|
| `ring`     | *K* distinct cards hit one merchant inside a short window (fan-in)    | **GNN** (graph)     |
| `velocity` | one card, many transactions in a short window                        | SSM / tabular       |
| `temporal` | a transaction at one of the card's **rarest hours-of-day**           | **SSM** (sequence)  |
| `category` | a transaction in one of the card's rarest categories                 | GLM (tabular)       |
| `geo`      | a merchant placed implausibly far from the card's home               | GLM (tabular)       |

**Controlled-benchmark invariant (leak-free):** an injected row matches the legit distribution on **every axis except the intended signature** — `amt` and the timestamp are sampled from the legit pools, and merchant coordinates reuse the legit home→merchant offset distribution (except `geo`). A detector can only succeed via the real signal, never a distributional artifact. **Overlap** events stamp two compatible signatures on one event, producing the `cross_border` ground truth.

---

## Headline results — the bake-off

Isolated test AUC: each signature's **solo rows vs. legit** (overlaps excluded, so no typology can borrow another's signal). Methodology and reproduction live in the bake-off scripts and notebooks 06–08.

| typology     | tabular GLM | structural oracle                  | learned neural        | LR-test gate                         |
|--------------|:-----------:|------------------------------------|-----------------------|--------------------------------------|
| **ring**     | 0.582       | **0.959** windowed merchant fan-in | **0.841** RingSAGE    | ✓ admitted · *G²*=9732, df 2, *p*≈0  |
| **temporal** | 0.702       | **0.877** card-relative hour-rarity| **0.806** TemporalSSM | ✓ admitted · *G²*=12247, df 2, *p*≈0 |
| velocity     | 0.882       | — (rolling 1 h count is tabular)   | —                     | tabular-solved                       |
| category     | 0.874       | —                                  | —                     | tabular                              |
| geo          | 1.000       | — (haversine distance is tabular)  | —                     | tabular-solved (null control)        |

**Reading:** `ring` and `temporal` are the two signatures invisible to a single table row — a ring is a *time-windowed* merchant fan-in (tabular ≈ chance), a temporal anomaly is *card-relative* (global `hour_sin/cos` tops out at 0.70). In both, the learned model sits **between tabular and the hand-crafted oracle** ("the network recovers most of the signal; the oracle is the ceiling"), and clears the LR-test admission gate overwhelmingly. `geo`/`velocity`/`category` stay tabular by design — `geo` is kept as the **null-LR-test negative control**.

### External validation — does the premise hold on real fraud?

The reason for planting is that *real* fraud is entangled across typologies, so clean signatures must be injected to measure recovery at all. An external fold on the fully-anonymized **IEEE-CIS** dataset confirms this: the planted oracles fall to chance on real `isFraud` (card-relative hour-rarity **0.45**, decayed-rate **0.49**, versus **0.88 / 0.91** on the planted signatures), while a tabular GLM over engineered count features carries real fraud at **0.83**. Real fraud is not a single planted typology — injection is *necessary*, not a shortcut. The per-card sequence machinery still clears the same LR-test gate, but its held-out lift is marginal once those engineered counts are present (the "effect size, not raw *p*" caveat, reproduced out of domain). Because IEEE-CIS is anonymized — no merchant id, coordinates, or clean card key — the typologies cannot be re-injected there, so this fold validates the **premise and representation relevance**, not the which-representation-recovers-which thesis, which needs the answer key only injection provides.

---

## Architecture

The benchmark validates the slots of a single production architecture: neural networks are **feature extractors only**, admitted into a statistical classifier through a per-label likelihood-ratio test. The GLM stays the inference engine.

```
                         ┌─────────────┐
                         │  Raw data   │  Sparkov · legit background
                         └──────┬──────┘
                                │
               ┌────────────────▼────────────────┐
               │   Controlled injection engine    │  src/inject.py
               │  ring velocity temporal cat geo  │  known signatures + answer key
               └────────────────┬────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
   ┌──────▼──────┐       ┌──────▼──────┐      ┌──────▼──────┐
   │  RingSAGE   │       │ TemporalSSM │      │  Classical  │
   │ card↔merch  │       │  per-card   │      │  features   │
   │  ±window    │       │  sequences  │      │  (tabular)  │
   └──────┬──────┘       └──────┬──────┘      └──────┬──────┘
          │  LR test ✓          │  LR test ✓          │
          └─────────────────────▼─────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Binary-relevance GLM │  K independent logits
                    │     (statsmodels)     │  one per fraud label
                    └───────────┬───────────┘
                                │
                       prediction vector L̂
```

A neural scalar enters the GLM design matrix **for a given label only after a significant nested likelihood-ratio test** (`BinaryRelevanceGLM.admit_extension`). An extension admitted for `ring` is not automatically admitted for `geo` — the gate is per-label. The logistic core keeps interpretable odds ratios, Wald tests, and calibration for every label regardless of which extensions are admitted.

---

## Two response tracks

CrossBorder carries two complementary label systems on the same data:

**1. Injected typologies (the benchmark answer key).** Ground truth for "which representation recovers which structure," read back from the `typology` / `inj_event` columns via `typology_dummies()` / `is_cross_border()`. This drives the bake-off above.

**2. Rule-based labels (the statistical-inference layer).** Five non-exclusive binary labels derived from raw features by fixed heuristics **before any model is fit** (`src/labels.py`), used for the categorical-data-analysis companion (χ², odds ratios, log-linear, calibration). Thresholds were **data-tuned to the Sparkov distribution** (see threshold diagnostic):

| Label | Name | Trigger (tuned) | Gated on confirmed fraud? |
|---|---|---|---|
| `L_V` | Velocity burst | ≥ 3 transactions from the same card in any **60-minute** window | Yes |
| `L_G` | Geographic anomaly | home → merchant distance **> 120 km** | No |
| `L_C` | Category anomaly | category **unseen** in the cardholder's prior 30-day history | No |
| `L_R` | Ring membership | ≥ 3 distinct fraudulent cards sharing a merchant in **± 72 h** | Yes |
| `L_T` | Temporal anomaly | hour outside the cardholder's **Tukey 1.0 × IQR** (global cold-start prior) | No |

The two systems are cross-checked as a loop-closure sanity check, but for the bake-off the **injection typology is ground truth**.

---

## Statistical analysis layer

Beyond the production classifier, the project carries a categorical inference layer over the rule-based labels:

| Analysis | Technique |
|---|---|
| Feature–label association | Pearson χ², Cramér's V across feature × label pairs |
| Odds ratios | 2×2 tables with Haldane–Anscombe correction and 95% CIs |
| Confounding control | Mantel–Haenszel stratified OR (`L_V × L_R` by `category`) |
| Label co-occurrence | Pairwise χ² and OR matrix |
| Feature × label 3-way | Log-linear Poisson GLM; conditional independence |
| **Label joint distribution** | Log-linear on the 2⁵ = 32-cell table; label-association graph |
| Calibration | Hosmer–Lemeshow per binary logit |
| Severity | Proportional-odds model on `|L|` ∈ {0 … 5} |
| Velocity counts | Poisson / Negative-Binomial regression on transaction rate |

---

## Tech stack

| Layer | Library |
|---|---|
| Statistical inference | `statsmodels` |
| Data wrangling | `pandas` · `numpy` |
| Statistical tests | `scipy.stats` |
| Graph neural network | `torch-geometric` — GraphSAGE over a bipartite `(card, merchant-time-bucket)` graph |
| Sequence model | `torch` + `scipy.signal` — fixed-*A* diagonal SSM (CPU) · `mamba-ssm` (GPU, optional) |
| Multi-label metrics | `scikit-learn` |
| Visualisation | `matplotlib` · `seaborn` |
| Notebooks | `jupyterlab` |

> An external-validity fold on IEEE-CIS is included (see *External validation* above). A gradient-boosting tabular comparison (`xgboost` · `lightgbm` · `tabpfn`) remains optional and is not required by the benchmark.

---

## Dataset

**Primary:** [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — Sparkov-generated, **CC0 public domain**.
1.3 M train · 556 K test · 983 cardholders · 693 merchants · 14 categories · Jan 2019 – Jun 2020.

After injection (`build_injected_dataset.py`): **1,312,269 train** rows / **563,507 test** rows, each at **1.76 % injected fraud**, with **5,600 / 2,408** cross-border (overlap) events.

**Secondary (benchmark only):** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — external validation fold, never used in label engineering or injection.

> Sparkov is simulator-generated; its fraud is cleaner and more separable than real production traffic (`geo` separating perfectly is a symptom of that). The benchmark answers *"which representation recovers which structure,"* **not** *"production-grade fraud detection."*

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/vybhav72954/cross-border-fraud.git
cd cross-border-fraud

pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn \
            jupyterlab tqdm pyarrow

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
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
```

Produces `data/raw/fraudTrain.csv` and `data/raw/fraudTest.csv` (~250 MB total).

### 3. Build the controlled dataset and run the bake-off

```bash
# one-shot: the whole pipeline in documented order
python run_all.py                      # build → baseline → benchmark → gnn → ssm → … → robustness

# …or step by step (run from the project root):
python scripts/00_build_dataset.py     # → data/processed/injected_{train,test}.parquet
python scripts/01_glm_baseline.py      # tabular reference grid (the line to beat)
python scripts/02_model_benchmark.py   # 5-model cross-validated bake-off → results/
python scripts/03_gnn_ring.py          # windowed fan-in + RingSAGE + LR-test gate (ring)
python scripts/04_ssm_temporal.py      # hour-rarity + TemporalSSM + LR-test gate (temporal)
```

All tables and figures are written to `results/` (gitignored).

### 4. Notebooks

```bash
jupyter lab
```

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_eda` | Schema validation, distributions, imbalance audit |
| 02 | `02_label_engineering` | Derives the rule-based labels · saves processed parquets ⚠️ run first |
| 03 | `03_cda_contingency` | χ², Cramér's V, odds ratios, Mantel–Haenszel, label co-occurrence |
| 04 | `04_glm_binary_relevance` | Rule-label GLM · OR tables · calibration · multi-label metrics |
| 05 | `05_log_linear` | Log-linear on feature tables + the 32-cell label joint distribution |
| 06 | `06_gnn` | **Ring slot:** windowed merchant fan-in · RingSAGE · LR-test admission |
| 07 | `07_mamba` | **Temporal slot:** card-relative hour-rarity · TemporalSSM · LR-test admission |
| 08 | `08_benchmark` | **Bake-off summary:** the full grid + tabular→learned→oracle comparison |

---

## Project structure

```
cross-border-fraud/
├── data/
│   ├── raw/                       # fraudTrain.csv, fraudTest.csv  (gitignored)
│   └── processed/                 # labeled + injected parquets    (gitignored)
├── notebooks/                     # 01_eda … 08_benchmark  (per-slot deep-dives)
├── src/
│   ├── inject.py                  # controlled fraud injection + answer-key readers
│   ├── labels.py                  # derive_labels() — rule-based, pre-model
│   ├── features.py                # GLM design-matrix construction
│   ├── evaluation.py              # multi-label metrics · LR test · H-L calibration
│   ├── benchmark.py               # multi-model cross-validated bake-off engine
│   ├── robustness.py              # injectors/oracles for the robustness suite
│   ├── external.py                # IEEE-CIS external-validity helpers
│   └── models/
│       ├── glm.py                 # BinaryRelevanceGLM (+ companion models) + LR-test gate
│       ├── gnn.py                 # merchant_window_features + RingSAGE
│       └── ssm.py                 # card_hour_rarity + TemporalSSM (+ velocity / selective)
├── scripts/                       # numbered pipeline, in run-order
│   ├── 00_build_dataset.py        # builds the controlled dataset
│   ├── 01_glm_baseline.py         # tabular reference grid
│   ├── 02_model_benchmark.py      # 5-model cross-validated bake-off
│   ├── 03_gnn_ring.py             # GNN ring slot + LR-test gate
│   ├── 04_ssm_temporal.py         # SSM temporal slot + LR-test gate
│   └── …  12_external_validity.py # velocity · selective · snapshot · category · geo · integration · robustness · external
├── results/                       # figures + tables + json  (gitignored)
├── run_all.py                     # one-shot pipeline driver
├── requirements.txt
└── pyproject.toml
```

---

## Key design invariants

**Plant, don't detect.** Fraud is injected with known signatures so detectors can be scored against an answer key; injected rows match the legit distribution on every axis except the intended signature.

**Labels before models.** Rule-based label thresholds are fixed constants in `src/labels.py`, declared before any training run. No model output informs a label.

**GLM is always the classifier.** GNN and SSM change what goes into `X`; they never replace the logit. Odds ratios, Wald tests, and Hosmer–Lemeshow calibration are reported for every label.

**LR test is the gate, per label.** An extension admitted for `ring` is not automatically admitted for `geo`. Each label gets its own nested test.

**Binary relevance, not power-set.** Independent binary logits are the production model. The power-set multinomial over top-*M* combinations is a companion analysis only.

---

## References

- Agresti, A. *Categorical Data Analysis*, 3rd ed. Wiley (2013)
- Hamilton, Ying & Leskovec. *Inductive Representation Learning on Large Graphs.* NeurIPS 2017
- Veličković et al. *Graph Attention Networks.* ICLR 2018
- Gu, Goel & Ré. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* ICLR 2022
- Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv 2312.00752 (2023)
- Read et al. *Classifier Chains for Multi-Label Classification.* Machine Learning 85 (2011)
- Zhang & Zhou. *A Review on Multi-Label Learning Algorithms.* IEEE TKDE 26(8) (2014)

---

<div align="center">
<sub>Dataset: Sparkov CC0 &nbsp;·&nbsp; Statistical core: statsmodels &nbsp;·&nbsp; Graph: PyTorch Geometric &nbsp;·&nbsp; Sequences: fixed-A diagonal SSM</sub>
</div>
