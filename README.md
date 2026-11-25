# DVDA-Net for Cross-Subject fNIRS Classification

Code for **DVDA-Net** (Dual-View Domain-Adaptive Representation Network).  
The goal is three-class fNIRS decoding when train and test subjects are different.

I built this for cross-subject motor-task classification on the open UFFT fNIRS dataset (Korea University / Bak et al.).

## What the model does

1. Normalize using **source subjects only** (no target leakage)
2. Build two views of each sample:
   - noise-robust view (NERM)
   - wavelet view (WTRM)
3. Shared feature extractor + view-specific attention encoders
4. Train in two stages:
   - **Stage I:** source pretraining (classification + supervised contrastive + view alignment)
   - **Stage II:** domain adaptation (GRL adversarial loss + class prototype matching)

## Results (subject-wise folds)

Setup: 30 subjects, **24 source / 6 target** per fold (5 folds).

| Fold | Accuracy | Weighted F1 |
|------|----------|-------------|
| Fold-1 | 88.48% | 88.56% |
| Fold-2 | 88.60% | 88.57% |
| Fold-3 | 90.05% | 90.08% |
| Fold-4 | 86.81% | 86.82% |
| Fold-5 | 95.78% | 95.78% |
| **Mean ± Std** | **89.94% ± 3.16%** | **89.96% ± 3.15%** |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the UFFT subject CSVs and place them in:

```text
data/ufft/
```

See `data/README.md` for the dataset link.

## Run

```bash
python run_cross_subject.py --data_dir data/ufft --out_dir outputs
```

Useful options:

```bash
python run_cross_subject.py --data_dir data/ufft --test_subjects_per_fold 6
python run_cross_subject.py --data_dir data/ufft --max_folds 1   # quick check
```

Outputs go to `outputs/` (fold metrics, confusion matrices, loss curves).

## Repo layout

```text
dvda_net/
  config.py     # hyperparameters
  data.py       # folds, loaders, source normalization
  model.py      # DVDA-Net architecture
  losses.py     # Stage I / Stage II losses
  train.py      # training loop
  viz.py        # feature helpers
run_cross_subject.py
requirements.txt
```

## Notes

- Default device is CUDA if available, otherwise CPU (CPU will be slow).
- Label mapping uses classes `{1,2,3} -> {0,1,2}`.
- This is research code; feel free to open an issue if something breaks on your machine.

## Author

**Saima Bibi**  
Department of Applied Artificial Intelligence  
Seoul National University of Science and Technology (SeoulTech)  
Email: bibisaima_810@seoultech.ac.kr

Co-authors: Fahad Shakeel, Jamila Akhter, Hammad Nazeer, Noman Naseer (Air University)
