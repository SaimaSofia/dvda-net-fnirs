"""Data loading, subject folds, and source-fitted normalization."""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from .config import ALLOWED_LABELS, LABEL_MAP, BATCH_SIZE, NUM_CLASSES


def make_subject_folds(folder_path, test_subjects_per_fold=1, seed=42, max_folds=None):
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if len(files) == 0:
        raise ValueError(f"No CSV files found in: {folder_path}")
    if test_subjects_per_fold < 1 or test_subjects_per_fold >= len(files):
        raise ValueError("test_subjects_per_fold must be >=1 and < number of subjects")

    rng = np.random.default_rng(seed)
    shuffled = [files[i] for i in rng.permutation(len(files))]
    folds = []

    for i in range(0, len(shuffled), test_subjects_per_fold):
        tgt_files = shuffled[i:i + test_subjects_per_fold]
        if len(tgt_files) == 0:
            continue
        src_files = [f for f in shuffled if f not in tgt_files]
        folds.append((src_files, tgt_files))

    if max_folds is not None:
        folds = folds[:max_folds]

    print(f"Total subject files: {len(files)}")
    print(f"Test subjects per fold: {test_subjects_per_fold}")
    print(f"Total folds: {len(folds)}")
    return folds


def load_arrays_from_subject_files(
    csv_files,
    class_col_candidates=("class", "label", "Label"),
    allowed_labels=(1, 2, 3),
    label_map=None
):
    all_x, all_y = [], []

    for f in csv_files:
        df = pd.read_csv(f)

        class_col = None
        for c in class_col_candidates:
            if c in df.columns:
                class_col = c
                break
        if class_col is None:
            raise ValueError(f"No class column found in {f}. Expected one of {class_col_candidates}")

        y_raw = pd.to_numeric(df[class_col], errors="coerce")
        valid = y_raw.notna().values

        x = df.drop(columns=[class_col]).values.astype(np.float32)[valid]
        y_raw = y_raw[valid].astype(int).values

        keep = np.isin(y_raw, allowed_labels)
        x = x[keep]
        y_raw = y_raw[keep]

        if len(y_raw) == 0:
            continue

        if label_map is not None:
            y = np.array([label_map[int(v)] for v in y_raw], dtype=np.int64)
        else:
            y = y_raw.astype(np.int64)

        all_x.append(x)
        all_y.append(y)

    if len(all_x) == 0:
        raise ValueError("No valid samples found after filtering labels.")

    x = np.concatenate(all_x, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    return x, y


def make_loader_from_arrays(x, y, batch_size=64, shuffle=True):
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def fit_source_normalizer(x_src):
    mean = x_src.mean(axis=0, keepdims=True)
    std = x_src.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalizer(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def label_counts(loader, name):
    ys = []
    for _, y in loader:
        ys.extend(y.numpy().tolist())
    ys = np.array(ys)
    print(f"{name} label counts -> 0:{(ys==0).sum()} 1:{(ys==1).sum()} 2:{(ys==2).sum()}")


def get_in_dim(loader):
    return next(iter(loader))[0].shape[1]


def split_loader(loader, train_ratio=0.8, batch_size=64):
    ds = loader.dataset
    n = len(ds)
    n_train = int(n * train_ratio)
    n_test = n - n_train
    tr_ds, te_ds = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(SEED))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    return tr_loader, te_loader


def compute_class_weights(loader, num_classes=3):
    ys = []
    for _, y in loader:
        ys.extend(y.numpy().tolist())
    ys = np.array(ys)

    counts = np.array([(ys == c).sum() for c in range(num_classes)], dtype=np.float64)
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)

