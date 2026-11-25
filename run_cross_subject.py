"""
Run cross-subject evaluation for DVDA-Net on UFFT fNIRS subject CSVs.

Example:
  python run_cross_subject.py --data_dir data/ufft
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dvda_net.config import (
    SEED,
    BATCH_SIZE,
    PRETRAIN_EPOCHS,
    FINETUNE_EPOCHS,
    TGT_TRAIN_RATIO,
    LAMBDA_CONTRAST,
    LAMBDA_ALIGN,
    LAMBDA_STYLE,
    USE_DUAL_VIEW,
    TEST_SUBJECTS_PER_FOLD,
    MAX_FOLDS,
    CLASS_NAMES,
    ALLOWED_LABELS,
    LABEL_MAP,
)
from dvda_net.utils import seed_everything
from dvda_net.data import (
    make_subject_folds,
    load_arrays_from_subject_files,
    make_loader_from_arrays,
    fit_source_normalizer,
    apply_normalizer,
    get_in_dim,
)
from dvda_net.train import train_dvda


def parse_args():
    p = argparse.ArgumentParser(description="DVDA-Net cross-subject fNIRS training")
    p.add_argument(
        "--data_dir",
        type=str,
        default="data/ufft",
        help="Folder with subject-wise CSV files",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Where to save metrics and plots",
    )
    p.add_argument("--test_subjects_per_fold", type=int, default=TEST_SUBJECTS_PER_FOLD)
    p.add_argument("--max_folds", type=int, default=MAX_FOLDS)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion").mkdir(exist_ok=True)
    (out_dir / "curves").mkdir(exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_dir}\n"
            "Download the Bak et al. UFFT fNIRS CSVs and put them here."
        )

    folds = make_subject_folds(
        str(data_dir),
        test_subjects_per_fold=args.test_subjects_per_fold,
        seed=args.seed,
        max_folds=args.max_folds,
    )
    print(f"Subjects folder: {data_dir}")
    print(f"Folds: {len(folds)} | test subjects / fold: {args.test_subjects_per_fold}")

    all_results = {}

    for i, (src_files, tgt_files) in enumerate(folds, start=1):
        name = f"Fold-{i}"
        print("\n" + "=" * 70)
        print(f"{name}: train subjects={len(src_files)} | test subjects={len(tgt_files)}")

        x_src, y_src = load_arrays_from_subject_files(
            src_files, allowed_labels=ALLOWED_LABELS, label_map=LABEL_MAP
        )
        x_tgt, y_tgt = load_arrays_from_subject_files(
            tgt_files, allowed_labels=ALLOWED_LABELS, label_map=LABEL_MAP
        )

        mean, std = fit_source_normalizer(x_src)
        x_src = apply_normalizer(x_src, mean, std)
        x_tgt = apply_normalizer(x_tgt, mean, std)

        src_loader = make_loader_from_arrays(x_src, y_src, batch_size=BATCH_SIZE, shuffle=True)
        tgt_loader = make_loader_from_arrays(x_tgt, y_tgt, batch_size=BATCH_SIZE, shuffle=True)

        src_dim = get_in_dim(src_loader)
        tgt_dim = get_in_dim(tgt_loader)

        out = train_dvda(
            src_loader,
            tgt_loader,
            name,
            src_dim,
            tgt_dim,
            lambda_contrast=LAMBDA_CONTRAST,
            lambda_align=LAMBDA_ALIGN,
            lambda_style=LAMBDA_STYLE,
            use_dual_view=USE_DUAL_VIEW,
            pretrain_epochs=PRETRAIN_EPOCHS,
            finetune_epochs=FINETUNE_EPOCHS,
            tgt_train_ratio=TGT_TRAIN_RATIO,
        )
        all_results[name] = out

        # confusion matrix
        cm = confusion_matrix(out["y_true"], out["y_pred"], labels=[0, 1, 2])
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
        )
        plt.title(f"{name} | Acc={out['accuracy']:.4f} | F1={out['f1']:.4f}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion" / f"{name}_cm.png", dpi=150)
        plt.close()

        # loss curve
        plt.figure(figsize=(6, 3.5))
        plt.plot(out["train_losses"], label="Train Loss")
        plt.plot(out["test_losses"], label="Test Loss")
        plt.axvline(PRETRAIN_EPOCHS - 0.5, color="r", linestyle="--", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{name} loss curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "curves" / f"{name}_loss.png", dpi=150)
        plt.close()

    rows = [
        {"fold": k, "accuracy": v["accuracy"], "f1_weighted": v["f1"]}
        for k, v in all_results.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fold_metrics.csv", index=False)

    mean_acc = df["accuracy"].mean()
    std_acc = df["accuracy"].std(ddof=0)
    mean_f1 = df["f1_weighted"].mean()
    std_f1 = df["f1_weighted"].std(ddof=0)

    print("\n" + "=" * 70)
    print("CROSS-SUBJECT SUMMARY")
    print(df.to_string(index=False))
    print(f"\nMean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean F1      : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
