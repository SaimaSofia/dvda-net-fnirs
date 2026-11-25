"""Two-stage training: source pretraining then domain-adaptive fine-tuning."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from .config import DEVICE, BATCH_SIZE, NUM_CLASSES, CLASS_NAMES
from .data import split_loader, compute_class_weights
from .model import DVDANet
from .losses import alignment_loss, style_diversity_loss, supcon_loss
from .viz import collect_features


def train_dvda(
    src_loader, tgt_loader, name,
    src_dim, tgt_dim,
    lambda_contrast=0.02,
    lambda_align=0.02,
    lambda_style=0.005,
    use_dual_view=True,
    pretrain_epochs=40,
    finetune_epochs=20,
    tgt_train_ratio=0.8
):
    print(f"\n==== {name} | src_dim={src_dim}, tgt_dim={tgt_dim} ====")
    print(f"Dual-view (NERM + WTRM) enabled: {use_dual_view}")

    tgt_train_loader, tgt_test_loader = split_loader(
        tgt_loader,
        train_ratio=tgt_train_ratio,
        batch_size=tgt_loader.batch_size if hasattr(tgt_loader, "batch_size") else BATCH_SIZE
    )
    print(f"Target split: {len(tgt_train_loader.dataset)} fine-tune / {len(tgt_test_loader.dataset)} test")

    model = DVDANet(src_dim, use_dual_view=use_dual_view).to(DEVICE)

    proj_tgt = None
    if tgt_dim != src_dim:
        proj_tgt = nn.Linear(tgt_dim, src_dim).to(DEVICE)

    class_weights = compute_class_weights(src_loader, num_classes=NUM_CLASSES)
    ce_task = nn.CrossEntropyLoss(weight=class_weights)
    ce_domain = nn.CrossEntropyLoss()

    # -------- Phase I --------
    print("Phase I: Pre-training on source...")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=3e-4)
    train_losses, test_losses = [], []

    for epoch in range(pretrain_epochs):
        model.train()
        total_loss = 0.0

        for x, y in src_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if use_dual_view:
                ylog, _, z_fused, z_nerm, z_wtrm = model(x, alpha=0.0, return_z=True)
                L_cls = ce_task(ylog, y)
                L_con = supcon_loss(z_fused, y)
                L_align = alignment_loss(z_nerm, z_wtrm)
                L_style = style_diversity_loss(z_nerm, z_wtrm)
                loss = L_cls + lambda_contrast * L_con + lambda_align * L_align + lambda_style * L_style
            else:
                ylog, _, z = model(x, alpha=0.0, return_z=True)
                loss = ce_task(ylog, y) + lambda_contrast * supcon_loss(z, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        train_losses.append(total_loss / max(1, len(src_loader)))

        model.eval()
        t_loss = 0.0
        with torch.no_grad():
            for x, y in tgt_test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if proj_tgt is not None:
                    x = proj_tgt(x)
                logits = model(x, alpha=0.0)[0]
                t_loss += ce_task(logits, y).item()
        test_losses.append(t_loss / max(1, len(tgt_test_loader)))

        if (epoch + 1) % 5 == 0:
            print(f"  Pretrain Epoch {epoch+1}/{pretrain_epochs} | Loss: {train_losses[-1]:.4f}")

    # t-SNE before adaptation
    z_src_before, y_src_before = collect_features(model, src_loader, proj=None, max_samples=1200)
    z_tgt_before, y_tgt_before = collect_features(model, tgt_test_loader, proj=proj_tgt, max_samples=1200)

    # Freeze shared encoder except last layer
    for p in model.shared_encoder.parameters():
        p.requires_grad = False
    for p in model.shared_encoder.net[-1].parameters():
        p.requires_grad = True

    # -------- Phase II --------
    print("Phase II: Fine-tuning with domain adaptation...")
    trainable_params = (
        list(model.shared_encoder.net[-1].parameters()) +
        list(model.classifier.parameters()) +
        list(model.domain_disc.parameters()) +
        (list(model.nerm_encoder.parameters()) if use_dual_view else []) +
        (list(model.wtrm_encoder.parameters()) if use_dual_view else []) +
        (list(proj_tgt.parameters()) if proj_tgt is not None else [])
    )

    opt = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    for epoch in range(finetune_epochs):
        model.train()
        total_loss = 0.0

        # Use max number of batches instead of zip(min(...))
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_train_loader)
        steps = max(len(src_loader), len(tgt_train_loader))

        for _ in range(steps):
            try:
                xs, ys = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                xs, ys = next(src_iter)

            try:
                xt, yt = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_train_loader)
                xt, yt = next(tgt_iter)

            xs, ys = xs.to(DEVICE), ys.to(DEVICE)
            xt, yt = xt.to(DEVICE), yt.to(DEVICE)

            if proj_tgt is not None:
                xt = proj_tgt(xt)

            _, ds = model(xs, alpha=0.5)
            yt_log, dt = model(xt, alpha=0.5)

            L_cls = ce_task(yt_log, yt)
            L_domain = (
                ce_domain(ds, torch.zeros(len(xs), device=DEVICE, dtype=torch.long)) +
                ce_domain(dt, torch.ones(len(xt), device=DEVICE, dtype=torch.long))
            )

            # More classification-focused
            loss = 0.95 * L_cls + 0.05 * L_domain

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            total_loss += loss.item()

        train_losses.append(total_loss / max(1, steps))

        model.eval()
        t_loss = 0.0
        with torch.no_grad():
            for x, y in tgt_test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if proj_tgt is not None:
                    x = proj_tgt(x)
                logits = model(x, alpha=0.0)[0]
                t_loss += ce_task(logits, y).item()
        test_losses.append(t_loss / max(1, len(tgt_test_loader)))

        scheduler.step(total_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Finetune Epoch {epoch+1}/{finetune_epochs} | Loss: {train_losses[-1]:.4f}")

    # t-SNE after adaptation
    z_src_after, y_src_after = collect_features(model, src_loader, proj=None, max_samples=1200)
    z_tgt_after, y_tgt_after = collect_features(model, tgt_test_loader, proj=proj_tgt, max_samples=1200)

    # Evaluation
    print("Evaluating on held-out target subjects...")
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for x, y in tgt_test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if proj_tgt is not None:
                x = proj_tgt(x)
            logits = model(x, alpha=0.0)[0]
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")

    tsne_pack = {
        "before": (z_src_before, y_src_before, z_tgt_before, y_tgt_before),
        "after": (z_src_after, y_src_after, z_tgt_after, y_tgt_after),
    }

    return {
        "model": model,
        "proj": proj_tgt,
        "accuracy": acc,
        "f1": f1,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "pretrain_epochs": pretrain_epochs,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_probs": np.array(y_probs),
        "tsne": tsne_pack
    }


