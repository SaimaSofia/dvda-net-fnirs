"""Optional feature / t-SNE plotting helpers."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .config import DEVICE


def collect_features(model, loader, proj=None, max_samples=1500):
    model.eval()
    feats, ys = [], []
    got = 0

    for x, y in loader:
        x = x.to(DEVICE)
        if proj is not None:
            x = proj(x)

        out = model(x, alpha=0.0, return_z=True)
        z = out[2].detach().cpu().numpy()

        feats.append(z)
        ys.append(y.numpy())
        got += len(y)
        if got >= max_samples:
            break

    feats = np.concatenate(feats, axis=0)[:max_samples]
    ys = np.concatenate(ys, axis=0)[:max_samples]
    return feats, ys


def run_tsne_and_plot(z_src, y_src, z_tgt, y_tgt, title):
    Z = np.concatenate([z_src, z_tgt], axis=0)
    Y = np.concatenate([y_src, y_tgt], axis=0)
    D = np.concatenate([np.zeros(len(y_src), dtype=int), np.ones(len(y_tgt), dtype=int)], axis=0)

    n = len(Z)
    perplexity = min(30, max(5, (n - 1) // 3))
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, init="pca", learning_rate="auto")
    Z2 = tsne.fit_transform(Z)

    plt.figure(figsize=(7.5, 6))
    domain_colors = {0: "#1f77b4", 1: "#ff7f0e"}
    class_markers = {0: "o", 1: "^", 2: "s"}

    for d in [0, 1]:
        for c in [0, 1, 2]:
            m = (D == d) & (Y == c)
            if m.sum() == 0:
                continue
            plt.scatter(
                Z2[m, 0], Z2[m, 1],
                c=domain_colors[d], marker=class_markers[c],
                s=18, alpha=0.75,
                label=f"{'Src' if d==0 else 'Tgt'} | C{c}"
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8, ncol=2)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

