"""Losses used in Stage I and Stage II."""

import torch
import torch.nn.functional as F


def alignment_loss(z_nerm, z_wtrm):
    z_nerm_n = F.normalize(z_nerm, dim=-1)
    z_wtrm_n = F.normalize(z_wtrm, dim=-1)
    cos_sim = (z_nerm_n * z_wtrm_n).sum(dim=-1).mean()
    return 2.0 - 2.0 * cos_sim


def style_diversity_loss(z_nerm, z_wtrm):
    z_nerm_n = F.normalize(z_nerm, dim=-1)
    z_wtrm_n = F.normalize(z_wtrm, dim=-1)
    G_nerm = torch.matmul(z_nerm_n, z_nerm_n.T)
    G_wtrm = torch.matmul(z_wtrm_n, z_wtrm_n.T)
    I = torch.eye(z_nerm.size(0), device=z_nerm.device)

    loss = torch.norm(G_nerm - I, p='fro')**2 + torch.norm(G_wtrm - I, p='fro')**2
    return loss / (z_nerm.size(0) ** 2 + 1e-8)


def supcon_loss(z, y, temperature=0.1):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    y = y.view(-1, 1)
    mask = (y == y.T).float()
    mask.fill_diagonal_(0)

    if mask.sum() < 1:
        return torch.tensor(0.0, device=z.device)

    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(sim) * (1 - torch.eye(len(z), device=z.device))
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-8)
    return (-(mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)).mean()

