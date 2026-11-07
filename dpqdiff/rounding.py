"""Output-preserving rounding of a quantized layer (DPQ-Diff, Eq. 11).

The rounding of every weight, up or down, is relaxed with a rectified sigmoid h(V) and
learned such that the output of the layer on its recorded (quantized) inputs matches
the full-precision output. The rounding regularizer is enabled after the first fifth of
the iterations with an exponent annealed from 20 to 2, after which every entry is
rounded to a hard decision.
"""
import torch
import torch.nn as nn

import engine as E

ZETA, GAMMA = 1.1, -0.1


def layer_out(mod, x, w):
    if isinstance(mod, nn.Linear):
        return nn.functional.linear(x, w, mod.bias)
    return mod._conv_forward(x, w, mod.bias)


def qdq_act(x, clip, qmax):
    if clip is None:
        return x
    s = max(clip, 1e-12) / qmax
    return torch.clamp(torch.round(x / s), -qmax, qmax) * s


def learn_rounding(mod, X, w_bits, clip, a_bits, iters=1500, lam=0.01, lr=1e-3, bs=1024, seed=0):
    """Returns the quantized weight of `mod` with learned rounding, and the output
    errors of the learned and of the nearest rounding on the recorded inputs X."""
    qw, qa = E.qmax_of(w_bits), E.qmax_of(a_bits)
    W = mod.weight.data.clone().float()
    s = E.channel_amax(W) / qw
    Wf = torch.floor(W / s)
    rest = (W / s - Wf).clamp(1e-4, 1 - 1e-4)
    V = nn.Parameter(-torch.log((ZETA - GAMMA) / (rest - GAMMA) - 1))
    opt = torch.optim.Adam([V], lr=lr)
    X = X.cuda()
    with torch.no_grad():
        target = layer_out(mod, X, W)
        Xq = qdq_act(X, clip, qa)
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = X.shape[0]
    warm = int(0.2 * iters)
    for it in range(iters):
        idx = torch.randint(0, n, (min(bs, n),), device="cuda", generator=g)
        h = torch.clamp(torch.sigmoid(V) * (ZETA - GAMMA) + GAMMA, 0, 1)
        Wq = torch.clamp(Wf + h, -qw, qw) * s
        out = layer_out(mod, Xq[idx], Wq)
        d = (out - target[idx]).pow(2)
        rec = d.reshape(d.shape[0], -1).sum(1).mean()
        if it < warm:
            loss = rec
        else:
            beta = 20.0 + (2.0 - 20.0) * (it - warm) / max(iters - warm, 1)
            loss = rec + lam * (1 - (2 * h - 1).abs().pow(beta)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = (torch.sigmoid(V) * (ZETA - GAMMA) + GAMMA >= 0.5).float()
        Wq = torch.clamp(Wf + h, -qw, qw) * s
        nearest = torch.clamp(torch.round(W / s), -qw, qw) * s
        e_learn = (layer_out(mod, Xq, Wq) - target).pow(2).mean().item()
        e_near = (layer_out(mod, Xq, nearest) - target).pow(2).mean().item()
    return Wq.to(mod.weight.dtype), e_learn, e_near
