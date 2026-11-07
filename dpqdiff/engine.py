"""DiffPTS forecasters under fake quantization: construction, step-aware hooks,
ancestral sampling, scoring, and storage accounting.

The forecaster is built with the official DiffPTS implementation (DIFFPTS_ROOT), and
the datasets are read from DPQ_DATA. It consists of the mean backbone f, the variance
backbone g, and the denoiser; every nn.Linear and nn.Conv1d layer of the three
networks is quantized.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.environ.get("DIFFPTS_ROOT", "DiffPTS"))

from src.experiments.DiffPTS import DiffPTSForecast  # noqa: E402
from src.layer.diffpts_utils import p_sample_loop  # noqa: E402
from src.metrics import CRPS, CRPSSum, PICP, QICE, ProbMAE, ProbMSE, ProbRMSE  # noqa: E402
from torchmetrics import MetricCollection  # noqa: E402

DATA_PATH = os.environ.get("DPQ_DATA", "data")
QUANT_TYPES = (nn.Linear, nn.Conv1d)
_T_STATE = {"t": None}  # reverse step of the running denoiser call, read by the hooks


class Experiment(DiffPTSForecast):
    """Official DiffPTS experiment; scoring is done by eval_pass on the GPU."""

    def _init_metrics(self):
        self.metrics = MetricCollection(metrics={
            "crps": CRPS(), "crps_sum": CRPSSum(), "qice": QICE(), "picp": PICP(),
            "mse": ProbMSE(), "mae": ProbMAE(), "rmse": ProbRMSE(),
        })
        self.metrics.to("cpu")
        self.task_pool = None


def backbone_of(dataset):
    """Mean backbone of the released DiffPTS configurations."""
    return "patchtst" if dataset.startswith("ETT") else "ns_transformer"


def pred_len_of(dataset):
    return 32 if dataset == "ILI" else 192


def build(dataset, seed, ckpt, batch_size=32, out="results"):
    """Restores a trained full-precision forecaster.

    Returns the experiment (data loaders and model) and its three networks."""
    exp = Experiment(
        dataset_type=dataset, data_path=DATA_PATH, save_dir=os.path.join(out, "scratch"),
        device="cuda:0", batch_size=batch_size, horizon=1, pred_len=pred_len_of(dataset),
        windows=168, cond_backbone=backbone_of(dataset),
    )
    exp._setup_run(seed)
    dev = torch.device("cuda:0")
    for mod, fname in [(exp.model, "model.pth"),
                       (exp.cond_pred_model, "cond_pred_model.pth"),
                       (exp.cond_pred_model_g, "cond_pred_model_g.pth")]:
        mod.load_state_dict(torch.load(os.path.join(ckpt, fname), map_location=dev))
        mod.eval()
    comps = {"f_cond": exp.cond_pred_model, "g_cond": exp.cond_pred_model_g,
             "denoiser": exp.model}
    return exp, comps


def set_mode(comps, train):
    for m in comps.values():
        m.train(train)


class ModelTWrapper:
    """Calls the denoiser and exposes its reverse step t to the activation hooks."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, *args, **kwargs):
        t = args[5] if len(args) > 5 else kwargs.get("t")
        if t is None:
            _T_STATE["t"] = None
        elif torch.is_tensor(t):
            _T_STATE["t"] = int(t.reshape(-1)[0].item())
        else:
            _T_STATE["t"] = int(t)
        try:
            return self.inner(*args, **kwargs)
        finally:
            _T_STATE["t"] = None

    def parameters(self):
        return self.inner.parameters()


class TrainTWrapper(ModelTWrapper):
    """As ModelTWrapper, but keeps per-sample steps (training batches) as a tensor."""

    def __call__(self, *args, **kwargs):
        t = args[5] if len(args) > 5 else kwargs.get("t")
        if t is None:
            _T_STATE["t"] = None
        elif torch.is_tensor(t):
            _T_STATE["t"] = t if t.numel() > 1 else int(t.reshape(-1)[0].item())
        else:
            _T_STATE["t"] = int(t)
        try:
            return self.inner(*args, **kwargs)
        finally:
            _T_STATE["t"] = None


# ----------------------------------------------------------------------------- quantizers
def weight_modules(components, scope=None):
    """Yields (network, layer name, layer) for every quantized layer."""
    for cname, comp in components.items():
        if scope is not None and cname not in scope:
            continue
        for mname, mod in comp.named_modules():
            if isinstance(mod, QUANT_TYPES):
                yield cname, mname, mod


def qmax_of(bits):
    return 2 ** (int(bits) - 1) - 1


def channel_amax(w):
    return (w.detach().abs().flatten(1).amax(dim=1).clamp(min=1e-12)
            .view(-1, *([1] * (w.dim() - 1))))


def qdq_weight_per_channel(w, qmax):
    """Symmetric round-to-nearest quantization with one scale per output channel."""
    if w.dim() < 2:
        s = max(float(w.detach().abs().max().item()), 1e-12) / qmax
        return torch.clamp(torch.round(w / s), -qmax, qmax) * s
    s = channel_amax(w) / qmax
    return torch.clamp(torch.round(w / s), -qmax, qmax) * s


def _is_static_buffer(name):
    return name.split(".")[-1] == "pe"  # sinusoidal tables are rebuilt at load time


def storage_accounting(components, w_bits, scope=None):
    """Bytes of the FP32 model and of the packed model (integer weights with one
    half-precision scale per output channel, all other parameters in FP32)."""
    qids = {id(m.weight) for _, _, m in weight_modules(components, scope)}
    out, tot_fp, tot_q = {}, 0, 0
    for cname, comp in components.items():
        fp = q = 0
        for _, p in comp.named_parameters():
            nb = p.numel() * 4
            fp += nb
            if id(p) in qids:
                q += int(np.ceil(p.numel() * w_bits / 8.0)) + 2 * p.shape[0]
            else:
                q += nb
        for bname, b in comp.named_buffers():
            if _is_static_buffer(bname):
                continue
            nb = b.numel() * b.element_size()
            fp += nb
            q += nb
        out[cname] = {"fp32_bytes": int(fp), "quant_bytes": int(q)}
        tot_fp += fp
        tot_q += q
    out["total"] = {"fp32_bytes": int(tot_fp), "quant_bytes": int(tot_q)}
    return out


# ----------------------------------------------------------------------------- sampling
@torch.no_grad()
def gen_batch(exp, model, batch_x, batch_x_mark, batch_y_mark, n_samples):
    """S sample paths per window with the ancestral sampler; returns (B, H, N, S)."""
    device = exp.device
    b = batch_x.shape[0]
    batch_y_mark_input = torch.concat(
        [batch_x_mark[:, -exp.label_len:, :], batch_y_mark], dim=1
    )
    dec_inp_pred = torch.zeros([b, exp.pred_len, exp.dataset.num_features]).to(device)
    dec_inp_label = batch_x[:, -exp.label_len:, :].to(device)
    dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

    fx, _ = exp.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
    gx = exp.cond_pred_model_g(batch_x)

    def tile(v):
        return v.repeat(n_samples, 1, 1, 1).transpose(0, 1).flatten(0, 1).to(device)

    x_t, xm_t, f_t, g_t = tile(batch_x), tile(batch_x_mark), tile(fx), tile(gx)
    m = exp.model
    seq = p_sample_loop(
        model, x_t, xm_t, f_t, g_t, f_t, m.num_timesteps,
        m.alphas, m.one_minus_alphas_bar_sqrt, m.alphas_cumprod,
        m.alphas_cumprod_sum, m.alphas_cumprod_prev, m.alphas_cumprod_sum_prev,
        m.betas_tilde, m.betas_bar, m.betas_tilde_m_1, m.betas_bar_m_1,
    )
    y0 = seq[-1]
    y0 = y0.reshape(b, n_samples, exp.pred_len, exp.dataset.num_features)
    return y0.permute(0, 2, 3, 1)


@torch.no_grad()
def batch_scores(outs, y):
    """outs (B, H, N, S), y (B, H, N) -> CRPS (ensemble estimator), squared and
    absolute error of the sample mean, and coverage and width of the 90% interval."""
    s = outs.double()
    yy = y.double()
    m = s.shape[-1]
    ss, _ = torch.sort(s, dim=-1)
    t1 = (s - yy.unsqueeze(-1)).abs().mean(-1)
    i = torch.arange(1, m + 1, device=s.device, dtype=s.dtype)
    t2 = ((2 * i - m - 1) * ss).sum(-1) / (m * m)
    crps = t1 - t2
    mean = s.mean(-1)
    lo = ss[..., int(round(0.05 * (m - 1)))]
    hi = ss[..., int(round(0.95 * (m - 1)))]
    cov = ((yy >= lo) & (yy <= hi)).double()
    return crps, (mean - yy) ** 2, (mean - yy).abs(), cov, (hi - lo)


@torch.no_grad()
def eval_pass(exp, model, loader, n_samples, seed, scored=True):
    """One pass over a loader with fixed sampling noise; returns the metrics."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    t0 = time.perf_counter()
    n_batches = 0
    sums = {"crps": 0.0, "mse": 0.0, "mae": 0.0, "cov90": 0.0, "width90": 0.0}
    count = 0
    for batch_x, batch_y, _, _, bxd, byd in loader:
        batch_x = batch_x.to(exp.device).float()
        batch_y = batch_y.to(exp.device).float()
        bxd = bxd.to(exp.device).float()
        byd = byd.to(exp.device).float()
        outs = gen_batch(exp, model, batch_x, bxd, byd, n_samples)
        torch.cuda.synchronize()
        if scored:
            crps, se, ae, cov, wid = batch_scores(outs, batch_y)
            sums["crps"] += float(crps.sum())
            sums["mse"] += float(se.sum())
            sums["mae"] += float(ae.sum())
            sums["cov90"] += float(cov.sum())
            sums["width90"] += float(wid.sum())
            count += crps.numel()
        n_batches += 1
    torch.cuda.synchronize()
    res = {"wall_s": time.perf_counter() - t0, "n_batches": n_batches}
    if scored:
        res["metrics"] = {k: v / max(count, 1) for k, v in sums.items()}
    return res
