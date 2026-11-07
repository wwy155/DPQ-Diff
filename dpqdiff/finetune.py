"""Distribution-preserving fine-tuning of DPQ-Diff.

Starts from the post-training pipeline (learned rounding and the step-aware lookup
table) and fine-tunes all networks end to end. Full-precision master weights are
quantized per output channel in every forward pass with the straight-through
estimator, and every activation is quantized with the calibrated range of its step
group. The objective is the ELBO of the quantized forecaster (Gaussian NLL of the
endpoint and denoising loss) plus anchors that tie its noise prediction, location, and
log-scale to the full-precision forecaster. After every epoch the validation CRPS is
computed with the deployed sampler, and the best pipeline on the validation windows,
including the post-training starting point, is kept.

    python finetune.py --dataset ETTh1 --ckpt checkpoints/ETTh1/1 --w-bits 4
"""
import argparse
import copy
import json
import os
import time

import torch

import calibration as C
import engine as E
from src.layer.diffpts_utils import cal_forward_noise, q_sample  # noqa: E402


class STEQuant(torch.autograd.Function):
    """Quantize-dequantize in the forward pass, identity in the backward pass."""

    @staticmethod
    def forward(ctx, x, scale, qmax):
        return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class FixedScaleFakeQuant(torch.nn.Module):
    """Per-output-channel weight quantizer on the integer grid of the calibrated pipeline."""

    def __init__(self, s, qmax):
        super().__init__()
        self.register_buffer("s", s)
        self.qmax = qmax

    def forward(self, w):
        return STEQuant.apply(w, self.s, self.qmax)


def attach_quantizers(comps, w_bits, init_path):
    """Freezes a full-precision copy of the pipeline as the anchor, loads the learned
    rounding, and registers the weight quantizers (master weights stay trainable)."""
    teacher = {}
    for k, v in comps.items():
        t = copy.deepcopy(v).eval()
        for p in t.parameters():
            p.requires_grad_(False)
        teacher[k] = t
    state = torch.load(init_path, map_location="cpu")
    qmax = E.qmax_of(w_bits)
    for cname, mname, mod in list(E.weight_modules(comps)):
        key = f"{cname}.{mname}"
        s = E.channel_amax(mod.weight.data) / qmax
        if key in state:
            with torch.no_grad():
                mod.weight.data.copy_(state[key].to(mod.weight.device, mod.weight.dtype))
        torch.nn.utils.parametrize.register_parametrization(mod, "weight", FixedScaleFakeQuant(s, qmax))
    return teacher


def _scale(table, name, group, qmax):
    d = table.get(name, {})
    if group in d and d[group] > 0:
        return d[group] / qmax
    vals = [v for v in d.values() if v > 0]
    return (max(vals) / qmax) if vals else None


def install_train_hooks(comps, table, G, n_steps, a_bits=8):
    """Straight-through activation quantization in the training forward pass; every
    sample is quantized with the range of the step group of its own reverse step."""
    qmax = E.qmax_of(a_bits)
    handles = []
    for cname, mname, mod in E.weight_modules(comps):
        use_t = cname == "denoiser"
        name = f"{cname}.{mname}"

        def hook(module, args, name=name, use_t=use_t):
            x = args[0]
            t = E._T_STATE["t"]
            if (not use_t) or t is None:
                s = _scale(table, name, "cond", qmax)
                if s is None:
                    return None
                sc = torch.tensor(s, dtype=x.dtype, device=x.device)
                return (STEQuant.apply(x, sc, qmax),) + tuple(args[1:])
            if torch.is_tensor(t) and t.numel() > 1:
                scales = []
                for ti in t.reshape(-1).tolist():
                    s = _scale(table, name, f"t{C.group_of(int(ti), G, n_steps)}", qmax)
                    if s is None:
                        return None
                    scales.append(s)
                sc = torch.tensor(scales, dtype=x.dtype, device=x.device)
                sc = sc.view(-1, *([1] * (x.dim() - 1)))
                return (STEQuant.apply(x, sc, qmax),) + tuple(args[1:])
            tv = int(t.reshape(-1)[0].item()) if torch.is_tensor(t) else int(t)
            s = _scale(table, name, f"t{C.group_of(tv, G, n_steps)}", qmax)
            if s is None:
                return None
            sc = torch.tensor(s, dtype=x.dtype, device=x.device)
            return (STEQuant.apply(x, sc, qmax),) + tuple(args[1:])
        handles.append(mod.register_forward_pre_hook(hook))
    return handles


def anchored_elbo(exp, wrapper, teacher, lam, batch_x, batch_y, batch_x_mark, batch_y_mark):
    """ELBO of the quantized forecaster plus the location, scale, and noise anchors."""
    n = batch_x.size(0)
    t = torch.randint(low=0, high=exp.model.num_timesteps, size=(n // 2 + 1,), device=exp.device)
    t = torch.cat([t, exp.model.num_timesteps - 1 - t], dim=0)[:n]  # antithetic steps
    bym_in = torch.concat([batch_x_mark[:, -exp.label_len:, :], batch_y_mark], dim=1)
    dec_inp = torch.cat([
        batch_x[:, -exp.label_len:, :].to(exp.device),
        torch.zeros([n, exp.pred_len, exp.dataset.num_features], device=exp.device),
    ], dim=1)

    fx, _ = exp.cond_pred_model(batch_x, batch_x_mark, dec_inp, bym_in)
    gx = exp.cond_pred_model_g(batch_x)
    nll = (0.5 * (torch.log(gx) + (fx - batch_y).square() / gx)).mean()
    e = torch.randn_like(batch_y)
    noise = e * torch.sqrt(cal_forward_noise(exp.model.betas_bar, gx, t))
    y_t = q_sample(batch_y, fx, exp.model.alphas_bar_sqrt,
                   exp.model.one_minus_alphas_bar_sqrt, t, noise=noise)
    out, _ = wrapper(batch_x, batch_x_mark, y_t, fx, gx, t)
    den = (e - out).square().mean()
    loss = den + nll
    if lam > 0:
        with torch.no_grad():
            f_t, _ = teacher["f_cond"](batch_x, batch_x_mark, dec_inp, bym_in)
            g_t = teacher["g_cond"](batch_x)
            o_t, _ = teacher["denoiser"](batch_x, batch_x_mark, y_t.detach(), f_t, g_t, t)
        anchor = ((out - o_t).square().mean() + (fx - f_t).square().mean()
                  + (torch.log(gx) - torch.log(g_t)).square().mean())
        loss = loss + lam * anchor
    return loss


def snapshot_state(components):
    return {cname: {k: v.detach().cpu().clone()
                    for k, v in list(comp.named_parameters()) + list(comp.named_buffers())}
            for cname, comp in components.items()}


def restore_state(components, snap):
    with torch.no_grad():
        for cname, comp in components.items():
            live = dict(list(comp.named_parameters()) + list(comp.named_buffers()))
            for k, v in snap[cname].items():
                live[k].data.copy_(v.to(live[k].device))


def evaluate(exp, comps, wrapper, table, loader, args, n_steps):
    E.set_mode(comps, False)
    hs = C.install_ranges(comps, table, n_steps, args.a_bits, args.groups)
    m = E.eval_pass(exp, wrapper, loader, args.n_samples, args.seed)["metrics"]
    for h in hs:
        h.remove()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt", required=True, help="directory of the FP32 forecaster")
    ap.add_argument("--out", default="results", help="directory of the post-training stage")
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train-batches", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=1.0, help="anchor weight lambda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=100)
    args = ap.parse_args()
    tag = f"w{args.w_bits}a{args.a_bits}"

    exp, comps = E.build(args.dataset, args.seed, args.ckpt, args.batch_size, args.out)
    wrapper = E.TrainTWrapper(exp.model)
    n_steps = exp.model.num_timesteps
    with open(os.path.join(args.out, f"ranges_a{args.a_bits}.json")) as f:
        table = json.load(f)
    size = E.storage_accounting(comps, args.w_bits)["total"]
    teacher = attach_quantizers(comps, args.w_bits, os.path.join(args.out, f"round_w{args.w_bits}.pt"))
    opt = torch.optim.Adam([p for comp in comps.values() for p in comp.parameters()], lr=args.lr)

    torch.manual_seed(args.seed)
    v = evaluate(exp, comps, wrapper, table, exp.val_loader, args, n_steps)
    best = {"crps": v["crps"], "epoch": 0, "state": snapshot_state(comps)}
    curve = [v["crps"]]
    print(f"[ft]     post-training  val CRPS {v['crps']:.4f}", flush=True)
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        E.set_mode(comps, True)
        th = install_train_hooks(comps, table, args.groups, n_steps, args.a_bits)
        for i, (bx, by, _, _, bxm, bym) in enumerate(exp.train_loader):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            bx, by = bx.to(exp.device).float(), by.to(exp.device).float()
            bxm, bym = bxm.to(exp.device).float(), bym.to(exp.device).float()
            loss = anchored_elbo(exp, wrapper, teacher, args.anchor, bx, by, bxm, bym)
            opt.zero_grad()
            loss.backward()
            opt.step()
        for h in th:
            h.remove()
        v = evaluate(exp, comps, wrapper, table, exp.val_loader, args, n_steps)
        curve.append(v["crps"])
        print(f"[ft]     epoch {epoch}        val CRPS {v['crps']:.4f}", flush=True)
        if v["crps"] < best["crps"]:
            best = {"crps": v["crps"], "epoch": epoch, "state": snapshot_state(comps)}
    seconds = time.perf_counter() - t0
    restore_state(comps, best["state"])
    test = evaluate(exp, comps, wrapper, table, exp.test_loader, args, n_steps)
    print(f"[ft]     selected epoch {best['epoch']}  test CRPS {test['crps']:.4f}  "
          f"MSE {test['mse']:.4f}", flush=True)

    torch.save({"state": best["state"], "ranges": table, "w_bits": args.w_bits,
                "a_bits": args.a_bits, "groups": args.groups},
               os.path.join(args.out, f"dpqdiff_{tag}.pt"))
    res = {"dataset": args.dataset, "seed": args.seed, "bits": tag, "test": test,
           "val_curve": curve, "selected_epoch": best["epoch"], "seconds": seconds,
           "fp32_bytes": size["fp32_bytes"], "quant_bytes": size["quant_bytes"]}
    with open(os.path.join(args.out, f"dpqdiff_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
