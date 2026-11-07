"""Calibration of DPQ-Diff on the validation windows.

The full-precision forecaster is run with S_cal sample paths, and the inputs of every
quantized layer are recorded, per reverse step for the denoiser and once per forecast
for the backbones. The recorded values give
  * error-minimizing clipping ranges (grid of 36 ratios between 0.3 and 1),
  * the step-group lookup table of the denoiser (G contiguous groups of steps), and
  * the layer inputs on which the rounding of every weight is learned.
"""
import numpy as np
import torch
import torch.nn as nn

import engine as E


class SampleObserver:
    """Abs-max and a bounded random subsample of values per (tensor, step)."""

    def __init__(self, per_call=1024, max_calls=24, seed=0):
        self.per_call, self.max_calls = per_call, max_calls
        self.samples, self.amax, self.calls = {}, {}, {}
        self.gen = torch.Generator(device="cuda").manual_seed(seed)

    def record(self, name, group, x):
        v = x.detach().reshape(-1)
        a = float(v.abs().max())
        d = self.amax.setdefault(name, {})
        d[group] = max(d.get(group, 0.0), a)
        key = (name, group)
        n_calls = self.calls.get(key, 0)
        self.calls[key] = n_calls + 1
        if v.numel() > self.per_call:
            idx = torch.randint(0, v.numel(), (self.per_call,), device=v.device,
                                generator=self.gen)
            v = v[idx]
        lst = self.samples.setdefault(name, {}).setdefault(group, [])
        if len(lst) < self.max_calls:
            lst.append(v.float().cpu())
        else:  # reservoir replacement keeps the subsample uniform over calls
            j = int(torch.randint(0, n_calls + 1, (1,)).item())
            if j < self.max_calls:
                lst[j] = v.float().cpu()


class InputCache:
    """Reservoir of layer inputs (rows for Linear, sequences for Conv1d) over calls."""

    def __init__(self, max_calls=48, rows=256, seqs=2, seed=0):
        self.max_calls, self.rows, self.seqs = max_calls, rows, seqs
        self.data, self.calls = {}, {}
        self.gen = torch.Generator(device="cuda").manual_seed(seed + 17)

    def record(self, name, mod, x):
        x = x.detach()
        if isinstance(mod, nn.Linear):
            v = x.reshape(-1, x.shape[-1])
            k = min(self.rows, v.shape[0])
        else:
            v = x
            k = min(self.seqs, v.shape[0])
        idx = torch.randint(0, v.shape[0], (k,), device=v.device, generator=self.gen)
        v = v[idx].float().cpu()
        n = self.calls.get(name, 0)
        self.calls[name] = n + 1
        lst = self.data.setdefault(name, [])
        if len(lst) < self.max_calls:
            lst.append(v)
        else:
            j = int(torch.randint(0, n + 1, (1,)).item())
            if j < self.max_calls:
                lst[j] = v

    def tensor(self, name):
        return torch.cat(self.data[name], dim=0)


def calibrate(exp, comps, wrapper, calib_samples, seed):
    """Records the inputs of every quantized layer on the validation windows."""
    obs = SampleObserver(seed=seed)
    cache = InputCache(seed=seed)
    hs = []
    for cname, mname, mod in E.weight_modules(comps):
        name = f"{cname}.{mname}"
        use_t = cname == "denoiser"

        def hook(module, args, name=name, use_t=use_t):
            t = E._T_STATE["t"]
            grp = f"t{int(t)}" if (use_t and t is not None) else "cond"
            obs.record(name, grp, args[0])
            cache.record(name, module, args[0])
            return None
        hs.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        E.eval_pass(exp, wrapper, exp.val_loader, calib_samples, seed, scored=False)
    for h in hs:
        h.remove()
    return obs, cache


def mse_clip(values, amax, qmax, grid=np.linspace(0.30, 1.0, 36)):
    """Clipping range that minimizes the quantization MSE of `values`."""
    x = values.cuda().double()
    best, best_c = None, amax
    for a in grid:
        c = max(a * amax, 1e-12)
        s = c / qmax
        err = ((torch.clamp(torch.round(x / s), -qmax, qmax) * s - x) ** 2).mean().item()
        if best is None or err < best:
            best, best_c = err, c
    return best_c


def trajectory_ranges(obs, n_steps, a_bits):
    """One error-minimizing range per tensor over all reverse steps; these ranges
    quantize the layer inputs while the rounding of the weights is learned."""
    qmax = E.qmax_of(a_bits)
    out = {}
    for name, groups in obs.samples.items():
        if name.startswith("denoiser."):
            parts, amax = [], 0.0
            for t in range(n_steps):
                g = f"t{t}"
                if g not in groups:
                    continue
                parts.append(torch.cat(groups[g]))
                amax = max(amax, obs.amax[name][g])
            if not parts:
                continue
            out[name] = mse_clip(torch.cat(parts), amax, qmax)
        elif "cond" in groups:
            out[name] = mse_clip(torch.cat(groups["cond"]), obs.amax[name]["cond"], qmax)
    return out


def group_of(t, G, n):
    return min(t * G // n, G - 1)


def step_group_ranges(comps, obs, n_steps, a_bits=8, G=5):
    """Lookup table of activation ranges: one error-minimizing range per step group
    for every denoiser layer and one range per tensor for the backbones."""
    qmax = E.qmax_of(a_bits)
    table = {}
    for cname, mname, _ in E.weight_modules(comps):
        name = f"{cname}.{mname}"
        if name not in obs.samples:
            continue
        smp, amx = obs.samples[name], obs.amax[name]
        if cname == "denoiser":
            row = {}
            for g in range(G):
                steps = [t for t in range(n_steps) if group_of(t, G, n_steps) == g and f"t{t}" in smp]
                amax = max(amx[f"t{t}"] for t in steps)
                vals = torch.cat([torch.cat(smp[f"t{t}"]) for t in steps])
                row[f"t{g}"] = float(mse_clip(vals, amax, qmax))
            table[name] = row
        elif smp.get("cond"):
            table[name] = {"cond": float(mse_clip(torch.cat(smp["cond"]), amx["cond"], qmax))}
    return table


def install_ranges(comps, table, n_steps, a_bits=8, G=5):
    """Fake-quantizes the input of every layer with its range from the lookup table;
    a denoiser layer uses the range of the step group of the running reverse step."""
    q = E.qmax_of(a_bits)
    hs = []
    for cname, mname, mod in E.weight_modules(comps):
        r = table.get(f"{cname}.{mname}")
        if r is None:
            continue
        if cname == "denoiser":
            tbl = [max(r[f"t{group_of(t, G, n_steps)}"], 1e-12) / q for t in range(n_steps)]

            def hook(module, args, tbl=tbl):
                t = E._T_STATE["t"]
                if t is None:
                    return None
                s = tbl[int(t)]
                return (torch.clamp(torch.round(args[0] / s), -q, q) * s,) + tuple(args[1:])
        else:
            s0 = max(r["cond"], 1e-12) / q

            def hook(module, args, s=s0):
                return (torch.clamp(torch.round(args[0] / s), -q, q) * s,) + tuple(args[1:])
        hs.append(mod.register_forward_pre_hook(hook))
    return hs
