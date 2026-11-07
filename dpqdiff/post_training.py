"""Post-training stage of DPQ-Diff.

Calibrates a trained full-precision DiffPTS forecaster on its validation windows,
learns the output-preserving rounding of every weight of the full pipeline (mean and
variance backbones, endpoint heads, and denoiser), and builds the step-aware lookup
table of activation ranges. The quantized forecaster is scored on the validation and
test windows with the sampling noise of the full-precision forecaster.

    python post_training.py --dataset ETTh1 --ckpt checkpoints/ETTh1/1 --w-bits 4
"""
import argparse
import json
import os
import time

import torch

import calibration as C
import engine as E
from rounding import learn_rounding


def fmt(m):
    return f"CRPS {m['crps']:.4f}  MSE {m['mse']:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt", required=True, help="directory of the FP32 forecaster")
    ap.add_argument("--out", default="results")
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--groups", type=int, default=5, help="step groups G of the denoiser")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=100, help="sample paths S")
    ap.add_argument("--calib-samples", type=int, default=100, help="sample paths S_cal")
    ap.add_argument("--iters", type=int, default=1500, help="rounding steps per layer")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = f"w{args.w_bits}a{args.a_bits}"

    exp, comps = E.build(args.dataset, args.seed, args.ckpt, args.batch_size, args.out)
    wrapper = E.ModelTWrapper(exp.model)
    n_steps = exp.model.num_timesteps

    fp_path = os.path.join(args.out, "fp32.json")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            fp32 = json.load(f)
    else:
        fp32 = {"val": E.eval_pass(exp, wrapper, exp.val_loader, args.n_samples, args.seed)["metrics"],
                "test": E.eval_pass(exp, wrapper, exp.test_loader, args.n_samples, args.seed)["metrics"],
                "storage": E.storage_accounting(comps, 32)["total"]["fp32_bytes"]}
        with open(fp_path, "w") as f:
            json.dump(fp32, f, indent=1)
    print(f"[fp32]   test {fmt(fp32['test'])}", flush=True)

    t0 = time.perf_counter()
    obs, cache = C.calibrate(exp, comps, wrapper, args.calib_samples, args.seed)
    clips = C.trajectory_ranges(obs, n_steps, args.a_bits)
    table = C.step_group_ranges(comps, obs, n_steps, args.a_bits, args.groups)
    t_cal = time.perf_counter() - t0
    rounding = {}
    for cname, mname, mod in list(E.weight_modules(comps)):
        name = f"{cname}.{mname}"
        if name in cache.data:
            wq, _, _ = learn_rounding(mod, cache.tensor(name), args.w_bits, clips.get(name),
                                      args.a_bits, args.iters, seed=args.seed)
        else:
            wq = E.qdq_weight_per_channel(mod.weight.data, E.qmax_of(args.w_bits))
        rounding[name] = wq.detach().cpu()
    torch.cuda.synchronize()
    t_all = time.perf_counter() - t0
    print(f"[post]   calibration {t_cal:.1f}s, calibration and rounding {t_all:.1f}s", flush=True)

    with torch.no_grad():
        for cname, mname, mod in E.weight_modules(comps):
            mod.weight.data.copy_(rounding[f"{cname}.{mname}"].to(mod.weight.device, mod.weight.dtype))
    hs = C.install_ranges(comps, table, n_steps, args.a_bits, args.groups)
    val = E.eval_pass(exp, wrapper, exp.val_loader, args.n_samples, args.seed)["metrics"]
    test = E.eval_pass(exp, wrapper, exp.test_loader, args.n_samples, args.seed)["metrics"]
    for h in hs:
        h.remove()
    size = E.storage_accounting(comps, args.w_bits)["total"]

    torch.save(rounding, os.path.join(args.out, f"round_w{args.w_bits}.pt"))
    with open(os.path.join(args.out, f"ranges_a{args.a_bits}.json"), "w") as f:
        json.dump(table, f)
    res = {"dataset": args.dataset, "seed": args.seed, "bits": tag, "groups": args.groups,
           "val": val, "test": test, "fp32_bytes": size["fp32_bytes"],
           "quant_bytes": size["quant_bytes"], "seconds": t_all}
    with open(os.path.join(args.out, f"post_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[post]   test {fmt(test)}  size {size['fp32_bytes'] / 1e6:.2f} -> "
          f"{size['quant_bytes'] / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
