"""Trains a full-precision DiffPTS forecaster with the official implementation and its
released configuration, and copies the selected checkpoint to --out.

    python train.py --dataset ETTh1 --seed 1 --out checkpoints/ETTh1/1
"""
import argparse
import os
import shutil

import engine as E

FILES = ("model.pth", "cond_pred_model.pth", "cond_pred_model_g.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="results/fp32_runs")
    args = ap.parse_args()

    exp = E.DiffPTSForecast(
        dataset_type=args.dataset, data_path=E.DATA_PATH, save_dir=args.runs, device="cuda:0",
        batch_size=32, horizon=1, pred_len=E.pred_len_of(args.dataset), windows=168,
        cond_backbone=E.backbone_of(args.dataset),
    )
    # the checkpoint is selected on the validation split; the test split is scored
    # once, for the selected checkpoint
    test, load = exp._test, exp._load_best_model
    final = {"on": False}

    def load_best(*a, **k):
        final["on"] = True
        return load(*a, **k)

    def test_final(*a, **k):
        return test(*a, **k) if final["on"] else {"crps": float("nan")}

    exp._load_best_model, exp._test = load_best, test_final
    res = exp.run(seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    for f in FILES:
        shutil.copy(os.path.join(exp.run_save_dir, f), os.path.join(args.out, f))
    print(f"[fp32]   {args.dataset} seed {args.seed}: {res}", flush=True)


if __name__ == "__main__":
    main()
