"""Prints the results of run.sh: python report.py results/<dataset>/<seed>"""
import json
import os
import sys

out = sys.argv[1]
with open(os.path.join(out, "fp32.json")) as f:
    fp32 = json.load(f)
rows = [("FP32", fp32["test"], fp32["storage"])]
for b in (8, 4):
    p = os.path.join(out, f"dpqdiff_w{b}a8.json")
    if os.path.exists(p):
        with open(p) as f:
            r = json.load(f)
        rows.append((f"DPQ-Diff W{b}A8", r["test"], r["quant_bytes"]))
print(f"{'model':<16}{'CRPS':>9}{'MSE':>9}{'size (MB)':>11}{'ratio':>8}")
for name, m, size in rows:
    print(f"{name:<16}{m['crps']:>9.4f}{m['mse']:>9.4f}{size / 1e6:>11.2f}"
          f"{fp32['storage'] / size:>7.1f}x")
