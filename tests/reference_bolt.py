"""PyTorch reference for the Chronos-Bolt Zig runner: same CSV formats as reference.py.

Pass the checkpoint id or a local directory as `model` (e.g. `amazon/chronos-bolt-small`). Every
(series, variate) row is forecast independently, matching the runner's univariate path.
"""
import argparse
import sys
import time

import numpy as np
import torch
from chronos import ChronosBoltPipeline

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("series")
ap.add_argument("--horizon", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--time", action="store_true")
a = ap.parse_args()


def blocks(path):
    out, cur = [], []
    for line in open(path).read().splitlines() + [""]:
        s = line.strip()
        if s.startswith("#"):
            continue
        if not s:
            if cur:
                out.append(cur)
                cur = []
            continue
        toks = s.split(",") if "," in s else s.split()
        cur.append([float(t) if t.strip().lower() not in ("", "nan", "na") else np.nan for t in toks])
    return np.array(out, dtype=np.float32)


ctx = blocks(a.series)
pipe = ChronosBoltPipeline.from_pretrained(a.model, torch_dtype=torch.float32)

# One independent 1-D input per (series, variate) row, as the Zig runner does.
series = [torch.tensor(ctx[s, v, :]) for s in range(ctx.shape[0]) for v in range(ctx.shape[1])]
t0 = time.perf_counter()
preds = pipe.predict(series, prediction_length=a.horizon)  # (n, num_quantiles, horizon)
if a.time:
    print(f"torch forecast={time.perf_counter() - t0:.3f}s", file=sys.stderr)

nvars = ctx.shape[1]
with open(a.out, "w") as fh:
    for idx in range(preds.shape[0]):
        s, v = divmod(idx, nvars)
        q = preds[idx].detach().cpu().numpy()
        for t in range(a.horizon):
            fh.write(f"{s},{v},{t + 1}," + ",".join(repr(float(x)) for x in q[:, t]) + "\n")
