"""PyTorch reference for the Chronos-2 Zig runner: same CSV formats as reference.py.

Pass the checkpoint id or a local directory as `model` (e.g. `amazon/chronos-2`). Every
(series, variate) row is forecast independently, matching the runner's v1 univariate path.
"""
import argparse
import sys
import time

import numpy as np
from chronos import Chronos2Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("series")
ap.add_argument("--horizon", type=int, required=True)
for k in ("past", "known", "cat-past", "cat-known"):
    ap.add_argument("--" + k)
ap.add_argument("--max-context", type=int)
ap.add_argument("--block-patches", type=int, default=32)
ap.add_argument("--no-symmetric", action="store_true")
ap.add_argument("--no-clip", action="store_true")
ap.add_argument("--out", required=True)
ap.add_argument("--time", action="store_true")
a = ap.parse_args()

if any((a.past, a.known, a.cat_past, a.cat_known)):
    sys.exit("Chronos-2 reference: covariates are not supported in the v1 runner")


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
pipe = Chronos2Pipeline.from_pretrained(a.model)

# One independent 1-D input per (series, variate) row, as the Zig runner does.
series = [ctx[s, v, :] for s in range(ctx.shape[0]) for v in range(ctx.shape[1])]
t0 = time.perf_counter()
preds = pipe.predict(series, prediction_length=a.horizon)
if a.time:
    print(f"torch forecast={time.perf_counter() - t0:.3f}s", file=sys.stderr)

nseries, nvars = ctx.shape[0], ctx.shape[1]
with open(a.out, "w") as fh:
    for idx, pred in enumerate(preds):
        s, v = divmod(idx, nvars)
        q = pred[0].detach().cpu().numpy()  # (num_quantiles, horizon)
        for t in range(a.horizon):
            fh.write(f"{s},{v},{t + 1}," + ",".join(repr(float(x)) for x in q[:, t]) + "\n")
