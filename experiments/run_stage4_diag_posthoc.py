"""Post-hoc sample diagnostics for a Stage 4B-3 pilot checkpoint.

Loads the EMA weights from ``models/diffusion_od_pilot/best.pt`` (or
another --ckpt), draws N samples conditioned on the first N val slices,
inverse-transforms to count space, and reports both the continuous-float
nonzero ratio AND the rounded-integer-count nonzero ratio.

Motivation: the in-loop ``marginal_compare`` uses ``np.count_nonzero``
on continuous floats. Because the forward normalization passes through
float32, ``inverse_transform`` of a zero entry returns ~1e-9 instead of
exactly 0, so real-data ``nonzero_ratio`` reads as 1.0 and the
generated-sample ratio (~23 %) is also continuous, not rounded counts.
The acceptance metric for the pilot has to be on rounded integer counts.

Run:
    python -m experiments.run_stage4_diag_posthoc \
        --ckpt models/diffusion_od_pilot/best.pt --n_samples 16

Writes ``results/stage4/train_pilot/diagnostics_posthoc.json`` (or the
sibling ``train_<profile>/`` for non-pilot checkpoints).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from experiments.run_stage4_train import (
    PROFILES,
    _TorchODDataset,
    _build_unet,
    _set_seed,
)
from src.data.od_dataset import ODDataset
from src.models.diffusion import GaussianDiffusion
from src.models.ema import EMA

REPO = Path(__file__).resolve().parents[1]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def _percentiles(arr: np.ndarray) -> dict[str, float]:
    qs = [0, 25, 50, 75, 90, 95, 99, 99.9, 100]
    vals = np.percentile(arr, qs)
    keys = ["p0", "p25", "p50", "p75", "p90", "p95", "p99", "p99_9", "max"]
    return {k: float(v) for k, v in zip(keys, vals)}


def _value_counts_top10(arr: np.ndarray) -> list[tuple[int, int]]:
    unique, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)
    return [(int(unique[i]), int(counts[i])) for i in order[:10]]


def _real_val_counts_raw(val_base: ODDataset) -> np.ndarray:
    """Read the raw int OD slices for val days, without round-tripping
    through normalization. This is the ground-truth integer reference."""
    raw_list: list[np.ndarray] = []
    for start in val_base._starts:
        sl = np.asarray(val_base._od[start : start + val_base.window], dtype=np.int64)
        raw_list.append(sl[0])  # window=1
    return np.stack(raw_list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt", type=Path,
        default=REPO / "models" / "diffusion_od_pilot" / "best.pt",
        help="Path to a saved checkpoint. The pilot directory is "
             "models/diffusion_od_<profile>/ for a successful run; the "
             "2026-05-20 invalid-split pilot lives at "
             "models/diffusion_od_pilot_invalid_old_split_debug/best.pt.",
    )
    parser.add_argument("--profile", default="pilot", choices=sorted(PROFILES))
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "diffusion.yaml")
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path. Default: "
                             "results/stage4/train_<profile>/diagnostics_posthoc.json")
    args = parser.parse_args()

    profile = dict(PROFILES[args.profile])
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  ckpt: {args.ckpt}")

    od_path = _resolve(cfg["input"]["od_path"])
    meta_path = _resolve(cfg["input"]["meta_path"])
    train_base = ODDataset(
        od_path, meta_path, "train",
        window=cfg["data"]["window"],
        pad_multiple=cfg["data"]["pad_multiple"],
        clip_val=cfg["data"]["clip_val"],
        split_cfg=cfg["data"]["split"],
    )
    val_base = ODDataset(
        od_path, meta_path, "val",
        window=cfg["data"]["window"],
        pad_multiple=cfg["data"]["pad_multiple"],
        norm_stats=train_base.norm_stats,
        split_cfg=cfg["data"]["split"],
    )
    val_loader = DataLoader(
        _TorchODDataset(val_base), batch_size=int(profile["batch_size"]),
        shuffle=False, num_workers=0,
    )

    # Real ground-truth integer OD slices for the val split.
    real_int = _real_val_counts_raw(val_base)
    print(f"real val slices: {real_int.shape}  dtype={real_int.dtype}")

    # Build model + EMA, load checkpoint, install EMA weights.
    model = _build_unet(profile, int(cfg["data"]["window"]),
                        int(cfg["model"]["time_emb_dim"]), device)
    diff = GaussianDiffusion(
        num_train_timesteps=int(profile["num_train_timesteps"]),
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    ).to(device)
    ema = EMA(model, decay=float(profile["ema_decay"])).to(device)
    ckpt = torch.load(args.ckpt, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    print(f"loaded step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")
    ema.store(model)
    ema.copy_to(model)
    model.eval()

    # Pull N conditions from the val loader.
    hours: list[int] = []
    dows: list[int] = []
    is_wks: list[int] = []
    for x, cond in val_loader:
        hours.extend(cond["hour"].tolist())
        dows.extend(cond["day_of_week"].tolist())
        is_wks.extend(cond["is_weekend"].tolist())
        if len(hours) >= args.n_samples:
            break
    hours = hours[: args.n_samples]
    dows = dows[: args.n_samples]
    is_wks = is_wks[: args.n_samples]

    # Sample in chunks of batch_size.
    chunk = int(profile["batch_size"])
    torch.manual_seed(args.seed)
    gen_chunks: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, args.n_samples, chunk):
            end = min(start + chunk, args.n_samples)
            n_c = end - start
            hour_t = torch.tensor(hours[start:end], device=device, dtype=torch.long)
            dow_t = torch.tensor(dows[start:end], device=device, dtype=torch.long)
            iw_t = torch.tensor(is_wks[start:end], device=device, dtype=torch.long)
            shape = (n_c, 1, train_base.pad_size, train_base.pad_size)
            samples = diff.ddim_sample(
                model, shape, hour_t, dow_t, iw_t,
                num_inference_steps=int(profile["num_inference_steps"]),
                guidance_scale=float(profile["guidance_scale"]),
            )
            samples_np = samples.detach().cpu().numpy()
            for i in range(n_c):
                counts = train_base.inverse_transform(samples_np[i])  # [1, Z, Z]
                gen_chunks.append(counts[0])
    dt = time.time() - t0
    print(f"sampled {args.n_samples} slices in {dt:.1f}s")

    gen_cont = np.stack(gen_chunks)              # [N, Z, Z] continuous floats
    gen_round = np.rint(np.maximum(gen_cont, 0.0)).astype(np.int64)

    # --- nonzero ratios ---
    cont_nz = float(np.count_nonzero(gen_cont)) / float(gen_cont.size)
    round_nz = float(np.count_nonzero(gen_round)) / float(gen_round.size)
    real_nz = float(np.count_nonzero(real_int)) / float(real_int.size)

    # --- quantiles (continuous) ---
    cont_q = _percentiles(gen_cont.ravel())
    round_q = _percentiles(gen_round.ravel().astype(np.float64))
    real_q = _percentiles(real_int.ravel().astype(np.float64))

    # --- value_counts of rounded counts (top 10) ---
    round_top = _value_counts_top10(gen_round.ravel())
    real_top = _value_counts_top10(real_int.ravel().astype(np.int64))

    # --- per-sample stats ---
    per_sample: list[dict[str, float]] = []
    for i in range(gen_cont.shape[0]):
        s_cont = gen_cont[i]
        s_round = gen_round[i]
        per_sample.append({
            "i": int(i),
            "hour": int(hours[i]),
            "dow": int(dows[i]),
            "is_weekend": int(is_wks[i]),
            "cont_sum": float(s_cont.sum()),
            "cont_nonzero_ratio": float(np.count_nonzero(s_cont)) / float(s_cont.size),
            "cont_max": float(s_cont.max()),
            "round_sum": int(s_round.sum()),
            "round_nonzero_ratio": float(np.count_nonzero(s_round)) / float(s_round.size),
            "round_max": int(s_round.max()),
        })

    # Real reference per-sample (first 16 val slices).
    real_per_sample: list[dict[str, Any]] = []
    for i in range(min(args.n_samples, real_int.shape[0])):
        s = real_int[i]
        real_per_sample.append({
            "i": int(i),
            "sum": int(s.sum()),
            "nonzero_ratio": float(np.count_nonzero(s)) / float(s.size),
            "max": int(s.max()),
        })

    out_path = args.out
    if out_path is None:
        out_path = REPO / "results" / "stage4" / f"train_{args.profile}" / "diagnostics_posthoc.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "ckpt": str(args.ckpt),
        "step": int(ckpt["step"]),
        "val_loss": float(ckpt["val_loss"]),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "guidance_scale": float(profile["guidance_scale"]),
        "num_inference_steps": int(profile["num_inference_steps"]),
        "nonzero_ratio": {
            "real_int": real_nz,
            "gen_continuous": cont_nz,
            "gen_rounded": round_nz,
        },
        "quantiles": {
            "gen_continuous": cont_q,
            "gen_rounded": round_q,
            "real_int": real_q,
        },
        "value_counts_top10": {
            "gen_rounded": round_top,
            "real_int": real_top,
        },
        "per_sample": per_sample,
        "real_per_sample_first_n": real_per_sample,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")

    # --- pretty console summary ---
    print("\n--- nonzero ratio ---")
    print(f"  real (int)           : {real_nz:.6%}")
    print(f"  gen continuous (>0)  : {cont_nz:.6%}")
    print(f"  gen rounded counts   : {round_nz:.6%}")
    print("\n--- gen continuous quantiles ---")
    for k, v in cont_q.items():
        print(f"  {k:<6}: {v:.6f}")
    print("\n--- gen rounded counts quantiles ---")
    for k, v in round_q.items():
        print(f"  {k:<6}: {v:.0f}")
    print("\n--- real int quantiles ---")
    for k, v in real_q.items():
        print(f"  {k:<6}: {v:.0f}")
    print("\n--- gen rounded top-10 value counts ---")
    for val, n in round_top:
        print(f"  {val:>4}: {n}")
    print("\n--- real int top-10 value counts ---")
    for val, n in real_top:
        print(f"  {val:>4}: {n}")
    print("\n--- per-sample (gen) ---")
    for s in per_sample:
        print(
            f"  i={s['i']:>2} hour={s['hour']:>2} dow={s['dow']} wknd={s['is_weekend']}  "
            f"cont(sum={s['cont_sum']:.2f} nz={s['cont_nonzero_ratio']:.4%} max={s['cont_max']:.2f})  "
            f"round(sum={s['round_sum']} nz={s['round_nonzero_ratio']:.4%} max={s['round_max']})"
        )


if __name__ == "__main__":
    main()
