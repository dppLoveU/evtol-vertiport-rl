"""Stage-4A smoke test: validate ODDataset on the real OD tensor.

This is a *validation-only* script. It does NOT train, build, or save
any model. It builds the train/val/test ``ODDataset`` over the real
``data/processed/od_evtol.npy``, caches the train-split norm stats, and
prints the shapes / padding / normalization / sparsity figures Stage 4B
will rely on.

Run:
    python -m experiments.run_stage4_dataset_smoke
    python -m experiments.run_stage4_dataset_smoke --config configs/diffusion.yaml
    python -m experiments.run_stage4_dataset_smoke \\
        --config configs/diffusion_12km_zpin_weighted.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data.od_dataset import ODDataset, save_norm_stats

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "diffusion.yaml"


def _resolve(path_str: str) -> Path:
    """Resolve a yaml path relative to repo root if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    od_path = _resolve(cfg["input"]["od_path"])
    meta_path = _resolve(cfg["input"]["meta_path"])
    data_cfg = cfg["data"]
    window = int(data_cfg["window"])
    pad_multiple = int(data_cfg["pad_multiple"])
    clip_val = float(data_cfg["clip_val"])
    norm_stats_path = _resolve(data_cfg["norm_stats_path"])
    split_cfg = data_cfg["split"]
    # Stage 4B-5B: normalization scheme. Default keeps old-config
    # behaviour unchanged (configs/diffusion.yaml /
    # configs/diffusion_12km.yaml carry no `scheme` key).
    scheme = data_cfg.get("scheme", "global_clip")

    print("=" * 68)
    print("STAGE 4A  --  ODDataset smoke test")
    print("=" * 68)
    print(f"  od_path        : {od_path}")
    print(f"  meta_path      : {meta_path}")
    print(f"  window         : {window}")
    print(f"  pad_multiple   : {pad_multiple}")
    print(f"  clip_val       : {clip_val}")
    print(f"  scheme         : {scheme}")

    # --- raw tensor sparsity (mmap; streamed, never fully resident) ------
    od = np.load(od_path, mmap_mode="r")
    n_total = int(od.size)
    n_nonzero = int(np.count_nonzero(od))
    print("\n--- raw OD tensor ---------------------------------------------")
    print(f"  shape          : {od.shape}  dtype={od.dtype}")
    print(f"  nonzero_ratio  : {n_nonzero / n_total:.6%}  ({n_nonzero} / {n_total})")
    print(f"  on-disk size   : {od_path.stat().st_size / 1e6:.1f} MB")

    # --- train split: builds + caches norm stats -------------------------
    train = ODDataset(
        od_path,
        meta_path,
        "train",
        window=window,
        pad_multiple=pad_multiple,
        clip_val=clip_val,
        scheme=scheme,
        split_cfg=split_cfg,
    )
    save_norm_stats(train.norm_stats, norm_stats_path)
    stats = train.norm_stats

    print("\n--- dimensions ------------------------------------------------")
    print(f"  T (n_slots)    : {train.n_slots}")
    print(f"  |Z| (n_zones)  : {train.n_zones}")
    print(f"  pad_size       : {train.pad_size}  ({train.n_zones} -> {train.pad_size})")
    pad_mb = window * train.pad_size**2 * 4 / 1e6
    print(f"  padded sample  : [{window}, {train.pad_size}, {train.pad_size}] "
          f"float32  ({pad_mb:.2f} MB)")

    print("\n--- norm stats (computed on TRAIN split) ----------------------")
    if scheme == "global_clip":
        print(f"  mu             : {stats['mu']:.6f}")
        print(f"  sigma          : {stats['sigma']:.6f}")
    else:  # zero_pinned_nonzero
        print(f"  mu_nz          : {stats['mu_nz']:.6f}")
        print(f"  sigma_nz       : {stats['sigma_nz']:.6f}")
    print(f"  clip_val       : {stats['clip_val']:.1f}")
    print(f"  scheme tag     : {stats.get('scheme', '(none)')}")
    print(f"  keys           : {sorted(stats.keys())}")
    print(f"  cached to      : {norm_stats_path}")

    # --- per-split datasets (val/test reuse the train stats) -------------
    val = ODDataset(od_path, meta_path, "val", window=window,
                    pad_multiple=pad_multiple, scheme=scheme,
                    norm_stats=stats, split_cfg=split_cfg)
    test = ODDataset(od_path, meta_path, "test", window=window,
                     pad_multiple=pad_multiple, scheme=scheme,
                     norm_stats=stats, split_cfg=split_cfg)

    print("\n--- splits ----------------------------------------------------")
    for ds in (train, val, test):
        s = ds.summary()
        print(f"  {s['split']:<5}: n_samples={s['n_samples']:>4}  "
              f"slot_range={s['split_slot_range']}")

    # --- sample inspection ----------------------------------------------
    print("\n--- sample checks ---------------------------------------------")
    for name, ds in (("train", train), ("val", val), ("test", test)):
        x, cond = ds[0]
        print(f"  {name}[0]: x.shape={x.shape}  dtype={x.dtype}  "
              f"min={x.min():.4f}  max={x.max():.4f}  mean={x.mean():.4f}")
        print(f"          finite={np.all(np.isfinite(x))}  condition={cond}")

    # --- zero-pin check (zpin scheme only) -------------------------------
    if scheme == "zero_pinned_nonzero":
        raw_10 = np.asarray(
            od[train._starts[10] : train._starts[10] + window], dtype=np.int64
        )
        x_10, _ = train[10]
        n_z = train.n_zones
        raw_inside = raw_10[..., :n_z, :n_z]
        x_inside = x_10[..., :n_z, :n_z]
        zero_mask = raw_inside == 0
        n_zero = int(zero_mask.sum())
        n_nz_in_sample = int((raw_inside > 0).sum())
        print("\n--- zpin pin check (train sample 10) --------------------------")
        if n_zero > 0:
            x_at_zero = x_inside[zero_mask]
            pin_ok = bool(np.all(x_at_zero == -1.0))
            print(f"  raw-zero entries : {n_zero}  pinned to -1.0: {pin_ok}  "
                  f"(min={float(x_at_zero.min()):.6f}, "
                  f"max={float(x_at_zero.max()):.6f})")
        else:
            print("  no zero entries in sample 10 (unexpected for sparse OD)")
        # And confirm the padded region is all -1.0 too.
        pad_block = np.concatenate(
            [x_10[..., n_z:, :].ravel(), x_10[..., :, n_z:].ravel()]
        )
        if pad_block.size > 0:
            pad_pin_ok = bool(np.all(pad_block == -1.0))
            print(f"  padded entries   : {pad_block.size}  pinned to -1.0: {pad_pin_ok}")
        print(f"  nonzero in slice : {n_nz_in_sample}")

    # --- inverse-transform round-trip -----------------------------------
    x, _ = train[10]
    back = train.inverse_transform(x)
    raw = np.asarray(od[train._starts[10] : train._starts[10] + window], dtype=np.float64)
    max_abs_err = float(np.max(np.abs(back - raw)))
    print("\n--- inverse transform (train sample 10) -----------------------")
    print(f"  recovered shape: {back.shape}  nonneg={np.all(back >= 0.0)}")
    print(f"  max|recovered - raw|: {max_abs_err:.4f}  "
          f"(nonzero clip_val={clip_val} -> large counts may not round-trip)")
    if scheme == "zero_pinned_nonzero":
        raw_zero_mask = raw == 0.0
        if raw_zero_mask.any():
            zero_back = back[raw_zero_mask]
            print(f"  raw-zero entries: {int(raw_zero_mask.sum())}  "
                  f"back min={float(zero_back.min()):.6e}  "
                  f"max={float(zero_back.max()):.6e}  "
                  f"all_zero={bool(np.all(zero_back == 0.0))}")

    print("\n" + "=" * 68)
    print("STAGE 4A smoke OK -- ODDataset builds, normalizes, pads, inverts.")
    print("=" * 68)


if __name__ == "__main__":
    main()
