

import argparse
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import anndata as ad
import yaml


NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
FEATURE_DIM      = 1024
N_SCALES         = 1 + len(NEIGHBOR_OFFSETS)              # 5 → 5120-dim output


#loaders
def _decode(raw):
    return np.array([b.decode("utf-8") if isinstance(b, bytes) else str(b) for b in raw])


def load_features(feature_path: str) -> dict:
    """Returns {barcode: np.array(1024,)}. First occurrence wins on duplicates."""
    with h5py.File(feature_path, "r") as f:
        barcodes = _decode(f["barcodes"][:])
        features = np.array(f["features"], dtype=np.float32)
    bc_to_feat = {}
    for i, bc in enumerate(barcodes):
        if bc not in bc_to_feat:
            bc_to_feat[bc] = features[i]
    return bc_to_feat


def load_positions(pos_path: str) -> tuple[dict, dict]:
    """Returns (barcode→grid_pos, grid_pos→barcode)."""
    df = pd.read_parquet(pos_path)
    bc_to_pos  = {}
    grid_to_bc = {}
    for _, row in df.iterrows():
        bc  = str(row["barcode"])
        pos = (int(row["array_row"]), int(row["array_col"]))
        if bc not in bc_to_pos:
            bc_to_pos[bc]  = pos
            grid_to_bc[pos] = bc
    return bc_to_pos, grid_to_bc


def load_targets(h5ad_path: str) -> tuple[dict, list]:
    """Returns ({barcode: np.array(n_proteins,)}, protein_names)."""
    adata = ad.read_h5ad(h5ad_path)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    X = X.astype(np.float32)
    bc_to_targ = {str(bc): X[i] for i, bc in enumerate(adata.obs_names)}
    return bc_to_targ, list(adata.var_names)


#per-sample builder
def build_sample(sample: dict) -> tuple[dict, list]:
    name = sample["name"]
    t0   = time.time()
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")

    bc_to_feat             = load_features(sample["feature_path"])
    bc_to_pos, grid_to_bc  = load_positions(sample["pos_path"])
    bc_to_targ, proteins   = load_targets(sample["h5ad_path"])

    print(f"  features  : {len(bc_to_feat):>8,} barcodes")
    print(f"  positions : {len(bc_to_pos):>8,} barcodes")
    print(f"  targets   : {len(bc_to_targ):>8,} barcodes")

    valid_barcodes = sorted(
        set(bc_to_feat) & set(bc_to_pos) & set(bc_to_targ)
    )
    print(f"  intersection              : {len(valid_barcodes):>8,} barcodes")

    n            = len(valid_barcodes)
    n_proteins   = len(proteins)
    features_out = np.zeros((n, N_SCALES * FEATURE_DIM), dtype=np.float32)
    targets_out  = np.zeros((n, n_proteins),              dtype=np.float32)
    missing      = defaultdict(int)

    for i, bc in enumerate(valid_barcodes):
        features_out[i, :FEATURE_DIM] = bc_to_feat[bc]
        targets_out[i]                 = bc_to_targ[bc]

        center_row, center_col = bc_to_pos[bc]
        for j, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
            nbr_bc = grid_to_bc.get((center_row + dr, center_col + dc))
            if nbr_bc and nbr_bc in bc_to_feat:
                s = (1 + j) * FEATURE_DIM
                features_out[i, s : s + FEATURE_DIM] = bc_to_feat[nbr_bc]
            else:
                missing[j] += 1

    neighbor_names = ["up", "down", "left", "right"]
    print("  Neighbor coverage (missing = edge / background):")
    for j, nm in enumerate(neighbor_names):
        pct = 100.0 * missing[j] / n
        print(f"    {nm:6s}: {missing[j]:>8,} missing  ({pct:.1f}%)")

    print(f"  Done in {time.time() - t0:.1f}s  —  "
          f"features {features_out.shape}, targets {targets_out.shape}")

    return {
        "features": torch.from_numpy(features_out),
        "targets":  torch.from_numpy(targets_out),
        "barcodes": valid_barcodes,
    }, proteins


#main
def parse_args():
    p = argparse.ArgumentParser(description="Build multiscale ReSPIRE feature .pt file")
    p.add_argument("--config",   required=True,
                   help="YAML file listing samples (name, feature_path, pos_path, h5ad_path)")
    p.add_argument("--output",   default="metaviperfeatures1.pt",
                   help="Output .pt file path (default: metaviperfeatures1.pt)")
    p.add_argument("--feat_dim", type=int, default=1024,
                   help="UNI embedding dimension per spot (default: 1024)")
    return p.parse_args()


def main():
    args = parse_args()

    global FEATURE_DIM, N_SCALES
    FEATURE_DIM = args.feat_dim
    N_SCALES    = 1 + len(NEIGHBOR_OFFSETS)

    print("=" * 60)
    print("  ReSPIRE — Multiscale Feature Construction")
    print(f"  Neighbor offsets : {NEIGHBOR_OFFSETS}")
    print(f"  Output dim       : {N_SCALES} × {FEATURE_DIM} = {N_SCALES * FEATURE_DIM}")
    print("=" * 60)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    samples = cfg["samples"]

    all_data     = {}
    barcode_map  = {}
    proteins     = None

    for sample in samples:
        result, proteins = build_sample(sample)
        all_data[sample["name"]]    = result
        barcode_map[sample["name"]] = result["barcodes"]

    payload = {
        "data": {
            name: {"features": d["features"], "targets": d["targets"]}
            for name, d in all_data.items()
        },
        "meta": {
            "proteins":             proteins,
            "barcodes_per_sample":  barcode_map,
            "neighbor_offsets":     NEIGHBOR_OFFSETS,
        },
    }

    torch.save(payload, args.output)
    print(f"\nSaved: {args.output}")
    print(f'  "input_dim": {N_SCALES * FEATURE_DIM}')


if __name__ == "__main__":
    main()