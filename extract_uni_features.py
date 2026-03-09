import os
import sys
import json
import argparse

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import pyvips
except ImportError:
    print("ERROR: pyvips not installed. Run: pip install pyvips")
    sys.exit(1)

try:
    from deepspot.utils.utils_image_hd import get_morphology_model_and_preprocess, crop_tile
except ImportError:
    print("ERROR: deepspot not installed or not on PYTHONPATH.")
    sys.exit(1)


#tile dataset
class VisiumTileDataset(Dataset):

    def __init__(self, barcodes, coords_map, image, spot_diam, preprocess_fn):
        self.barcodes     = barcodes
        self.coords_map   = coords_map
        self.image        = image
        self.spot_diam    = spot_diam
        self.preprocess   = preprocess_fn

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, i):
        bc  = self.barcodes[i]
        row = self.coords_map[bc]
        try:
            tile = crop_tile(self.image, row["x_px"], row["y_px"], self.spot_diam)
        except Exception:
            tile = np.zeros((self.spot_diam, self.spot_diam, 3), dtype=np.uint8)
        return self.preprocess(tile)


#extracting features

def extract_features(
    visium_root: str,
    pa_h5ad: str,
    save_dir: str,
    model_name: str,
    batch_size: int,
    num_workers: int,
):
    os.makedirs(save_dir, exist_ok=True)
    feature_file = os.path.join(save_dir, f"{model_name}_features.h5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # if the bank already exists, this portion will skip over
    if os.path.exists(feature_file):
        print(f"Feature bank already exists: {feature_file}")
        print("Skipping extraction. Delete the file to re-run.")
    else:
        print(f"Creating feature bank ({model_name.upper()})...")

        # load barcodes from NES h5ad
        adata = sc.read_h5ad(pa_h5ad)
        unique_barcodes = list(adata.obs_names.unique())

        # load spatial positions
        pos_path = os.path.join(visium_root, "spatial", "tissue_positions.parquet")
        pos_df = pd.read_parquet(pos_path)
        if "barcode" in pos_df.columns:
            pos_df = pos_df.set_index("barcode")

        # intersect with barcodes that have positions
        common = pos_df.index.intersection(unique_barcodes)
        pos_df = pos_df.loc[common]
        unique_barcodes = list(common)
        print(f"  Spots with positions: {len(unique_barcodes):,}")

        # compute pixel coordinates from scale factors
        sf_path = os.path.join(visium_root, "spatial", "scalefactors_json.json")
        with open(sf_path, "r") as f:
            sf = json.load(f)
        hires_scalef = sf["tissue_hires_scalef"]
        pos_df["x_px"] = pos_df["pxl_col_in_fullres"] * hires_scalef
        pos_df["y_px"] = pos_df["pxl_row_in_fullres"] * hires_scalef
        spot_diam = max(32, int(sf["spot_diameter_fullres"] * hires_scalef))

        # load model
        print(f"Loading {model_name.upper()} model...")
        model, preprocess, _ = get_morphology_model_and_preprocess(model_name, device="cpu")
        model = model.to(device).eval()

        # streaming image
        img_path = os.path.join(visium_root, "spatial", "tissue_hires_image.png")
        main_img = pyvips.Image.new_from_file(img_path, access="random")
        coords_map = pos_df[["x_px", "y_px"]].to_dict("index")

        # dataloader
        dataset = VisiumTileDataset(unique_barcodes, coords_map, main_img, spot_diam, preprocess)
        loader  = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        # determine embedding dim from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            emb_dim = model(dummy).shape[1]
        print(f"  Embedding dimension: {emb_dim}")

        # extract and stream into HDF5
        print(f"Extracting {len(unique_barcodes):,} spots...")
        with h5py.File(feature_file, "w") as f:
            feat_dset = f.create_dataset("features", (len(unique_barcodes), emb_dim), dtype="float32")
            bc_dset   = f.create_dataset("barcodes", (len(unique_barcodes),),
                                         dtype=h5py.special_dtype(vlen=str))
            bc_dset[:] = unique_barcodes

            start = 0
            with torch.no_grad():
                for batch in tqdm(loader, desc="  Extracting"):
                    emb = model(batch.to(device)).cpu().numpy()
                    feat_dset[start : start + len(emb)] = emb
                    start += len(emb)

        del model
        torch.cuda.empty_cache()
        print(f"Feature bank saved: {feature_file}")
    #load and report
   
    with h5py.File(feature_file, "r") as f:
        n_spots, emb_dim = f["features"].shape
    print(f"\nFeature bank summary:")
    print(f"  File     : {feature_file}")
    print(f"  Spots    : {n_spots:,}")
    print(f"  Emb dim  : {emb_dim}")

#cli

def parse_args():
    p = argparse.ArgumentParser(description="Extract UNI features for Visium HD spots")
    p.add_argument("--visium_root",  required=True,
                   help="Visium HD root dir (contains spatial/ subfolder)")
    p.add_argument("--pa_h5ad",      required=True,
                   help="NES H5AD produced by run_metaviper.py (used for barcode list)")
    p.add_argument("--save_dir",     required=True,
                   help="Directory to write <model_name>_features.h5")
    p.add_argument("--model_name",   default="uni",
                   choices=["uni", "conch", "resnet"],
                   help="Foundation model to use (default: uni)")
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--num_workers",  type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    extract_features(
        visium_root=args.visium_root,
        pa_h5ad=args.pa_h5ad,
        save_dir=args.save_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()