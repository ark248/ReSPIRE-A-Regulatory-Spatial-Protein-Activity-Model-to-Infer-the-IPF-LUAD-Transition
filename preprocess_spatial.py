import os
import sys
import argparse
import traceback
from pathlib import Path

import scanpy as sc
import pandas as pd
import anndata as ad


#quick helpers to load in data faster
def load_data(path_str: str, fmt: str, sample_id: str) -> ad.AnnData:
    path = Path(path_str)
    print(f"[{sample_id}] Loading '{fmt}' from: {path}")

    if fmt == "dense_txt":
        adata = sc.read_text(path, delimiter="\t").T

    elif fmt == "dense_csv":
        adata = sc.read_csv(path)

    elif fmt == "dense_tsv":
        df = pd.read_csv(path, sep="\t", index_col=0)
        adata = ad.AnnData(df.T)

    elif fmt == "standard_10x":
        adata = sc.read_10x_mtx(path, var_names="gene_symbols", cache=False)

    elif fmt == "custom_10x":
        mtx_files = list(path.glob("*matrix*.mtx*"))
        if not mtx_files:
            available = [f.name for f in path.glob("*")]
            raise FileNotFoundError(
                f"No .mtx file found in {path}. Contents: {available}"
            )
        print(f"  -> matrix : {mtx_files[0].name}")
        adata = sc.read_mtx(mtx_files[0]).T

        feat_files = list(path.glob("*features*.tsv*")) or list(path.glob("*genes*.tsv*"))
        if feat_files:
            print(f"  -> features: {feat_files[0].name}")
            genes = pd.read_csv(feat_files[0], sep="\t", header=None)
            adata.var_names = genes[1].values if genes.shape[1] > 1 else genes[0].values

        bar_files = list(path.glob("*barcodes*.tsv*"))
        if bar_files:
            print(f"  -> barcodes: {bar_files[0].name}")
            barcodes = pd.read_csv(bar_files[0], sep="\t", header=None)
            adata.obs_names = barcodes[0].values

    else:
        raise ValueError(
            f"Unknown file_type '{fmt}'. "
            "Choose from: standard_10x, custom_10x, dense_txt, dense_tsv, dense_csv"
        )

    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    print(f"  Loaded: {adata.n_obs:,} cells  x  {adata.n_vars:,} genes")
    return adata


def preprocess(adata: ad.AnnData, sample_id: str) -> tuple[ad.AnnData, list]:
    print("\nPreprocessing...")

    sc.pp.filter_cells(adata, min_genes=200)
    print(f"  After cell filter  : {adata.n_obs:,} cells")

    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After gene filter  : {adata.n_vars:,} genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    hvg_list = adata.var_names[adata.var["highly_variable"]].tolist()
    print(f"  HVGs identified    : {len(hvg_list):,}  (full gene set retained)")

    return adata, hvg_list


#main

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Visium HD data → H5AD")
    p.add_argument("--input_path", required=True,
                   help="Path to input data (folder for 10x, file for dense formats)")
    p.add_argument("--sample_id",  required=True,
                   help="Human-readable sample identifier (used in logs)")
    p.add_argument("--file_type",  required=True,
                   choices=["standard_10x", "custom_10x", "dense_txt", "dense_tsv", "dense_csv"],
                   help="Input format")
    p.add_argument("--output_h5ad", required=True,
                   help="Path to write the output .h5ad file")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        adata = load_data(args.input_path, args.file_type, args.sample_id)
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        traceback.print_exc()
        sys.exit(1)

    adata, hvg_list = preprocess(adata, args.sample_id)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_h5ad)), exist_ok=True)
    print(f"\nSaving to: {args.output_h5ad}")
    adata.write_h5ad(args.output_h5ad)

    print("\n" + "=" * 70)
    print(f"  Sample : {args.sample_id}")
    print(f"  Output : {args.output_h5ad}")
    print(f"  Cells  : {adata.n_obs:,}")
    print(f"  Genes  : {adata.n_vars:,}")
    print(f"  HVGs   : {len(hvg_list):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()