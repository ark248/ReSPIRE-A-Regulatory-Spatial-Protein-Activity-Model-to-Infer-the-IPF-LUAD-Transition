import os
import sys
import argparse
import time

import scanpy as sc
import pandas as pd
import pyviper


#helpers to load function

def load_interactome(network_path: str, name: str, regul_size: int):
    #Load an ARACNe3 TSV and wrap it as a pyviper Interactome.
    print(f"  Loading {name}: {network_path}")
    if not os.path.isfile(network_path):
        print(f"ERROR: Network file not found: {network_path}")
        sys.exit(1)
    df = pd.read_csv(network_path, sep="\t")
    regulon = pyviper.pp.aracne3_to_regulon(net_file=None, net_df=df, regul_size=regul_size)
    interactome = pyviper.Interactome(name, regulon)
    print(f"    {interactome.size()} regulons loaded")
    return interactome


#main
def run_metaviper(
    input_h5ad: str,
    network_stromal: str,
    network_epithelial: str,
    network_immune: str,
    output_h5ad: str,
    njobs: int,
    regul_size: int,
):
    t0 = time.time()


    #load the spatial expression data
    print(f"\n[1/4] Loading spatial data: {input_h5ad}")
    if not os.path.isfile(input_h5ad):
        print(f"ERROR: Input H5AD not found: {input_h5ad}")
        sys.exit(1)
    adata = sc.read_h5ad(input_h5ad)
    print(f"  {adata.n_obs:,} spots  x  {adata.n_vars:,} genes")
    print(f"  First 3 barcodes: {list(adata.obs_names[:3])}")

    #loading the GRN interactomes
    print(f"\n[2/4] Loading GRN interactomes (regul_size={regul_size})...")
    interactome_stromal    = load_interactome(network_stromal,    "lung_stromal",    regul_size)
    interactome_epithelial = load_interactome(network_epithelial, "lung_epithelial", regul_size)
    interactome_immune     = load_interactome(network_immune,     "lung_immune",     regul_size)

    #filtering network targets to the genes present in the spatial data
    print(f"\n[3/4] Filtering network targets to match expression data...")
    for itm in [interactome_stromal, interactome_epithelial, interactome_immune]:
        itm.filter_targets(adata.var_names)
    print("  Done.")

    #run metaVIPER
    print(f"\n[4/4] Running metaVIPER ({njobs} cores)...")
    viper_t0 = time.time()

    nes = pyviper.viper(
        gex_data=adata,
        interactome=[interactome_stromal, interactome_epithelial, interactome_immune],
        enrichment="area",
        eset_filter=False,
        njobs=njobs,
        verbose=True,
    )

    print(f"  Completed in {(time.time() - viper_t0) / 60:.1f} min")
    print(f"  Output shape: {nes.n_obs:,} spots  x  {nes.n_vars:,} proteins")
    #checking barcodes
    if list(adata.obs_names) == list(nes.obs_names):
        print("  Barcode check: PASSED")
    else:
        print("  WARNING: output barcodes do not match input — check pyviper version")

    #save
    os.makedirs(os.path.dirname(os.path.abspath(output_h5ad)), exist_ok=True)
    print(f"\nSaving to: {output_h5ad}")
    nes.write_h5ad(output_h5ad)

    elapsed = (time.time() - t0) / 60
    print(f"  Total runtime : {elapsed:.1f} min")
    print(f"  Output        : {output_h5ad}")
    print(f"  Spots         : {nes.n_obs:,}")
    print(f"  Proteins      : {nes.n_vars:,}")
    


#CLI

def parse_args():
    p = argparse.ArgumentParser(description="Run metaVIPER on a Visium HD H5AD")
    p.add_argument("--input_h5ad",           required=True)
    p.add_argument("--network_stromal",       required=True,
                   help="Path to stromal ARACNe3 TSV (consolidated-net_defaultid.tsv)")
    p.add_argument("--network_epithelial",    required=True,
                   help="Path to epithelial ARACNe3 TSV")
    p.add_argument("--network_immune",        required=True,
                   help="Path to immune ARACNe3 TSV")
    p.add_argument("--output_h5ad",           required=True,
                   help="Path to write NES output H5AD")
    p.add_argument("--njobs",    type=int, default=32,
                   help="CPU cores for metaVIPER (default: 32)")
    p.add_argument("--regul_size", type=int, default=100,
                   help="Max targets per regulon (default: 100)")
    return p.parse_args()


def main():
    args = parse_args()
    run_metaviper(
        input_h5ad=args.input_h5ad,
        network_stromal=args.network_stromal,
        network_epithelial=args.network_epithelial,
        network_immune=args.network_immune,
        output_h5ad=args.output_h5ad,
        njobs=args.njobs,
        regul_size=args.regul_size,
    )


if __name__ == "__main__":
    main()
