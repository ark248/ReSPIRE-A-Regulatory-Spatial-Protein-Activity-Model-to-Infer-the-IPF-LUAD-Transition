import argparse
import gc
import os
import time

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset

L.seed_everything(42)
torch.set_float32_matmul_precision("medium")

CONFIG = {
    "input_dim":          5120,
    "hidden_dims":        [512, 256],
    "dropout":            0.35,
    "batch_size":         4096,
    "lr":                 1e-4,
    "min_lr":             1e-6,
    "warmup_epochs":      3,
    "weight_decay":       1e-3,
    "epochs":             40,
    "patience":           12,
    "grad_clip":          1.0,
    "correlation_weight": 0.3,
    "min_protein_std":    0.05,
    "num_workers":        4,
    "output_dir":         "predictions",
}

SAMPLES = {
    "Lung1_S1": "LUAD",
    "Lung1_S2": "LUAD",
    "Lung2_S1": "LUAD",
    "Lung3_S1": "LUAD",
    "Lung6_S1": "Fibrosis",
    "Lung6_S2": "Fibrosis",
}

PANEL_PROTEINS = {
    "TF":      ["NKX2-1", "MYC", "STAT3", "YAP1", "TEAD1", "SMAD3", "SMAD4"],
    "coTF":    ["BRD4", "BRD2", "EP300", "CREBBP", "CTCF"],
    "SigSurf": ["EGFR", "PDGFRA", "PDGFRB", "TGFBR1", "TGFBR2", "CXCR4", "ITGB1"],
}
ALL_PANEL = [p for group in PANEL_PROTEINS.values() for p in group]


#dataset
class ProteinDataset(Dataset):
    def __init__(self, features_f16: torch.Tensor, targets_f16: torch.Tensor):
        self.x = features_f16
        self.y = targets_f16

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx].float()


#model architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.needs_proj = in_dim != out_dim
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        if self.needs_proj:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.proj(x) if self.needs_proj else x) + self.block(x)


class ProteinPredictor(L.LightningModule):
    def __init__(self, n_proteins: int, target_mean=None, target_std=None, protein_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=["target_mean", "target_std", "protein_names"])
        self.n_proteins    = n_proteins
        self.corr_weight   = CONFIG["correlation_weight"]
        self.protein_names = protein_names

        if target_mean is not None:
            self.register_buffer("target_mean", target_mean)
            self.register_buffer("target_std",  target_std)

        dims = [CONFIG["input_dim"]] + CONFIG["hidden_dims"]
        self.encoder = nn.Sequential(
            *[ResidualBlock(dims[i], dims[i + 1], CONFIG["dropout"])
              for i in range(len(dims) - 1)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.GELU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(dims[-1], n_proteins),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    @staticmethod
    def _pearson_r(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p   = pred   - pred.mean(0, keepdim=True)
        t   = target - target.mean(0, keepdim=True)
        num = (p * t).sum(0)
        den = p.pow(2).sum(0).sqrt() * t.pow(2).sum(0).sqrt() + 1e-8
        return (num / den).mean()

    def _loss(self, pred, target):
        mse = F.mse_loss(pred, target)
        if target.shape[0] >= 128:
            r = self._pearson_r(pred, target)
            return mse - self.corr_weight * r, mse, r
        return mse, mse, torch.tensor(0.0, device=pred.device)

    def training_step(self, batch, _):
        loss, _, r = self._loss(self(batch[0]), batch[1])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_r",    r,    prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, mse, r = self._loss(self(batch[0]), batch[1])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_mse",  mse, prog_bar=True, sync_dist=True)
        self.log("val_r",    r,   prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )

        def lr_lambda(epoch):
            if epoch < CONFIG["warmup_epochs"]:
                return 0.1 + 0.9 * (epoch / CONFIG["warmup_epochs"])
            progress = (epoch - CONFIG["warmup_epochs"]) / max(
                CONFIG["epochs"] - CONFIG["warmup_epochs"], 1
            )
            return max(
                CONFIG["min_lr"] / CONFIG["lr"],
                0.5 * (1 + np.cos(np.pi * progress)),
            )

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


#loading the data
def load_all_data(data_file: str):
    t0 = time.time()
    print(f"Loading: {data_file}")
    payload = torch.load(data_file, weights_only=False)
    data               = payload["data"]
    all_protein_names  = payload["meta"]["proteins"]
    barcodes_per_sample = payload["meta"].get("barcodes_per_sample", {})

    print(f"  Proteins: {len(all_protein_names):,}")
    for name, d in data.items():
        label = SAMPLES.get(name, "?")
        print(f"  {name:12s}  {len(d['features']):>7,} spots  [{label}]")
    print(f"  Loaded in {time.time() - t0:.0f}s\n")

    return data, all_protein_names, barcodes_per_sample


#normalization per-fold
def prepare_fold(data, all_protein_names, held_out_sample):
    #computing the welford online mean/std from the training samples only

    train_names   = [n for n in data.keys() if n != held_out_sample]
    input_dim     = CONFIG["input_dim"]
    n_all_proteins = len(all_protein_names)

    n_total     = 0
    input_mean  = np.zeros(input_dim,     dtype=np.float64)
    input_m2    = np.zeros(input_dim,     dtype=np.float64)
    target_mean = np.zeros(n_all_proteins, dtype=np.float64)
    target_m2   = np.zeros(n_all_proteins, dtype=np.float64)

    for name in train_names:
        x_np = data[name]["features"].numpy().astype(np.float64)
        y_np = data[name]["targets"].numpy().astype(np.float64)
        for i in range(len(x_np)):
            n_total += 1
            dx = x_np[i] - input_mean
            input_mean += dx / n_total
            input_m2   += dx * (x_np[i] - input_mean)
            dy = y_np[i] - target_mean
            target_mean += dy / n_total
            target_m2   += dy * (y_np[i] - target_mean)
        del x_np, y_np

    input_std  = np.clip(np.sqrt(input_m2  / max(n_total - 1, 1)), 1e-6, None).astype(np.float32)
    target_std = np.clip(np.sqrt(target_m2 / max(n_total - 1, 1)), 1e-6, None).astype(np.float32)
    input_mean  = input_mean.astype(np.float32)
    target_mean = target_mean.astype(np.float32)

    keep_mask    = target_std >= CONFIG["min_protein_std"]
    keep_indices = np.where(keep_mask)[0]
    protein_names    = [all_protein_names[i] for i in keep_indices]
    target_mean_kept = target_mean[keep_mask]
    target_std_kept  = target_std[keep_mask]

    def pack_samples(sample_names):
        xs, ys = [], []
        for name in sample_names:
            x = ((data[name]["features"].numpy() - input_mean) / input_std).astype(np.float16)
            y = ((data[name]["targets"].numpy()[:, keep_indices] - target_mean_kept) / target_std_kept).astype(np.float16)
            xs.append(torch.from_numpy(x))
            ys.append(torch.from_numpy(y))
        return torch.cat(xs), torch.cat(ys)

    train_x, train_y = pack_samples(train_names)
    val_x,   val_y   = pack_samples([held_out_sample])

    return (
        ProteinDataset(train_x, train_y),
        ProteinDataset(val_x,   val_y),
        protein_names,
        torch.from_numpy(target_mean_kept),
        torch.from_numpy(target_std_kept),
        keep_indices,
    )


#single fold
def train_fold(fold_idx: int, held_out: str, data, all_protein_names, output_dir: str):
    print(f"  Fold {fold_idx}  |  held-out: {held_out}  [{SAMPLES.get(held_out, '?')}]")

    train_ds, val_ds, protein_names, y_mean, y_std, keep_indices = \
        prepare_fold(data, all_protein_names, held_out)

    print(f"  Train spots : {len(train_ds):,}")
    print(f"  Val spots   : {len(val_ds):,}")
    print(f"  Proteins    : {len(protein_names):,}")

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True,
    )

    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    model = ProteinPredictor(
        n_proteins=len(protein_names),
        target_mean=y_mean,
        target_std=y_std,
        protein_names=protein_names,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=fold_dir, filename="best_model",
            monitor="val_loss", save_top_k=1, mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=CONFIG["patience"], mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs        = CONFIG["epochs"],
        gradient_clip_val = CONFIG["grad_clip"],
        callbacks         = callbacks,
        logger            = CSVLogger(fold_dir, name="metrics"),
        enable_progress_bar = True,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"\n  Best checkpoint : {callbacks[0].best_model_path}")
    print(f"  Best val_loss   : {callbacks[0].best_model_score:.4f}")

    # save predictions on the held-out sample
    best = ProteinPredictor.load_from_checkpoint(callbacks[0].best_model_path)
    best.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds.append(best(xb).numpy())
    preds = np.concatenate(preds, axis=0)
    pred_path = os.path.join(fold_dir, f"predictions_{held_out}.npy")
    np.save(pred_path, preds)
    print(f"  Predictions saved: {pred_path}  shape={preds.shape}")

    del model, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()


#cli

def parse_args():
    p = argparse.ArgumentParser(description="Train ReSPIRE MLP (LOO cross-validation)")
    p.add_argument("--data_file",  required=True,
                   help="Path to metaviperfeatures1.pt from build_multiscale_features.py")
    p.add_argument("--output_dir", default=CONFIG["output_dir"],
                   help=f"Output directory (default: {CONFIG['output_dir']})")
    p.add_argument("--fold", type=int, default=None,
                   help="Run a single fold by 0-based index. Omit to run all folds.")
    return p.parse_args()


def main():
    args = parse_args()

    # allow output_dir override from CLI while keeping CONFIG in sync
    CONFIG["data_file"]  = args.data_file
    CONFIG["output_dir"] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

   
    print(f"  Architecture : {CONFIG['input_dim']} → "
          f"{' → '.join(str(d) for d in CONFIG['hidden_dims'])} → n_proteins")
    print(f"  Epochs       : {CONFIG['epochs']}  |  Patience : {CONFIG['patience']}")
    print(f"  LR           : {CONFIG['lr']}  |  Batch : {CONFIG['batch_size']}")


    data, all_protein_names, _ = load_all_data(args.data_file)
    sample_names = list(data.keys())

    folds = list(enumerate(sample_names))
    if args.fold is not None:
        if args.fold >= len(folds):
            print(f"ERROR: --fold {args.fold} out of range (0–{len(folds)-1})")
            return
        folds = [folds[args.fold]]

    t0 = time.time()
    for fold_idx, held_out in folds:
        train_fold(fold_idx, held_out, data, all_protein_names, args.output_dir)

    print(f"\nAll folds complete in {(time.time() - t0) / 60:.1f} min")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()