#!/usr/bin/env python3
"""
MEG Encoding — BERT Layer-wise Ridge Regression (GPU)
======================================================
Equivalent of meg_encoding_ridge_regression_bert_layers.ipynb
but GPU-accelerated via torch.linalg.svd.

Input:
  notebooks/bert-base-lw-20.npy       -- BERT features, 12 layers
  notebooks/sub{01-27}-meg-data.npy   -- preprocessed MEG, 27 subjects

Output:
  meg_predictions/bert_L{00-11}/
    corr_sub{01-27}.npy   -- Pearson r, shape (16848,) = (208*81,)
    r2_sub{01-27}.npy     -- R2,        shape (16848,)

Method (matches paper Appendix E exactly):
  - SVD ridge regression
  - 4-fold leave-one-story-out CV
  - 16 lambda values log-spaced 10^1 to 10^3
  - Per-voxel lambda selection
  - Final weights refit on all training data with best lambda per voxel

GPU memory: capped at ~25GB via voxel chunking
CPU cores:  limit via taskset when launching

Usage:
  conda activate meg_env
  nohup taskset -c 0-15 python meg_encoding_bert_gpu.py > bert_encoding_log.txt 2>&1 &
  echo $!
"""

import numpy as np
import torch
import os
import time
from pathlib import Path
from sklearn.model_selection import KFold

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PROJECT_DIR   = Path("/home/mtech1/Desktop/meg_project")
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
PRED_DIR      = PROJECT_DIR / "meg_predictions"

FEATURE_FILE  = NOTEBOOKS_DIR / "bert-base-lw-20.npy"

N_LAYERS   = 12
N_SUBJECTS = 27
N_VOXELS   = 208 * 81   # 16848

SUBJECTS = [str(i).zfill(2) for i in range(1, N_SUBJECTS + 1)]

# Lambda grid: 16 values log-spaced 10^1 to 10^3 (paper Appendix E)
LAMBDAS = np.logspace(1, 3, 16).astype(np.float32)

# Voxel chunk size for GPU — keeps memory under 25GB
# At n_train~6000, float32: 6000 * 4000 * 4 bytes = ~96MB per chunk, well within budget
GPU_VOXEL_CHUNK = 4000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── GPU RIDGE REGRESSION ─────────────────────────────────────────────────────

def cross_val_ridge_gpu(X_train, Y_train):
    """
    SVD ridge regression with per-voxel lambda selection on GPU.
    Matches cross_val_ridge(..., method='svd') from the original notebooks.

    X_train: np.float32 (n_train, n_feat)
    Y_train: np.float32 (n_train, n_voxels)
    Returns:
        weights:      np.float32 (n_feat, n_voxels)
        best_lambdas: np.float32 (n_voxels,)
    """
    n_train, n_feat = X_train.shape
    n_vox = Y_train.shape[1]
    n_lam = len(LAMBDAS)

    X_gpu = torch.tensor(X_train, dtype=torch.float32, device=device)

    # CV error accumulator — kept on CPU
    cv_error = np.zeros((n_lam, n_vox), dtype=np.float32)

    # 4-fold CV within training data for lambda selection
    kf = KFold(n_splits=4)
    for trn_idx, val_idx in kf.split(np.arange(n_train)):
        X_trn = X_gpu[trn_idx]
        X_val = X_gpu[val_idx]

        # SVD once per fold — X_trn is small (n_train*0.75 x 768)
        U, s, Vh = torch.linalg.svd(X_trn, full_matrices=False)

        for v_start in range(0, n_vox, GPU_VOXEL_CHUNK):
            v_end  = min(v_start + GPU_VOXEL_CHUNK, n_vox)
            Y_trn  = torch.tensor(Y_train[trn_idx][:, v_start:v_end],
                                  dtype=torch.float32, device=device)
            Y_val  = torch.tensor(Y_train[val_idx][:, v_start:v_end],
                                  dtype=torch.float32, device=device)
            Y_var  = Y_val.var(dim=0).clamp(min=1e-10)
            UtY    = U.T @ Y_trn   # (k, v_chunk)

            for lam_idx, lmbda in enumerate(LAMBDAS):
                d      = s / (s ** 2 + lmbda)
                W      = Vh.T @ (d.unsqueeze(1) * UtY)    # (p, v_chunk)
                Y_pred = X_val @ W                          # (n_val, v_chunk)
                ss_res = (Y_val - Y_pred).pow(2).mean(dim=0)
                r2     = 1.0 - ss_res / Y_var
                cv_error[lam_idx, v_start:v_end] += (1.0 - r2).cpu().numpy()

            del Y_trn, Y_val, UtY
            torch.cuda.empty_cache()

    # Best lambda index per voxel
    best_lam_idx = np.argmin(cv_error, axis=0)   # (n_vox,)

    # Refit on ALL training data with best lambda per voxel
    U, s, Vh = torch.linalg.svd(X_gpu, full_matrices=False)
    weights   = np.zeros((n_feat, n_vox), dtype=np.float32)

    for lam_idx in range(n_lam):
        vox_mask = best_lam_idx == lam_idx
        if not vox_mask.any():
            continue
        lmbda = LAMBDAS[lam_idx]
        d     = s / (s ** 2 + lmbda)

        for v_start in range(0, n_vox, GPU_VOXEL_CHUNK):
            v_end       = min(v_start + GPU_VOXEL_CHUNK, n_vox)
            chunk_mask  = vox_mask[v_start:v_end]
            if not chunk_mask.any():
                continue
            Y_chunk = torch.tensor(Y_train[:, v_start:v_end][:, chunk_mask],
                                   dtype=torch.float32, device=device)
            W_chunk = Vh.T @ (d.unsqueeze(1) * (U.T @ Y_chunk))
            weights[:, v_start:v_end][:, chunk_mask] = W_chunk.cpu().numpy()
            del Y_chunk, W_chunk
            torch.cuda.empty_cache()

    best_lambdas = np.array([LAMBDAS[i] for i in best_lam_idx])
    return weights, best_lambdas


# ─── PEARSON R ────────────────────────────────────────────────────────────────

def pearson_r(y_real, y_pred):
    """Vectorized Pearson r across all voxels. Matches corr() in original code."""
    real_z = (y_real - y_real.mean(0)) / (y_real.std(0) + 1e-10)
    pred_z = (y_pred - y_pred.mean(0)) / (y_pred.std(0) + 1e-10)
    return np.mean(real_z * pred_z, axis=0)


# ─── PER SUBJECT PER LAYER ────────────────────────────────────────────────────

def encode_subject_layer(sub, layer_idx, features, data, out_dir):
    corr_path = out_dir / f"corr_sub{sub}.npy"
    r2_path   = out_dir / f"r2_sub{sub}.npy"

    if corr_path.exists() and r2_path.exists():
        print(f"    sub{sub} L{layer_idx:02d}: already done, skipping")
        return

    all_preds = []
    all_reals = []

    kf = KFold(n_splits=4)
    for train_folds, test_fold in kf.split(np.arange(4)):
        x_train = np.concatenate(
            [np.array(features[i][layer_idx]) for i in train_folds], axis=0
        ).astype(np.float32)
        x_test  = np.array(features[test_fold[0]][layer_idx]).astype(np.float32)

        y_train = np.concatenate(
            [data[i].reshape(len(data[i]), N_VOXELS) for i in train_folds],
            axis=0
        ).astype(np.float32)
        y_test  = data[test_fold[0]].reshape(
            len(data[test_fold[0]]), N_VOXELS
        ).astype(np.float32)

        weights, _ = cross_val_ridge_gpu(x_train, y_train)
        y_pred     = x_test @ weights

        all_preds.append(y_pred)
        all_reals.append(y_test)

    all_reals = np.vstack(all_reals)
    all_preds = np.vstack(all_preds)

    corr_vals = pearson_r(all_reals, all_preds)
    ss_res    = np.mean((all_reals - all_preds) ** 2, axis=0)
    ss_tot    = np.var(all_reals, axis=0)
    r2_vals   = np.nan_to_num(1.0 - ss_res / (ss_tot + 1e-10))

    np.save(str(corr_path), np.nan_to_num(corr_vals))
    np.save(str(r2_path),   np.nan_to_num(r2_vals))
    print(f"    sub{sub} L{layer_idx:02d}: "
          f"max_r={corr_vals.max():.4f}  mean_r={corr_vals.mean():.5f}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MEG Encoding — BERT (GPU)")
    print(f"  Features: {FEATURE_FILE.name}")
    print(f"  Subjects: {N_SUBJECTS}  Layers: 0-{N_LAYERS-1}")
    print(f"  Lambdas:  {LAMBDAS[0]:.1f}–{LAMBDAS[-1]:.1f}  ({len(LAMBDAS)} values)")
    print(f"  Device:   {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:      {props.name}, {props.total_memory/1e9:.1f}GB")
    print("=" * 60)

    print(f"\nLoading BERT features...")
    features = np.load(str(FEATURE_FILE), allow_pickle=True)
    print(f"  Word counts per story: {[np.array(features[s][0]).shape[0] for s in range(4)]}")

    t_total = time.time()

    for layer_idx in range(N_LAYERS):
        print(f"\n{'─'*50}")
        print(f"Layer {layer_idx:02d}")
        t_layer = time.time()

        out_dir = PRED_DIR / f"bert_L{layer_idx:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for sub in SUBJECTS:
            meg_path = NOTEBOOKS_DIR / f"sub{sub}-meg-data.npy"
            if not meg_path.exists():
                print(f"    sub{sub}: MEG file not found, skipping")
                continue

            t_sub = time.time()
            data  = np.load(str(meg_path), allow_pickle=True)
            encode_subject_layer(sub, layer_idx, features, data, out_dir)
            print(f"    sub{sub} finished in {time.time()-t_sub:.0f}s")
            del data

        print(f"Layer {layer_idx:02d} complete in "
              f"{(time.time()-t_layer)/60:.1f} min")

    elapsed = (time.time() - t_total) / 3600
    print(f"\nAll done in {elapsed:.2f} hours")
    print(f"Results saved to: {PRED_DIR}/bert_L*/")


if __name__ == "__main__":
    main()
