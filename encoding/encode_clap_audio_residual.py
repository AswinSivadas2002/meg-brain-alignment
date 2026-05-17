#!/usr/bin/env python3
"""
MEG Encoding — CLAP Audio Residual Ridge Regression (GPU)
==========================================================
Right side of Figure 4 — CLAP audio with BERT (text) removed.

If predictivity shows no improvement after text removal → confirms
speech embeddings don't gain meaningful semantic benefits from text
(paper's finding: asymmetric knowledge transfer).

Input:
  notebooks/clap_audio_residual_bert_synth.npy  -- residual features, 5 layers
  notebooks/sub{01-27}-meg-data.npy       -- preprocessed MEG, 27 subjects

Output:
  meg_predictions/clap_audio_residual_synth_L{00-04}/
    corr_sub{01-27}.npy   -- Pearson r, shape (16848,)
    r2_sub{01-27}.npy     -- R2,        shape (16848,)

Usage:
  conda activate meg_env
  nohup taskset -c 0-15 python3 -u meg_encoding_clap_audio_residual_gpu.py > clap_audio_residual_encoding_log.txt 2>&1 &
  echo $!
"""

import numpy as np
import torch
import time
from pathlib import Path
from sklearn.model_selection import KFold

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PROJECT_DIR   = Path("/home/mtech1/Desktop/meg_project")
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
PRED_DIR      = PROJECT_DIR / "meg_predictions"

FEATURE_FILE  = NOTEBOOKS_DIR / "clap_audio_residual_bert_synth.npy"

N_LAYERS   = 5    # HTSAT stages
N_SUBJECTS = 27
N_VOXELS   = 208 * 81   # 16848

SUBJECTS = [str(i).zfill(2) for i in range(1, N_SUBJECTS + 1)]
LAMBDAS  = np.logspace(1, 3, 16).astype(np.float32)
GPU_VOXEL_CHUNK = 4000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── GPU RIDGE REGRESSION (identical to other encoding scripts) ───────────────

def cross_val_ridge_gpu(X_train, Y_train):
    n_train, n_feat = X_train.shape
    n_vox = Y_train.shape[1]
    n_lam = len(LAMBDAS)

    X_gpu    = torch.tensor(X_train, dtype=torch.float32, device=device)
    cv_error = np.zeros((n_lam, n_vox), dtype=np.float32)

    kf = KFold(n_splits=4)
    for trn_idx, val_idx in kf.split(np.arange(n_train)):
        X_trn = X_gpu[trn_idx]
        X_val = X_gpu[val_idx]
        U, s, Vh = torch.linalg.svd(X_trn, full_matrices=False)

        for v_start in range(0, n_vox, GPU_VOXEL_CHUNK):
            v_end  = min(v_start + GPU_VOXEL_CHUNK, n_vox)
            Y_trn  = torch.tensor(Y_train[trn_idx][:, v_start:v_end],
                                  dtype=torch.float32, device=device)
            Y_val  = torch.tensor(Y_train[val_idx][:, v_start:v_end],
                                  dtype=torch.float32, device=device)
            Y_var  = Y_val.var(dim=0).clamp(min=1e-10)
            UtY    = U.T @ Y_trn

            for lam_idx, lmbda in enumerate(LAMBDAS):
                d      = s / (s ** 2 + lmbda)
                W      = Vh.T @ (d.unsqueeze(1) * UtY)
                Y_pred = X_val @ W
                ss_res = (Y_val - Y_pred).pow(2).mean(dim=0)
                r2     = 1.0 - ss_res / Y_var
                cv_error[lam_idx, v_start:v_end] += (1.0 - r2).cpu().numpy()

            del Y_trn, Y_val, UtY
            torch.cuda.empty_cache()

    best_lam_idx = np.argmin(cv_error, axis=0)
    U, s, Vh = torch.linalg.svd(X_gpu, full_matrices=False)
    weights  = np.zeros((n_feat, n_vox), dtype=np.float32)

    for lam_idx in range(n_lam):
        vox_mask = best_lam_idx == lam_idx
        if not vox_mask.any():
            continue
        lmbda = LAMBDAS[lam_idx]
        d     = s / (s ** 2 + lmbda)

        for v_start in range(0, n_vox, GPU_VOXEL_CHUNK):
            v_end      = min(v_start + GPU_VOXEL_CHUNK, n_vox)
            chunk_mask = vox_mask[v_start:v_end]
            if not chunk_mask.any():
                continue
            Y_chunk = torch.tensor(Y_train[:, v_start:v_end][:, chunk_mask],
                                   dtype=torch.float32, device=device)
            W_chunk = Vh.T @ (d.unsqueeze(1) * (U.T @ Y_chunk))
            weights[:, v_start:v_end][:, chunk_mask] = W_chunk.cpu().numpy()
            del Y_chunk, W_chunk
            torch.cuda.empty_cache()

    return weights, np.array([LAMBDAS[i] for i in best_lam_idx])


def pearson_r(y_real, y_pred):
    real_z = (y_real - y_real.mean(0)) / (y_real.std(0) + 1e-10)
    pred_z = (y_pred - y_pred.mean(0)) / (y_pred.std(0) + 1e-10)
    return np.mean(real_z * pred_z, axis=0)


def encode_subject_layer(sub, layer_idx, features, data, out_dir):
    corr_path = out_dir / f"corr_sub{sub}.npy"
    r2_path   = out_dir / f"r2_sub{sub}.npy"

    if corr_path.exists() and r2_path.exists():
        print(f"    sub{sub} L{layer_idx:02d}: already done, skipping")
        return

    all_preds, all_reals = [], []
    kf = KFold(n_splits=4)

    for train_folds, test_fold in kf.split(np.arange(4)):
        # Residual features are already stacked arrays
        x_train = np.concatenate(
            [features[i][layer_idx] for i in train_folds], axis=0
        ).astype(np.float32)
        x_test  = features[test_fold[0]][layer_idx].astype(np.float32)

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


def main():
    print("=" * 60)
    print("MEG Encoding — CLAP Audio Residual (GPU)")
    print(f"  Features: {FEATURE_FILE.name}")
    print(f"  Subjects: {N_SUBJECTS}  Layers: 0-{N_LAYERS-1}")
    print(f"  Device:   {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:      {props.name}, {props.total_memory/1e9:.1f}GB")
    print("=" * 60)

    if not FEATURE_FILE.exists():
        print(f"\nERROR: {FEATURE_FILE} not found.")
        print("Run compute_residuals_audio.py first.")
        return

    print(f"\nLoading CLAP audio residual features...")
    features = np.load(str(FEATURE_FILE), allow_pickle=True)
    for l in range(N_LAYERS):
        print(f"  L{l:02d}: {features[0][l].shape}")

    t_total = time.time()

    for layer_idx in range(N_LAYERS):
        print(f"\n{'─'*50}")
        print(f"Layer {layer_idx:02d}  (dim={features[0][layer_idx].shape[1]})")
        t_layer = time.time()

        out_dir = PRED_DIR / f"clap_audio_residual_synth_L{layer_idx:02d}"
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

    print(f"\nAll done in {(time.time()-t_total)/3600:.2f} hours")
    print(f"Results: {PRED_DIR}/clap_audio_residual_synth_L*/")
    print("\nNext: compare vs clap_audio_L*/ to see if text removal affects audio")


if __name__ == "__main__":
    main()
