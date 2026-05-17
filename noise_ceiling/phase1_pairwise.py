#!/usr/bin/env python3
"""
Noise Ceiling — Matches Paper's Noise Ceiling.ipynb Exactly
============================================================
Srijith et al. EMNLP 2025.

Phase 1: Kernel ridge brain→brain predictions (GPU + global PCA)
  - For each source→target pair (702 total):
    - PCA compress both to N_PCA=1000 dims (global PCA, fit once)
    - Kernel ridge: source → target in PCA space (GPU)
    - 4-fold leave-one-story-out CV
    - Pearson r per voxel → predict_{target}_with_{source}.npy

Phase 2: Bootstrap exponential fit (matches paper Cell 4 exactly)
  - For each target subject:
    - All combinations of source subjects at each subsample size
    - Bootstrap 100 times per voxel
    - Fit f(s) = v0 * (1 - exp(-s/t0))
    - Median v0 = ceiling estimate
  - Parallelized across subjects with joblib

Output:
  noise_ceiling_preds/
    predict_{target}_with_{source}.npy  -- Pearson r (16848,)
    subject_{target}_kernel_ridge.npy   -- ceiling per subject (16848,)
    ceiling_upper.npy / ceiling_kfold.npy / ceiling_lower.npy

Usage:
  nohup taskset -c 0-15 python3 -u noise_ceiling.py > noise_ceiling_log.txt 2>&1 &
"""

import numpy as np
import torch
import time
import gc
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd
from scipy.optimize import curve_fit
from itertools import combinations
from joblib import Parallel, delayed

PROJECT_DIR   = Path("/home/mtech1/Desktop/meg_project")
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
PRED_DIR      = PROJECT_DIR / "noise_ceiling_preds"

N_SUBJECTS      = 27
N_PCA           = 1000
N_BOOTSTRAP     = 100
N_JOBS          = 16

SUBJECTS    = [str(i).zfill(2) for i in range(1, N_SUBJECTS + 1)]
SUBJECTS_INT = list(range(1, N_SUBJECTS + 1))
LAMBDAS     = torch.logspace(-6, 8, steps=15)
device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Cap GPU memory to 10GB to leave headroom for other processes
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(10/33.7)


# ── GLOBAL PCA (fit once on all data) ────────────────────────────────────────

def fit_global_pca(all_flat, n_components):
    """
    Fit PCA using randomized SVD on CPU.
    (231147, 16848) is too large for GPU SVD (14.7GB).
    Randomized SVD is fast for top-k components and gives
    identical results to full PCA for top-1000 components.
    """
    print(f"  Fitting global PCA n={n_components} (CPU randomized SVD)...")
    mean   = all_flat.mean(axis=0)
    Y_cent = (all_flat - mean).astype(np.float32)
    t0 = time.time()
    _, _, Vt = randomized_svd(Y_cent, n_components=n_components,
                               random_state=42, n_iter=4)
    print(f"  SVD done in {time.time()-t0:.1f}s")
    return Vt.astype(np.float32), mean


def project(Y, components, mean):
    return ((Y - mean) @ components.T).astype(np.float32)

def unproject(Y_pca, components, mean):
    return (Y_pca @ components + mean).astype(np.float32)


# ── KERNEL RIDGE (GPU) ────────────────────────────────────────────────────────

def kernel_ridge_gpu(K_train, Y_pca, K_test, n_splits=4):
    """Kernel ridge in PCA space. GPU-accelerated."""
    lambdas = LAMBDAS.to(device)
    K_tr = torch.tensor(K_train, dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y_pca,   dtype=torch.float32, device=device)
    K_te = torch.tensor(K_test,  dtype=torch.float32, device=device)

    n_train = K_tr.shape[0]
    n_comps = Y_tr.shape[1]
    kf      = KFold(n_splits=n_splits)
    val_err = torch.zeros((len(lambdas), n_comps), device=device)

    for tr_idx, vl_idx in kf.split(np.arange(n_train)):
        Ktr = K_tr[np.ix_(tr_idx, tr_idx)]
        Kvl = K_tr[np.ix_(vl_idx, tr_idx)]
        Ytr = Y_tr[tr_idx]
        Yvl = Y_tr[vl_idx]
        for i, lmbda in enumerate(lambdas):
            I     = torch.eye(len(tr_idx), device=device)
            alpha = torch.linalg.solve(Ktr + lmbda * I, Ytr)
            Yp    = Kvl @ alpha
            ss_r  = torch.mean((Yvl - Yp) ** 2, dim=0)
            ss_t  = torch.var(Yvl, dim=0, unbiased=False).clamp(min=1e-8)
            val_err[i] += 1.0 - (1.0 - ss_r / ss_t)

    best  = torch.argmin(val_err, dim=0)
    I_f   = torch.eye(n_train, device=device)
    pred  = torch.zeros((K_te.shape[0], n_comps), device=device)

    for i, lmbda in enumerate(lambdas):
        mask = best == i
        if mask.sum() == 0:
            continue
        alpha         = torch.linalg.solve(K_tr + lmbda * I_f, Y_tr[:, mask])
        pred[:, mask] = K_te @ alpha

    result = pred.cpu().numpy()
    del K_tr, Y_tr, K_te, pred
    torch.cuda.empty_cache()
    return result


# ── PHASE 1: BRAIN→BRAIN PREDICTIONS ─────────────────────────────────────────

def run_phase1(data_dict, components, mean):
    """
    Compute all 702 source→target predictions using global PCA + kernel ridge.
    """
    total = N_SUBJECTS * (N_SUBJECTS - 1)
    done  = sum(1 for t in SUBJECTS for s in SUBJECTS
                if s != t and (PRED_DIR/f"predict_{t}_with_{s}.npy").exists())

    print(f"\n{'='*60}")
    print(f"Phase 1: Brain→Brain ({total} pairs, {done} already done)")
    print(f"{'='*60}")

    t0 = time.time()
    nd = 0

    for target in SUBJECTS:
        for source in SUBJECTS:
            if source == target:
                continue
            fname = PRED_DIR / f"predict_{target}_with_{source}.npy"
            if fname.exists():
                continue

            print(f"  {source}→{target} ({done+nd+1}/{total})...", flush=True)

            # Flatten all stories
            X_all = np.concatenate(
                [data_dict[source][s].reshape(len(data_dict[source][s]), -1)
                 for s in range(4)], axis=0).astype(np.float32)
            Y_all = np.concatenate(
                [data_dict[target][s].reshape(len(data_dict[target][s]), -1)
                 for s in range(4)], axis=0).astype(np.float32)

            # Project to PCA space using global components
            X_pca = project(X_all, components, mean)
            Y_pca = project(Y_all, components, mean)

            sizes  = [len(data_dict[source][s]) for s in range(4)]
            bounds = np.cumsum([0] + sizes)
            preds  = np.zeros_like(Y_pca)

            for fold in range(4):
                vl = np.arange(bounds[fold], bounds[fold+1])
                tr = np.concatenate([np.arange(bounds[f], bounds[f+1])
                                     for f in range(4) if f != fold])

                Xtr_z = X_pca[tr]; mu = Xtr_z.mean(0); sd = Xtr_z.std(0)+1e-8
                Xtr_z = (Xtr_z - mu) / sd
                Xvl_z = (X_pca[vl] - mu) / sd

                preds[vl] = kernel_ridge_gpu(
                    Xtr_z @ Xtr_z.T, Y_pca[tr],
                    Xvl_z @ Xtr_z.T
                )

            # Pearson r per voxel (in original space via unproject)
            Y_pred_full = unproject(preds, components, mean)
            p_z   = Y_pred_full - Y_pred_full.mean(0)
            a_z   = Y_all - Y_all.mean(0)
            denom = np.sqrt((p_z**2).sum(0) * (a_z**2).sum(0)) + 1e-8
            corr  = ((p_z * a_z).sum(0) / denom).astype(np.float32)

            np.save(str(fname), corr)
            nd += 1

            el  = time.time() - t0
            rem = (el / nd) * (total - done - nd)
            print(f"    mean_r={corr.mean():.4f}  "
                  f"elapsed={el/60:.1f}m  remaining≈{rem/60:.1f}m",
                  flush=True)

    print(f"Phase 1 done. New pairs: {nd}")


# ── PHASE 2: BOOTSTRAP (matches paper Cell 4 exactly) ────────────────────────

def exponential_func(x, v0, t0):
    return v0 * (1 - np.exp(-x / t0))


def bootstrap_function(x, y):
    popt, _ = curve_fit(exponential_func, x, y, maxfev=6000)
    return popt[0]


def bootstrap_one_subject(target_int, pred_dir_str, subjects_int,
                           n_bootstrap, seed):
    """
    Matches paper's Noise Ceiling.ipynb Cell 4 exactly.
    target_int: integer subject number (1-27)
    """
    import numpy as np
    from itertools import combinations
    from pathlib import Path

    np.random.seed(seed)
    pred_dir     = Path(pred_dir_str)
    target       = str(target_int).zfill(2)
    out_path     = pred_dir / f"subject_{target_int}_kernel_ridge.npy"

    if out_path.exists():
        return target_int, np.load(str(out_path))

    source_ints  = [i for i in subjects_int if i != target_int]
    n_voxels     = 208 * 81

    subsample_corrs = []
    subsample_sizes = []

    # All combinations at each subsample size — matches paper exactly
    for subsample_size in range(2, len(subjects_int) + 1):
        subsamples = list(combinations(source_ints, subsample_size - 1))
        for subsample in subsamples:
            preds = []
            for src_int in subsample:
                src    = str(src_int).zfill(2)
                fname  = pred_dir / f"predict_{target}_with_{src}.npy"
                preds.append(np.load(str(fname)))
            mean_pred = np.mean(preds, axis=0)
            subsample_corrs.append(mean_pred)
            subsample_sizes.append(subsample_size)

        print(f"  Sub {target_int}: done subsample size {subsample_size}",
              flush=True)

    subsample_corrs = np.array(subsample_corrs)   # (n_combos, 16848)
    subsample_sizes = np.array(subsample_sizes)

    # Bootstrap per voxel — matches paper Cell 4 exactly
    ceilings = []
    for pc_ind in range(n_voxels):
        bootstraps = []
        y = subsample_corrs[:, pc_ind]
        n_samples = len(subsample_sizes)
        for b in range(n_bootstrap):
            inds = np.random.choice(range(n_samples), n_samples, replace=True)
            try:
                v0 = bootstrap_function(subsample_sizes[inds], y[inds])
                bootstraps.append(v0)
            except Exception:
                bootstraps.append(float(np.nanmean(y)))
        ceilings.append(np.median(bootstraps))

    ceiling = np.array(ceilings, dtype=np.float32)
    np.save(str(out_path), ceiling)
    return target_int, ceiling


def run_phase2():
    print(f"\n{'='*60}")
    print(f"Phase 2: Bootstrap Ceiling ({N_BOOTSTRAP} bootstraps, "
          f"{N_JOBS} workers)")

    done      = [i for i in SUBJECTS_INT
                 if (PRED_DIR/f"subject_{i}_kernel_ridge.npy").exists()]
    remaining = [i for i in SUBJECTS_INT if i not in done]
    print(f"  Done: {len(done)}  Remaining: {len(remaining)}")
    print(f"{'='*60}")

    all_c = {i: np.load(str(PRED_DIR/f"subject_{i}_kernel_ridge.npy"))
             for i in done}

    if remaining:
        results = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(bootstrap_one_subject)(
                target_int   = t,
                pred_dir_str = str(PRED_DIR),
                subjects_int = SUBJECTS_INT,
                n_bootstrap  = N_BOOTSTRAP,
                seed         = 42 + i,
            )
            for i, t in enumerate(remaining)
        )
        for t, c in results:
            all_c[t] = c
            print(f"  Sub {t}: mean={c.mean():.4f}  max={c.max():.4f}")

    return np.array([all_c[i] for i in SUBJECTS_INT])


# ── PHASE 3: SAVE FINAL FILES ─────────────────────────────────────────────────

def run_phase3(all_ceilings):
    print(f"\n{'='*60}")
    print("Phase 3: Saving")
    upper = all_ceilings.mean(axis=0)
    lower = np.percentile(all_ceilings, 10, axis=0)
    np.save(str(PRED_DIR/"ceiling_upper.npy"), upper)
    np.save(str(PRED_DIR/"ceiling_kfold.npy"), upper)
    np.save(str(PRED_DIR/"ceiling_lower.npy"), lower)
    print(f"  upper mean={upper.mean():.4f}  max={upper.max():.4f}")
    print(f"  positive: {(upper>0).mean()*100:.1f}%")
    print(f"  Saved to {PRED_DIR}/")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Noise Ceiling — Global PCA + GPU Kernel Ridge + joblib")
    print(f"  Subjects: {N_SUBJECTS}  PCA: {N_PCA}  "
          f"Bootstraps: {N_BOOTSTRAP}  Workers: {N_JOBS}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}, {p.total_memory/1e9:.1f}GB")
    print("=" * 60)

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Load all MEG data
    print("\nLoading MEG data...")
    data_dict = {}
    for sub in SUBJECTS:
        path = NOTEBOOKS_DIR / f"sub{sub}-meg-data.npy"
        if not path.exists():
            print(f"  WARNING: sub{sub} not found")
            continue
        data_dict[sub] = np.load(str(path), allow_pickle=True)
        print(f"  Loaded sub{sub}")
    print(f"Loaded {len(data_dict)}/{N_SUBJECTS} subjects")

    # Fit global PCA once on all subjects' data
    print("\nFitting global PCA...")
    all_flat = np.concatenate([
        np.concatenate([data_dict[sub][s].reshape(len(data_dict[sub][s]), -1)
                        for s in range(4)], axis=0)
        for sub in SUBJECTS if sub in data_dict
    ], axis=0).astype(np.float32)
    print(f"  All data shape: {all_flat.shape}")
    components, mean = fit_global_pca(all_flat, N_PCA)
    del all_flat
    gc.collect()
    print(f"  Components: {components.shape}")

    t0 = time.time()

    # Phase 1: brain→brain kernel ridge predictions
    run_phase1(data_dict, components, mean)
    del data_dict, components, mean
    gc.collect()

    # Phase 2: bootstrap exponential fit (paper-exact)
    all_ceilings = run_phase2()

    # Phase 3: save
    run_phase3(all_ceilings)

    print(f"\nAll done in {(time.time()-t0)/3600:.2f} hours")
    print("Use ceiling_kfold.npy to normalize model predictions.")


if __name__ == "__main__":
    main()
