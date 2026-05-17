#!/usr/bin/env python3
"""
Noise Ceiling — Phase 2 Only (Bootstrap)
=========================================
Run this after Phase 1 (702 predict_*.npy files) is complete.

Fix for disk I/O bottleneck: loads ALL 702 prediction files into RAM
once at startup (~47MB total), then all bootstrap computations run
entirely in memory with zero disk reads during the loop.

Matches paper's Noise Ceiling.ipynb Cell 4 exactly:
  - All combinations per subsample size (capped at 50 for sizes > 5)
  - 100 bootstraps per voxel
  - exponential fit f(s) = v0*(1-exp(-s/t0))
  - median v0 = ceiling estimate

Output:
  noise_ceiling_preds/subject_{1-27}_kernel_ridge.npy
  noise_ceiling_preds/ceiling_upper.npy
  noise_ceiling_preds/ceiling_kfold.npy
  noise_ceiling_preds/ceiling_lower.npy

Usage:
  nohup taskset -c 0-15 python3 -u noise_ceiling_phase2.py > noise_ceiling_phase2_log.txt 2>&1 &
  echo $!
"""

import numpy as np
import time
from pathlib import Path
from scipy.optimize import curve_fit
from itertools import combinations

PROJECT_DIR = Path("/home/mtech1/Desktop/meg_project")
PRED_DIR    = PROJECT_DIR / "noise_ceiling_preds"

N_SUBJECTS  = 27
N_BOOTSTRAP = 100
N_VOXELS    = 208 * 81   # 16848
MAX_COMBOS  = 50         # cap per subsample size > 5

SUBJECTS_INT = list(range(1, N_SUBJECTS + 1))
SUBJECTS     = [str(i).zfill(2) for i in SUBJECTS_INT]


def exponential_func(x, v0, t0):
    return v0 * (1 - np.exp(-x / t0))


def load_all_predictions():
    """
    Load all 702 predict_{target}_with_{source}.npy files into RAM.
    Returns dict: preds[target_int][source_int] = np.array (16848,)
    Total size: 702 * 16848 * 4 bytes ≈ 47MB
    """
    print("Loading all 702 prediction files into RAM...")
    t0   = time.time()
    preds = {}
    for t in SUBJECTS_INT:
        preds[t] = {}
        target = str(t).zfill(2)
        for s in SUBJECTS_INT:
            if s == t:
                continue
            source = str(s).zfill(2)
            fname  = PRED_DIR / f"predict_{target}_with_{source}.npy"
            preds[t][s] = np.load(str(fname))
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"({702 * N_VOXELS * 4 / 1e6:.0f}MB)")
    return preds


def compute_ceiling_one_subject(target_int, preds, seed):
    """
    Compute noise ceiling for one target subject.
    All data already in RAM — no disk I/O.
    Matches paper Cell 4 exactly.
    """
    np.random.seed(seed)
    out_path = PRED_DIR / f"subject_{target_int}_kernel_ridge.npy"
    if out_path.exists():
        print(f"  Sub {target_int}: already done, loading")
        return target_int, np.load(str(out_path))

    source_ints     = [i for i in SUBJECTS_INT if i != target_int]
    subsample_corrs = []
    subsample_sizes = []

    for subsample_size in range(2, N_SUBJECTS + 1):
        all_combos = list(combinations(source_ints, subsample_size - 1))

        # Cap at MAX_COMBOS for large sizes
        if len(all_combos) > MAX_COMBOS:
            idx     = np.random.choice(len(all_combos), MAX_COMBOS,
                                        replace=False)
            sampled = [all_combos[i] for i in idx]
        else:
            sampled = all_combos

        for combo in sampled:
            # Average predictions from this combo — all from RAM
            mean_pred = np.mean(
                [preds[target_int][src] for src in combo], axis=0
            )
            subsample_corrs.append(mean_pred)
            subsample_sizes.append(subsample_size)

        print(f"  Sub {target_int}: done subsample size {subsample_size}  "
              f"({len(sampled)} combos)", flush=True)

    subsample_corrs = np.array(subsample_corrs)   # (n_points, 16848)
    subsample_sizes = np.array(subsample_sizes)
    n_samples       = len(subsample_sizes)

    # Bootstrap per voxel — matches paper Cell 4 exactly
    ceilings = []
    t0 = time.time()
    for pc_ind in range(N_VOXELS):
        bootstraps = []
        y = subsample_corrs[:, pc_ind]
        for b in range(N_BOOTSTRAP):
            inds = np.random.choice(n_samples, n_samples, replace=True)
            try:
                popt, _ = curve_fit(
                    exponential_func,
                    subsample_sizes[inds], y[inds],
                    maxfev=6000
                )
                bootstraps.append(popt[0])
            except Exception:
                bootstraps.append(float(np.nanmean(y)))
        ceilings.append(np.median(bootstraps))

    ceiling = np.array(ceilings, dtype=np.float32)
    np.save(str(out_path), ceiling)
    print(f"  Sub {target_int}: DONE  "
          f"mean={ceiling.mean():.4f}  max={ceiling.max():.4f}  "
          f"bootstrap time={time.time()-t0:.0f}s")
    return target_int, ceiling


def main():
    print("=" * 60)
    print("Noise Ceiling Phase 2 — Bootstrap (in-memory)")
    print(f"  Subjects:   {N_SUBJECTS}")
    print(f"  Bootstraps: {N_BOOTSTRAP}")
    print(f"  Max combos: {MAX_COMBOS} per size > 5")
    print(f"  Voxels:     {N_VOXELS}")
    print("=" * 60)

    # Load all predictions into RAM once
    preds = load_all_predictions()

    t_total = time.time()
    all_ceilings = {}

    # Process subjects in parallel — preds already in RAM, no disk I/O
    from joblib import Parallel, delayed
    N_JOBS = 16
    print(f"\nRunning bootstrap for {N_SUBJECTS} subjects ({N_JOBS} workers)...")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(compute_ceiling_one_subject)(target_int, preds, seed=42+i)
        for i, target_int in enumerate(SUBJECTS_INT)
    )
    for target_int, ceiling in results:
        all_ceilings[target_int] = ceiling

    # Save final ceiling files
    print(f"\n{'='*60}")
    print("Saving final ceiling files...")
    arr      = np.array([all_ceilings[i] for i in SUBJECTS_INT])
    upper    = arr.mean(axis=0)
    lower    = np.percentile(arr, 10, axis=0)

    np.save(str(PRED_DIR / "ceiling_upper.npy"), upper)
    np.save(str(PRED_DIR / "ceiling_kfold.npy"), upper)
    np.save(str(PRED_DIR / "ceiling_lower.npy"), lower)

    print(f"  upper mean={upper.mean():.4f}  max={upper.max():.4f}")
    print(f"  positive:  {(upper > 0).mean()*100:.1f}%")
    print(f"  Saved to {PRED_DIR}/")
    print(f"\nAll done in {(time.time()-t_total)/3600:.2f} hours")
    print("Use ceiling_kfold.npy to normalize model predictions.")


if __name__ == "__main__":
    main()
