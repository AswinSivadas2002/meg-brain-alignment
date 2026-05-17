#!/usr/bin/env python3
"""
Residual Feature Computation: CLAP Text - Wav2Vec
==================================================
Implements the residual approach from Srijith et al. (EMNLP 2025) Section 4.5
and Appendix D.

Purpose:
  Remove the linear contribution of unimodal speech (Wav2Vec2) from
  multimodal text (CLAP text) embeddings. The residual represents the
  part of CLAP text that CANNOT be explained by unimodal speech features.

  If MEG predictivity drops at ~200ms after this removal, it confirms
  that CLAP text's 200ms boost comes specifically from speech information
  (Figure 4 left side of the paper).

Method (matches paper Appendix D exactly):
  For each CLAP text layer:
    For each CV fold (leave-one-story-out):
      1. Train ridge regression: Wav2Vec → CLAP_text  (on 3 stories)
      2. Predict: Wav2Vec_test → CLAP_text_predicted  (on 1 story)
      3. Residual = CLAP_text_test - CLAP_text_predicted
  Stack residuals across folds → (n_words_total, 768) per layer

  The residual is the CLAP text embedding with speech information
  linearly removed.

Input:
  notebooks/clap_text_features.npy   -- CLAP text, 12 layers, 768-dim
  notebooks/wav2vec_features.npy     -- Wav2Vec2, 12 layers, 768-dim

Output:
  notebooks/clap_text_residual_wav2vec.npy
  Same format as clap_text_features.npy:
    np.array (4,) dtype=object
    Each element: dict {layer_idx 0-11: np.array (n_words, 768)}

Pure CPU/numpy — no GPU needed.

Usage:
  conda activate meg_env
  nohup taskset -c 0-15 python -u compute_residuals.py > residual_log.txt 2>&1 &
  echo $!
"""

import numpy as np
from numpy.linalg import svd
from sklearn.model_selection import KFold
import time
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PROJECT_DIR   = Path("/home/mtech1/Desktop/meg_project")
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

CLAP_TEXT_FILE = NOTEBOOKS_DIR / "clap_text_features.npy"
WAV2VEC_FILE   = NOTEBOOKS_DIR / "wav2vec_features.npy"
OUTPUT_FILE    = NOTEBOOKS_DIR / "clap_text_residual_wav2vec.npy"

N_LAYERS_CLAP    = 12   # CLAP text layers
N_LAYERS_WAV2VEC = 12   # Wav2Vec2 layers

# Lambda grid for ridge regression (source → target feature space)
# Using same range as encoding model
LAMBDAS = np.logspace(1, 3, 16)

STORY_WORD_COUNTS = {0: 668, 1: 1503, 2: 2637, 3: 3753}

# ─── RIDGE REGRESSION (CPU numpy) ─────────────────────────────────────────────

def ridge_svd(X, Y, lmbda):
    """SVD ridge regression. X: (n,p), Y: (n,d) → weights: (p,d)"""
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s ** 2 + lmbda)
    return Vt.T @ (np.diag(d) @ (U.T @ Y))


def cross_val_ridge_predict(X_train, Y_train, X_test):
    """
    Fit ridge regression X→Y on training data with CV lambda selection,
    then predict on test data.

    Uses inner 4-fold CV on training data to select best lambda
    per output dimension, then refits on all training data.

    Returns: Y_pred (n_test, d)
    """
    n_train, n_feat = X_train.shape
    n_out = Y_train.shape[1]
    n_lam = len(LAMBDAS)

    # Inner CV for lambda selection
    cv_error = np.zeros((n_lam, n_out))
    kf = KFold(n_splits=4)

    for trn_idx, val_idx in kf.split(np.arange(n_train)):
        X_trn, X_val = X_train[trn_idx], X_train[val_idx]
        Y_trn, Y_val = Y_train[trn_idx], Y_train[val_idx]
        Y_var = Y_val.var(axis=0).clip(min=1e-10)

        U, s, Vt = svd(X_trn, full_matrices=False)

        for lam_idx, lmbda in enumerate(LAMBDAS):
            d      = s / (s ** 2 + lmbda)
            W      = Vt.T @ (np.diag(d) @ (U.T @ Y_trn))
            Y_pred = X_val @ W
            ss_res = np.mean((Y_val - Y_pred) ** 2, axis=0)
            cv_error[lam_idx] += 1.0 - ss_res / Y_var

    # Best lambda per output dimension (max R2 = min error)
    best_lam_idx = np.argmax(cv_error, axis=0)   # (n_out,)

    # Refit on all training data
    weights = np.zeros((n_feat, n_out))
    for lam_idx in range(n_lam):
        mask = best_lam_idx == lam_idx
        if not mask.any():
            continue
        W = ridge_svd(X_train, Y_train[:, mask], LAMBDAS[lam_idx])
        weights[:, mask] = W

    return X_test @ weights   # (n_test, n_out)


# ─── RESIDUAL COMPUTATION ─────────────────────────────────────────────────────

def compute_residuals_one_layer(clap_text_stories, wav2vec_stories,
                                 clap_layer_idx, wav2vec_layer_idx):
    """
    Compute CLAP text residuals after removing Wav2Vec for one layer pair.

    Uses leave-one-story-out CV (4 folds) matching the encoding model setup.

    clap_layer_idx:    which CLAP text layer to use as target
    wav2vec_layer_idx: which Wav2Vec layer to use as source (speech)

    Returns: residuals per story, same structure as input features
             list of 4 arrays, each (n_words_story, 768)
    """
    kf = KFold(n_splits=4)
    residuals_by_story = [None] * 4

    for train_folds, test_fold in kf.split(np.arange(4)):
        test_idx = test_fold[0]

        # Source: Wav2Vec features (speech)
        X_train = np.concatenate(
            [wav2vec_stories[i][wav2vec_layer_idx] for i in train_folds],
            axis=0
        ).astype(np.float32)
        X_test = wav2vec_stories[test_idx][wav2vec_layer_idx].astype(np.float32)

        # Target: CLAP text features (multimodal text)
        Y_train = np.concatenate(
            [clap_text_stories[i][clap_layer_idx] for i in train_folds],
            axis=0
        ).astype(np.float32)
        Y_test = clap_text_stories[test_idx][clap_layer_idx].astype(np.float32)

        # Predict CLAP text from Wav2Vec, compute residual
        Y_pred = cross_val_ridge_predict(X_train, Y_train, X_test)
        residual = Y_test - Y_pred   # (n_test_words, 768)

        residuals_by_story[test_idx] = residual

    return residuals_by_story


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Residual Computation: CLAP Text - Wav2Vec")
    print("  Method: ridge regression (Wav2Vec → CLAP text)")
    print("  CV:     4-fold leave-one-story-out")
    print("  Layers: CLAP text L0-11, Wav2Vec L0-11 (matched)")
    print("=" * 60)

    if OUTPUT_FILE.exists():
        print(f"\nOutput already exists: {OUTPUT_FILE}")
        print("Delete to re-run.")
        return

    # ── Load features ──
    print(f"\nLoading CLAP text features...")
    clap_text = np.load(str(CLAP_TEXT_FILE), allow_pickle=True)
    print(f"  Stories: {len(clap_text)}")
    print(f"  Word counts: {[clap_text[s][0].shape[0] for s in range(4)]}")

    print(f"\nLoading Wav2Vec features...")
    wav2vec = np.load(str(WAV2VEC_FILE), allow_pickle=True)
    print(f"  Stories: {len(wav2vec)}")
    print(f"  Word counts: {[wav2vec[s][0].shape[0] for s in range(4)]}")

    # ── Compute residuals layer by layer ──
    # Use matched layers: CLAP text layer i ← remove Wav2Vec layer i
    # This is the standard approach — matched layer indices
    # Result: 12 residual layers, each (n_words, 768) per story

    # Output structure: list of 4 story dicts
    # residual_features[story][layer] = (n_words, 768)
    residual_features = [{} for _ in range(4)]

    t_total = time.time()

    for layer_idx in range(N_LAYERS_CLAP):
        print(f"\nLayer {layer_idx:02d}...")
        t_layer = time.time()

        residuals_by_story = compute_residuals_one_layer(
            clap_text, wav2vec,
            clap_layer_idx=layer_idx,
            wav2vec_layer_idx=layer_idx,  # matched layers
        )

        for story_idx in range(4):
            residual_features[story_idx][layer_idx] = residuals_by_story[story_idx]

        # Quick stats
        all_res = np.concatenate(residuals_by_story, axis=0)
        orig    = np.concatenate([clap_text[s][layer_idx] for s in range(4)], axis=0)
        print(f"  Original CLAP text: mean={orig.mean():.4f}  std={orig.std():.4f}")
        print(f"  Residual:           mean={all_res.mean():.4f}  std={all_res.std():.4f}")
        print(f"  Variance removed:   {(1 - all_res.var()/orig.var())*100:.1f}%")
        print(f"  Time: {(time.time()-t_layer)/60:.1f} min")

    print(f"\n{'='*60}")
    print(f"All layers done in {(time.time()-t_total)/60:.1f} min")

    # ── Save ──
    print("\nSaving...")
    np.save(str(OUTPUT_FILE), np.array(residual_features, dtype=object))
    print(f"  {OUTPUT_FILE}")

    # ── Verify ──
    print("\nVerifying...")
    loaded = np.load(str(OUTPUT_FILE), allow_pickle=True)
    for s in range(4):
        print(f"  Story {s}: L6={loaded[s][6].shape}")

    print("\nDone.")
    print("Next steps:")
    print("  1. Run meg_encoding_clap_residual_gpu.py on GPU")
    print("     (uses clap_text_residual_wav2vec.npy as input)")
    print("  2. Compare predictivity curves:")
    print("     CLAP text (raw) vs CLAP text (residual)")
    print("     Drop at ~200ms = speech info confirmed")


if __name__ == "__main__":
    main()
