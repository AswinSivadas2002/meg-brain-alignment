#!/usr/bin/env python3
"""
Residual Feature Computation: CLAP Audio - BERT
================================================
Right side of Figure 4 in Srijith et al. (EMNLP 2025).

Purpose:
  Remove the linear contribution of unimodal text (BERT) from
  multimodal audio (CLAP audio) embeddings.

  Answers: does CLAP audio gain anything from text information?
  If predictivity drops after removal → text info confirmed in CLAP audio
  If no drop → speech embeddings don't benefit from text (paper's finding)

Method (matches paper Appendix D):
  For each CLAP audio layer (0-4) × each BERT layer (matched or best):
    For each CV fold (leave-one-story-out):
      1. Train ridge: BERT → CLAP_audio  (on 3 stories)
      2. Predict: BERT_test → CLAP_audio_predicted
      3. Residual = CLAP_audio_test - CLAP_audio_predicted

  Note: BERT has 12 layers (768-dim), CLAP audio has 5 layers (variable dim)
  We use BERT L6 (best layer) as the unimodal text source for all audio layers.
  This matches the paper's approach of using best unimodal representative.

Input:
  notebooks/clap_audio_features_synth.npy   -- CLAP audio, 5 layers, variable dim
  notebooks/bert-base-lw-20.npy       -- BERT, 12 layers, 768-dim

Output:
  notebooks/clap_audio_residual_bert_synth.npy
  np.array (4,) dtype=object
  Each element: dict {layer_idx 0-4: np.array (n_words, C)}
  where C is the channel dim of each HTSAT stage.

Pure CPU/numpy. ~10-20 minutes.

Usage:
  conda activate meg_env
  nohup taskset -c 0-15 python3 -u compute_residuals_audio.py > residual_audio_log.txt 2>&1 &
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

CLAP_AUDIO_FILE = NOTEBOOKS_DIR / "clap_audio_features_synth.npy"
BERT_FILE       = NOTEBOOKS_DIR / "bert-base-lw-20.npy"
OUTPUT_FILE     = NOTEBOOKS_DIR / "clap_audio_residual_bert_synth.npy"

N_LAYERS_AUDIO = 5    # HTSAT stages
BERT_LAYER     = 6    # Use BERT L6 as unimodal text source (paper's best layer)

LAMBDAS = np.logspace(1, 3, 16)

# ─── RIDGE REGRESSION ─────────────────────────────────────────────────────────

def ridge_svd(X, Y, lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s ** 2 + lmbda)
    return Vt.T @ (np.diag(d) @ (U.T @ Y))


def cross_val_ridge_predict(X_train, Y_train, X_test):
    """Ridge regression with CV lambda selection, predict on test."""
    n_train = X_train.shape[0]
    n_out   = Y_train.shape[1]
    n_lam   = len(LAMBDAS)

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

    best_lam_idx = np.argmax(cv_error, axis=0)
    weights = np.zeros((X_train.shape[1], n_out))

    for lam_idx in range(n_lam):
        mask = best_lam_idx == lam_idx
        if not mask.any():
            continue
        weights[:, mask] = ridge_svd(X_train, Y_train[:, mask], LAMBDAS[lam_idx])

    return X_test @ weights


# ─── RESIDUAL COMPUTATION ─────────────────────────────────────────────────────

def compute_residuals_one_layer(clap_audio_stories, bert_stories,
                                audio_layer_idx, bert_layer_idx):
    """
    Remove BERT (unimodal text) from CLAP audio for one layer.
    Returns list of 4 residual arrays, one per story.
    """
    kf = KFold(n_splits=4)
    residuals_by_story = [None] * 4

    for train_folds, test_fold in kf.split(np.arange(4)):
        test_idx = test_fold[0]

        # Source: BERT features (unimodal text)
        X_train = np.concatenate(
            [np.array(bert_stories[i][bert_layer_idx]) for i in train_folds],
            axis=0
        ).astype(np.float32)
        X_test = np.array(bert_stories[test_idx][bert_layer_idx]).astype(np.float32)

        # Target: CLAP audio features (multimodal audio)
        Y_train = np.concatenate(
            [clap_audio_stories[i][audio_layer_idx] for i in train_folds],
            axis=0
        ).astype(np.float32)
        Y_test = clap_audio_stories[test_idx][audio_layer_idx].astype(np.float32)

        Y_pred   = cross_val_ridge_predict(X_train, Y_train, X_test)
        residual = Y_test - Y_pred

        residuals_by_story[test_idx] = residual

    return residuals_by_story


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Residual Computation: CLAP Audio - BERT")
    print(f"  Source (text):  BERT L{BERT_LAYER} (768-dim)")
    print(f"  Target (audio): CLAP audio L0-{N_LAYERS_AUDIO-1} (variable dim)")
    print(f"  CV: 4-fold leave-one-story-out")
    print("=" * 60)

    if OUTPUT_FILE.exists():
        print(f"\nOutput already exists: {OUTPUT_FILE}")
        print("Delete to re-run.")
        return

    print(f"\nLoading CLAP audio features...")
    clap_audio = np.load(str(CLAP_AUDIO_FILE), allow_pickle=True)
    for l in range(N_LAYERS_AUDIO):
        print(f"  L{l:02d}: {clap_audio[0][l].shape}")

    print(f"\nLoading BERT features (using L{BERT_LAYER})...")
    bert = np.load(str(BERT_FILE), allow_pickle=True)
    print(f"  Word counts: {[len(bert[s][BERT_LAYER]) for s in range(4)]}")

    residual_features = [{} for _ in range(4)]
    t_total = time.time()

    for audio_layer_idx in range(N_LAYERS_AUDIO):
        print(f"\nAudio Layer {audio_layer_idx:02d} "
              f"(dim={clap_audio[0][audio_layer_idx].shape[1]})...")
        t_layer = time.time()

        residuals_by_story = compute_residuals_one_layer(
            clap_audio, bert,
            audio_layer_idx=audio_layer_idx,
            bert_layer_idx=BERT_LAYER,
        )

        for story_idx in range(4):
            residual_features[story_idx][audio_layer_idx] = \
                residuals_by_story[story_idx]

        # Stats
        all_res = np.concatenate(residuals_by_story, axis=0)
        orig    = np.concatenate(
            [clap_audio[s][audio_layer_idx] for s in range(4)], axis=0
        )
        print(f"  Original: mean={orig.mean():.4f}  std={orig.std():.4f}")
        print(f"  Residual: mean={all_res.mean():.4f}  std={all_res.std():.4f}")
        print(f"  Variance removed: {(1 - all_res.var()/orig.var())*100:.1f}%")
        print(f"  Time: {(time.time()-t_layer)/60:.1f} min")

    print(f"\n{'='*60}")
    print(f"All layers done in {(time.time()-t_total)/60:.1f} min")

    print("\nSaving...")
    np.save(str(OUTPUT_FILE), np.array(residual_features, dtype=object))
    print(f"  {OUTPUT_FILE}")

    print("\nVerifying...")
    loaded = np.load(str(OUTPUT_FILE), allow_pickle=True)
    for s in range(4):
        for l in range(N_LAYERS_AUDIO):
            print(f"  Story {s} L{l:02d}: {loaded[s][l].shape}")

    print("\nDone.")
    print("Next: run meg_encoding_clap_audio_residual_gpu.py")


if __name__ == "__main__":
    main()
