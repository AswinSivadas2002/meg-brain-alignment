# MEG Brain Alignment

Replication of Srijith et al. EMNLP 2025 — predicting MEG brain activity word-by-word from BERT, CLAP, and Wav2Vec representations using ridge regression.

## What this project does

For each word in 4 naturalistic spoken stories (MEG-MASC dataset, 27 subjects):
1. Preprocesses raw MEG (bandpass filter, ICA artifact removal, epoching)
2. Extracts features from BERT, CLAP text, CLAP audio, Wav2Vec at every layer
3. Computes residual features (CLAP text minus Wav2Vec, CLAP audio minus BERT)
4. Trains SVD ridge regression to predict MEG signal from each feature set
5. Normalizes Pearson r by a noise ceiling (exponential saturation curve fit)
6. Plots normalized predictivity over time to compare models

## Repository structure

```
preprocessing/   — bandpass filter, ICA (Picard + ICAlabel), epoching
features/        — BERT, CLAP text, CLAP audio (synth), Wav2Vec extraction + residuals
encoding/        — SVD GPU ridge regression, per-voxel lambda, 4-fold CV
noise_ceiling/   — Phase 1 pairwise predictions, Phase 2 exponential fit
plotting/        — normalized predictivity timecourses, layer profiles
assets/          — ICA component visualization images
```

## Key methodological decisions

- ICA artifact removal using Picard algorithm (100 components, extended=True for ICAlabel
  compatibility). Components classified by ICAlabel — a pretrained neural network classifier
  that assigns probabilities across categories: brain, eye, heart, muscle, line noise, channel
  noise. Components with >70% non-brain probability are removed. The paper uses no ICA —
  our cleaner signal shifts best layers slightly earlier (BERT L4 vs paper L6, CLAP text L2
  vs paper L4) due to artifact-contaminated components that inflated later-layer correlations
  in the paper's data.
- CLAP audio uses TTS synthesized isolated words (gTTS + ffmpeg, 1s, 48kHz)
  Reason: story clips have coarticulation noise that degrades CLAP audio predictivity 3-4x
- Wav2Vec uses naturalistic story audio clips (16kHz)
  Reason: unimodal speech model trained on continuous speech — synthesis degrades it
- Ridge regression: SVD-based on GPU, per-voxel lambda selection up to 1e3, inner 4-fold CV for lambda
- Outer CV: 4-fold leave-one-story-out (4 stories = 4 folds)
- Noise ceiling: exponential v0*(1-exp(-s/t0)) fit over cross-subject accuracy, 100 bootstraps

## Models (HuggingFace)

- BERT:    bert-base-uncased
- CLAP:    laion/clap-htsat-unfused
- Wav2Vec: facebook/wav2vec2-base

## Environment

- conda env: meg_env
- GPU: NVIDIA RTX PRO 4500 Blackwell (sm_120), 32GB VRAM
- PyTorch nightly required (sm_120 not supported in stable)
- Python 3.10+

## Data (not in repo — stored on lab workstation)

- Raw MEG: MEG-MASC dataset (BIDS format)
- Extracted features: ~/Desktop/meg_project/features/*.npy
- Encoding outputs: ~/Desktop/meg_project/meg_predictions/
- Noise ceiling: ~/Desktop/meg_project/noise_ceiling_preds/ceiling_kfold.npy
- Plots: ~/Desktop/meg_project/plots/

## Run order

1. preprocessing/meg_preprocess.py
2. features/synthesize_audio.py
3. features/extract_clap.py
4. features/extract_wav2vec.py
5. features/compute_clap_text_residual.py
6. features/compute_clap_audio_residual.py
7. encoding/encode_bert.py  (run all 6 encoding scripts simultaneously on GPU)
8. noise_ceiling/phase1_pairwise.py
9. noise_ceiling/phase2_curve_fit.py
10. plotting/plot_results.py
