# MEG Brain Alignment

Replication study of **Srijith et al., EMNLP 2025**:
*Aligning Text/Speech Representations from Multimodal Models with MEG Brain Activity During Listening*

We replicate the text-side analysis using CLAP (multimodal) and BERT (unimodal text)
on the full MEG-MASC dataset (27 subjects, 4 stories, 8,567 words).

## Key Finding

Text representations (CLAP text, BERT) predict MEG brain activity far better than
speech representations (CLAP audio, Wav2Vec) throughout the full 0–600ms window.
CLAP text shows a distinct 200ms auditory peak absent in BERT, confirming that
multimodal training causes text encoders to absorb acoustic-phonological knowledge.

## Setup

```bash
conda create -n meg_env python=3.10
conda activate meg_env
pip install -r requirements.txt
```

PyTorch nightly is required for RTX PRO 4500 (sm_120) support:

```bash
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

## Run Order

```bash
python preprocessing/meg_preprocess.py
python features/synthesize_audio.py
python features/extract_clap.py
python features/extract_wav2vec.py
python features/compute_clap_text_residual.py
python features/compute_clap_audio_residual.py

# Run all encoding scripts simultaneously on GPU
python encoding/encode_bert.py &
python encoding/encode_clap_text.py &
python encoding/encode_clap_audio.py &
python encoding/encode_clap_text_residual.py &
python encoding/encode_clap_audio_residual.py &
python encoding/encode_wav2vec.py &
wait

python noise_ceiling/phase1_pairwise.py
python noise_ceiling/phase2_curve_fit.py
python plotting/plot_results.py
```

## Dataset

MEG-MASC (Gwilliams et al. 2023) — 27 participants, 4 stories, 208 MEG sensors, 1000Hz.
Not included in this repo. Download from OpenNeuro: https://openneuro.org/datasets/ds004356

## Reference

Srijith, P., et al. (2025). Aligning Text/Speech Representations from Multimodal Models
with MEG Brain Activity During Listening. EMNLP 2025, pp. 34469–34486.
