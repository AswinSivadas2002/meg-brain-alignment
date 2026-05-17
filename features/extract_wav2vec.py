#!/usr/bin/env python3
"""
Wav2Vec2 Feature Extraction for MEG Brain Encoding
====================================================
Equivalent of BERT_Features.ipynb but for Wav2Vec2 (unimodal speech).

Model: facebook/wav2vec2-base
  - 12 transformer layers, 768-dim hidden states
  - Input: raw audio waveform at 16000Hz
  - No text input — audio only

Key differences from BERT/CLAP text extraction:
  - No context window — each word gets its own audio clip independently
    (speech models process audio, not word sequences)
  - Hidden states are (seq_len, 768) per clip — mean pool over time frames
    to get one vector per word per layer
  - Audio resampled to 16000Hz (Wav2Vec2 requirement, not 48kHz like CLAP)

Output:
  notebooks/wav2vec_features.npy
  Shape: np.array (4,) dtype=object
  Each element: dict {layer_idx 0-11: np.array (n_words, 768)}
  Matches bert-base-lw-20.npy format exactly.

Usage:
  conda activate meg_env
  nohup taskset -c 0-15 python -u wav2vec_feature_extraction.py > wav2vec_log.txt 2>&1 &
  echo $!

Runs on CPU. Expected time: ~3-5 hours for all 4 stories.
"""

import numpy as np
import torch
import librosa
import pandas as pd
import glob
import time as tm
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PROJECT_DIR   = Path("/home/mtech1/Desktop/meg_project")
DATA_DIR      = PROJECT_DIR / "data"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
AUDIO_DIR     = DATA_DIR / "stimuli" / "audio"

WAV2VEC_MODEL_ID = "facebook/wav2vec2-base"
TARGET_SR        = 16000   # Wav2Vec2 requires 16kHz
CLIP_DURATION    = 1.0     # seconds per word clip
N_LAYERS         = 12      # 12 transformer layers

STORIES = {
    0: {"task": 0, "name": "cable_spool_fort", "chunks": list(range(6))},
    1: {"task": 1, "name": "easy_money",       "chunks": list(range(8))},
    2: {"task": 2, "name": "lw1",              "chunks": None},
    3: {"task": 3, "name": "the_black_willow", "chunks": list(range(12))},
}

STORY_WORD_COUNTS = {0: 668, 1: 1503, 2: 2637, 3: 3753}

device = torch.device("cpu")

# ─── WORD LIST + ONSETS ───────────────────────────────────────────────────────

def load_word_onsets(story_idx):
    """
    Load word list and onset/offset times from sub-01 events.tsv.
    Same as CLAP extraction — uses trial_type column with eval().
    """
    task    = STORIES[story_idx]["task"]
    pattern = str(DATA_DIR / "sub-01" / "ses-0" / "meg" /
                  f"sub-01_ses-0_task-{task}_events.tsv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"events.tsv not found: {pattern}")

    df = pd.read_csv(files[0], sep='\t')

    # Parse trial_type column (Python dict strings)
    if 'kind' not in df.columns and 'trial_type' in df.columns:
        parsed = df['trial_type'].apply(
            lambda x: eval(x) if isinstance(x, str) else {}
        )
        df = pd.concat([df.drop(columns=['trial_type']),
                        pd.DataFrame(list(parsed))], axis=1)

    words_df  = df[df['kind'] == 'word'].sort_values('onset').reset_index(drop=True)
    word_list = words_df['word'].tolist()
    onsets    = words_df['onset'].values
    durations = words_df['duration'].values if 'duration' in words_df.columns \
                else np.full(len(words_df), 0.3)
    offsets   = onsets + durations

    n = len(word_list)
    if n != STORY_WORD_COUNTS[story_idx]:
        print(f"  WARNING story {story_idx}: expected {STORY_WORD_COUNTS[story_idx]} words, got {n}")

    print(f"  Story {story_idx}: {n} words loaded from events.tsv")
    return word_list, list(zip(onsets, offsets))


# ─── AUDIO UTILITIES ──────────────────────────────────────────────────────────

def load_and_concat_audio(story_idx):
    """Load full story audio, concat chunks if needed, resample to 16kHz."""
    info   = STORIES[story_idx]
    name   = info["name"]
    chunks = info["chunks"]

    if chunks is None:
        audio, sr = librosa.load(str(AUDIO_DIR / f"{name}.wav"), sr=None, mono=True)
    else:
        parts = []
        for c in chunks:
            candidates = sorted(AUDIO_DIR.glob(f"{name}_{c}*.wav"))
            if not candidates:
                raise FileNotFoundError(f"Missing chunk {c} for '{name}' in {AUDIO_DIR}")
            part, sr = librosa.load(str(candidates[0]), sr=None, mono=True)
            parts.append(part)
        audio = np.concatenate(parts)

    if sr != TARGET_SR:
        print(f"  Resampling {sr}Hz → {TARGET_SR}Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    print(f"  Story {story_idx} ({name}): {len(audio)/TARGET_SR:.1f}s loaded")
    return audio


def extract_word_clip(audio, onset_sec, offset_sec):
    """Extract fixed-duration clip for one word, zero-padded if needed."""
    clip_len     = int(CLIP_DURATION * TARGET_SR)
    onset_sample = int(onset_sec * TARGET_SR)
    raw          = audio[onset_sample : int(offset_sec * TARGET_SR)]
    clip         = np.zeros(clip_len, dtype=np.float32)
    clip[:min(len(raw), clip_len)] = raw[:clip_len]
    return clip


# ─── WAV2VEC2 FEATURE EXTRACTION ─────────────────────────────────────────────

@torch.inference_mode()
def extract_wav2vec_features_story(story_idx, model, processor, audio):
    """
    Extract Wav2Vec2 layer-wise hidden states for all words in a story.

    For each word:
      1. Slice audio clip at word onset
      2. Run through Wav2Vec2 encoder
      3. Mean pool over time frames → one vector per layer per word

    Returns:
        features: dict {layer_idx 0-11: np.array (n_words, 768)}
    """
    word_list, word_onsets = load_word_onsets(story_idx)
    n_words = len(word_list)

    layer_reps = {l: [] for l in range(N_LAYERS)}
    start_time = tm.time()

    for word_idx in range(n_words):
        onset_sec, offset_sec = word_onsets[word_idx]
        audio_clip = extract_word_clip(audio, onset_sec, offset_sec)

        # Process audio clip
        inputs = processor(
            audio_clip,
            return_tensors="pt",
            sampling_rate=TARGET_SR,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: tuple of 13 — [CNN features, L1..L12]
        # Index 0 = CNN feature extractor output (not a transformer layer)
        # Index 1-12 = transformer layers — matches BERT convention
        hidden_states = outputs.hidden_states

        for layer_idx in range(N_LAYERS):
            h = hidden_states[layer_idx + 1]          # (1, time_frames, 768)
            # Mean pool over time frames → (768,)
            vec = h[0].mean(dim=0).cpu().numpy()
            layer_reps[layer_idx].append(vec)

        if (word_idx + 1) % 100 == 0:
            elapsed   = tm.time() - start_time
            remaining = elapsed / (word_idx + 1) * (n_words - word_idx - 1)
            print(f"  [{word_idx+1}/{n_words}] elapsed={elapsed/60:.1f}m  "
                  f"remaining≈{remaining/60:.1f}m")
            start_time = tm.time()

    features = {l: np.stack(layer_reps[l]) for l in range(N_LAYERS)}
    return features


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Wav2Vec2 Feature Extraction")
    print(f"  Model:   {WAV2VEC_MODEL_ID}")
    print(f"  Device:  {device}")
    print(f"  Layers:  0-{N_LAYERS-1} (768-dim, mean pooled over time)")
    print(f"  Audio:   {CLIP_DURATION}s clips @ {TARGET_SR}Hz")
    print(f"  Note:    No context window — audio only, one clip per word")
    print("=" * 60)

    out_path = NOTEBOOKS_DIR / "wav2vec_features.npy"

    if out_path.exists():
        print(f"\nOutput already exists: {out_path}")
        print("Delete to re-run.")
        return

    print(f"\nLoading Wav2Vec2 model ({WAV2VEC_MODEL_ID})...")
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL_ID)
    model     = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL_ID).to(device)
    model.eval()
    print("Model loaded.")

    all_features = []

    for story_idx in range(4):
        print(f"\n{'─'*50}")
        print(f"Story {story_idx}: {STORIES[story_idx]['name']}")

        audio    = load_and_concat_audio(story_idx)
        features = extract_wav2vec_features_story(story_idx, model, processor, audio)

        all_features.append(features)

        for l in [0, 6, 11]:
            f = features[l]
            print(f"  L{l:02d}: {f.shape}  mean={f.mean():.4f}  std={f.std():.4f}")

    print(f"\n{'='*60}")
    print("Saving...")
    np.save(str(out_path), np.array(all_features, dtype=object))
    print(f"  {out_path}")

    print("\nVerifying...")
    loaded = np.load(str(out_path), allow_pickle=True)
    for s in range(4):
        print(f"  Story {s}: L6={loaded[s][6].shape}")

    print("\nDone.")
    print("Next: use wav2vec_features.npy for residual analysis")
    print("      (CLAP text features - Wav2Vec features → residual encoding)")


if __name__ == "__main__":
    main()
