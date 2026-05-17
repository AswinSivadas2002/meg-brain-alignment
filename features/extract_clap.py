#!/usr/bin/env python3
"""
CLAP Feature Extraction for MEG Brain Encoding
===============================================
Direct equivalent of BERT_Features.ipynb, but for CLAP.

Text encoder:  RoBERTa-based, 12 transformer layers, 768-dim hidden states
Audio encoder: HTSAT, 5 stages, shapes (B,C,H,W) — mean pooled spatially

Output files (saved to notebooks/):
  clap_text_features_synth.npy   -- dict {layer 0-11: (n_words, 768)} per story
  clap_audio_features_synth.npy  -- dict {layer 0-4:  (n_words, C)}   per story

Usage:
  conda activate meg_env
  nohup taskset -c 0-15 python clap_feature_extraction.py > clap_log.txt 2>&1 &
"""

import numpy as np
import torch
import librosa
import pandas as pd
import glob
import time as tm
from pathlib import Path
from transformers import ClapModel, ClapProcessor
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PROJECT_DIR   = Path("/home/mtech1/Desktop/meg_project")
DATA_DIR      = PROJECT_DIR / "data"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
AUDIO_DIR     = DATA_DIR / "stimuli" / "audio"
SYNTH_DIR = PROJECT_DIR / "synthesized_words"

CLAP_MODEL_ID   = "laion/clap-htsat-unfused"
TARGET_SR       = 48000
CONTEXT_LEN     = 20
CLIP_DURATION   = 1.0
N_LAYERS_TEXT   = 12   # RoBERTa transformer layers
N_LAYERS_AUDIO  = 5    # HTSAT stages (confirmed from model inspection)
REMOVE_CHARS    = [",", "\"", "@"]

STORIES = {
    0: {"task": 0, "name": "cable_spool_fort", "chunks": list(range(6))},
    1: {"task": 1, "name": "easy_money",       "chunks": list(range(8))},
    2: {"task": 2, "name": "lw1",              "chunks": None},
    3: {"task": 3, "name": "the_black_willow", "chunks": list(range(12))},
}

STORY_WORD_COUNTS = {0: 668, 1: 1503, 2: 2637, 3: 3753}

device = torch.device("cpu")

# ─── WORD LIST LOADING ────────────────────────────────────────────────────────

def load_word_list(story_idx):
    task    = STORIES[story_idx]["task"]
    pattern = str(DATA_DIR / "sub-01" / "ses-0" / "meg" /
                  f"sub-01_ses-0_task-{task}_events.tsv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"events.tsv not found: {pattern}")

    df = pd.read_csv(files[0], sep='\t')

    # MEG-MASC stores metadata as Python dict string in 'trial_type' column
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
    info, name, chunks = STORIES[story_idx], STORIES[story_idx]["name"], STORIES[story_idx]["chunks"]

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


def extract_word_clip(audio, onset_sec, offset_sec, word=None):
    """Load from synthesized word audio if available, else fall back to story clip."""
    clip_len = int(CLIP_DURATION * TARGET_SR)
    
    if word is not None:
        syn_path = SYNTH_DIR / f"{word.lower().strip()}.wav"
        if syn_path.exists():
            import soundfile as sf
            clip, _ = sf.read(str(syn_path), dtype='float32')
            # Ensure exactly clip_len samples
            if len(clip) >= clip_len:
                return clip[:clip_len]
            else:
                return np.pad(clip, (0, clip_len - len(clip)))
    
    # Fallback to story audio clip
    onset_sample = int(onset_sec * TARGET_SR)
    raw  = audio[onset_sample : int(offset_sec * TARGET_SR)]
    clip = np.zeros(clip_len, dtype=np.float32)
    clip[:min(len(raw), clip_len)] = raw[:clip_len]
    return clip

# ─── TEXT HIDDEN STATES ───────────────────────────────────────────────────────

def get_target_token_indices(word_seq, tokenizer, target_word_idx):
    word_ind_to_token_ind = {}
    seq_tokens = []
    n_tokens   = 0
    for i, word in enumerate(word_seq):
        word_ind_to_token_ind[i] = []
        for token in tokenizer.tokenize(word):
            if token not in REMOVE_CHARS:
                seq_tokens.append(token)
                word_ind_to_token_ind[i].append(n_tokens)
                n_tokens += 1
    return seq_tokens, word_ind_to_token_ind[target_word_idx]


@torch.inference_mode()
def extract_text_hidden_states(word_seq, target_word_idx, model, tokenizer):
    seq_tokens, token_inds = get_target_token_indices(word_seq, tokenizer, target_word_idx)

    if not token_inds:
        return {l: np.zeros(768) for l in range(N_LAYERS_TEXT)}

    tokens_tensor = torch.tensor(
        [tokenizer.convert_tokens_to_ids(seq_tokens)]
    ).to(device)

    hidden_states = model.text_model(
        input_ids=tokens_tensor,
        output_hidden_states=True,
        return_dict=True,
    ).hidden_states  # tuple of 13: [embedding, L1..L12]

    layer_vecs = {}
    for layer_idx in range(N_LAYERS_TEXT):
        h_np = hidden_states[layer_idx + 1][0].cpu().numpy()   # (seq_len, 768)
        layer_vecs[layer_idx] = np.mean(h_np[token_inds, :], axis=0)  # (768,)
    return layer_vecs


# ─── MAIN EXTRACTION LOOP ─────────────────────────────────────────────────────

@torch.inference_mode()
def extract_clap_features_story(story_idx, model, processor, audio):
    word_list, word_onsets = load_word_list(story_idx)
    n_words   = len(word_list)
    tokenizer = processor.tokenizer

    text_layer_reps  = {l: [] for l in range(N_LAYERS_TEXT)}
    audio_layer_reps = {l: [] for l in range(N_LAYERS_AUDIO)}

    start_time = tm.time()

    for word_idx in range(n_words):

        # ── Text: sliding context window ──
        if word_idx < CONTEXT_LEN:
            word_seq        = word_list[:word_idx + 1]
            target_word_idx = word_idx
        else:
            word_seq        = word_list[word_idx - CONTEXT_LEN + 1 : word_idx + 1]
            target_word_idx = CONTEXT_LEN - 1

        text_vecs = extract_text_hidden_states(word_seq, target_word_idx, model, tokenizer)
        for l in range(N_LAYERS_TEXT):
            text_layer_reps[l].append(text_vecs[l])

        # ── Audio: word clip → HTSAT encoder → mean pool spatial dims ──
        onset_sec, offset_sec = word_onsets[word_idx]
        word_str = word_list[word_idx]

        audio_clip = extract_word_clip(audio, onset_sec, offset_sec, word=word_str)
        audio_inputs = processor(audio=audio_clip, return_tensors="pt", sampling_rate=TARGET_SR)
        audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

        audio_hidden = model.audio_model(
            **audio_inputs,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states  # tuple of 5, each (1, C, H, W)

        for l in range(N_LAYERS_AUDIO):
            # mean pool over spatial dims H, W → (C,)
            a_vec = audio_hidden[l][0].mean(dim=(-2, -1)).cpu().numpy()
            audio_layer_reps[l].append(a_vec)

        if (word_idx + 1) % 100 == 0:
            elapsed   = tm.time() - start_time
            remaining = elapsed / (word_idx + 1) * (n_words - word_idx - 1)
            print(f"  [{word_idx+1}/{n_words}] elapsed={elapsed/60:.1f}m  "
                  f"remaining≈{remaining/60:.1f}m")
            start_time = tm.time()

    text_features  = {l: np.stack(text_layer_reps[l])  for l in range(N_LAYERS_TEXT)}
    audio_features = {l: np.stack(audio_layer_reps[l]) for l in range(N_LAYERS_AUDIO)}
    return text_features, audio_features


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CLAP Feature Extraction")
    print(f"  Model:         {CLAP_MODEL_ID}")
    print(f"  Device:        {device}")
    print(f"  Context:       {CONTEXT_LEN} words")
    print(f"  Text layers:   0-{N_LAYERS_TEXT-1}  (768-dim)")
    print(f"  Audio layers:  0-{N_LAYERS_AUDIO-1} (HTSAT stages, variable C-dim)")
    print("=" * 60)

    text_out  = NOTEBOOKS_DIR / "clap_text_features_synth.npy"
    audio_out = NOTEBOOKS_DIR / "clap_audio_features_synth.npy"

    if text_out.exists() and audio_out.exists():
        print("\nOutput files already exist. Delete to re-run.")
        return

    print(f"\nLoading CLAP model...")
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    model     = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
    model.eval()
    print("Model loaded.")

    all_text_features  = []
    all_audio_features = []

    for story_idx in range(4):
        print(f"\n{'─'*50}")
        print(f"Story {story_idx}: {STORIES[story_idx]['name']}")

        audio = load_and_concat_audio(story_idx)
        text_feats, audio_feats = extract_clap_features_story(
            story_idx, model, processor, audio
        )

        all_text_features.append(text_feats)
        all_audio_features.append(audio_feats)

        for l in [0, N_LAYERS_TEXT-1]:
            t = text_feats[l]
            print(f"  text  L{l:02d}: {t.shape}  mean={t.mean():.4f}  std={t.std():.4f}")
        for l in [0, N_LAYERS_AUDIO-1]:
            a = audio_feats[l]
            print(f"  audio L{l:02d}: {a.shape}  mean={a.mean():.4f}  std={a.std():.4f}")

    print(f"\n{'='*60}")
    print("Saving...")
    np.save(str(text_out),  np.array(all_text_features,  dtype=object))
    np.save(str(audio_out), np.array(all_audio_features, dtype=object))
    print(f"  {text_out}")
    print(f"  {audio_out}")

    print("\nVerifying...")
    t = np.load(str(text_out),  allow_pickle=True)
    a = np.load(str(audio_out), allow_pickle=True)
    for s in range(4):
        print(f"  Story {s}: text L6={t[s][6].shape}, audio L0={a[s][0].shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
