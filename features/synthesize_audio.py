#!/usr/bin/env python3
"""
synthesize_word_audio.py
Synthesizes each unique word from all 4 stories into isolated .wav files
using gTTS (Google TTS) + ffmpeg for resampling.

Output: /home/mtech1/Desktop/meg_project/synthesized_words/{word}.wav
Run: python3 synthesize_word_audio.py
"""

import os
import io
import time
import numpy as np
import pandas as pd
from pathlib import Path
from gtts import gTTS
import subprocess

PROJECT_DIR  = Path("/home/mtech1/Desktop/meg_project")
DATA_DIR     = PROJECT_DIR / "data"
OUT_DIR      = PROJECT_DIR / "synthesized_words"
OUT_DIR.mkdir(exist_ok=True)

TARGET_SR  = 48000   # CLAP expects 48kHz
TARGET_DUR = 48000   # 1 second = 48000 samples

SUBJECT = "01"       # any subject — word list is same across subjects
STORIES = [0, 1, 2, 3]

# ── Collect all unique words ───────────────────────────────────────────────────
def get_all_words():
    all_words = set()
    for task in STORIES:
        events = list(DATA_DIR.glob(
            f"sub-{SUBJECT}/ses-0/meg/*task-{task}*_events.tsv"
        ))
        if not events:
            print(f"  No events.tsv for task {task}, skipping")
            continue
        df = pd.read_csv(events[0], sep='\t')
        
        # Parse the trial_type dict string column
        for _, row in df.iterrows():
            try:
                desc = eval(row['trial_type'])
                if desc.get('kind') == 'word':
                    word = desc.get('word', '').lower().strip()
                    if word and word.isalpha():
                        all_words.add(word)
            except:
                continue
        
        print(f"  Story {task}: done")
    
    print(f"Total unique words: {len(all_words)}")
    return sorted(all_words)
# ── Synthesize one word via gTTS → mp3 → wav via ffmpeg ───────────────────────
def synthesize_word(word, out_path):
    if out_path.exists():
        return True  # already done

    try:
        # gTTS → mp3 bytes in memory
        tts = gTTS(text=word, lang='en', slow=False)
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)

        # Write mp3 to temp file
        tmp_mp3 = str(out_path).replace('.wav', '_tmp.mp3')
        with open(tmp_mp3, 'wb') as f:
            f.write(mp3_buf.read())

        # ffmpeg: mp3 → wav, resample to 48kHz, mono, trim/pad to exactly 1 second
        cmd = [
            'ffmpeg', '-y',
            '-i', tmp_mp3,
            '-ar', str(TARGET_SR),   # resample to 48kHz
            '-ac', '1',              # mono
            '-t', '1.0',             # trim to 1 second max
            str(out_path)
        ]
        result = subprocess.run(cmd, capture_output=True)

        # Clean up temp mp3
        if os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)

        # Check output exists and pad to exactly 1 second if shorter
        if out_path.exists():
            import soundfile as sf
            audio, sr = sf.read(str(out_path))
            if len(audio) < TARGET_DUR:
                audio = np.pad(audio, (0, TARGET_DUR - len(audio)))
                sf.write(str(out_path), audio, TARGET_SR)
            return True

    except Exception as e:
        print(f"  ERROR on '{word}': {e}")
        return False

    return False

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Collecting words from all 4 stories...")
    words = get_all_words()
    print(f"Total unique words: {len(words)}")

    # Filter already done
    remaining = [w for w in words if not (OUT_DIR / f"{w}.wav").exists()]
    print(f"Already done: {len(words) - len(remaining)}  |  Remaining: {len(remaining)}")

    print("\nSynthesizing...")
    failed = []
    for i, word in enumerate(remaining):
        out_path = OUT_DIR / f"{word}.wav"
        ok = synthesize_word(word, out_path)
        if not ok:
            failed.append(word)

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(remaining)} done")

        # Rate limit — gTTS is a Google API, be polite
        time.sleep(0.3)

    print(f"\nDone. {len(remaining) - len(failed)} synthesized, {len(failed)} failed.")
    if failed:
        print(f"Failed words: {failed[:20]}")
        # Save failed list for retry
        with open(OUT_DIR / "failed_words.txt", 'w') as f:
            f.write('\n'.join(failed))

if __name__ == "__main__":
    main()
