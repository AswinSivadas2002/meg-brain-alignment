#!/usr/bin/env python3
# =========================================
# MEG ICA STRUCTURE PIPELINE — V2
# Compares v2 preprocessing vs v1
# =========================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mne
import mne_bids
from mne.preprocessing import ICA
from mne_icalabel import label_components
import pickle
import pprint
import time
from collections import Counter

start_time = time.time()
print("=== STARTING MEG ICA PIPELINE V2 ===")

# ─── CONFIG ───────────────────────────────
SUBJECT  = "13"
SESSION  = "0"
TASK     = "0"
ROOT     = "/home/mtech1/Desktop/meg_project/data"
SAVE_DIR = "/home/mtech1/Desktop/meg_project/notebooks"

# ─── STEP 1: Load raw ─────────────────────
bids_path = mne_bids.BIDSPath(
    subject=SUBJECT, session=SESSION, task=TASK,
    datatype="meg", root=ROOT
)
print("Loading raw data...")
raw = mne_bids.read_raw_bids(bids_path)
raw.load_data()
print(f"Channels: {len(raw.ch_names)}, Sampling rate: {raw.info['sfreq']}")

# ─── STEP 2: Before ICA plot ──────────────
data_before = raw.get_data(picks='meg')
times = raw.times

plt.figure(figsize=(10, 4))
plt.plot(times[:2000], data_before[0, :2000])
plt.title("Before ICA (V2)")
plt.xlabel("Time"); plt.ylabel("Amplitude")
plt.savefig(f"{SAVE_DIR}/before_ica_v2.png")
plt.close()
print("Saved before_ica_v2.png")

# ─── STEP 3: Filter ───────────────────────
print("Filtering...")
raw.filter(0.5, 30.0)

# ─── STEP 4: Fit ICA ──────────────────────
print("Fitting ICA (extended infomax via Picard)...")
t_ica = time.time()

ica = ICA(
    n_components=100,
    method='picard',
    random_state=42,
    max_iter=200,
    fit_params=dict(ortho=False, extended=True)
)
ica.fit(raw, picks='meg')
print(f"ICA done in {time.time()-t_ica:.0f}s  ({ica.n_components_} components)")

# ─── STEP 5: Relabel mag→eeg for ICAlabel ─
print("Relabeling mag→eeg for ICAlabel...")
mag_chs = [ch for ch in raw.ch_names
           if raw.get_channel_types([ch])[0] == 'mag']
mapping_to_eeg  = {ch: 'eeg' for ch in mag_chs}
mapping_back    = {ch: 'mag' for ch in mag_chs}

raw.set_channel_types(mapping_to_eeg)

ch_pos = {}
for ch_info in raw.info['chs']:
    if ch_info['ch_name'] in mag_chs:
        loc = ch_info['loc'][:3]
        if not np.all(loc == 0):
            ch_pos[ch_info['ch_name']] = loc

if ch_pos:
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, on_missing='ignore')
    print(f"Montage set with {len(ch_pos)} positions")

# ─── STEP 6: ICAlabel ─────────────────────
print("Running ICAlabel...")
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ic_labels = label_components(raw, ica, method="iclabel")

    labels = ic_labels["labels"]
    probs  = ic_labels["y_pred_proba"]

    exclude = [
        i for i, (label, prob) in enumerate(zip(labels, probs))
        if label not in ("brain", "other") and prob.max() > 0.70
    ]

    label_counts = Counter(labels)
    excluded_labels = [labels[i] for i in exclude]
    ica_method_used = "iclabel"
    print(f"ICAlabel excluded {len(exclude)} components: {excluded_labels}")
    print(f"All component labels: {dict(label_counts)}")

except Exception as e:
    print(f"ICAlabel failed ({e}), falling back to EOG/ECG")
    try:
        eog_idx, _ = ica.find_bads_eog(raw, ch_name='MEG 110')
    except Exception:
        eog_idx = []
    try:
        ecg_idx, _ = ica.find_bads_ecg(raw)
    except Exception:
        ecg_idx = []
    exclude = list(set(list(eog_idx) + list(ecg_idx)))
    labels = ["unknown"] * ica.n_components_
    label_counts = {}
    excluded_labels = exclude
    ica_method_used = "eog_ecg_fallback"

finally:
    raw.set_channel_types(mapping_back)
    print("Reverted to mag ✓")

ica.exclude = exclude

# ─── STEP 7: Apply ICA ────────────────────
print("Applying ICA...")
raw_clean = raw.copy()
if len(ica.exclude) > 0:
    ica.apply(raw_clean)

# ─── STEP 8: After ICA plot ───────────────
data_after = raw_clean.get_data(picks='meg')

plt.figure(figsize=(10, 4))
plt.plot(times[:2000], data_after[0, :2000])
plt.title("After ICA (V2)")
plt.xlabel("Time"); plt.ylabel("Amplitude")
plt.savefig(f"{SAVE_DIR}/after_ica_v2.png")
plt.close()
print("Saved after_ica_v2.png")

# ─── STEP 9: Comparison plot ──────────────
plt.figure(figsize=(10, 4))
plt.plot(times[:2000], data_before[0, :2000], label="Before ICA", alpha=0.7)
plt.plot(times[:2000], data_after[0, :2000],  label="After ICA V2", alpha=0.7)
plt.legend(); plt.title("Before vs After ICA — V2")
plt.xlabel("Time"); plt.ylabel("Amplitude")
plt.savefig(f"{SAVE_DIR}/comparison_v2.png")
plt.close()
print("Saved comparison_v2.png")

# ─── STEP 10: V1 vs V2 comparison plot ────
print("Loading V1 structure for comparison...")
try:
    with open(f"{SAVE_DIR}/meg_structure_updated.pkl", "rb") as f:
        v1_data = pickle.load(f)
    v1_signal = v1_data.get("example_epoch", None)

    if v1_signal is not None:
        # V1 used fixed-length epochs, V2 uses word-onset epochs
        # Just compare the raw cleaned signal directly
        with open(f"{SAVE_DIR}/before_ica.pkl", "rb") as f:
            before_data = pickle.load(f)
        before_signal = before_data.get("example_epoch", None)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

        axes[0].plot(data_before[0, :2000], color='gray')
        axes[0].set_title("Before ICA (raw)")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(v1_data.get("example_epoch", np.zeros(801))[0], color='orange')
        axes[1].set_title("After ICA — V1 (EOG/ECG fallback)")
        axes[1].set_ylabel("Amplitude")

        axes[2].plot(data_after[0, :2000], color='royalblue')
        axes[2].set_title("After ICA — V2 (ICAlabel)")
        axes[2].set_ylabel("Amplitude")

        plt.suptitle("ICA Comparison: V1 vs V2", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/comparison_v1_vs_v2.png", dpi=150)
        plt.close()
        print("Saved comparison_v1_vs_v2.png")
except Exception as e:
    print(f"Could not load V1 structure for comparison: {e}")

# ─── STEP 11: Epoching ────────────────────
print("Creating fixed-length epochs for structure inspection...")
events = mne.make_fixed_length_events(raw_clean, duration=1.0)
epochs = mne.Epochs(
    raw_clean, events,
    tmin=-0.2, tmax=0.6,
    baseline=(-0.2, 0),
    preload=True,
    picks="meg"
)
X = epochs.get_data()
print(f"Epoch shape: {X.shape}")

# ─── STEP 12: Channel info ────────────────
channel_types = [mne.channel_type(raw.info, i) for i in range(len(raw.ch_names))]
channel_info  = []
for idx, ch in enumerate(raw.info['chs']):
    channel_info.append({
        "name":     ch['ch_name'],
        "type":     mne.channel_type(raw.info, idx),
        "location": ch['loc'][:3].tolist()
    })

print("\nChannel type distribution:")
print(Counter(channel_types))

# ─── STEP 13: Build structure ─────────────
meg_structure_v2 = {
    "n_channels":              len(raw.ch_names),
    "sampling_rate":           raw.info['sfreq'],
    "ica_n_components":        ica.n_components_,
    "ica_excluded_components": ica.exclude,
    "ica_excluded_labels":     excluded_labels,
    "ica_all_label_counts":    dict(label_counts),
    "ica_method":              ica_method_used,
    "data_shape":              X.shape,
    "epochs":                  X.shape[0],
    "channels":                X.shape[1],
    "timepoints":              X.shape[2],
    "description":             "MEG data structured as (epochs, channels, timepoints) — V2 preprocessing",
    "channel_names":           raw.ch_names,
    "channel_types":           channel_types,
    "channel_locations":       channel_info,
    "example_epoch":           X[0],
}

pprint.pprint({k: v for k, v in meg_structure_v2.items()
               if k != "example_epoch" and k != "channel_locations"
               and k != "channel_names" and k != "channel_types"})

# ─── STEP 14: Save ────────────────────────
out_path = f"{SAVE_DIR}/meg_structure_v2.pkl"
with open(out_path, "wb") as f:
    pickle.dump(meg_structure_v2, f)
print(f"\nSaved: {out_path}")

# ─── STEP 15: Print V1 vs V2 diff ─────────
print("\n" + "="*50)
print("V1 vs V2 COMPARISON SUMMARY")
print("="*50)
try:
    with open(f"{SAVE_DIR}/meg_structure_updated.pkl", "rb") as f:
        v1 = pickle.load(f)
    print(f"{'Metric':<30} {'V1':>15} {'V2':>15}")
    print("-"*60)
    print(f"{'n_channels':<30} {v1.get('n_channels','?'):>15} {meg_structure_v2['n_channels']:>15}")
    print(f"{'ica_n_components':<30} {v1.get('ica_n_components','?'):>15} {meg_structure_v2['ica_n_components']:>15}")
    print(f"{'excluded components':<30} {str(len(v1.get('ica_excluded_components',[]))) :>15} {str(len(meg_structure_v2['ica_excluded_components'])):>15}")
    print(f"{'excluded labels':<30} {'eog/ecg only':>15} {str(excluded_labels):>15}")
    print(f"{'epoch shape':<30} {str(v1.get('data_shape','?')):>15} {str(meg_structure_v2['data_shape']):>15}")
    print(f"{'ICA method':<30} {'eog_ecg_fallback':>15} {'iclabel':>15}")
except Exception as e:
    print(f"Could not load V1 for comparison: {e}")

print(f"\nTotal time: {(time.time()-start_time)/60:.2f} minutes")
print("=== DONE ===")
