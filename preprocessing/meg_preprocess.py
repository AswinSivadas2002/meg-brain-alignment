#!/usr/bin/env python3
"""
MEG Preprocessing — v2 (ICALabel with mag→eeg relabeling)
==========================================================
Changes from v1:
  - Relabel only mag channels (208) as 'eeg' using safe string-type check
    so ICAlabel accepts the data (ref_meg and misc channels untouched)
  - Build DigMontage from existing 3D positions in raw.info
  - ALWAYS revert back to mag after ICAlabel (needed for picks='meg')
  - n_components=100 (no Maxwell filter on MEG-MASC)
  - Fallback to EOG/ECG correlation if ICAlabel still fails

Run:
  nohup taskset -c 0-17 python3 -u meg_preprocess_multiprogramming_v2.py \
      > preprocess_log.txt 2>&1 &
  echo $!
"""

import numpy as np
import os
import multiprocessing as mp
import time


# ── Configuration ──────────────────────────────────────────────────────────
root     = "/home/mtech1/Desktop/meg_project/data"
save_dir = "/home/mtech1/Desktop/meg_project/notebooks"

subjects = [str(i).zfill(2) for i in range(1, 28)]  # 01 to 27

N_WORKERS = 5   # adjust based on available RAM (~8-12 GB per worker)


def process_subject(args):
    """Process one complete subject — all 4 stories sequentially."""

    # ── Thread limiting — must be FIRST before any numpy import ────────────
    import os
    os.environ["OMP_NUM_THREADS"]      = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"]      = "2"

    sub, root_path, save_path = args

    # ── Per-process imports ─────────────────────────────────────────────────
    import numpy as np
    import mne
    import mne_bids
    import pandas as pd
    import warnings
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    from wordfreq import zipf_frequency

    mne.set_log_level("WARNING")

    # ── Skip if already done ────────────────────────────────────────────────
    out_path = os.path.join(save_path, f"sub{sub}-meg-data.npy")
    if os.path.exists(out_path):
        print(f"sub{sub} already exists, skipping", flush=True)
        return sub, True

    ph_info = pd.read_csv(os.path.join(save_path, "phoneme_info.csv"))

    # ────────────────────────────────────────────────────────────────────────
    def meg_preprocessing(raw):
        """Epoch around phoneme onsets, baseline correct, clip."""
        meta = []
        for annot in raw.annotations:
            desc_dict = eval(annot["description"])
            desc_dict.update({
                "onset":    annot["onset"],
                "duration": annot["duration"]
            })
            meta.append(desc_dict)

        meta = pd.DataFrame(meta)
        meta["intercept"] = 1.0

        phonemes = meta.query('kind == "phoneme"').copy()
        for ph, group in phonemes.groupby("phoneme"):
            base_ph = ph.split("_")[0]
            match   = ph_info.query("phoneme == @base_ph")
            if len(match) == 1:
                meta.loc[group.index, "voiced"] = (
                    match.iloc[0].phonation == "v")
            else:
                meta.loc[group.index, "voiced"] = False

        meta["is_word"] = False
        words = meta.query('kind == "word"').copy()
        meta.loc[words.index + 1, "is_word"]  = True
        meta.loc[words.index + 1, "wordfreq"] = (
            words.word.apply(
                lambda x: zipf_frequency(x, "en")).values
        )
        meta = meta.query('kind == "phoneme"').copy()

        sfreq  = raw.info["sfreq"]
        events = np.zeros((len(meta), 3), dtype=int)
        events[:, 0] = (meta["onset"] * sfreq).astype(int)
        events[:, 2] = 1

        epochs = mne.Epochs(
            raw, events,
            tmin=-0.2, tmax=0.6,
            baseline=None,
            decim=10,
            preload=True,
            picks="meg",
            event_repeated="drop",
            metadata=meta
        )

        data   = epochs.get_data()
        thresh = np.percentile(np.abs(data), 99)
        data   = np.clip(data, -thresh, thresh)
        epochs._data = data
        epochs.apply_baseline(baseline=(-0.2, 0.0))
        return epochs

    # ────────────────────────────────────────────────────────────────────────
    def relabel_mag_to_eeg(raw):
        """
        Relabel only the 208 mag channels to 'eeg'.
        ref_meg (16) and misc (32) channels are left untouched.
        Uses safe string-type check — no FIFF constant imports needed.
        Builds DigMontage from 3D positions already stored in raw.info
        so ICAlabel can compute component topographies.
        Returns reverse mapping (eeg→mag) for reverting after ICAlabel.
        """
        # Safe version: check string type directly, no FIFF constants
        mag_chs = [ch for ch in raw.ch_names
                   if raw.get_channel_types([ch])[0] == 'mag']

        print(f"    Relabeling {len(mag_chs)} mag → eeg "
              f"(ref_meg and misc untouched)", flush=True)

        mapping_to_eeg = {ch: 'eeg' for ch in mag_chs}
        raw.set_channel_types(mapping_to_eeg)

        # Build DigMontage from existing 3D sensor positions in raw.info
        # ch_info['loc'][:3] = x, y, z in metres in head coordinate frame
        ch_pos = {}
        for ch_info in raw.info['chs']:
            if ch_info['ch_name'] in mag_chs:
                loc = ch_info['loc'][:3]
                if not np.all(loc == 0):   # skip channels with no position
                    ch_pos[ch_info['ch_name']] = loc

        if ch_pos:
            montage = mne.channels.make_dig_montage(
                ch_pos=ch_pos, coord_frame='head'
            )
            raw.set_montage(montage, on_missing='ignore')
            print(f"    Montage set with {len(ch_pos)}/{len(mag_chs)} "
                  f"channel positions", flush=True)
        else:
            print(f"    WARNING: no 3D positions found in raw.info — "
                  f"ICAlabel topographies may be unreliable", flush=True)

        # Reverse mapping to undo relabeling after ICAlabel
        mapping_back = {ch: 'mag' for ch in mag_chs}
        return mapping_back

    # ────────────────────────────────────────────────────────────────────────
    def run_icalabel(raw, ica, sub, task):
        """
        Attempt ICAlabel after relabeling mag→eeg.
        The finally block ALWAYS reverts back to mag before returning,
        whether ICAlabel succeeds or fails — this is critical because
        picks='meg' in epoching requires channels to be typed as mag.
        Returns list of excluded component indices, or None if failed.
        """
        mapping_back = relabel_mag_to_eeg(raw)

        try:
            ic_labels = label_components(raw, ica, method="iclabel")
            labels    = ic_labels["labels"]
            probs     = ic_labels["y_pred_proba"]

            exclude = [
                i for i, (label, prob) in
                enumerate(zip(labels, probs))
                if label not in ("brain", "other")
                and prob.max() > 0.70
            ]

            excluded_labels = [labels[i] for i in exclude]
            print(f"  sub{sub} story{task}: ICALabel excluded "
                  f"{len(exclude)} components: {excluded_labels}",
                  flush=True)

            # Full breakdown — useful for checking if classifications
            # are sensible (eye blink/heartbeat should dominate excluded)
            label_counts = {}
            for l in labels:
                label_counts[l] = label_counts.get(l, 0) + 1
            print(f"  sub{sub} story{task}: all component labels: "
                  f"{label_counts}", flush=True)

            return exclude

        except Exception as e:
            print(f"  sub{sub} story{task}: ICALabel failed ({e})",
                  flush=True)
            return None

        finally:
            # ALWAYS revert to mag — picks='meg' needs this
            raw.set_channel_types(mapping_back)
            print(f"  sub{sub} story{task}: reverted to mag ✓", flush=True)

    # ────────────────────────────────────────────────────────────────────────
    def process_one_story(task):
        """Process one story for this subject with ICA."""
        checkpoint = os.path.join(
            save_path, f"sub{sub}-story{task}-checkpoint.npy"
        )
        if os.path.exists(checkpoint):
            print(f"  sub{sub} story{task}: loading checkpoint", flush=True)
            return np.load(checkpoint, allow_pickle=True)

        try:
            bids_path = mne_bids.BIDSPath(
                subject=sub, session="0", task=str(task),
                datatype="meg", root=root_path
            )

            raw = mne_bids.read_raw_bids(bids_path)
            raw.load_data()

            # ── Step 1: Bandpass filter ────────────────────────────────────
            raw.filter(0.5, 30.0, n_jobs=1)
            print(f"  sub{sub} story{task}: filtered", flush=True)

            # ── Step 2: Fit ICA ────────────────────────────────────────────
            # n_components=100: MEG-MASC has no Maxwell filter → full rank
            print(f"  sub{sub} story{task}: fitting ICA...", flush=True)
            t_ica = time.time()

            ica = ICA(
                n_components=100,
                method='picard',
                max_iter=200,
                random_state=42,
                fit_params=dict(ortho=False,extended=True)
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ica.fit(raw, picks='meg')
                converged = not any(
                    "did not converge" in str(x.message) for x in w
                )

            print(f"  sub{sub} story{task}: "
                  f"ICA {'converged ✓' if converged else 'WARNING: did not converge'} "
                  f"in {time.time()-t_ica:.0f}s "
                  f"({ica.n_components_} components)", flush=True)

            # ── Step 3: ICALabel with mag→eeg relabeling ───────────────────
            exclude = run_icalabel(raw, ica, sub, task)

            if exclude is None:
                # ICAlabel failed even after relabeling → use fallback
                print(f"  sub{sub} story{task}: using EOG/ECG fallback",
                      flush=True)
                try:
                    eog_idx, _ = ica.find_bads_eog(raw, ch_name='MEG 110')
                except Exception:
                    eog_idx = []
                try:
                    ecg_idx, _ = ica.find_bads_ecg(raw)
                except Exception:
                    ecg_idx = []
                exclude = list(set(list(eog_idx) + list(ecg_idx)))
                print(f"  sub{sub} story{task}: fallback excluded "
                      f"{len(exclude)} components", flush=True)

            ica.exclude = exclude

            # ── Step 4: Apply ICA ──────────────────────────────────────────
            if len(ica.exclude) > 0:
                ica.apply(raw)
                print(f"  sub{sub} story{task}: ICA applied, "
                      f"removed {len(ica.exclude)} components", flush=True)
            else:
                print(f"  sub{sub} story{task}: no components excluded",
                      flush=True)

            # ── Step 5: Epoch + baseline + clip ───────────────────────────
            epochs = meg_preprocessing(raw)
            epochs.metadata["half"] = np.round(
                np.linspace(0, 1.0, len(epochs))).astype(int)
            epochs.metadata["task"]    = str(task)
            epochs.metadata["session"] = "0"

            word_epochs = epochs["is_word == True"]
            X_words     = word_epochs.get_data() * 1e13

            np.save(checkpoint, X_words)
            print(f"  sub{sub} story{task} done — shape {X_words.shape}",
                  flush=True)
            return X_words

        except Exception as e:
            print(f"  sub{sub} story{task} FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    # ── Process all 4 stories sequentially ────────────────────────────────
    print(f"\nWorker starting sub{sub}...", flush=True)
    t_sub   = time.time()
    megdata = []
    for task in range(4):
        X = process_one_story(task)
        megdata.append(X)

    if all(m is not None for m in megdata):
        np.save(out_path, np.array(megdata, dtype=object))
        print(f"Saved sub{sub} in {(time.time()-t_sub)/60:.1f} min ✓",
              flush=True)
        for task in range(4):
            ck = os.path.join(
                save_path, f"sub{sub}-story{task}-checkpoint.npy"
            )
            if os.path.exists(ck):
                os.remove(ck)
        return sub, True
    else:
        print(f"sub{sub} incomplete — not saved", flush=True)
        return sub, False


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import psutil

    ram_gb    = psutil.virtual_memory().available / 1e9
    cpu_count = os.cpu_count()
    print(f"Available RAM: {ram_gb:.1f} GB")
    print(f"CPU cores:     {cpu_count}")
    print(f"Workers:       {N_WORKERS}")
    print(f"Subjects:      {len(subjects)}")

    remaining = [
        s for s in subjects
        if not os.path.exists(
            os.path.join(save_dir, f"sub{s}-meg-data.npy")
        )
    ]
    print(f"Remaining:     {len(remaining)}/27\n")

    if not remaining:
        print("All subjects already done!")
    else:
        args_list = [(sub, root, save_dir) for sub in remaining]

        t_start = time.time()
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=N_WORKERS) as pool:
            results = pool.map(process_subject, args_list)

        print("\n── Results ──────────────────────────────────")
        for sub, success in results:
            status = "✓" if success else "✗ FAILED"
            print(f"  sub{sub}: {status}")

        elapsed = (time.time() - t_start) / 3600
        print(f"\nTotal time: {elapsed:.2f} hours")
