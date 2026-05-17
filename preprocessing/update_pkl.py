# =========================================
# UPDATE EXISTING PKL WITH CHANNEL INFO
# (NO ICA RERUN NEEDED)
# =========================================

import pickle
import mne
import mne_bids
from collections import Counter

print("Loading existing structure...")

with open("meg_structure.pkl", "rb") as f:
    data = pickle.load(f)

# ------------------------------
# Load raw (FAST)
# ------------------------------
print("Loading raw data (fast)...")

bids_path = mne_bids.BIDSPath(
    subject="13",
    session="0",
    task="0",
    datatype="meg",
    root="/home/mtech1/Desktop/meg_project/data"
)

raw = mne_bids.read_raw_bids(bids_path)

# ------------------------------
# Extract channel info (FIXED VERSION)
# ------------------------------
print("Extracting channel info...")

# Compatible with older MNE versions
channel_types = [mne.channel_type(raw.info, i) for i in range(len(raw.ch_names))]

channel_info = []
for idx, ch in enumerate(raw.info['chs']):
    channel_info.append({
        "name": ch['ch_name'],
        "type": mne.channel_type(raw.info, idx),
        "location": ch['loc'][:3].tolist()
    })

# Show distribution
print("\nChannel type distribution:")
print(Counter(channel_types))

# ------------------------------
# Add to existing structure
# ------------------------------
data["channel_names"] = raw.ch_names
data["channel_types"] = channel_types
data["channel_locations"] = channel_info

print("\nAdded channel metadata to structure.")

# ------------------------------
# Save updated file
# ------------------------------
print("Saving updated structure...")

with open("meg_structure_updated.pkl", "wb") as f:
    pickle.dump(data, f)

print("\n✅ DONE — saved as meg_structure_updated.pkl")
