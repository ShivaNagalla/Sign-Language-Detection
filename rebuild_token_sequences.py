import os
import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

# Path to transcripts
corpus_dir = r"C:\Users\manug\Sign-Language-Detection\rwth-boston-104\corpus"

phrases = []
recording_ids = []

# Iterate over .txt files
for fname in os.listdir(corpus_dir):
    if fname.endswith(".txt"):
        rec_id = os.path.splitext(fname)[0]
        fpath = os.path.join(corpus_dir, fname)

        try:
            with open(fpath, "r") as f:
                phrase = f.read().strip().lower()
            if phrase:
                recording_ids.append(rec_id)
                phrases.append(phrase)
            else:
                print(f"⚠️ Empty transcript: {fname}")
        except Exception as e:
            print(f"❌ Could not read {fname}: {e}")

print(f"📄 Found {len(phrases)} transcript files")

# Tokenize and pad
token_sequences = tokenizer.texts_to_sequences(phrases)
token_sequences = pad_sequences(token_sequences, padding="post")

# Save results
np.save("token_sequences.npy", token_sequences)
np.save("recording_ids.npy", np.array(recording_ids))

print("✅ Saved token_sequences.npy and recording_ids.npy")