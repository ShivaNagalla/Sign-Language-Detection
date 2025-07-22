# build_dataset.py

import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# ─── 1. Parse transcripts from the corpus XML ──────────────────────────

def parse_corpus(corpus_xml_path):
    """
    Parse the RWTH-Boston-104 corpus file to extract:
      recording_ids: ['001', '003', ...]
      phrases      : ['john write homework', 'ix-1p see john yesterday ix', ...]
    """
    tree = ET.parse(corpus_xml_path)
    root = tree.getroot()
    recording_ids, phrases = [], []

    for rec in root.findall('recording'):
        rec_id = rec.attrib['name']
        orth = rec.find('.//orth')
        if orth is None or not orth.text:
            continue
        # clean: lowercase, strip [silence], collapse spaces
        text = orth.text.lower()
        text = text.replace('[silence]', '').strip()
        text = ' '.join(text.split())
        recording_ids.append(rec_id)
        phrases.append(text)

    return recording_ids, phrases

# ─── 2. Load & prepare tokenizer ────────────────────────────────────────

def load_tokenizer(json_path=r'C:\Users\manug\Sign-Language-Detection\tokenizer.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        return tokenizer_from_json(f.read())

# ─── 3. Load keypoints arrays ──────────────────────────────────────────

def load_keypoints_for_ids(keypoint_dir, recording_ids):
    keypoint_seqs = []
    for rec_id in recording_ids:
        # Pattern to match all segments of this recording
        pattern = os.path.join(keypoint_dir, f"{rec_id}_*_keypoints.npy")
        matches = sorted(glob.glob(pattern))  # sort to keep order by segment index
        
        if not matches:
            raise FileNotFoundError(f"No keypoints files found for recording ID {rec_id} with pattern {pattern}")
        
        # Load and concatenate all segments along time axis (axis=0)
        segments = []
        for fpath in matches:
            kp = np.load(fpath)
            if kp.size == 0:
                raise ValueError(f"Empty keypoints in file {fpath}")
            segments.append(kp.astype(np.float32))
        
        full_seq = np.concatenate(segments, axis=0)
        keypoint_seqs.append(full_seq)
    
    return keypoint_seqs

# ─── 4. Pad sequences ───────────────────────────────────────────────────

def pad_keypoint_sequences(sequences, max_len=None):
    if not sequences:
        raise ValueError("No keypoint sequences found! The input list is empty.")
    if max_len is None:
        max_len = max(seq.shape[0] for seq in sequences)
    feat_dim = sequences[0].shape[1]
    padded = np.zeros((len(sequences), max_len, feat_dim), dtype=np.float32)
    for i, seq in enumerate(sequences):
        L = seq.shape[0]
        padded[i, :L, :] = seq
    return padded

def pad_token_sequences(sequences, max_len=None):
    return pad_sequences(sequences, maxlen=max_len, padding='post')

# ─── 5. Build tf.data.Dataset ──────────────────────────────────────────

def build_dataset(corpus_xml, keypoint_dir, tokenizer_json,
                  batch_size=32, shuffle=True, 
                  max_kp_len=None, max_tok_len=None):
    # 1. transcripts
    rec_ids, phrases = parse_corpus(corpus_xml)

    # 2. tokenizer + tokenization
    tokenizer = load_tokenizer(tokenizer_json)
    token_seqs = tokenizer.texts_to_sequences(phrases)
    token_seqs = pad_token_sequences(token_seqs, max_len=max_tok_len)

    # 3. keypoints
    kp_seqs = load_keypoints_for_ids(keypoint_dir, rec_ids)
    kp_seqs = pad_keypoint_sequences(kp_seqs, max_len=max_kp_len)

    # 4. wrap into tf.data
    X = tf.convert_to_tensor(kp_seqs)           # shape (N, T_kp, feat)
    y = tf.convert_to_tensor(token_seqs)        # shape (N, T_tok)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=X.shape[0])
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, tokenizer

# ─── 6. Example usage ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Adjust these paths:
    CORPUS_XML     = r"C:\Users\manug\Sign-Language-Detection\rwth-boston-104\corpus\train.sentences.pronunciations.corpus"
    KEYPOINT_DIR   = r"C:\Users\manug\Sign-Language-Detection\data\keypoints"           # where you saved 001.npy, 003.npy, ...
    TOKENIZER_JSON = r"C:\Users\manug\Sign-Language-Detection\tokenizer.json"

    ds, tokenizer = build_dataset(
        corpus_xml = CORPUS_XML,
        keypoint_dir = KEYPOINT_DIR,
        tokenizer_json = TOKENIZER_JSON,
        batch_size = 16,
        shuffle = True
    )

    print("Dataset ready. Sample batch shapes:")
    for X_batch, y_batch in ds.take(1):
        print("  keypoints batch:", X_batch.shape)   # (B, T_kp, feat)
        print("  tokens batch:   ", y_batch.shape)   # (B, T_tok)
    print("Vocab size:", len(tokenizer.word_index) + 1)