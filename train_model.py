import os
import xml.etree.ElementTree as ET
import glob
import numpy as np
import tensorflow as tf
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

def parse_corpus(corpus_xml_path):
    tree = ET.parse(corpus_xml_path)
    root = tree.getroot()
    rec_ids, phrases = [], []
    for rec in root.findall('recording'):
        rec_id = rec.attrib['name']
        orth = rec.find('.//orth')
        if orth is None or not orth.text:
            continue
        text = orth.text.lower().replace('[silence]', '').strip()
        text = ' '.join(text.split())
        rec_ids.append(rec_id)
        phrases.append(text)
    return rec_ids, phrases

def load_tokenizer(path):
    with open(path, 'r', encoding='utf-8') as f:
        return tokenizer_from_json(f.read())

def load_keypoints(keypoint_dir, rec_ids):
    seqs = []
    for rid in rec_ids:
        pattern = os.path.join(keypoint_dir, f"{rid}_*_keypoints.npy")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"Warning: no files for {rid}")
            continue
        parts = [np.load(fp) for fp in files]
        seq = np.concatenate(parts, axis=0)
        seqs.append(seq.astype(np.float32))
    return seqs

# Positional Encoding Layer
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_encoding[np.newaxis, ...], tf.float32)

class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_enc = positional_encoding(seq_len, d_model)
    def call(self, x):
        return x + self.pos_enc[:, :tf.shape(x)[1], :]

# Transformer Model Builder
def build_transformer(max_kp_len, feat_dim, max_tok_len, vocab_size,
                      d_model=128, num_heads=4, ff_dim=256):
    enc_inputs = layers.Input(shape=(max_kp_len, feat_dim), name='enc_inputs')
    enc_inputs_proj = layers.Dense(d_model)(enc_inputs)  # project 1629 → 128
    x = PositionalEncoding(max_kp_len, d_model)(enc_inputs_proj)
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    enc_output = layers.LayerNormalization(epsilon=1e-6)(attn_out + x)

    # Decoder
    dec_inputs = layers.Input(shape=(max_tok_len-1,), name='dec_inputs')
    emb = layers.Embedding(vocab_size, d_model)(dec_inputs)
    x2 = PositionalEncoding(max_tok_len-1, d_model)(emb)
    x2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x2, enc_output)
    x2 = layers.LayerNormalization(epsilon=1e-6)(x2 + emb)
    outputs = layers.Dense(vocab_size, activation='softmax')(x2)

    return tf.keras.Model([enc_inputs, dec_inputs], outputs)

# Data Preparation
def make_dataset(corpus_xml, keypoint_dir, tokenizer_json, batch_size=16):
    rec_ids, phrases = parse_corpus(corpus_xml)
    tokenizer = load_tokenizer(tokenizer_json)
    token_seqs = tokenizer.texts_to_sequences(phrases)
    max_tok_len = max(len(s) for s in token_seqs)
    token_seqs = pad_sequences(token_seqs, maxlen=max_tok_len, padding='post')

    kp_seqs = load_keypoints(keypoint_dir, rec_ids)
    
    # Add this check right here:
    feat_dims = [s.shape[1] for s in kp_seqs]
    if len(set(feat_dims)) > 1:
        raise ValueError(f"Inconsistent feature dimensions found: {set(feat_dims)}")

    max_kp_len = max(s.shape[0] for s in kp_seqs)
    feat_dim = kp_seqs[0].shape[1]
    padded_kp = np.zeros((len(kp_seqs), max_kp_len, feat_dim), np.float32)
    for i, s in enumerate(kp_seqs): 
        padded_kp[i, :s.shape[0]] = s

    # prepare decoder inputs/outputs
    dec_in = token_seqs[:, :-1]
    dec_out = token_seqs[:, 1:]

    # build tf dataset
    ds = tf.data.Dataset.from_tensor_slices(((padded_kp, dec_in), dec_out))
    ds = ds.shuffle(len(padded_kp)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, max_kp_len, feat_dim, max_tok_len, tokenizer


if __name__ == '__main__':
    # Paths
    CORPUS_XML = r"C:\Users\manug\Sign-Language-Detection\rwth-boston-104\corpus\train.sentences.pronunciations.corpus"
    KEYPOINT_DIR = r"C:\Users\manug\Sign-Language-Detection\data\keypoints"
    TOKENIZER_JSON = r"C:\Users\manug\Sign-Language-Detection\tokenizer.json"

    # Build dataset
    ds, max_kp_len, feat_dim, max_tok_len, tokenizer = make_dataset(
        CORPUS_XML, KEYPOINT_DIR, TOKENIZER_JSON)

    vocab_size = len(tokenizer.word_index) + 1

    # Build model
    model = build_transformer(max_kp_len, feat_dim, max_tok_len, vocab_size)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Train
    model.fit(ds, epochs=20)

    # Save
    model.save("asl_phrase_transformer.h5")
    print("Saved model to asl_phrase_transformer.h5")
