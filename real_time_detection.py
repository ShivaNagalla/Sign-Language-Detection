import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import json

# --- Config ---
SEQUENCE_LENGTH = 30
THRESHOLD = 0.8

# Paths
TOKENIZER_JSON = r"C:\Users\manug\Sign-Language-Detection\tokenizer.json"
MODEL_PATH = r"C:\Users\manug\Sign-Language-Detection\asl_phrase_transformer.h5"

# Load tokenizer
with open(TOKENIZER_JSON, 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        pos = np.arange(self.seq_len)[:, None]
        i = np.arange(self.d_model)[None, :]
        angle_rates = 1 / np.power(10000., (2 * (i // 2)) / self.d_model)
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + tf.cast(self.pos_encoding, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model
        })
        return config

model = load_model(MODEL_PATH, custom_objects={'PositionalEncoding': PositionalEncoding})

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Constants from your dataset (adjust if needed)
POSE_LEN = 33 * 4
FACE_LEN = 468 * 3
LH_LEN = 21 * 3
RH_LEN = 21 * 3

# Buffer to hold frames
seq_buffer = deque(maxlen=SEQUENCE_LENGTH)

def get_full_keypoints(results):
    # Extract pose
    pose_vals = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose_vals += [lm.x, lm.y, lm.z, lm.visibility]
    pose_vals += [0] * (POSE_LEN - len(pose_vals))

    # Extract face
    face_vals = []
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            face_vals += [lm.x, lm.y, lm.z]
    face_vals += [0] * (FACE_LEN - len(face_vals))

    # Extract left hand
    lh_vals = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh_vals += [lm.x, lm.y, lm.z]
    lh_vals += [0] * (LH_LEN - len(lh_vals))

    # Extract right hand
    rh_vals = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh_vals += [lm.x, lm.y, lm.z]
    rh_vals += [0] * (RH_LEN - len(rh_vals))

    # Concatenate all
    full_vector = pose_vals + face_vals + lh_vals + rh_vals
    return np.array(full_vector, dtype=np.float32)

def decode_tokens(token_ids):
    # Remove padding (0) tokens and map indices back to words
    words = []
    for t in token_ids:
        if t == 0:
            continue
        word = tokenizer.index_word.get(t, '')
        if word == '':
            continue
        words.append(word)
    return ' '.join(words)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(0)
    predicted_phrase = ''
    token_buffer = deque(maxlen=5)  # For smoothing predicted tokens
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)
        
        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        # Extract features and append to buffer
        full_kp = get_full_keypoints(results)
        seq_buffer.append(full_kp)
        
        # Predict when buffer full
        if len(seq_buffer) == SEQUENCE_LENGTH:
            input_seq = np.expand_dims(np.array(seq_buffer), axis=0)  # Shape (1, 30, feature_dim)
            
            # Prepare decoder input tokens: start token (usually 1) followed by padding
            # Here, we simplify by feeding zeros for decoder input (you can improve with proper start tokens)
            max_tok_len = model.input[1].shape[1] + 1  # Decoder input length, adjust if needed
            dec_in = np.zeros((1, max_tok_len-1), dtype=np.int32)
            
            # Predict token probabilities
            preds = model.predict([input_seq, dec_in], verbose=0)  # (1, T, vocab_size)
            
            # Take argmax tokens for each timestep
            pred_tokens = np.argmax(preds[0], axis=-1)
            
            # Smoothing: keep recent predicted tokens and take most common
            token_buffer.append(tuple(pred_tokens))
            # Just take last prediction for now
            decoded = decode_tokens(pred_tokens)
            predicted_phrase = decoded
        
        # Display predicted phrase
        cv2.rectangle(frame, (0,0), (640,40), (245,117,16), -1)
        cv2.putText(frame, predicted_phrase, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.imshow('ASL Phrase Transformer', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()