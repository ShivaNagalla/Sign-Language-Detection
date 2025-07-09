import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import os
from tensorflow.keras.models import load_model



class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

        # build (1
        #, seq_len, d_model) sinusoidal table
        pos = np.arange(seq_len)[:, None]
        i   = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000., (2 * (i // 2)) / d_model)
        angle_rads  = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        # cast to match (could be float16 under mixed precision)
        return x + tf.cast(self.pos_encoding, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model
        })
        return cfg
    
# --- CONFIG ---
DATA_PATH = './ASL_Citizen'
VIDEOS_PATH = os.path.join(DATA_PATH, 'videos')
SEQUENCE_LENGTH = 30
THRESHOLD = 0.8

# Dynamically load ACTIONS from the trained label encoder
  # Access the output layer

# --- Feature Settings ---
POSE_LEN = 33 * 4
FACE_LEN = 468 * 3
LH_LEN = 21 * 3
RH_LEN = 21 * 3
LH_START = POSE_LEN + FACE_LEN
RH_START = LH_START + LH_LEN

model = load_model(
    'asl_transformer_model.h5',
    custom_objects={'PositionalEncoding': PositionalEncoding}
)

        

# Load the trained model with custom PositionalEncoding

num_classes = model.output_shape[-1]
ACTIONS = [str(i) for i in range(num_classes)]


# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_hand_features(full_kp: np.ndarray) -> np.ndarray:
    lh = full_kp[LH_START:LH_START + LH_LEN].reshape(21, 3)
    rh = full_kp[RH_START:RH_START + RH_LEN].reshape(21, 3)
    lh -= lh[0] if len(lh) > 0 else 0
    rh -= rh[0] if len(rh) > 0 else 0
    feat = np.concatenate([lh.flatten(), rh.flatten()])
    max_d = np.max(np.linalg.norm(feat.reshape(-1, 3), axis=1)) if feat.size > 0 else 1
    return feat / (max_d + 1e-6)

# Buffers
seq_buffer = deque(maxlen=SEQUENCE_LENGTH)
prob_buffer = deque(maxlen=10)
sentence = []

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res = holistic.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(img, res.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, res.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # 1) Pose
        pose_vals = []
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark:
                pose_vals += [lm.x, lm.y, lm.z, lm.visibility]
        pose_vals += [0] * (POSE_LEN - len(pose_vals))

        # 2) Face
        face_vals = []
        if res.face_landmarks:
            for lm in res.face_landmarks.landmark:
                face_vals += [lm.x, lm.y, lm.z]
        face_vals += [0] * (FACE_LEN - len(face_vals))

        # 3) Left hand
        lh_vals = []
        if res.left_hand_landmarks:
            for lm in res.left_hand_landmarks.landmark:
                lh_vals += [lm.x, lm.y, lm.z]
        lh_vals += [0] * (LH_LEN - len(lh_vals))

        # 4) Right hand
        rh_vals = []
        if res.right_hand_landmarks:
            for lm in res.right_hand_landmarks.landmark:
                rh_vals += [lm.x, lm.y, lm.z]
        rh_vals += [0] * (RH_LEN - len(rh_vals))

        # Combine into one fixed-length vector
        full = pose_vals + face_vals + lh_vals + rh_vals
        arr  = np.array(full, dtype=np.float32)
        feat = extract_hand_features(arr)
        seq_buffer.append(feat)

        arr = np.array(full)
        feat = extract_hand_features(arr)
        seq_buffer.append(feat)

        if len(seq_buffer) == SEQUENCE_LENGTH:
            inp = np.expand_dims(np.array(seq_buffer), axis=0)
            proba = model.predict(inp, verbose=0)[0]
            prob_buffer.append(proba)
            avg = np.mean(prob_buffer, axis=0)
            idx = np.argmax(avg)
            if avg[idx] > THRESHOLD:
                if not sentence or sentence[-1] != ACTIONS[idx]:
                    sentence.append(ACTIONS[idx])
                if len(sentence) > 5:
                    sentence = sentence[-5:]

        cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('ASL Transformer Detection', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()