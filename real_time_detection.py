import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# --- CONFIG ---
ACTIONS         = ['please', 'thankyou', 'sorry']
SEQUENCE_LENGTH = 30
THRESHOLD       = 0.8

POSE_LEN  = 33 * 4
FACE_LEN  = 468 * 3
LH_LEN    = 21 * 3
RH_LEN    = 21 * 3
LH_START  = POSE_LEN + FACE_LEN
RH_START  = LH_START + LH_LEN

# Updated PositionalEncoding to accept **kwargs
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        pos = np.arange(seq_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.cast(angle_rads[None, ...], tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# Load the trained Transformer model with custom objects
model = tf.keras.models.load_model(
    'asl_transformer_model.h5',
    custom_objects={'PositionalEncoding': PositionalEncoding}
)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def extract_hand_features(full_kp: np.ndarray) -> np.ndarray:
    lh = full_kp[LH_START:LH_START + LH_LEN].reshape(21, 3)
    rh = full_kp[RH_START:RH_START + RH_LEN].reshape(21, 3)
    lh -= lh[0]  # wrist → origin
    rh -= rh[0]
    feat = np.concatenate([lh.flatten(), rh.flatten()])
    max_d = np.max(np.linalg.norm(feat.reshape(-1, 3), axis=1))
    return feat / (max_d + 1e-6)

# Buffers for real-time detection
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

        # Process frame with MediaPipe
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res = holistic.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw landmarks (optional)
        mp_drawing.draw_landmarks(
            img, res.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            img, res.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            img, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )

        # Build full keypoint vector
        full = []
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark:
                full += [lm.x, lm.y, lm.z, lm.visibility]
        else:
            full += [0] * POSE_LEN

        if res.face_landmarks:
            for lm in res.face_landmarks.landmark:
                full += [lm.x, lm.y, lm.z]
        else:
            full += [0] * FACE_LEN

        if res.left_hand_landmarks:
            for lm in res.left_hand_landmarks.landmark:
                full += [lm.x, lm.y, lm.z]
        else:
            full += [0] * LH_LEN

        if res.right_hand_landmarks:
            for lm in res.right_hand_landmarks.landmark:
                full += [lm.x, lm.y, lm.z]
        else:
            full += [0] * RH_LEN

        arr = np.array(full)
        feat = extract_hand_features(arr)
        seq_buffer.append(feat)

        # Once we have 30 frames, make a prediction
        if len(seq_buffer) == SEQUENCE_LENGTH:
            inp = np.expand_dims(np.array(seq_buffer), axis=0)  # (1, 30, 126)
            proba = model.predict(inp, verbose=0)[0]
            prob_buffer.append(proba)

            avg = np.mean(prob_buffer, axis=0)
            idx = np.argmax(avg)
            if avg[idx] > THRESHOLD:
                if not sentence or sentence[-1] != ACTIONS[idx]:
                    sentence.append(ACTIONS[idx])
                if len(sentence) > 5:
                    sentence = sentence[-5:]

        # Overlay the result on the frame
        cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(
            img, ' '.join(sentence), (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.imshow('ASL Transformer Detection', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()