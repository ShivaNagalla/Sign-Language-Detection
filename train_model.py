import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
# Folder settings
DATA_PATH       = './MP_Data'
ACTIONS         = ['please', 'thankyou', 'sorry']
NO_SEQUENCES    = 30
SEQUENCE_LENGTH = 30

POSE_LEN  = 33 * 4
FACE_LEN  = 468 * 3
LH_LEN    = 21 * 3
RH_LEN    = 21 * 3
LH_START  = POSE_LEN + FACE_LEN
RH_START  = LH_START + LH_LEN

# --- FEATURE EXTRACTION ---
def extract_hand_features(full_kp: np.ndarray) -> np.ndarray:
    """
    Slice out the two hands, translate so each wrist is origin,
    then scale so max distance across both hands is 1.
    Returns 126-dim vector.
    """
    lh = full_kp[LH_START:LH_START + LH_LEN].reshape(21, 3)
    rh = full_kp[RH_START:RH_START + RH_LEN].reshape(21, 3)
    lh -= lh[0]  # wrist → origin
    rh -= rh[0]
    feat = np.concatenate([lh.flatten(), rh.flatten()])
    max_d = np.max(np.linalg.norm(feat.reshape(-1, 3), axis=1))
    return feat / (max_d + 1e-6)

# --- LOAD DATA ---
sequences, labels = [], []
label_map = {a: i for i, a in enumerate(ACTIONS)}

for action in ACTIONS:
    for seq in range(NO_SEQUENCES):
        base = os.path.join(DATA_PATH, action, str(seq))
        window = []
        for f in range(SEQUENCE_LENGTH):
            arr = np.load(os.path.join(base, f"{f}.npy"))
            window.append(extract_hand_features(arr))
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)  # (N, 30, 126)
y = tf.keras.utils.to_categorical(labels, num_classes=len(ACTIONS))

# --- SPLIT ---
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.2, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5,
    stratify=np.argmax(y_tmp, axis=1), random_state=42
)

# --- TRANSFORMER MODEL ---
d_model    = 128
num_heads  = 4
d_ff       = 512
num_layers = 4
drop_rate  = 0.1

# Positional Encoding Layer
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

# Single Transformer Encoder Block with explicit Add layers
def transformer_encoder(x):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn = Dropout(drop_rate)(attn)
    add1 = Add()([x, attn])  
    out1 = LayerNormalization(epsilon=1e-6)(add1)
    ff = Dense(d_ff, activation='relu')(out1)
    ff = Dense(d_model)(ff)
    ff = Dropout(drop_rate)(ff)
    add2 = Add()([out1, ff])  
    return LayerNormalization(epsilon=1e-6)(add2)

# Build the model
inputs = Input(shape=(SEQUENCE_LENGTH, X.shape[2])) 
x = Dense(d_model)(inputs)
x = PositionalEncoding(SEQUENCE_LENGTH, d_model)(x)

for _ in range(num_layers):
    x = transformer_encoder(x)

x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(Dense(64, activation='relu')(x))
outputs = Dense(len(ACTIONS), activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- TRAIN ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)

print("Test accuracy", model.evaluate(X_test, y_test))
model.save('asl_transformer_model.h5')