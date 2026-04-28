import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.utils import to_categorical

# Paths
DATA_PATH = "processed"
LABEL_FILE = "labels.txt"

# Load labels
data = []
labels = []

with open(LABEL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        file_name = parts[0]
        sentence = parts[1:]

        npy_path = os.path.join(DATA_PATH, file_name + ".npy")

        if os.path.exists(npy_path):
            x = np.load(npy_path)
            x = x.reshape(x.shape[0], -1)  # IMPORTANT FIX
            data.append(x)
            labels.append(sentence)

# Build vocabulary
all_words = set(word for sentence in labels for word in sentence)
word2idx = {word: i+1 for i, word in enumerate(sorted(all_words))}
idx2word = {i: w for w, i in word2idx.items()}

# Convert labels to numbers
y = []
for sentence in labels:
    y.append([word2idx[word] for word in sentence])

# Pad sequences
X = pad_sequences(data, padding='post', dtype='float32')
y = pad_sequences(y, padding='post')

# One-hot encode y
num_classes = len(word2idx) + 1
y = to_categorical(y, num_classes=num_classes)

# Model
model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
print("X shape:", X.shape)
print("y shape:", y.shape)
print("y sample:", y[0])
print(model.output_shape)
# Train
model.fit(X, y, epochs=10, batch_size=4)

# Save model
model.save("lip_reading_model.h5")

print("Model training complete!")