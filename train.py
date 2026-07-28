import os
import pickle
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ==========================================================
# Create model folder if it doesn't exist
# ==========================================================

os.makedirs("model", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# ==========================================================
# Add Labels
# Fake = 0
# True = 1
# ==========================================================

fake["label"] = 0
true["label"] = 1

# ==========================================================
# Merge Dataset
# ==========================================================

data = pd.concat([fake, true], ignore_index=True)

# Shuffle dataset
data = data.sample(frac=1, random_state=42)
data.reset_index(drop=True, inplace=True)

print("Dataset Shape:", data.shape)

# ==========================================================
# Features and Labels
# ==========================================================

X = data["text"]
y = data["label"]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# Tokenization
# ==========================================================

max_words = 10000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# ==========================================================
# Padding
# ==========================================================

max_len = 300

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# ==========================================================
# Build Many-to-One RNN Model
# ==========================================================

model = Sequential()

model.add(
    Embedding(
        input_dim=max_words,
        output_dim=64,
        input_length=max_len
    )
)

model.add(SimpleRNN(64))

model.add(Dense(1, activation="sigmoid"))

# ==========================================================
# Compile Model
# ==========================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================================
# Model Summary
# ==========================================================

model.summary()

# ==========================================================
# Early Stopping
# ==========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# ==========================================================
# Train Model
# ==========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# ==========================================================
# Evaluate Model
# ==========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("Test Loss:", round(loss, 4))

# ==========================================================
# Save Model
# ==========================================================

model.save("model/fake_news_rnn.h5")

# ==========================================================
# Save Tokenizer
# ==========================================================

with open("model/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

print("\nModel Saved Successfully!")

# ==========================================================
# Plot Accuracy
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# ==========================================================
# Plot Loss
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()