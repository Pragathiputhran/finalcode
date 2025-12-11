import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, TimeDistributed
from sklearn.model_selection import train_test_split
import pickle

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("chatbot_dataset_en_ar.csv")   # Must contain: Question, Answer

questions = df["question"].astype(str).values
answers = df["answer"].astype(str).values

# ===============================
# BASIC CLEANING (English + Arabic)
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9ء-ي؟،!.\s]", " ", text)   # English + Arabic allowed
    return text

questions = [clean_text(q) for q in questions]
answers = [clean_text(a) for a in answers]

# ===============================
# TOKENIZATION
# ===============================
VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
MAX_LEN = 40

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(questions + answers)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

X = tokenizer.texts_to_sequences(questions)
y = tokenizer.texts_to_sequences(answers)

X = pad_sequences(X, maxlen=MAX_LEN, padding="post")
y = pad_sequences(y, maxlen=MAX_LEN, padding="post")

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# ===============================
# BUILD BiLSTM MODEL
# ===============================
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(TimeDistributed(Dense(256, activation='relu')))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(VOCAB_SIZE, activation='softmax')))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# y should be integer sequences of shape [batch, seq_len] for
# `sparse_categorical_crossentropy` when the model predicts
# a distribution at each time step (shape [batch, seq_len, vocab]).
# Do not expand dims here.

# ===============================
# TRAINING
# ===============================
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# ===============================
# SAVE MODEL
# ===============================
model.save("bilstm_chatbot_model.h5")

print("Training complete. Model saved as bilstm_chatbot_model.h5")
print("Tokenizer saved as tokenizer.pkl")
