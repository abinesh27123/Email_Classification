import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the model
model = tf.keras.models.load_model("email_classifier.h5", compile=False)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Preprocessing function (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Streamlit UI configuration
st.set_page_config(page_title="Email Classification", page_icon="📧")
st.title("📧 Email Classification App")
st.markdown("Detect whether an email is **Spam** or **Ham** using an LSTM model.")

# Text input area with customized height and placeholder
email_text = st.text_area("Enter Email Text:", height=320)

# Button to classify
if st.button("🔍 Classify Email"):
    if email_text.strip():
        # Preprocess and prepare input
        processed_text = preprocess_text(email_text)
        seq = tokenizer.texts_to_sequences([processed_text])
        padded_seq = pad_sequences(seq, maxlen=100)

        # Prediction
        prediction = model.predict(padded_seq)
        label = "Spam" if np.argmax(prediction) == 1 else "Ham"

        # Display result
        st.subheader(f"📢 Prediction: {label}")
    else:
        st.warning("⚠️ Please enter some email text first.")
