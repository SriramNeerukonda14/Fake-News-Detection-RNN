import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# Load Model and Tokenizer
# ==========================================

model = load_model("model/fake_news_rnn.h5")

with open("model/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# ==========================================
# Constants
# ==========================================

MAX_LEN = 300

# ==========================================
# Prediction Function
# ==========================================

def predict_news(news_text):
    sequence = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)

    prediction = model.predict(padded, verbose=0)[0][0]

    if prediction >= 0.5:
        return "🟢 Real News", prediction
    else:
        return "🔴 Fake News", prediction

# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection using RNN (Many-to-One)")
st.write("Enter a news article or headline below to determine whether it is **Fake** or **Real**.")

news = st.text_area(
    "News Text",
    height=250,
    placeholder="Paste the news article or headline here..."
)

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        result, confidence = predict_news(news)

        st.subheader("Prediction")

        if "Real" in result:
            st.success(result)
        else:
            st.error(result)

        st.write(f"**Confidence Score:** {confidence:.4f}")