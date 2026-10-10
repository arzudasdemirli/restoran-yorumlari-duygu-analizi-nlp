import streamlit as st
import numpy as np
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# ------------------------------------------------
# SAYFA AYARI
# ------------------------------------------------
st.set_page_config(
    page_title="Yapay Sinir Ağları - Duygu Analizi",
    layout="centered"
)

st.title("🧠 Yapay Sinir Ağları Dersi Projesi")
st.subheader("Restoran Yorumları Üzerinden Duygu Analizi")

st.markdown("---")

# ------------------------------------------------
# MODELLERİ YÜKLE
# ------------------------------------------------
@st.cache_resource
def load_models():
    lstm_model = load_model("lstm_model.h5")

    tokenizer_bert = BertTokenizer.from_pretrained("bert_sentiment")
    bert_model = BertForSequenceClassification.from_pretrained("bert_sentiment")

    return lstm_model, tokenizer_bert, bert_model

lstm_model, tokenizer_bert, bert_model = load_models()

# ------------------------------------------------
# LSTM TOKENIZER (COLAB'TA KAYDETTİĞİN)
# ------------------------------------------------
@st.cache_resource
def load_lstm_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

# Eğer tokenizer.pkl yoksa LSTM seçimini kapatabiliriz
try:
    tokenizer_lstm = load_lstm_tokenizer()
    lstm_ready = True
except:
    lstm_ready = False

# ------------------------------------------------
# MODEL SEÇİMİ
# ------------------------------------------------
model_list = ["İyileştirilmiş BERT"]
if lstm_ready:
    model_list.insert(0, "İyileştirilmiş LSTM")

model_choice = st.selectbox("🔍 Kullanılacak Model", model_list)

text = st.text_area("✍️ Restoran Yorumunu Giriniz", height=120)

# ------------------------------------------------
# TAHMİN
# ------------------------------------------------
if st.button("Duygu Analizi Yap"):
    if text.strip() == "":
        st.warning("Lütfen bir yorum giriniz.")
    else:
        if model_choice == "İyileştirilmiş BERT":
            inputs = tokenizer_bert(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=160
            )
            with torch.no_grad():
                outputs = bert_model(**inputs)

            pred = torch.argmax(outputs.logits, dim=1).item()

        else:
            seq = tokenizer_lstm.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=200)
            pred = np.argmax(lstm_model.predict(padded), axis=1)[0]

        label_map = {
            0: "Negatif 😡",
            1: "Nötr 😐",
            2: "Pozitif 😊"
        }

        st.success(f"**Tahmin Edilen Duygu:** {label_map[pred]}")

st.markdown("---")
st.caption("İyileştirilmiş LSTM ve BERT modelleri kullanılarak geliştirilmiştir.")
