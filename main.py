import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import re
import random

# Judul Aplikasi
st.title("Sistem Rekomendasi Produk")

# Load Data
df = pd.read_csv("data_produk.csv")
#st.dataframe(df)  # Tampilkan data awal

# Filter data dengan deskripsi yang tidak null
df = df[df['description'].notnull()]

# Preprocessing
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
sastrawi = StopWordRemoverFactory()
stopworda = sastrawi.get_stop_words()
factory=StemmerFactory()
Stemmer = factory.create_stemmer()

def clean_text(text):
  text= text.lower()
  text = clean_spcl.sub(' ', text)
  text = clean_symbol.sub(' ', text)
  text = ' '.join(word for word in text.split() if word not in stopworda)
  return text

# Bersihkan deskripsi
df['desc_clean'] = df['description'].apply(clean_text)
df.set_index('item_name', inplace=True)

# TF-IDF dan Cosine Similarity
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1)
tfidf_matrix = tf.fit_transform(df['desc_clean'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index)

# Fungsi Rekomendasi
def recommendations(keyword, top=10):
    recommended_produk = []

    # Filter produk yang mengandung keyword di nama produk
    filtered_indices = indices[indices.str.contains(keyword, case=False, na=False)]

    if filtered_indices.empty:
        return ["Tidak ada produk yang cocok dengan kata kunci."]

    # Ambil indeks semua produk yang mengandung keyword
    filtered_indexes = filtered_indices.index

    # Buat sub-matriks cos_sim hanya untuk produk yang relevan
    if len(filtered_indexes) == 1:  # Jika hanya ada satu produk
        relevant_cos_sim = [1.0]  # Skor 1 untuk dirinya sendiri
        score_series = pd.Series(relevant_cos_sim, index=filtered_indexes)
    else:
        relevant_cos_sim = cos_sim[filtered_indexes, :][:, filtered_indexes]
        # Hitung kesamaan rata-rata untuk produk yang cocok
        score_series = pd.Series(relevant_cos_sim.mean(axis=0), index=filtered_indexes)

    # Urutkan dan ambil rekomendasi teratas
    score_series = score_series.sort_values(ascending=False)
    top_indexes = list(score_series.iloc[:top].index)

    for i in top_indexes:
        recommended_produk.append(
            f"{df.index[i]} - Skor: {score_series[i]:.2f}"
        )

    return recommended_produk

# Antarmuka Streamlit untuk mencari rekomendasi
keyword = st.text_input("Masukkan Kata Kunci Produk :")
top_n = st.slider("Jumlah Rekomendasi :", 1, 20, 5)

if st.button("Cari Rekomendasi"):
    hasil = recommendations(keyword, top_n)
    st.write("Rekomendasi Produk :")
    
    # Menampilkan hasil rekomendasi per baris
    for rekomendasi in hasil:
        st.write(rekomendasi)  # Setiap rekomendasi muncul di baris baru
