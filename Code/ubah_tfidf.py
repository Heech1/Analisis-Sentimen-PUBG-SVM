import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

print("Mulai ngubah kata jadi matriks angka (TF-IDF)...")

df = pd.read_csv('data_pubg_tahap3.csv')

df['Teks_Stemming'] = df['Teks_Stemming'].fillna('')

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['Teks_Stemming'])


print("\n=== HASIL TRANSFORMASI TF-IDF ===")
print(f"Ukuran Matriks Lu Sekarang : {X.shape}")
print(f"Total Baris (Data Ulasan)  : {X.shape[0]}")
print(f"Total Kolom (Dimensi/Kata) : {X.shape[1]}")


with open('tfidf_model.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("\nBeres bro! Otak TF-IDF udah disimpen dengan nama 'tfidf_model.pkl'.")