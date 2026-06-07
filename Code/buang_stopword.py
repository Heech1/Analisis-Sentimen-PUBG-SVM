import pandas as pd
import nltk
from nltk.corpus import stopwords

print("Mulai proses buang kata-kata ga penting versi FINAL... Tunggu bentar bro!")


df = pd.read_csv('data_pubg_tahap1.csv')

nltk.download('stopwords')
daftar_stopword = stopwords.words('indonesian')

kata_tambahan = [
    'game', 'pubg', 'nya', 'yg', 'aja', 'udah', 'gak', 'ga', 'ya', 'di', 'ke', 'buat', 
    'ini', 'itu', 'dan', 'dari', 'gua', 'gw', 'gue', 'lu', 'lo', 'sih', 'banget', 'bgt', 
    'dong', 'pas', 'kan', 'kok', 'deh', 'pun', 'lah', 'mah', 'kalo', 'kalau', 'biar', 
    'ada', 'sama', 'kayak', 'terus', 'trus', 'lagi', 'yang', 'buat', 'dalam', 'untuk',
    'aja', 'saja', 'juga', 'aku', 'kamu', 'kita', 'kami', 'mereka', 'dia', 'bikin', 'jadi'
]
daftar_stopword.extend(kata_tambahan)

def hapus_stopword(teks):
    if pd.isna(teks): 
        return ""
    kata_kata = str(teks).split()
    kata_bersih = [kata for kata in kata_kata if kata not in daftar_stopword]
    return ' '.join(kata_bersih)

df['Teks_Stopword'] = df['Teks_Bersih'].apply(hapus_stopword)


nama_file = 'data_pubg_tahap2.csv'
df.to_csv(nama_file, index=False)

print(f"Beres bro! Kata gaul dan kata hubung udah dibasmi tuntas!")