import pandas as pd
import re

print("Mulai ngebersihin data... Tunggu bentar ya bro!")


df = pd.read_csv('data_sentimen_pubg_baru.csv')


df['Teks_Ulasan'] = df['Teks_Ulasan'].astype(str).replace(r'\n|\r', ' ', regex=True)


def bersihin_teks(teks):
    if pd.isna(teks): 
        return ""
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks


df['Teks_Bersih'] = df['Teks_Ulasan'].apply(bersihin_teks)


nama_file_baru = 'data_pubg_tahap1.csv'
df.to_csv(nama_file_baru, index=False)

print(f"Beres bro! Data bersih udah disimpen aman di {nama_file_baru}")