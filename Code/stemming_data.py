import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

print("Mulai proses Stemming...")
print("Peringatan: Ini bakal makan waktu agak lama (2-5 menit). Ditinggal ngopi aja bro, jangan di-close!")

df = pd.read_csv('data_pubg_tahap2.csv')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming_teks(teks):
    if pd.isna(teks): 
        return ""
   
    return stemmer.stem(str(teks))


print("Sedang mencukur ribuan kata... Harap sabar...")
df['Teks_Stemming'] = df['Teks_Stopword'].apply(stemming_teks)


nama_file = 'data_pubg_tahap3.csv'
df.to_csv(nama_file, index=False)

print(f"AKHIRNYA BERES JUGA BRO! Data siap tempur disimpen di {nama_file}")