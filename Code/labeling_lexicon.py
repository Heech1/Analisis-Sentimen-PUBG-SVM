import pandas as pd

print("Mulai proses Labeling pakai Kamus Lexicon InSet... Gas bro!")

# 1. Baca data ulasan lu yang udah di-stemming kemaren (tahap 3)
df = pd.read_csv('data_pubg_tahap3.csv')
df['Teks_Stemming'] = df['Teks_Stemming'].fillna('')

# 2. Buka kamus InSet dari dosen lu
kamus_positif = pd.read_csv('positive.tsv', sep='\t', header=None, names=['kata', 'bobot'])
kamus_negatif = pd.read_csv('negative.tsv', sep='\t', header=None, names=['kata', 'bobot'])

dict_lexicon = {}

# KITA PERBAIKI DI SINI: Paksa kolom 'bobot' jadi tipe data Integer (Angka)
for index, row in kamus_positif.iterrows():
    try:
        # int() fungsinya buat ngubah teks "-4" jadi angka mutlak -4
        dict_lexicon[row['kata']] = int(row['bobot'])
    except:
        pass # Kalo ada baris kosong di TSV, cuekin aja

for index, row in kamus_negatif.iterrows():
    try:
        dict_lexicon[row['kata']] = int(row['bobot'])
    except:
        pass

# 3. Bikin pos satpam buat ngitung skor per kalimat
def hitung_skor_lexicon(teks):
    skor_total = 0
    kata_kata = str(teks).split() 
    
    for kata in kata_kata:
        if kata in dict_lexicon:
            skor_total += dict_lexicon[kata] # Sekarang udah sama-sama angka, pasti bisa ditambahin!
            
    return skor_total

# 4. Bikin aturan cap sentimen berdasarkan total skornya
def tentukan_sentimen(skor):
    if skor > 0:
        return 'Positif'
    elif skor < 0:
        return 'Negatif'
    else:
        return 'Netral' 

# 5. Eksekusi perhitungannya
print("Mesin lagi ngitung skor kata per kata, harap sabar...")
df['Skor_Lexicon'] = df['Teks_Stemming'].apply(hitung_skor_lexicon)
df['Sentimen'] = df['Skor_Lexicon'].apply(tentukan_sentimen)

# 6. Pamerin hasilnya ke terminal
print("\n=== HASIL LABELING LEXICON ===")
print(df['Sentimen'].value_counts())

# 7. Simpan file final baru
nama_file = 'data_pubg_siap_svm_lexicon.csv'
df.to_csv(nama_file, index=False)

print(f"\nBeres bro! Labeling ala skripsi kating udah kelar disimpen di {nama_file}")