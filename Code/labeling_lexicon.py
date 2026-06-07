import pandas as pd

print("Mulai proses Labeling pakai Kamus Lexicon InSet... Gas bro!")

df = pd.read_csv('data_pubg_tahap3.csv')
df['Teks_Stemming'] = df['Teks_Stemming'].fillna('')

kamus_positif = pd.read_csv('positive.tsv', sep='\t', header=None, names=['kata', 'bobot'])
kamus_negatif = pd.read_csv('negative.tsv', sep='\t', header=None, names=['kata', 'bobot'])

dict_lexicon = {}

for index, row in kamus_positif.iterrows():
    try:
        
        dict_lexicon[row['kata']] = int(row['bobot'])
    except:
        pass 

for index, row in kamus_negatif.iterrows():
    try:
        dict_lexicon[row['kata']] = int(row['bobot'])
    except:
        pass

def hitung_skor_lexicon(teks):
    skor_total = 0
    kata_kata = str(teks).split() 
    
    for kata in kata_kata:
        if kata in dict_lexicon:
            skor_total += dict_lexicon[kata] 
            
    return skor_total

def tentukan_sentimen(skor):
    if skor > 0:
        return 'Positif'
    elif skor < 0:
        return 'Negatif'
    else:
        return 'Netral' 

print("Mesin lagi ngitung skor kata per kata, harap sabar...")
df['Skor_Lexicon'] = df['Teks_Stemming'].apply(hitung_skor_lexicon)
df['Sentimen'] = df['Skor_Lexicon'].apply(tentukan_sentimen)

print("\n=== HASIL LABELING LEXICON ===")
print(df['Sentimen'].value_counts())

nama_file = 'data_pubg_siap_svm_lexicon.csv'
df.to_csv(nama_file, index=False)

print(f"\nBeres bro! Labeling ala skripsi kating udah kelar disimpen di {nama_file}")