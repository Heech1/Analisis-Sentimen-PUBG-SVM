from google_play_scraper import Sort, reviews
import pandas as pd

print("Mulai narik data ulang... Tunggu bentar!")


result, _ = reviews('com.tencent.ig', lang='id', country='id', sort=Sort.NEWEST, count=2000)

data_ulasan = []
for review in result:
    data_ulasan.append({
        'Tanggal': review['at'],
        'Teks_Ulasan': review['content'],
        'Rating': review['score']
    })

df = pd.DataFrame(data_ulasan)


df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df = df[df['Tanggal'] >= '2026-04-28']


df.to_csv('data_sentimen_pubg_baru.csv', index=False)
print(f"Beres! Sisa {len(df)} ulasan murni versi 4.4.0 tanpa perlu dihapus manual!")