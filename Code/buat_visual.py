import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle

print("Mula melukis graf dan Word Cloud... Tunggu sekejap bro!")

df = pd.read_csv('data_pubg_siap_svm_lexicon.csv')
df['Teks_Stemming'] = df['Teks_Stemming'].fillna('')


plt.figure(figsize=(8, 6))
sns.countplot(x='Sentimen', data=df, palette='viridis', order=['Negatif', 'Netral', 'Positif'])
plt.title('Taburan Sentimen Ulasan PUBG (Lexicon InSet)')
plt.xlabel('Kategori Sentimen')
plt.ylabel('Jumlah Ulasan')
plt.savefig('grafik_bar_sentimen.png')
plt.close()
print("- Grafik Bar Sentimen dah siap disimpan (grafik_bar_sentimen.png)")


teks_positif = ' '.join(df[df['Sentimen'] == 'Positif']['Teks_Stemming'])
teks_negatif = ' '.join(df[df['Sentimen'] == 'Negatif']['Teks_Stemming'])


wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(teks_positif)
plt.figure(figsize=(10, 5))
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Ulasan Positif')
plt.savefig('wordcloud_positif.png')
plt.close()


wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(teks_negatif)
plt.figure(figsize=(10, 5))
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Ulasan Negatif')
plt.savefig('wordcloud_negatif.png')
plt.close()
print("- Word Cloud Positif & Negatif dah siap disimpan")



df_2_kelas = df[df['Sentimen'] != 'Netral']

with open('tfidf_model.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('model_svm_lexicon_2kelas.pkl', 'rb') as file:
    model_svm = pickle.load(file)

X = vectorizer.transform(df_2_kelas['Teks_Stemming'])
y = df_2_kelas['Sentimen']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
prediksi = model_svm.predict(X_test)


cm = confusion_matrix(y_test, prediksi, labels=['Negatif', 'Positif'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
plt.title('Confusion Matrix - SVM 2 Kelas (Akurasi 91.67%)')
plt.xlabel('Tebakan AI (Predicted)')
plt.ylabel('Kunci Jawapan Asli (Actual)')
plt.savefig('confusion_matrix_2kelas.png')
plt.close()
print("- Confusion Matrix dah siap disimpan (confusion_matrix_2kelas.png)")

print("\nSelesai bro! Semua 4 fail gambar dah ada dalam folder projek kau.")