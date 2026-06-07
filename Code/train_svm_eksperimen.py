import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("=== EKSPERIMEN SVM: 3 KELAS VS 2 KELAS ===\n")

df = pd.read_csv('data_pubg_siap_svm_lexicon.csv')
df['Teks_Stemming'] = df['Teks_Stemming'].fillna('')

with open('tfidf_model.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


print(">> MULAI TRAINING VERSI 1 (Positif, Netral, Negatif)")
X_3 = vectorizer.transform(df['Teks_Stemming'])
y_3 = df['Sentimen']


X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=42)

model_3 = SVC(kernel='linear')
model_3.fit(X_train_3, y_train_3)
pred_3 = model_3.predict(X_test_3)

print(f"AKURASI 3 KELAS: {accuracy_score(y_test_3, pred_3) * 100:.2f}%")
print(classification_report(y_test_3, pred_3))
print("-" * 50)


print("\n>> MULAI TRAINING VERSI 2 (Murni Positif vs Negatif)")

df_2_kelas = df[df['Sentimen'] != 'Netral']

X_2 = vectorizer.transform(df_2_kelas['Teks_Stemming'])
y_2 = df_2_kelas['Sentimen']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

model_2 = SVC(kernel='linear')
model_2.fit(X_train_2, y_train_2)
pred_2 = model_2.predict(X_test_2)

print(f"AKURASI 2 KELAS: {accuracy_score(y_test_2, pred_2) * 100:.2f}%")
print(classification_report(y_test_2, pred_2))

with open('model_svm_lexicon_2kelas.pkl', 'wb') as file:
    pickle.dump(model_2, file)

print("\nBeres bro! Eksperimen kelar dan model terbaik udah disimpen.")