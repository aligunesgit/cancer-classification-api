# notebooks/train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("=== VERİ YÜKLEME ===")
# Veriyi yükle
train_path = '/Users/aligunes/MALENOMIA/cancer-classification/data/model_features_effb0_train.txt'
test_path = '/Users/aligunes/MALENOMIA/cancer-classification/data/model_features_effb0_test.txt'

df_train = pd.read_csv(train_path, header=None, sep=r'\s+')
df_test = pd.read_csv(test_path, header=None, sep=r'\s+')

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

# X ve y'yi ayır
X_train = df_train.iloc[:, 1:].values  # Feature'lar (1-1000 sütunlar)
y_train = df_train.iloc[:, 0].values.astype(int)  # Label (0. sütun)

X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values.astype(int)

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train dağılımı: 0={sum(y_train==0)}, 1={sum(y_train==1)}")

# Model eğit
print("\n=== MODEL EĞİTİMİ ===")
print("Random Forest eğitiliyor...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,  # Tüm CPU'ları kullan
    verbose=1
)

model.fit(X_train, y_train)
print("✅ Model eğitimi tamamlandı!")

# Tahmin yap
print("\n=== DEĞERLENDİRME ===")
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
print("\n=== Detaylı Rapor ===")
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)']))

# Confusion matrix
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# Modeli kaydet
print("\n=== MODEL KAYDETME ===")
model_dir = '/Users/aligunes/MALENOMIA/cancer-classification/app/models'
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, f'{model_dir}/cancer_model.pkl')
print(f"✅ Model kaydedildi: {model_dir}/cancer_model.pkl")

# Feature importance (en önemli 10 feature)
print("\n=== En Önemli 10 Feature ===")
feature_importance = model.feature_importances_
top_features = np.argsort(feature_importance)[-10:][::-1]
for i, idx in enumerate(top_features, 1):
    print(f"{i}. Feature {idx}: {feature_importance[idx]:.6f}")

print("\n🎉 İşlem tamamlandı!")