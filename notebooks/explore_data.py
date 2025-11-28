# notebooks/explore_data.py
import pandas as pd
import numpy as np

train_path = '/Users/aligunes/MALENOMIA/cancer-classification/data/model_features_effb0_train.txt'
test_path = '/Users/aligunes/MALENOMIA/cancer-classification/data/model_features_effb0_test.txt'

# Train data
df_train = pd.read_csv(train_path, header=None, sep=r'\s+')

print("=== TRAIN DATA ===")
print(f"Shape: {df_train.shape}")
print(f"Satır: {df_train.shape[0]}, Sütun: {df_train.shape[1]}")

# İlk sütun (label)
print("\n=== İLK SÜTUN (LABEL) ===")
print(f"İlk 20 değer:\n{df_train.iloc[:20, 0]}")
print(f"\nUnique değerler: {df_train.iloc[:, 0].unique()}")
print(f"\nLabel dağılımı:\n{df_train.iloc[:, 0].value_counts()}")

# 0 ve 1'e çevir
df_train[0] = df_train[0].astype(int)
print(f"\nInteger'a çevrilmiş label dağılımı:\n{df_train.iloc[:, 0].value_counts()}")

# Test data
df_test = pd.read_csv(test_path, header=None, sep=r'\s+')
print("\n=== TEST DATA ===")
print(f"Shape: {df_test.shape}")
df_test[0] = df_test[0].astype(int)
print(f"Label dağılımı:\n{df_test.iloc[:, 0].value_counts()}")

# Features (1. sütundan sonrası)
print("\n=== FEATURES ===")
print(f"Feature sayısı: {df_train.shape[1] - 1}")
print(f"\nİlk 3 satır, ilk 10 feature:")
print(df_train.iloc[:3, 1:11])