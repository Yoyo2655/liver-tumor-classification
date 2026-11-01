import os
import pandas as pd
import warnings
import joblib
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 1. CHEMINS
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_dir = os.path.abspath(os.path.join(root_dir, ".."))

path_multi = os.path.join(root_dir, "dataset", "multislice_excel_basic.xlsx")
models_dir = os.path.join(root_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# =========================================================
# 2. CHARGEMENT DONNÉES
# =========================================================
print("⏳ Chargement du dataset multi-slice...")

df_multi = pd.read_excel(path_multi)

df_filtered = df_multi[
    df_multi['temps_inj'].isin(['ART', 'VEIN', 'TARD', 'PORT'])
].copy()

df_clean = df_filtered.iloc[:, 15:].copy()
df_clean = df_clean[df_clean['classe_name'] != 'Mixtes'].copy()

# =========================================================
# 3. FEATURES / LABELS
# =========================================================
exclusions_finales = ['classe_name', 'temps_inj', 'patient_num', 'slice_num']

features_ia = [
    col for col in df_clean.columns
    if col not in exclusions_finales
]

X = df_clean[features_ia].fillna(0)

le = LabelEncoder()

df_clean = df_clean.copy()
y = le.fit_transform(df_clean['classe_name'])

groups = df_clean['patient_num']
X = df_clean[features_ia].fillna(0)

# =========================================================
# 4. SPLIT PATIENT + STRATIFIÉ
# =========================================================
print("📊 Split stratifié par patient...")

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    sgkf.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# =========================================================
# 5. NORMALISATION (IMPORTANT : train only)
# =========================================================
print("📏 Normalisation...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 6. MODÈLE MLP
# =========================================================
print("🧠 Entraînement du MLP...")

clf_nn = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=700,
    random_state=42
)

clf_nn.fit(X_train_scaled, y_train)

# évaluation du modèle
print("\n📊 ÉVALUATION DU MODÈLE")
y_pred = clf_nn.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy globale : {acc * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\n📉 Matrice de confusion :")
print(cm)

print("\n📄 Classification report :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =========================================================
# 7. SAUVEGARDE
# =========================================================
print("💾 Sauvegarde...")

joblib.dump(clf_nn, os.path.join(models_dir, "neural_network_model.joblib"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))

joblib.dump(
    (
        df_clean,
        df_clean.index[train_idx].tolist(),
        df_clean.index[test_idx].tolist(),
        features_ia,
        len(df_multi)
    ),
    os.path.join(models_dir, "dataset_split.joblib")
)

print("✅ Entraînement terminé avec succès !")
