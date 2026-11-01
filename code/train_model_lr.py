"""
train_model_lr.py
-----------------
Alternative à train_model_nn.py : Régression Logistique avec class_weight='balanced'.
Meilleur que le MLP sur petits datasets déséquilibrés.

Sauvegarde dans les mêmes fichiers joblib → diagnostic_patient.py fonctionne sans modification.
"""

import os
import pandas as pd
import warnings
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 1. CHEMINS
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.abspath(os.path.join(current_dir, ".."))
path_multi  = os.path.join(root_dir, "dataset", "multislice_excel_basic_clean.xlsx")
models_dir  = os.path.join(root_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# =========================================================
# 2. CHARGEMENT
# =========================================================
print("⏳ Chargement du dataset multi-slice...")
df_multi = pd.read_excel(path_multi)

df_filtered = df_multi[df_multi['temps_inj'].isin(['ART', 'VEIN', 'TARD', 'PORT'])].copy()
df_clean    = df_filtered.iloc[:, 15:].copy()
df_clean    = df_clean[df_clean['classe_name'] != 'Mixtes'].copy()

# =========================================================
# 3. FEATURES / LABELS
# =========================================================
exclusions  = ['classe_name', 'temps_inj', 'patient_num', 'slice_num']
features_ia = [c for c in df_clean.columns if c not in exclusions]

le = LabelEncoder()
df_clean = df_clean.copy()
y        = le.fit_transform(df_clean['classe_name'])
groups   = df_clean['patient_num']
X        = df_clean[features_ia].fillna(0)

# =========================================================
# 4. SPLIT PAR PATIENT STRATIFIÉ
# =========================================================
print("📊 Split stratifié par patient...")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# =========================================================
# 5. NORMALISATION
# =========================================================
print("📏 Normalisation...")
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =========================================================
# 6. MODÈLE : RÉGRESSION LOGISTIQUE ÉQUILIBRÉE
# class_weight='balanced' : pénalise les erreurs CCK
# proportionnellement à leur rareté → compense le déséquilibre
# =========================================================
print("🧠 Entraînement de la Régression Logistique (class_weight=balanced)...")
clf = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    random_state=42,
    solver='lbfgs',
    C=1.0
)
clf.fit(X_train_scaled, y_train)

# =========================================================
# 7. ÉVALUATION
# =========================================================
print("\n📊 ÉVALUATION DU MODÈLE")
y_pred = clf.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy globale : {acc * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\n📉 Matrice de confusion :")
print(f"        Prédit CCK  Prédit CHC")
print(f"Vrai CCK   {cm[0][0]:<10} {cm[0][1]}")
print(f"Vrai CHC   {cm[1][0]:<10} {cm[1][1]}")

print("\n📄 Classification report :")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# =========================================================
# 8. SAUVEGARDE (mêmes noms → diagnostic_patient.py inchangé)
# =========================================================
print("💾 Sauvegarde...")
joblib.dump(clf,    os.path.join(models_dir, "neural_network_model.joblib"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
joblib.dump(le,     os.path.join(models_dir, "label_encoder.joblib"))
joblib.dump(
    (df_clean, df_clean.index[train_idx].tolist(),
     df_clean.index[test_idx].tolist(), features_ia, len(df_multi)),
    os.path.join(models_dir, "dataset_split.joblib")
)

# Sauvegarde des Mixtes en joblib → plus besoin de relire l'Excel au démarrage
df_mixtes = df_multi[df_multi['temps_inj'].isin(['ART', 'PORT', 'VEIN', 'TARD'])].copy()
df_mixtes = df_mixtes.iloc[:, 15:].copy()
df_mixtes = df_mixtes[df_mixtes['classe_name'] == 'Mixtes'].copy()
joblib.dump(df_mixtes, os.path.join(models_dir, "df_mixtes.joblib"))

print("✅ Entraînement terminé !")
