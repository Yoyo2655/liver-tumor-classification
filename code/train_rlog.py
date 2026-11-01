import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

base_dir = os.path.dirname(os.path.abspath(__file__))

print("Script :", base_dir)
print("Dataset :", os.path.join(base_dir, "..", "dataset"))

print("--- 1. Chargement et filtrage des données ---")
df_clinique = pd.read_excel(
    os.path.join(base_dir, "..", "dataset", "Descriptif_patients.xlsx")
)

df_radiomique = pd.read_excel(
    os.path.join(base_dir, "..", "dataset", "multislice_excel_basic.xlsx")
)

# Réalignement direct : attribution propre du patient_num d'origine
# pour lier chaque ligne radiomique à sa ligne clinique correspondante
if 'patient_num' not in df_radiomique.columns:
    df_radiomique['patient_num'] = df_clinique['patient_num']

# Si 'classe_name' (le diagnostic) est absent de la radiomique, on le prend de la clinique
if 'classe_name' not in df_radiomique.columns:
    df_all = pd.merge(
        df_radiomique,
        df_clinique[['patient_num', 'classe_name', 'Age_at_disease', 'Gender']],
        on='patient_num',
        how='left'
    )
else:
    # Si 'classe_name' est déjà présent, on fusionne juste le reste
    df_all = pd.merge(
        df_radiomique,
        df_clinique[['patient_num', 'Age_at_disease', 'Gender']],
        on='patient_num',
        how='left'
    )

# CONSIGNES : Exclure les tumeurs 'Mixtes' pour ne garder que CHC et CCK
df_filtered = df_all[df_all['classe_name'].isin(['CHC', 'CCK'])].copy()

# S'assurer que le diagnostic reste une chaîne de caractères (str) propre
df_filtered['classe_name'] = df_filtered['classe_name'].astype(str)

# Nettoyage des colonnes de diagnostics PyRadiomics
id_column = df_filtered.columns[0]
cols_diagnostics = [col for col in df_filtered.columns if col.startswith('diagnostics_') or col == id_column]
df_filtered = df_filtered.drop(columns=cols_diagnostics)

# Conversion des colonnes texte en colonnes numériques (One-Hot)
df_filtered = pd.get_dummies(df_filtered, columns=['Gender'], drop_first=True)
if 'temps_inj' in df_filtered.columns:
    df_filtered = pd.get_dummies(df_filtered, columns=['temps_inj'], drop_first=True)

print("--- 2. Séparation stricte par Patient (80/20) ---")
patients_uniques = df_filtered['patient_num'].unique()
labels_patients = df_filtered.drop_duplicates(subset=['patient_num'])['classe_name']

# Séparation sécurisée des patients uniques
patients_train, patients_test = train_test_split(
    patients_uniques, test_size=0.20, random_state=42, stratify=labels_patients
)

# Création des DataFrames d'apprentissage et de test
df_train_slices = df_filtered[df_filtered['patient_num'].isin(patients_train)].copy()
df_test_slices = df_filtered[df_filtered['patient_num'].isin(patients_test)].copy()

# Séparation X et y pour l'entraînement
X_train = df_train_slices.drop(columns=['patient_num', 'classe_name'])
y_train = df_train_slices['classe_name']

# Pour le fichier de test disponible au terminal : on force l'écriture textuelle 
# de 'patient_num' et 'classe_name' tout à gauche, puis les autres colonnes de features
features_cols = list(X_train.columns)
X_test = df_test_slices[['patient_num', 'classe_name'] + features_cols]

# Sauvegarde physique dans le dossier dédié
os.makedirs("data_split", exist_ok=True)
X_train.to_csv("data_split/X_train.csv", index=False)
y_train.to_csv("data_split/y_train.csv", index=False)
X_test.to_csv("data_split/dataset_test_disponible.csv", index=False)

# =====================================================================
# --- 3. Normalisation et Entraînement de l'IA (AVEC PONDÉRATION) ---
# =====================================================================
print("--- 3. Normalisation et Entraînement de l'IA ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Pondération manuelle : On donne un poids de 1 au CHC, et on teste un poids 
# modéré de 2.0 pour le CCK (au lieu du calcul automatique trop agressif)
poids_personnalises = {'CHC': 1.0, 'CCK': 2.0}

model = LogisticRegression(
    penalty='elasticnet',
    l1_ratio=1.0,
    solver='saga',
    C=1.0,                    # Augmenté à 1.0 pour des prédictions plus tranchées
    max_iter=5000,
    tol=1e-3,
    class_weight=poids_personnalises,  # <--- Utilisation de notre dictionnaire
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Sauvegarde des objets entraînés
with open("model_lasso.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n[SUCCÈS] Entraînement terminé.")
print(f" -> Patients d'apprentissage : {len(patients_train)} ({len(df_train_slices)} slices)")
print(f" -> Patients de test sauvegardés : {len(patients_test)} ({len(df_test_slices)} slices)")
print(" -> Fichiers enregistrés sous './data_split/'")