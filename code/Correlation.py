import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Chargement des fichiers
chemin_radiomics = r"C:\Users\comme\Downloads\Cancer_foie\global_excel_resampled.xlsx"
chemin_patients = r"C:\Users\comme\Downloads\Cancer_foie\Descriptif_patients.xlsx"

print("1. Chargement des données...")
df_radiomics = pd.read_excel(chemin_radiomics)
df_patients = pd.read_excel(chemin_patients)

# 2. Identification de la colonne de liaison et fusion
col_link_radiomics = None
for col in df_radiomics.columns:
    if col.lower() in ['patient_num', 'patient', 'id', 'num_patient', 'patient_id', 'patient_name']:
        col_link_radiomics = col
        break

if col_link_radiomics:
    print(f"\n[Liaison] Fusion des fichiers via '{col_link_radiomics}' et 'patient_num'...")
    df_radiomics_renamed = df_radiomics.rename(columns={col_link_radiomics: 'patient_num'})
    df_combined = pd.merge(df_radiomics_renamed, df_patients[['patient_num', 'classe_name']], on='patient_num', how='inner')
    print(f"-> Tableau fusionné obtenu : {df_combined.shape[0]} lignes de mesures alignées.")
else:
    print("\n[Attention] Aucune colonne de liaison trouvée. Alignement par défaut...")
    df_combined = df_radiomics.copy()
    df_combined['classe_name'] = pd.Series(np.resize(df_patients['classe_name'].values, len(df_radiomics)))

# 3. Récupération robuste de la colonne cible (gestion des suffixes de fusion _x / _y)
colonne_label_reelle = None
for col in df_combined.columns:
    if 'classe_name' in col:
        colonne_label_reelle = col
        break

if colonne_label_reelle:
    print(f"-> Colonne cible identifiée après fusion : '{colonne_label_reelle}'")
    labels = df_combined[colonne_label_reelle]
else:
    raise KeyError("Impossible de trouver la colonne contenant 'classe_name' après la fusion.")

# 4. Séparation et nettoyage de la matrice de caractéristiques (X)
X_brut = df_combined.select_dtypes(include=[np.number])

# On retire les colonnes de compte ou d'identifiants de la matrice d'analyse X
colonnes_id = [col for col in X_brut.columns if 'id' in col.lower() or 'patient' in col.lower() or 'num' in col.lower()]
if colonnes_id:
    X_brut = X_brut.drop(columns=colonnes_id)

X_brut = X_brut.dropna(how='all', axis=1)  # Supprime les colonnes 100% vides
X_brut = X_brut.fillna(X_brut.median())     # Bouche les NaN par la médiane
colonnes_constantes = X_brut.columns[X_brut.nunique() <= 1]
if len(colonnes_constantes) > 0:
    X_brut = X_brut.drop(columns=colonnes_constantes)

# Élimination des redondances (Corrélation > 0.85)
corr_matrix = X_brut.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
X_nettoye = X_brut.drop(columns=to_drop)

print(f"-> Variables finales retenues pour l'ACP : {X_nettoye.shape[1]}")

# 5. Calcul de l'ACP
print("\n3. Calcul de l'ACP...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_nettoye)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

variance_expliquee = pca.explained_variance_ratio_ * 100
print(f"Variance expliquée par l'Axe 1 : {variance_expliquee[0]:.2f}%")
print(f"Variance expliquée par l'Axe 2 : {variance_expliquee[1]:.2f}%")
print(f"Variance totale résumée en 2D : {sum(variance_expliquee):.2f}%")

# 6. Affichage du graphique final
print("\n4. Affichage du graphique...")
# 5. Calcul de l'ACP
print("\n3. Calcul de l'ACP...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_nettoye)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

variance_expliquee = pca.explained_variance_ratio_ * 100
print(f"Variance expliquée par l'Axe 1 : {variance_expliquee[0]:.2f}%")
print(f"Variance expliquée par l'Axe 2 : {variance_expliquee[1]:.2f}%")
print(f"Variance totale résumée en 2D : {sum(variance_expliquee):.2f}%")

# 6. Affichage du graphique final (Corrigé !)
print("\n4. Affichage du graphique...")
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set1', s=65, alpha=0.75)

plt.title("Analyse Biostatistique : ACP multiphasique des tumeurs hépatiques", fontsize=13, pad=15)
plt.xlabel(f"Composante Principale 1 ({variance_expliquee[0]:.1f}%)")
plt.ylabel(f"Composante Principale 2 ({variance_expliquee[1]:.1f}%)") # <- Corrigé ici !
plt.legend(title="Diagnostic (Histologie)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()