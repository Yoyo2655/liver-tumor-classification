import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Chemins et chargement des données
chemin_radiomics = r"C:\Users\comme\Downloads\Cancer_foie\global_excel_resampled.xlsx"
chemin_patients = r"C:\Users\comme\Downloads\Cancer_foie\Descriptif_patients.xlsx"

print("Chargement des fichiers...")
df_radiomics = pd.read_excel(chemin_radiomics)
df_patients = pd.read_excel(chemin_patients)

# 2. Exploration rapide dans le terminal
print("\n--- Exploration préliminaire ---")
print(f"Nombre de tumeurs analysées : {df_radiomics.shape[0]}")
print(f"Nombre total de colonnes initiales : {df_radiomics.shape[1]}")

# 3. Préparation des données pour la Heatmap
# On ne garde que les colonnes numériques (les paramètres radiomiques)
numeric_features = df_radiomics.select_dtypes(include=[np.number])

# Si vos colonnes contiennent un identifiant patient (ex: 'ID' ou 'patient'), on le retire
colonnes_a_exclure = [col for col in numeric_features.columns if 'id' in col.lower() or 'patient' in col.lower()]
if colonnes_a_exclure:
    print(f"Exclusion des colonnes d'identification : {colonnes_a_exclure}")
    numeric_features = numeric_features.drop(columns=colonnes_a_exclure)

print(f"Nombre de caractéristiques radiomiques à corréler : {numeric_features.shape[1]}")

# 4. Calcul de la matrice de corrélation (Pearson)
print("\nCalcul de la matrice de corrélation...")
corr_matrix = numeric_features.corr()

# 5. Affichage de la Heatmap
print("Génération du graphique...")
plt.figure(figsize=(14, 12))

# On utilise la palette 'coolwarm' (Bleu = corrélation négative, Rouge = positive)
sns.heatmap(corr_matrix, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            xticklabels=False,  # Masqués pour éviter d'avoir 200 textes illisibles
            yticklabels=False,  # Masqués pour la lisibilité globale
            cbar_kws={'label': 'Coefficient de corrélation (r)'})

plt.title("Analyse Biostatistique : Heatmap de corrélation des paramètres radiomiques", fontsize=14, pad=15)
plt.tight_layout()

# Affichage de la fenêtre graphique
plt.show()