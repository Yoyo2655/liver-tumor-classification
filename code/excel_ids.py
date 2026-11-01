import os
import pandas as pd

# Définition des chemins vers tes fichiers Excel
# (Ajuste les noms exacts de tes fichiers si nécessaire)
EXCEL_DIAG = "dataset/Descriptif_patients.xlsx"
EXCEL_MULTI = "dataset/multislice_excel_basic.xlsx"

def corriger_fichiers_excel():
    print("=" * 70)
    print("🔄 CORRECTION DES IDENTIFIANTS PATIENTS UNIQUES")
    print("=" * 70)
    
    if not os.path.exists(EXCEL_DIAG) or not os.path.exists(EXCEL_MULTI):
        print("❌ Erreur : Vérifie les chemins de tes fichiers Excel.")
        return

    # 1. Chargement des données
    print("⏳ Lecture des fichiers Excel...")
    df_diag = pd.read_excel(EXCEL_DIAG)
    df_multi = pd.read_excel(EXCEL_MULTI)

    # On s'assure que les colonnes nécessaires existent et n'ont pas d'espaces
    for df in [df_diag, df_multi]:
        df.columns = df.columns.str.strip()

    # 2. Construction d'une cartographie (Mapping) unique
    # On crée une clé unique combinant l'ancien numéro et le nom de la classe
    print("🧠 Génération des nouveaux identifiants uniques globaux...")
    
    # On extrait toutes les paires (ancien_num, classe) uniques présentes dans le dataset principal
    # On utilise df_multi ou df_diag selon celui qui est le plus complet
    combinaisons_uniques = df_multi[['patient_num', 'classe_name']].drop_duplicates().sort_values(by=['classe_name', 'patient_num'])
    
    mapping_patients = {}
    nouveau_id = 1
    
    for _, row in combinaisons_uniques.iterrows():
        ancien_id = row['patient_num']
        classe = row['classe_name']
        
        # Le patient (1, 'CHC') devient le patient 1, le (1, 'CCK') devient le patient 2, etc.
        mapping_patients[(ancien_id, classe)] = nouveau_id
        nouveau_id += 1

    print(f"✨ {len(mapping_patients)} patients réels et uniques ont été identifiés au total.")

    # 3. Application de la correction sur les deux DataFrames
    def appliquer_nouveau_id(df):
        # Fonction interne pour remplacer l'ID ligne par ligne via le mapping
        nouveaux_nums = []
        for _, row in df.iterrows():
            cle = (row['patient_num'], row['classe_name'])
            nouveaux_nums.append(mapping_patients.get(cle, row['patient_num']))
        df['patient_num'] = nouveaux_nums
        return df

    print("📝 Mise à jour des lignes dans le tableau Diagnostic...")
    df_diag_corrige = appliquer_nouveau_id(df_diag)
    
    print("📝 Mise à jour des lignes dans le tableau Multislice...")
    df_multi_corrige = appliquer_nouveau_id(df_multi)

    # 4. Sauvegarde des fichiers corrigés
    # Par sécurité, on peut créer des versions "_clean.xlsx" ou écraser les anciens.
    # Ici on crée des fichiers propres pour éviter de détruire tes originaux par erreur.
    path_diag_out = EXCEL_DIAG.replace(".xlsx", "_clean.xlsx")
    path_multi_out = EXCEL_MULTI.replace(".xlsx", "_clean.xlsx")

    print("\n💾 Sauvegarde des nouveaux fichiers Excel...")
    df_diag_corrige.to_excel(path_diag_out, index=False)
    df_multi_corrige.to_excel(path_multi_out, index=False)

    print("=" * 70)
    print("✅ CORRECTION TERMINÉE AVEC SUCCÈS !")
    print(f"👉 Nouveau fichier Diag  : {path_diag_out}")
    print(f"👉 Nouveau fichier Multi : {path_multi_out}")
    print("=" * 70)
    print("💡 Pense à modifier ton script d'entraînement pour lire ces versions '_clean.xlsx'.")

if __name__ == "__main__":
    corriger_fichiers_excel()