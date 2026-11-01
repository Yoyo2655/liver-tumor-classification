import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# 1. CHARGEMENT ET ENTRAÎNEMENT DU MODÈLE
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

path_global = os.path.join(root_dir, "dataset", "global_excel_resampled.xlsx")
path_desc = os.path.join(root_dir, "dataset", "Descriptif_patients.xlsx")

df_global = pd.read_excel(path_global)
df_desc = pd.read_excel(path_desc)

# Fusion propre des données par Patient et Type
df_total = pd.merge(df_global, df_desc, on=['patient_num', 'classe_name'], how='inner')

# Sélection des descripteurs de texture/forme (Radiomique)
radiomic_features = [
    'original_shape_Elongation',
    'original_shape_Sphericity',
    'original_firstorder_Entropy',         # Chaos/Hétérogénéité des gris
    'original_firstorder_Mean',            # Intensité moyenne du gris
    'original_firstorder_Energy',          # Uniformité de la texture
    'original_glcm_Contrast',              # Contraste local des pixels
]

X = df_total[radiomic_features].fillna(0)

# ENCODAGE SÉCURISÉ DES CLASSES DE CANCER (CHC, CCK, Mixtes)
le = LabelEncoder()
y = le.fit_transform(df_total['classe_name'])

# Entraînement de l'arbre
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# Dictionnaire de correspondance propre pour l'affichage complet
nom_complet_cancer = {
    'CHC': 'Carcinome hépatocellulaire (CHC)',
    'CCK': 'Cholangiocarcinome (CCK)',
    'Mixtes': 'Tumeur Mixte'
}

# =====================================================================
# 2. INTERFACE DE PRÉDICTION INTERACTIVE ET EXPLICATIVE
# =====================================================================
def analyser_slice_index(index_ligne):
    if not (0 <= index_ligne < len(df_total)):
        print(f"❌ Index {index_ligne} invalide (Total de lignes : {len(df_total)}).")
        return

    slice_data = df_total.iloc[index_ligne]
    features_patient = slice_data[radiomic_features].fillna(0).to_frame().T
    
    # 1. Calcul des probabilités par type de cancer
    probabilites = clf.predict_proba(features_patient)[0]
    
    # Association des probabilités aux vrais noms textuels via le LabelEncoder
    resultats_probas = {le.classes_[i]: probabilites[i] for i in range(len(le.classes_))}
    classes_triees = sorted(resultats_probas.items(), key=lambda item: item[1], reverse=True)
    
    code_principal, proba_principale = classes_triees[0]
    code_secondaire, proba_secondaire = classes_triees[1] if len(classes_triees) > 1 else ("Aucun", 0.0)

    type_principal = nom_complet_cancer.get(code_principal, code_principal)
    type_secondaire = nom_complet_cancer.get(code_secondaire, code_secondaire)

    # 2. Identification du facteur principal qui a fait décider l'IA
    importances = clf.feature_importances_
    valeurs_patient = features_patient.iloc[0].values
    
    # Calcul d'un score d'impact (valeur de la feature pondérée par son importance globale)
    scores_impact = {radiomic_features[i]: importances[i] * abs(valeurs_patient[i]) for i in range(len(radiomic_features))}
    facteur_principal = max(scores_impact, key=scores_impact.get)

    traductions_facteurs = {
        'original_firstorder_Entropy': "l'Entropie du premier ordre (votre niveau de gris est extrêmement chaotique et hétérogène)",
        'original_firstorder_Mean': "l'Intensité Moyenne (la clarté globale de gris de la lésion)",
        'original_firstorder_Energy': "l'Énergie du premier ordre (l'uniformité de la masse tumorale)",
        'original_glcm_Contrast': "le Contraste GLCM (les variations brusques d'intensité entre pixels voisins)",
        'original_shape_Sphericity': "la Sphéricité (la régularité et la rondeur des contours géométriques)",
        'original_shape_Elongation': "l'Élongation de la forme (l'étirement géométrique de la tumeur)"
    }
    
    facteur_explique = traductions_facteurs.get(facteur_principal, facteur_principal)

    # --- AFFICHAGE DU COMPTE-RENDU ---
    print("\n" + "="*80)
    print(f" 🩺 COMPTE-RENDU RADIOMIQUE INTERACTIF - COUPE LIGNE {index_ligne}")
    print("="*80)
    print(f"▶ DIAGNOSTIC DE L'IA   : {type_principal} (Confiance : {proba_principale * 100:.1f}%)")
    print(f"▶ RISQUE SECONDAIRE    : {type_secondaire} (Probabilité : {proba_secondaire * 100:.1f}%)")
    print(f"▶ VÉRITÉ HISTOLOGIQUE  : {nom_complet_cancer.get(slice_data['classe_name'], slice_data['classe_name'])} [Patient n°{slice_data['patient_num']}]")
    print("-"*80)
    print(f"▶ FACTEUR PRINCIPAL DE DÉCISION :")
    print(f"  L'IA s'est principalement basée sur {facteur_explique}.")
    print("-"*80)
    print(f"▶ VALEURS BRUTES DES VARIABLES CLÉS DE LA SLICE :")
    print(f"  - Entropie (Hétérogénéité du gris) : {slice_data['original_firstorder_Entropy']:.3f}")
    print(f"  - Intensité moyenne du gris        : {slice_data['original_firstorder_Mean']:.1f}")
    print(f"  - Sphéricité (Rondeur des contours): {slice_data['original_shape_Sphericity']:.3f}")
    print("="*80 + "\n")

# =====================================================================
# 3. INTERACTION TERMINAL
# =====================================================================
if __name__ == "__main__":
    print("👋 Système d'analyse par slice prêt.")
    try:
        choix_index = input(f"Entrez le numéro de la ligne à analyser (0 à {len(df_total)-1}) : ").strip()
        analyser_slice_index(int(choix_index))
    except ValueError:
        print("❌ Veuillez entrer un nombre entier valide.")