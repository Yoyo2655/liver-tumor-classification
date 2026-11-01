import pickle
import pandas as pd
import numpy as np

# 1. Chargement du dataset de test disponible
try:
    df_test = pd.read_csv("data_split/dataset_test_disponible.csv")
except FileNotFoundError:
    print("[ERREUR] Le fichier de test est introuvable. Veuillez d'abord lancer 'python train_rlog.py'")
    exit()

# Trouver la liste unique des patients non utilisés pour l'entraînement
patients_test_dispo = df_test['patient_num'].unique()
patients_test_dispo.sort()

print("\n========================================================")
print("LISTE DES PATIENTS NON UTILISÉS POUR L'ENTRAÎNEMENT :")
print(list(patients_test_dispo))
print("========================================================\n")

# 2. Demande utilisateur sur le terminal
choix = input("Entrez le numéro d'un patient à tester parmi la liste ci-dessus : ")

try:
    patient_choisi = int(choix)
except ValueError:
    print("[ERREUR] Veuillez entrer un numéro entier valide.")
    exit()

if patient_choisi not in patients_test_dispo:
    print(f"[ERREUR] Le patient {patient_choisi} n'est pas dans la liste des données de test externes.")
    exit()

# 3. Isolation des données du patient choisi (toutes ses slices)
df_patient = df_test[df_test['patient_num'] == patient_choisi].copy()
vrai_diagnostic = str(df_patient['classe_name'].iloc[0])

# Préparation des caractéristiques pour le modèle (on enlève les colonnes de métadonnées)
X_patient = df_patient.drop(columns=['patient_num', 'classe_name'])

# 4. Chargement de l'IA et du scaler entraînés
with open("model_lasso.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 5. Normalisation et Prédiction
X_patient_scaled = scaler.transform(X_patient)
probabilites_slices = model.predict_proba(X_patient_scaled)

# Moyenne des probabilités de toutes ses coupes (slices)
probabilites_moyennes = np.mean(probabilites_slices, axis=0)
classes = model.classes_
dict_probs = dict(zip(classes, probabilites_moyennes))

classe_predite = max(dict_probs, key=dict_probs.get)
fiabilite = dict_probs[classe_predite] * 100

# 6. Affichage des résultats sur le terminal
print("\n" + "="*50)
print(f"RÉSULTATS DE L'IA POUR LE PATIENT N°{patient_choisi}")
print("="*50)
print(f"Vrai diagnostic (Base de données) : {vrai_diagnostic}")
print(f"Diagnostic prédit par l'IA        : {classe_predite}")
print(f"Indice de fiabilité (Certitude)   : {fiabilite:.1f}%")
print(f"Détail des votes de l'IA          : CHC ({dict_probs.get('CHC', 0)*100:.1f}%) | CCK ({dict_probs.get('CCK', 0)*100:.1f}%)")
print("-" * 50)

# 7. Explication des critères impactants (Coefficients LASSO)
# En binaire avec liblinear/saga, coef_[0] correspond aux poids de la deuxième classe dans model.classes_
coefficients = model.coef_[0]

# Alignement strict des noms de colonnes avec les poids du modèle
features_importance = pd.DataFrame({'Variable': X_patient.columns, 'Coefficient': coefficients})
features_impactantes = features_importance[features_importance['Coefficient'] != 0].copy()

# Extraction des 5 plus gros coefficients de la décision
top_features = features_impactantes.reindex(features_impactantes['Coefficient'].abs().nlargest(5).index)

# On détermine quelle classe biologique est favorisée par un coefficient positif
classe_positive = model.classes_[1] if len(model.classes_) > 1 else "CHC"
classe_negative = model.classes_[0] if len(model.classes_) > 1 else "CCK"

print("Pourquoi l'IA a pris cette décision ? (Éléments clés retenus) :")
for idx, row in top_features.iterrows():
    coeff = row['Coefficient']
    if coeff > 0:
        role = f"Pousse vers un diagnostic de {classe_positive}"
    else:
        role = f"Pousse vers un diagnostic de {classe_negative}"
    print(f" - Variable '{row['Variable']}' (poids: {coeff:.3f}) -> {role}")
print("="*50 + "\n")