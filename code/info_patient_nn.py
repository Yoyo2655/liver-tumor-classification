import os
import pandas as pd
import warnings
import joblib

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
models_path = os.path.join(root_dir, "models")

if not os.path.exists(os.path.join(models_path, "neural_network_model.joblib")):
    print("❌ Modèle non trouvé. Lance d'abord : python code/train_model_nn.py")
    exit()

# Chargement du modèle et des index
clf_nn = joblib.load(os.path.join(models_path, "neural_network_model.joblib"))
scaler = joblib.load(os.path.join(models_path, "scaler.joblib"))
le = joblib.load(os.path.join(models_path, "label_encoder.joblib"))
df_clean, train_idx_list, test_idx_list, features_ia, total_lignes_excel = joblib.load(os.path.join(models_path, "dataset_split.joblib"))

set_train = set(train_idx_list)
set_test = set(test_idx_list)

nom_complet_cancer = {
    'CHC': 'Carcinome hépatocellulaire (CHC)',
    'CCK': 'Cholangiocarcinome (CCK)',
    'Mixtes': 'Tumeur Mixte'
}

# =====================================================================
# AJOUT : GÉNÉRATION AUTOMATIQUE DE SUGGESTIONS POUR L'UTILISATEUR
# =====================================================================
def afficher_suggestions_lignes():
    print("\n" + "="*60)
    print(" 📋 GUIDE DE SÉLECTION DES LIGNES POUR TA DÉMO")
    print("="*60)
    
    # Trouver des exemples de Test (Inconnus) pour chaque type de cancer
    print("🟢 LIGNES DE TEST (Patients 100% inconnus par l'IA) :")
    df_test_seulement = df_clean.loc[list(set_test)]
    for cancer in ['CHC', 'CCK', 'Mixtes']:
        lignes_cancer = df_test_seulement[df_test_seulement['classe_name'] == cancer].index.tolist()
        if lignes_cancer:
            # On prend les 3 premiers exemples trouvés
            exemples = ", ".join(str(idx) for idx in lignes_cancer[:3])
            print(f"  - Pour tester du {cancer:<6} -> Lignes conseillées : [{exemples}]")
            
    # Trouver des exemples de Train (Déjà vus)
    print("\n🟡 LIGNES DE TRAIN (Patients déjà appris par l'IA - Confiance proche de 100%) :")
    df_train_seulement = df_clean.loc[list(set_train)]
    for cancer in ['CHC', 'CCK', 'Mixtes']:
        lignes_cancer = df_train_seulement[df_train_seulement['classe_name'] == cancer].index.tolist()
        if lignes_cancer:
            exemples = ", ".join(str(idx) for idx in lignes_cancer[:2])
            print(f"  - Pour revoir du {cancer:<6} -> Lignes conseillées : [{exemples}]")
            
    print("="*60 + "\n")


def analyser_slice_unique(index_ligne):
    if not (0 <= index_ligne < total_lignes_excel):
        print(f"❌ L'index {index_ligne} n'existe pas. Le fichier contient {total_lignes_excel} lignes (0 à {total_lignes_excel-1}).")
        return

    if index_ligne not in df_clean.index:
        print(f"⚠️ La ligne {index_ligne} correspond à une phase 'TARD' (exclue du modèle pour limiter le bruit).")
        print("👉 Regarde la liste ci-dessus et choisis une ligne valide en phase ART ou VEIN.")
        return

    slice_data = df_clean.loc[index_ligne]
    
    if index_ligne in set_test:
        statut_patient = "🔴 PATIENT 100% INCONNU (Vrai test à l'aveugle)"
    else:
        statut_patient = "🟡 PATIENT DÉJÀ VU À L'ENTRAÎNEMENT (Risque de surconfiance)"

    features_patient = slice_data[features_ia].fillna(0).to_frame().T
    features_patient_scaled = scaler.transform(features_patient)
    
    probabilites = clf_nn.predict_proba(features_patient_scaled)[0]
    resultats_probas = {le.classes_[i]: probabilites[i] for i in range(len(le.classes_))}
    classes_triees = sorted(resultats_probas.items(), key=lambda item: item[1], reverse=True)
    
    code_principal, proba_principale = classes_triees[0]
    code_secondaire, proba_secondaire = classes_triees[1] if len(classes_triees) > 1 else ("Aucun", 0.0)

    print("\n" + "="*80)
    print(f" 🧠 COMPTE-RENDU PRÉDICTIF SÉCURISÉ - LIGNE EXCEL DIRECTE {index_ligne}")
    print("="*80)
    print(f"▶ SÉCURITÉ DATA SCIENCE: {statut_patient}")
    print(f"▶ INFOS COUPE         : Patient n°{slice_data['patient_num']} | Slice ID : N°{slice_data['slice_num']} | Phase : {slice_data['temps_inj']}")
    print(f"▶ VÉRITÉ HISTOLOGIQUE  : {nom_complet_cancer.get(slice_data['classe_name'], slice_data['classe_name'])}")
    print("-"*80)
    print(f"▶ DIAGNOSTIC DE L'IA   : {nom_complet_cancer.get(code_principal, code_principal)} (Confiance : {proba_principale * 100:.1f}%)")
    print(f"▶ RISQUE SECONDAIRE    : {nom_complet_cancer.get(code_secondaire, code_secondaire)} (Probabilité : {proba_secondaire * 100:.1f}%)")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("👋 Application connectée aux index réels du fichier Excel.")
    
    # On affiche automatiquement le guide des lignes avant de demander la saisie
    afficher_suggestions_lignes()
    
    try:
        choix_index = input("Entrez le numéro de la LIGNE à analyser (parmi les suggestions ci-dessus) : ").strip()
        analyser_slice_unique(int(choix_index))
    except ValueError:
        print("❌ Saisie invalide.")