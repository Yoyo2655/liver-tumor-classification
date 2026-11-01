import os
import pandas as pd
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 1. CHARGEMENT
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
path_multi = os.path.join(root_dir, "dataset", "multislice_excel_basic.xlsx")

df_multi = pd.read_excel(path_multi)
df_clean = df_multi.iloc[:, 15:].copy()

exclusions_finales = ['classe_name', 'temps_inj', 'patient_num', 'slice_num']
features_ia = [col for col in df_clean.columns if col not in exclusions_finales]

X = df_clean[features_ia].fillna(0)
le = LabelEncoder()
y = le.fit_transform(df_clean['classe_name'])

# =====================================================================
# 2. SÉPARATION STRICTE PAR PATIENT (ANTI-TRICHE)
# =====================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df_clean['patient_num']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Listes d'index pour savoir où se trouve la ligne demandée
set_train = set(df_clean.index[train_idx])
set_test = set(df_clean.index[test_idx])

# =====================================================================
# 3. ENTRAÎNEMENT
# =====================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf_nn = MLPClassifier(
    hidden_layer_sizes=(64, 32), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    random_state=42
)
print("⚙️ Entraînement du réseau de neurones (Split par Patient connecté)...")
clf_nn.fit(X_train_scaled, y_train)

nom_complet_cancer = {
    'CHC': 'Carcinome hépatocellulaire (CHC)',
    'CCK': 'Cholangiocarcinome (CCK)',
    'Mixtes': 'Tumeur Mixte'
}

# =====================================================================
# 4. FONCTION DE PRÉDICTION SUR LA LIGNE DE TON CHOIX
# =====================================================================
def analyser_slice_specifique(index_ligne):
    if not (0 <= index_ligne < len(df_clean)):
        print(f"❌ Index {index_ligne} invalide (Total de lignes : {len(df_clean)}).")
        return

    # Détection du statut de la ligne pour le jury
    if index_ligne in set_test:
        statut_patient = "🔴 PATIENT 100% INCONNU (Vrai test à l'aveugle)"
    else:
        statut_patient = "🟡 PATIENT DÉJÀ VU (Présent dans le Train Set - Risque de surconfiance)"

    slice_data = df_clean.loc[index_ligne]
    features_patient = slice_data[features_ia].fillna(0).to_frame().T
    
    # Inférence (on utilise le scaler du Train)
    features_patient_scaled = scaler.transform(features_patient)
    probabilites = clf_nn.predict_proba(features_patient_scaled)[0]
    
    resultats_probas = {le.classes_[i]: probabilites[i] for i in range(len(le.classes_))}
    classes_triees = sorted(resultats_probas.items(), key=lambda item: item[1], reverse=True)
    
    code_principal, proba_principale = classes_triees[0]
    code_secondaire, proba_secondaire = classes_triees[1] if len(classes_triees) > 1 else ("Aucun", 0.0)

    print("\n" + "="*80)
    print(f" 🧠 COMPTE-RENDU IA - COUPE SPÉCIFIQUE (Ligne Excel : {index_ligne})")
    print("="*80)
    print(f"▶ STATUT DU TEST       : {statut_patient}")
    print(f"▶ INFOS COUPE         : Patient n°{slice_data['patient_num']} | Slice ID : N°{slice_data['slice_num']} | Phase : {slice_data['temps_inj']}")
    print(f"▶ VÉRITÉ HISTOLOGIQUE  : {nom_complet_cancer.get(slice_data['classe_name'], slice_data['classe_name'])}")
    print("-"*80)
    print(f"▶ DIAGNOSTIC DE L'IA   : {nom_complet_cancer.get(code_principal, code_principal)} (Confiance : {proba_principale * 100:.1f}%)")
    print(f"▶ RISQUE SECONDAIRE    : {nom_complet_cancer.get(code_secondaire, code_secondaire)} (Probabilité : {proba_secondaire * 100:.1f}%)")
    print("="*80 + "\n")

if __name__ == "__main__":
    print(f"👋 Système prêt. Base de {len(df_clean)} lignes.")
    print("-" * 60)
    try:
        choix_index = input(f"Entrez le numéro de la ligne à inspecter (0 à {len(df_clean)-1}) : ").strip()
        analyser_slice_specifique(int(choix_index))
    except ValueError:
        print("❌ Saisie invalide : Veuillez entrer un nombre entier.")