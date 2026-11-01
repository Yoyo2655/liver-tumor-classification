"""
diagnostic_patient.py
---------------------
Donne un diagnostic complet pour UN patient en agrégeant toutes ses slices.

Logique :
  - Prédiction sur chaque slice via le réseau de neurones
  - Moyenne des probabilités PAR PHASE (ART, PORT, TARD, VEIN)
    → chaque phase a une voix égale, peu importe le nombre de slices
  - Score final = moyenne des scores de phase → distribution en %
  - Si CHC et CCK sont proches (<20% d'écart) → flag "Mixte possible"
  - Les 4 phases sont toujours affichées (absente = signalée)

Utilisation :
    python code/diagnostic_patient.py

Pré-requis : avoir lancé train_model_nn.py au moins une fois.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 1. CHARGEMENT DU MODÈLE ET DES DONNÉES
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.abspath(os.path.join(current_dir, ".."))
models_path = os.path.join(root_dir, "models")

for fname in ["neural_network_model.joblib", "scaler.joblib",
              "label_encoder.joblib", "dataset_split.joblib", "df_mixtes.joblib"]:
    if not os.path.exists(os.path.join(models_path, fname)):
        print(f"❌ Fichier manquant : {fname}")
        print("   Lance d'abord : python code/train_model_nn.py")
        exit()

clf_nn  = joblib.load(os.path.join(models_path, "neural_network_model.joblib"))
scaler  = joblib.load(os.path.join(models_path, "scaler.joblib"))
le      = joblib.load(os.path.join(models_path, "label_encoder.joblib"))
df_clean, train_idx_list, test_idx_list, features_ia, total_lignes_excel = \
    joblib.load(os.path.join(models_path, "dataset_split.joblib"))

set_train = set(train_idx_list)
set_test  = set(test_idx_list)

# Chargement des Mixtes depuis joblib (pré-extrait à l'entraînement → rapide)
df_mixtes       = joblib.load(os.path.join(models_path, "df_mixtes.joblib"))
patients_mixtes = sorted(df_mixtes['patient_num'].unique().tolist())

# Les 4 phases standard dans l'ordre médical logique
PHASES_STANDARD = ['ART', 'PORT', 'VEIN', 'TARD']

NOM_CANCER = {
    'CHC':    'Carcinome hépatocellulaire (CHC)',
    'CCK':    'Cholangiocarcinome (CCK)',
    'Mixtes': 'Tumeur Mixte (Mixtes)',
}

# ---------------------------------------------------------------
# SEUILS DE DÉCISION
# P(CCK) > SEUIL_HAUT_CCK  → CCK confirmé
# P(CCK) < SEUIL_BAS_CCK   → CHC confirmé
# Entre les deux            → Non concluant
#
# Ces seuils sont ajustables selon la tolérance clinique :
# - Abaisser SEUIL_HAUT_CCK → plus sensible au CCK (moins de faux négatifs CCK)
# - Monter  SEUIL_BAS_CCK   → plus exigeant pour dire CHC
# ---------------------------------------------------------------
SEUIL_HAUT_CCK = 0.60   # au-dessus → CCK confirmé
SEUIL_BAS_CCK  = 0.40   # en-dessous → CHC confirmé
# Entre 0.40 et 0.60 → Non concluant

# Seuil par phase : si une phase est entre ces bornes, elle est "non conclusive"
SEUIL_PHASE_HAUT = 0.65
SEUIL_PHASE_BAS  = 0.35

SEUIL_MIXTE = 0.20   # si écart entre les 2 premières classes < 20%

# =====================================================================
# 2. FEATURES DÉTERMINANTES
# =====================================================================
def scores_features_explicatifs(slice_values: pd.Series, classe_gagnante: str) -> pd.Series:
    """Delta normalisé entre la slice et la moyenne des autres classes."""
    autres_classes  = [c for c in le.classes_ if c != classe_gagnante]
    masque_autres   = df_clean['classe_name'].isin(autres_classes)
    moyennes_autres = df_clean.loc[masque_autres, features_ia].mean()
    stds_autres     = df_clean.loc[masque_autres, features_ia].std().replace(0, 1)
    delta = (slice_values[features_ia] - moyennes_autres).abs() / stds_autres
    return delta.sort_values(ascending=False)


# =====================================================================
# 3. DIAGNOSTIC COMPLET D'UN PATIENT
# =====================================================================
def diagnostiquer_patient(patient_num: int, est_mixte: bool = False):

    if est_mixte:
        slices_patient = df_mixtes[df_mixtes['patient_num'] == patient_num]
        statut = "🟣 PATIENT MIXTE — Jamais vu à l'entraînement (modèle entraîné CHC/CCK uniquement)"
    else:
        slices_patient = df_clean[df_clean['patient_num'] == patient_num]

    if slices_patient.empty:
        print(f"\n❌ Patient n°{patient_num} introuvable.")
        return

    if not est_mixte:
        idx_patient = set(slices_patient.index)
        est_test_   = bool(idx_patient & set_test)
        est_train_  = bool(idx_patient & set_train)
        if est_test_ and not est_train_:
            statut = "🔴 PATIENT 100% INCONNU — Test à l'aveugle"
        elif est_train_ and not est_test_:
            statut = "🟡 PATIENT D'ENTRAÎNEMENT — Risque de surconfiance"
        else:
            statut = "🟠 PATIENT MIXTE (slices réparties entre train et test)"

    # ----------------------------------------------------------------
    # AGRÉGATION PAR PHASE (même logique que l'entraînement)
    # On moyenne les features par phase → un vecteur par phase
    # puis on prédit sur ce vecteur agrégé
    # ----------------------------------------------------------------
    phases_presentes = slices_patient['temps_inj'].values
    resultats_par_phase = {}

    for phase in PHASES_STANDARD:
        masque = (phases_presentes == phase)
        if masque.sum() == 0:
            resultats_par_phase[phase] = None   # phase absente
            continue
        # Moyenne des features de la phase → 1 vecteur agrégé
        X_phase  = slices_patient.loc[slices_patient['temps_inj'] == phase, features_ia].fillna(0)
        X_mean   = X_phase.mean(axis=0).values.reshape(1, -1)
        X_scaled = scaler.transform(X_mean)
        probas_phase = clf_nn.predict_proba(X_scaled)[0]
        idx_cck = list(le.classes_).index('CCK')
        p_cck   = probas_phase[idx_cck]

        # Seuil par phase
        if p_cck >= SEUIL_PHASE_HAUT:
            idx_cls   = idx_cck
            conclusif = True
        elif p_cck <= SEUIL_PHASE_BAS:
            idx_cls   = 1 - idx_cck   # CHC
            conclusif = True
        else:
            idx_cls   = probas_phase.argmax()
            conclusif = False

        resultats_par_phase[phase] = {
            'probas':     probas_phase,
            'classe_idx': idx_cls,
            'classe':     le.classes_[idx_cls],
            'confiance':  probas_phase[idx_cls],
            'n_slices':   int(masque.sum()),
            'conclusif':  conclusif,
            'p_cck':      p_cck,
        }

    # Score global = moyenne des probas sur toutes les phases présentes
    phases_actives = [r for r in resultats_par_phase.values() if r is not None]
    if not phases_actives:
        print("❌ Aucune phase utilisable pour ce patient.")
        return

    scores_globaux = np.mean([r['probas'] for r in phases_actives], axis=0)
    scores_globaux = scores_globaux / scores_globaux.sum()

    idx_cck  = list(le.classes_).index('CCK')
    p_cck_global = scores_globaux[idx_cck]

    # ---------------------------------------------------------------
    # DÉCISION FINALE AVEC SEUILS
    # ---------------------------------------------------------------
    if p_cck_global >= SEUIL_HAUT_CCK:
        verdict      = 'CCK'
        conclusif    = True
    elif p_cck_global <= SEUIL_BAS_CCK:
        verdict      = 'CHC'
        conclusif    = True
    else:
        verdict      = 'NON_CONCLUANT'
        conclusif    = False

    ordre    = np.argsort(scores_globaux)[::-1]
    classe_1 = le.classes_[ordre[0]]
    score_1  = scores_globaux[ordre[0]]
    classe_2 = le.classes_[ordre[1]] if len(ordre) > 1 else None
    score_2  = scores_globaux[ordre[1]] if len(ordre) > 1 else 0.0
    ecart    = score_1 - score_2
    mixte_flag = (ecart < SEUIL_MIXTE)

    n_phases_presentes = len(phases_actives)
    phases_conclusives = sum(1 for r in phases_actives if r['conclusif'])
    coherence_phases   = phases_conclusives / n_phases_presentes
    score_fiabilite    = 0.50 * abs(p_cck_global - 0.5) * 2 + 0.50 * coherence_phases

    # Phase la plus confiante pour la classe principale → slice médiane comme référence
    phase_ref = max(
        [p for p, r in resultats_par_phase.items() if r is not None and r['classe'] == verdict] or
        [p for p, r in resultats_par_phase.items() if r is not None],
        key=lambda p: resultats_par_phase[p]['confiance'],
        default=None
    )
    if phase_ref is None:
        phase_ref = next(p for p, r in resultats_par_phase.items() if r is not None)

    slices_phase_ref = slices_patient[slices_patient['temps_inj'] == phase_ref]
    slice_ref = slices_phase_ref.iloc[len(slices_phase_ref) // 2]
    delta_features = scores_features_explicatifs(slice_ref, verdict if verdict != 'NON_CONCLUANT' else classe_1)
    top_features   = delta_features.head(5)

    # =====================================================================
    # 4. AFFICHAGE
    # =====================================================================
    sep = "=" * 80
    vérité = 'Mixtes' if est_mixte else slices_patient['classe_name'].iloc[0]

    print("\n" + sep)
    print(f"  🏥  RAPPORT DIAGNOSTIQUE — PATIENT N°{patient_num}")
    print(sep)
    print(f"  Statut               : {statut}")
    print(f"  Nombre de slices     : {len(slices_patient)}")
    print(f"  Vérité histologique  : {NOM_CANCER.get(vérité, vérité)}")
    print()

    # --- Résultat par phase ---
    print("─" * 80)
    print("  📊  ANALYSE PAR PHASE D'INJECTION")
    print(f"      (seuils : CCK confirmé > {SEUIL_PHASE_HAUT*100:.0f}% | CHC confirmé < {SEUIL_PHASE_BAS*100:.0f}% | entre les deux = non concluant)")
    print("─" * 80)
    for phase in PHASES_STANDARD:
        res = resultats_par_phase[phase]
        if res is None:
            print(f"  {phase:<6}  ░░ Phase absente pour ce patient")
        else:
            detail = "  ".join(
                f"{le.classes_[i]} {res['probas'][i]*100:.0f}%"
                for i in range(len(le.classes_))
            )
            if res['conclusif']:
                statut_phase = f"→ {res['classe']} ✓"
            else:
                statut_phase = f"→ ⚠️  Non concluant ({res['classe']} à {res['confiance']*100:.0f}%)"
            print(f"  {phase:<6}  ({res['n_slices']:2d} slices)  [{detail}]  {statut_phase}")

    # --- Score global ---
    print()
    print("─" * 80)
    print("  🧮  SCORE GLOBAL (moyenne des phases présentes)")
    print(f"      (seuils : > {SEUIL_HAUT_CCK*100:.0f}% CCK → CCK | < {SEUIL_BAS_CCK*100:.0f}% CCK → CHC | entre les deux → Non concluant)")
    print("─" * 80)
    for i in ordre:
        cls   = le.classes_[i]
        score = scores_globaux[i]
        barre = "█" * int(score * 40)
        print(f"  {cls:<8}  {score*100:5.1f}%  {barre}")

    print()
    if verdict == 'NON_CONCLUANT':
        print(f"  ❓  DIAGNOSTIC NON CONCLUANT")
        print(f"      P(CCK) = {p_cck_global*100:.1f}% — zone d'incertitude ({SEUIL_BAS_CCK*100:.0f}%-{SEUIL_HAUT_CCK*100:.0f}%)")
        print(f"      → Bilan complémentaire nécessaire (biopsie, avis multidisciplinaire)")
        print(f"      → Tendance : {NOM_CANCER.get(classe_1, classe_1)} ({score_1*100:.1f}%)")
    elif mixte_flag:
        print(f"  ⚠️  DIAGNOSTIC PROBABLE : {NOM_CANCER.get(verdict, verdict)}")
        print(f"      Écart faible ({ecart*100:.1f}%) — tumeur potentiellement atypique ou mixte")
    else:
        print(f"  ✅  DIAGNOSTIC FINAL   : {NOM_CANCER.get(verdict, verdict)}")

    if est_mixte:
        print()
        print(f"  ⚠️  ATTENTION — INTERPRÉTATION LIMITÉE POUR CE CAS MIXTE")
        print(f"      Le modèle a été entraîné uniquement sur CHC et CCK.")
        print(f"      Il est calibré pour détecter le CCK agressivement (class_weight=balanced)")
        print(f"      → Les scores CCK sont probablement surestimés sur les tumeurs Mixtes.")
        print(f"      → Ce résultat indique la composante dominante, pas un diagnostic absolu.")

    print(f"  🎯  SCORE DE FIABILITÉ : {score_fiabilite*100:.1f}%"
          f"  ({phases_conclusives}/{n_phases_presentes} phases conclusives"
          f" | P(CCK) global {p_cck_global*100:.1f}%)")


    # --- Features déterminantes ---
    print()
    print("─" * 80)
    print("  🧬  CARACTÉRISTIQUES LES PLUS DÉTERMINANTES")
    classe_ref = verdict if verdict != 'NON_CONCLUANT' else classe_1
    print(f"      (depuis la slice la plus confiante pour {classe_ref},"
          " comparée aux autres classes)")
    print("─" * 80)
    for feat, delta in top_features.items():
        valeur    = slice_ref[feat]
        nom_court = feat.replace("original_", "").replace("_", " ")
        print(f"  ▶ {nom_court:<42}  val = {valeur:8.3f}  |  écart = {delta:.2f}σ")

    print()
    print(sep + "\n")


# =====================================================================
# 5. POINT D'ENTRÉE
# =====================================================================
if __name__ == "__main__":

    # Patients 100% dans le test set, listés par classe
    # Patients avec labels ambigus (apparaissent dans plusieurs classes) → exclus
    labels_par_patient = df_clean.groupby('patient_num')['classe_name'].nunique()
    patients_ambigus   = set(labels_par_patient[labels_par_patient > 1].index.tolist())

    patients_test = sorted(
        df_clean.loc[df_clean.index.isin(set_test), 'patient_num'].unique().tolist()
    )
    patients_test = [p for p in patients_test if p not in patients_ambigus]

    print("=" * 60)
    print("  🏥  DIAGNOSTIC MULTI-SLICE PAR PATIENT")
    print("  (patients 100% inconnus du modèle uniquement)")
    print("=" * 60)
    for cls in sorted(df_clean['classe_name'].unique()):
        pts_cls = sorted(
            df_clean.loc[
                df_clean.index.isin(set_test) & (df_clean['classe_name'] == cls),
                'patient_num'
            ].unique().tolist()
        )
        pts_cls = [p for p in pts_cls if p not in patients_ambigus]
        print(f"  {cls:<8} → {pts_cls}")
    print(f"  {'Mixtes':<8} → {patients_mixtes}  (jamais vus à l'entraînement)")
    print()

    try:
        choix = input("  Numéro du patient : ").strip()
        num   = int(choix)
        if num in patients_ambigus:
            print(f"⚠️  Patient n°{num} exclu — labels ambigus dans le dataset")
            print(f"   Ce patient apparaît dans plusieurs classes (CHC, CCK, Mixtes).")
            print(f"   Son diagnostic ne peut pas être évalué de façon fiable.")
        elif num in patients_mixtes:
            diagnostiquer_patient(num, est_mixte=True)
        elif num not in patients_test:
            print(f"❌ Patient n°{num} non disponible en test.")
            print(f"   Choisis parmi : {patients_test}")
        else:
            diagnostiquer_patient(num)
    except ValueError:
        print("❌ Saisie invalide : entrez un nombre entier.")
