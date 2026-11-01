import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Mode non-interactif pour la génération de fichiers en tâche de fond
plt.switch_backend('Agg')
warnings.filterwarnings("ignore")

# Configuration esthétique globale
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(current_dir)

try:
    import diagnostic_patient as dp
except ImportError:
    print("❌ Impossible de trouver 'diagnostic_patient.py' dans le même dossier.")
    exit()

# =====================================================================
# 1. COLLECTE DES DONNÉES AVANCÉES
# =====================================================================
def collecter_donnees_avancees():
    records_patients = []
    records_phases_details = []
    
    labels_par_patient = dp.df_clean.groupby('patient_num')['classe_name'].nunique()
    patients_ambigus = set(labels_par_patient[labels_par_patient > 1].index.tolist())
    
    all_patients = []
    for p in dp.df_clean.loc[dp.df_clean.index.isin(dp.set_test), 'patient_num'].unique():
        if p not in patients_ambigus: all_patients.append((p, "TEST", False))
    for p in dp.df_clean.loc[dp.df_clean.index.isin(dp.set_train), 'patient_num'].unique():
        if p not in patients_ambigus: all_patients.append((p, "TRAIN", False))
    for p in dp.patients_mixtes:
        all_patients.append((p, "MIXTE", True))

    print(f"⏳ Extraction des données complexes pour {len(all_patients)} patients...")

    idx_cck = list(dp.le.classes_).index('CCK')
    
    for patient_num, groupe, est_mixte in all_patients:
        slices_patient = dp.df_mixtes[dp.df_mixtes['patient_num'] == patient_num] if est_mixte else dp.df_clean[dp.df_clean['patient_num'] == patient_num]
        if slices_patient.empty: continue
        verite = "Mixtes" if est_mixte else slices_patient['classe_name'].iloc[0]

        # Calcul de la discordance interne des slices
        total_slices = len(slices_patient)
        slices_ia_data = slices_patient[dp.features_ia].fillna(0)
        slices_scaled = dp.scaler.transform(slices_ia_data)
        probas_all_slices = dp.clf_nn.predict_proba(slices_scaled)
        preds_all_slices = dp.le.classes_[probas_all_slices.argmax(axis=1)]
        
        # Taux de slices discordantes (à contre-courant de sa vraie nature)
        if verite in ['CHC', 'CCK']:
            nb_discordantes = sum(1 for pred in preds_all_slices if pred != verite)
            taux_discordance = (nb_discordantes / total_slices) * 100
        else:
            taux_discordance = np.nan # Pas de sens strict de "discordance" sur un mixte pur

        # Calcul par phase pour les courbes ROC
        probas_phases = []
        for phase in dp.PHASES_STANDARD:
            X_phase = slices_patient.loc[slices_patient['temps_inj'] == phase, dp.features_ia].fillna(0)
            if X_phase.empty: continue
            
            X_scaled = dp.scaler.transform(X_phase.mean(axis=0).values.reshape(1, -1))
            proba_p = dp.clf_nn.predict_proba(X_scaled)[0]
            p_cck_p = proba_p[idx_cck]
            probas_phases.append(proba_p)
            
            # Record au niveau de la phase
            records_phases_details.append({
                'patient_num': patient_num,
                'phase': phase,
                'verite': verite,
                'p_cck': p_cck_p
            })

        if not probas_phases: continue
        scores_globaux = np.mean(probas_phases, axis=0)
        p_cck_global = scores_globaux[idx_cck] / scores_globaux.sum()

        records_patients.append({
            'patient_num': patient_num,
            'statut': groupe,
            'verite': verite,
            'p_cck_global': p_cck_global,
            'taux_discordance': taux_discordance
        })
        
    return pd.DataFrame(records_patients), pd.DataFrame(records_phases_details)


# =====================================================================
# 2. LES 4 GRAPHIC GENERATORS
# =====================================================================

def plot_courbes_roc(df_patients, df_phases, output_dir):
    """ Graphique 1 : Multi-ROC Curves (Global vs Phases individuelles) """
    plt.figure(figsize=(8, 7))
    
    # On filtre uniquement sur les classes binaires pures pour le calcul ROC
    df_p_clean = df_patients[df_patients['verite'].isin(['CHC', 'CCK'])].copy()
    y_true_global = (df_p_clean['verite'] == 'CCK').astype(int)
    
    # 1. ROC Globale
    fpr, tpr, _ = roc_curve(y_true_global, df_p_clean['p_cck_global'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Score Global Patient (AUC = {roc_auc:.3f})', color='black', linewidth=3)
    
    # 2. ROC par Phase
    colors = {'ART': '#e377c2', 'PORT': '#1f77b4', 'VEIN': '#2ca02c', 'TARD': '#9467bd'}
    df_ph_clean = df_phases[df_phases['verite'].isin(['CHC', 'CCK'])].copy()
    
    for phase in dp.PHASES_STANDARD:
        sub_ph = df_ph_clean[df_ph_clean['phase'] == phase]
        if sub_ph.empty: continue
        # Aligner avec y_true via le patient_num
        merged = pd.merge(sub_ph, df_p_clean[['patient_num', 'verite']], on='patient_num')
        y_true_ph = (merged['verite_y'] == 'CCK').astype(int)
        
        fpr_p, tpr_p, _ = roc_curve(y_true_ph, merged['p_cck'])
        auc_p = auc(fpr_p, tpr_p)
        plt.plot(fpr_p, tpr_p, label=f'Phase {phase} (AUC = {auc_p:.3f})', color=colors.get(phase), linestyle='--')

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Taux de Faux Positifs (1 - Spécificité)", fontweight='bold')
    plt.ylabel("Taux de Vrais Positifs (Sensibilité)", fontweight='bold')
    plt.title("📈 Courbes ROC Multi-Phases & Diagnostic Global", fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right")
    
    path = os.path.join(output_dir, "advanced_courbes_roc.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 🖼️  Graphique 1/4 sauvegardé : {path}")


def plot_swarm_probabilities(df_patients, output_dir):
    """ Graphique 2 : Swarm Plot / Stripplot des probabilités avec lignes de seuils """
    plt.figure(figsize=(9, 6))
    
    # Ordre médical des classes sur l'axe X
    ordre_x = ['CHC', 'Mixtes', 'CCK']
    
    sns.stripplot(data=df_patients, x='verite', y='p_cck_global', hue='statut', 
                  order=ordre_x, size=7, palette='Dark2', jitter=0.25, alpha=0.85)
    
    # Tracé des seuils décisionnels configurés dans ton projet
    plt.axhline(y=dp.SEUIL_HAUT_CCK, color='#d62728', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.axhline(y=dp.SEUIL_BAS_CCK, color='#1f77b4', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Ajout du texte d'explication des zones
    plt.text(2.45, 0.80, 'Zone CCK', color='#d62728', weight='bold', fontsize=10)
    plt.text(2.45, 0.49, 'Zone Incertitude', color='orange', weight='bold', fontsize=10)
    plt.text(2.45, 0.15, 'Zone CHC', color='#1f77b4', weight='bold', fontsize=10)

    plt.xlabel("Vérité Histologique (Diagnostic Réel)", fontweight='bold')
    plt.ylabel("Probabilité P(CCK) Globale affectée au Patient", fontweight='bold')
    plt.title("📊 Séparabilité des Classes & Distribution des Scores P(CCK)", fontsize=13, fontweight='bold', pad=15)
    plt.ylim(-0.05, 1.05)
    plt.legend(title="Statut", loc='upper left')
    
    path = os.path.join(output_dir, "advanced_separation_probabilities.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 🖼️  Graphique 2/4 sauvegardé : {path}")


def plot_histogramme_discordance(df_patients, output_dir):
    """ Graphique 3 : Taux de discordance interne des slices """
    plt.figure(figsize=(8, 6))
    
    # On ne regarde que CHC et CCK (les mixtes n'ont pas de "bonne" ou "mauvaise" slice définie)
    df_pure = df_patients[df_patients['verite'].isin(['CHC', 'CCK'])]
    
    sns.histplot(data=df_pure, x='taux_discordance', hue='verite', kde=True, 
                 bins=15, multiple="layer", palette=['#1f77b4', '#d62728'], alpha=0.5)
    
    plt.xlabel("% de Slices en Contradiction avec le Diagnostic Réel", fontweight='bold')
    plt.ylabel("Nombre de Patients", fontweight='bold')
    plt.title("🧬 Indice d'Hétérogénéité : Taux de Slices 'Orphelines' par Patient", fontsize=13, fontweight='bold', pad=15)
    
    path = os.path.join(output_dir, "advanced_discordance_slices.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 🖼️  Graphique 3/4 sauvegardé : {path}")


def plot_importance_features_global(output_dir):
    """ Graphique 4 : Feature Importance Macro (Explicabilité Globale) """
    plt.figure(figsize=(9, 6))
    
    # Calcul de l'importance globale moyenne basée sur la logique de ta fonction delta
    # On simule un échantillon global pour extraire les features les plus changeantes
    masque_chc = dp.df_clean['classe_name'] == 'CHC'
    masque_cck = dp.df_clean['classe_name'] == 'CCK'
    
    moyennes_chc = dp.df_clean.loc[masque_chc, dp.features_ia].mean()
    moyennes_cck = dp.df_clean.loc[masque_cck, dp.features_ia].mean()
    std_global = dp.df_clean[dp.features_ia].std().replace(0, 1)
    
    # Delta moyen normalisé entre les deux pathologies majeures
    importance_macro = (moyennes_chc - moyennes_cck).abs() / std_global
    top_macro = importance_macro.sort_values(ascending=False).head(10)
    
    clean_names = [f.replace("original_", "").replace("_", " ") for f in top_macro.index]
    
    sns.barplot(x=top_macro.values, y=clean_names, palette="flare")
    plt.xlabel("Pouvoir Discriminant Moyen (Écart standardisé macro σ)", fontweight='bold')
    plt.title("🧬 Top 10 des Biomarqueurs Radiomiques les plus Discriminants", fontsize=13, fontweight='bold', pad=15)
    
    path = os.path.join(output_dir, "advanced_importance_features.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 🖼️  Graphique 4/4 sauvegardé : {path}")


# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 CALCULS ET CRÉATION DES GRAPHICS SCIENTIFIQUES AVANCÉS")
    print("=" * 70)

    # 1. Extraction unique des tables
    df_p, df_ph = collecter_donnees_avancees()
    
    # 2. Folder management
    output_dir = os.path.join(dp.root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Exportation des 4 images distinctes vers : {output_dir}\n")

    # 3. Runs
    plot_courbes_roc(df_p, df_ph, output_dir)
    plot_swarm_probabilities(df_p, output_dir)
    plot_histogramme_discordance(df_p, output_dir)
    plot_importance_features_global(output_dir)

    print("\n" + "=" * 70)
    print("✅ TOUT EST PRÊT ! Les rapports visuels avancés ont été ajoutés.")
    print("=" * 70)