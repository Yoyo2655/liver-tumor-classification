import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Force matplotlib à générer les images en tâche de fond
plt.switch_backend('Agg') 
warnings.filterwarnings("ignore")

# Configuration esthétique globale
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(current_dir)

try:
    import diagnostic_patient as dp
except ImportError:
    print("❌ Impossible de trouver 'diagnostic_patient.py' dans le même dossier.")
    exit()

def dessiner_et_sauvegarder(patient_num, est_mixte=False, output_dir=""):
    if est_mixte:
        slices_patient = dp.df_mixtes[dp.df_mixtes['patient_num'] == patient_num]
        statut = "Patient Mixte (Inconnu)"
    else:
        slices_patient = dp.df_clean[dp.df_clean['patient_num'] == patient_num]

    if slices_patient.empty:
        return

    if not est_mixte:
        idx_patient = set(slices_patient.index)
        est_test_   = bool(idx_patient & dp.set_test)
        est_train_  = bool(idx_patient & dp.set_train)
        if est_test_ and not est_train_:
            statut = "Test à l'aveugle (100% Inconnu)"
        elif est_train_ and not est_test_:
            statut = "Patient d'Entraînement"
        else:
            statut = "Patient Mixte (Train/Test)"

    phases_presentes = slices_patient['temps_inj'].values
    resultats_par_phase = {}

    for phase in dp.PHASES_STANDARD:
        masque = (phases_presentes == phase)
        if masque.sum() == 0:
            resultats_par_phase[phase] = None
            continue
        
        X_phase  = slices_patient.loc[slices_patient['temps_inj'] == phase, dp.features_ia].fillna(0)
        X_mean   = X_phase.mean(axis=0).values.reshape(1, -1)
        X_scaled = dp.scaler.transform(X_mean)
        probas_phase = dp.clf_nn.predict_proba(X_scaled)[0]
        idx_cck = list(dp.le.classes_).index('CCK')
        p_cck   = probas_phase[idx_cck]

        if p_cck >= dp.SEUIL_PHASE_HAUT:
            idx_cls, conclusif = idx_cck, True
        elif p_cck <= dp.SEUIL_PHASE_BAS:
            idx_cls, conclusif = 1 - idx_cck, True
        else:
            idx_cls, conclusif = probas_phase.argmax(), False

        resultats_par_phase[phase] = {
            'probas': probas_phase,
            'classe': dp.le.classes_[idx_cls],
            'n_slices': int(masque.sum()),
            'conclusif': conclusif,
            'p_cck': p_cck
        }

    phases_actives = [r for r in resultats_par_phase.values() if r is not None]
    if not phases_actives:
        return

    scores_globaux = np.mean([r['probas'] for r in phases_actives], axis=0)
    scores_globaux = scores_globaux / scores_globaux.sum()
    
    idx_cck = list(dp.le.classes_).index('CCK')
    idx_chc = list(dp.le.classes_).index('CHC')
    p_cck_global = scores_globaux[idx_cck]

    if p_cck_global >= dp.SEUIL_HAUT_CCK:
        verdict = 'CCK'
    elif p_cck_global <= dp.SEUIL_BAS_CCK:
        verdict = 'CHC'
    else:
        verdict = 'NON_CONCLUANT'

    phases_conclusives = sum(1 for r in phases_actives if r['conclusif'])
    score_fiabilite = 0.50 * abs(p_cck_global - 0.5) * 2 + 0.50 * (phases_conclusives / len(phases_actives))

    # ---- Ligne 111 corrigée ici ----
    phase_ref = max(
        [p for p, r in resultats_par_phase.items() if r is not None and r['classe'] == verdict] or
        [p for p, r in resultats_par_phase.items() if r is not None],
        key=lambda p: resultats_par_phase[p]['probas'][list(dp.le.classes_).index(resultats_par_phase[p]['classe'])],
        default=None
    )
    if phase_ref is None:
        phase_ref = next(p for p, r in resultats_par_phase.items() if r is not None)

    slices_phase_ref = slices_patient[slices_patient['temps_inj'] == phase_ref]
    slice_ref = slices_phase_ref.iloc[len(slices_phase_ref) // 2]
    delta_features = dp.scores_features_explicatifs(slice_ref, verdict if verdict != 'NON_CONCLUANT' else dp.le.classes_[scores_globaux.argmax()])
    top_features = delta_features.head(5)

    vérité = 'Mixtes' if est_mixte else slices_patient['classe_name'].iloc[0]

    # Génération du Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Rapport Diagnostique Visuel — Patient N°{patient_num}\n({statut})", fontsize=16, fontweight='bold', y=0.98)
    colors_dict = {'CHC': '#1f77b4', 'CCK': '#d62728', 'NON_CONCLUANT': '#7f7f7f', 'Mixtes': '#9467bd'}

    # Graph 1
    ax1 = axes[0, 0]
    phases_labels = []
    chc_p, cck_p = [], []
    for phase in dp.PHASES_STANDARD:
        res = resultats_par_phase[phase]
        phases_labels.append(f"{phase}\n({res['n_slices']} slc)" if res else f"{phase}\n(Absente)")
        chc_p.append(res['probas'][idx_chc] if res else 0)
        cck_p.append(res['probas'][idx_cck] if res else 0)

    x = np.arange(len(dp.PHASES_STANDARD))
    width = 0.35
    ax1.bar(x - width/2, chc_p, width, label='Probabilité CHC', color=colors_dict['CHC'], alpha=0.8)
    ax1.bar(x + width/2, cck_p, width, label='Probabilité CCK', color=colors_dict['CCK'], alpha=0.8)
    ax1.axhline(y=dp.SEUIL_PHASE_HAUT, color='#d62728', linestyle='--', alpha=0.6, label='Seuil Phase CCK')
    ax1.axhline(y=dp.SEUIL_PHASE_BAS, color='#1f77b4', linestyle='--', alpha=0.6, label='Seuil Phase CHC')
    ax1.set_title('📊 Prédictions par Phase d\'Injection', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases_labels)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right')

    # Graph 2
    ax2 = axes[0, 1]
    ax2.barh(['Score Global P(CCK)'], [p_cck_global], color=colors_dict.get(verdict, '#7f7f7f'), height=0.4)
    ax2.axvspan(0, dp.SEUIL_BAS_CCK, color='blue', alpha=0.08, label='Zone CHC Confirmé')
    ax2.axvspan(dp.SEUIL_BAS_CCK, dp.SEUIL_HAUT_CCK, color='orange', alpha=0.08, label='Zone Incertitude')
    ax2.axvspan(dp.SEUIL_HAUT_CCK, 1, color='red', alpha=0.08, label='Zone CCK Confirmé')
    ax2.axvline(x=p_cck_global, color='black', linestyle='-', linewidth=2.5)
    ax2.set_xlim(0, 1)
    ax2.set_title(f'🧮 Verdict Final : {verdict}', fontweight='bold')
    
    txt_details = f"Vérité Histologique : {dp.NOM_CANCER.get(vérité, vérité)}\nP(CCK) Global = {p_cck_global*100:.1f}%\nP(CHC) Global = {(1-p_cck_global)*100:.1f}%"
    ax2.text(0.02, 0.2, txt_details, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

    # Graph 3
    ax3 = axes[1, 0]
    fiab_pct = score_fiabilite * 100
    ax3.pie([fiab_pct, 100 - fiab_pct], startangle=90, colors=['#2ca02c', '#f7dcdb'], wedgeprops=dict(width=0.35, edgecolor='w'))
    ax3.text(0, 0, f"{fiab_pct:.1f}%", ha='center', va='center', fontsize=22, fontweight='bold', color='#1e5a1e')
    ax3.set_title('🎯 Indice de Fiabilité Clinique', fontweight='bold')

    # Graph 4
    ax4 = axes[1, 1]
    feats_clean_names = [f.replace("original_", "").replace("_", " ") for f in top_features.index]
    sns.barplot(x=top_features.values, y=feats_clean_names, ax=ax4, palette="crest")
    ax4.set_xlabel('Écart à la moyenne (en Sigma σ)')
    ax4.set_title('🧬 Top 5 des Caractéristiques Discriminantes', fontweight='bold')

    plt.tight_layout()
    
    filename = f"patient_{patient_num}_verdict_{verdict}.png"
    if est_mixte:
        filename = f"patient_{patient_num}_MIXTE_verdict_{verdict}.png"
        
    fig_path = os.path.join(output_dir, filename)
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 CALCUL ET GÉNÉRATION DES PLOTS POUR TOUS LES PATIENTS")
    print("=" * 70)

    labels_par_patient = dp.df_clean.groupby('patient_num')['classe_name'].nunique()
    patients_ambigus   = set(labels_par_patient[labels_par_patient > 1].index.tolist())

    patients_test = sorted([p for p in dp.df_clean.loc[dp.df_clean.index.isin(dp.set_test), 'patient_num'].unique().tolist() if p not in patients_ambigus])
    patients_train = sorted([p for p in dp.df_clean.loc[dp.df_clean.index.isin(dp.set_train), 'patient_num'].unique().tolist() if p not in patients_ambigus])
    patients_mixtes = dp.patients_mixtes

    output_dir = os.path.join(dp.root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Les graphiques (.png) seront créés dans :\n👉 {output_dir}\n")

    print(f"⏳ [1/3] Génération des {len(patients_test)} patients de TEST...")
    for p in patients_test:
        dessiner_et_sauvegarder(p, est_mixte=False, output_dir=output_dir)

    print(f"⏳ [2/3] Génération des {len(patients_mixtes)} patients MIXTES...")
    for p in patients_mixtes:
        dessiner_et_sauvegarder(p, est_mixte=True, output_dir=output_dir)

    print(f"⏳ [3/3] Génération des {len(patients_train)} patients de TRAIN...")
    for p in patients_train:
        dessiner_et_sauvegarder(p, est_mixte=False, output_dir=output_dir)

    print("\n" + "=" * 70)
    print(f"✅ TERMINÉ ! Tous les graphiques ont été déposés dans :")
    print(f"👉 {output_dir}")
    print("=" * 70)