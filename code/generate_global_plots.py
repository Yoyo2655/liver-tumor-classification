import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Mode non-interactif pour générer proprement les fichiers images en tâche de fond
plt.switch_backend('Agg')
warnings.filterwarnings("ignore")

# Configuration esthétique globale
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11

# Import de ton script pour récupérer le dataset et les modèles chargés
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(current_dir)

try:
    import diagnostic_patient as dp
except ImportError:
    print("❌ Impossible de trouver 'diagnostic_patient.py' dans le même dossier.")
    exit()

def collecter_donnees_cohortes():
    """ Boucle sur l'ensemble des patients pour extraire la base de données globale """
    records = []
    
    labels_par_patient = dp.df_clean.groupby('patient_num')['classe_name'].nunique()
    patients_ambigus = set(labels_par_patient[labels_par_patient > 1].index.tolist())
    
    all_patients = []
    # Patients Test
    for p in dp.df_clean.loc[dp.df_clean.index.isin(dp.set_test), 'patient_num'].unique():
        if p not in patients_ambigus: all_patients.append((p, "TEST", False))
    # Patients Train
    for p in dp.df_clean.loc[dp.df_clean.index.isin(dp.set_train), 'patient_num'].unique():
        if p not in patients_ambigus: all_patients.append((p, "TRAIN", False))
    # Patients Mixtes
    for p in dp.patients_mixtes:
        all_patients.append((p, "MIXTE", True))

    print(f"⏳ Extraction et calcul des métriques pour {len(all_patients)} patients...")

    for patient_num, groupe, est_mixte in all_patients:
        if est_mixte:
            slices_patient = dp.df_mixtes[dp.df_mixtes['patient_num'] == patient_num]
            verite = "Mixtes"
        else:
            slices_patient = dp.df_clean[dp.df_clean['patient_num'] == patient_num]
            verite = slices_patient['classe_name'].iloc[0]

        if slices_patient.empty: continue

        phases_presentes = slices_patient['temps_inj'].values
        phases_conclusives = 0
        phases_totales_actives = 0
        probas_phases = []
        stats_phases = {}
        
        idx_cck = list(dp.le.classes_).index('CCK')
        idx_chc = list(dp.le.classes_).index('CHC')
        
        for phase in dp.PHASES_STANDARD:
            masque = (phases_presentes == phase)
            if masque.sum() == 0: 
                stats_phases[phase] = 'ABSENTE'
                continue
            
            X_phase = slices_patient.loc[slices_patient['temps_inj'] == phase, dp.features_ia].fillna(0)
            X_scaled = dp.scaler.transform(X_phase.mean(axis=0).values.reshape(1, -1))
            probas_p = dp.clf_nn.predict_proba(X_scaled)[0]
            
            p_cck = probas_p[idx_cck]
            phases_totales_actives += 1
            probas_phases.append(probas_p)
            
            if p_cck >= dp.SEUIL_PHASE_HAUT or p_cck <= dp.SEUIL_PHASE_BAS:
                phases_conclusives += 1
                stats_phases[phase] = 'COCONCLUSIVE'
            else:
                stats_phases[phase] = 'NON_CONCLUANTE'

        if not probas_phases: continue
        
        scores_globaux = np.mean(probas_phases, axis=0)
        p_cck_global = scores_globaux[idx_cck] / scores_globaux.sum()

        if p_cck_global >= dp.SEUIL_HAUT_CCK: verdict = 'CCK'
        elif p_cck_global <= dp.SEUIL_BAS_CCK: verdict = 'CHC'
        else: verdict = 'NON_CONCLUANT'

        score_fiabilite = 0.50 * abs(p_cck_global - 0.5) * 2 + 0.50 * (phases_conclusives / phases_totales_actives)

        rec = {
            'patient_num': patient_num,
            'statut': groupe,
            'n_slices': len(slices_patient),
            'verite': verite,
            'verdict': verdict,
            'fiabilite': score_fiabilite,
            'p_cck_global': p_cck_global
        }
        for phase in dp.PHASES_STANDARD:
            rec[f'status_{phase}'] = stats_phases.get(phase, 'ABSENTE')
            
        records.append(rec)
        
    return pd.DataFrame(records)

# =====================================================================
# FONCTIONS DE GÉNÉRATION DES PLOTS INDIVIDUELS
# =====================================================================

def plot_matrice_confusion(df, output_dir):
    """ Fichier 1 : Matrice de confusion avec colonne Non Concluant """
    df_eval = df[df['verite'].isin(['CHC', 'CCK'])]
    classes_labels = ['CHC', 'CCK', 'NON_CONCLUANT']
    
    cm = confusion_matrix(df_eval['verite'].tolist(), df_eval['verdict'].tolist(), labels=classes_labels[:2])
    
    cm_complete = np.zeros((2, 3), dtype=int)
    cm_complete[:, :2] = cm
    cm_complete[0, 2] = len(df_eval[(df_eval['verite']=='CHC') & (df_eval['verdict']=='NON_CONCLUANT')])
    cm_complete[1, 2] = len(df_eval[(df_eval['verite']=='CCK') & (df_eval['verdict']=='NON_CONCLUANT')])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_complete, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes_labels, yticklabels=classes_labels[:2], cbar=False, annot_kws={"size": 14})
    plt.xlabel('Verdict du Modèle (Prédiction)', fontweight='bold')
    plt.ylabel('Vérité Histologique (Réel)', fontweight='bold')
    plt.title('🎯 Matrice de Confusion Clinique (Niveau Patients)', fontsize=13, fontweight='bold', pad=15)
    
    path = os.path.join(output_dir, "macro_matrice_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 🖼️  Image sauvegardée : {path}")

def plot_distribution_fiabilite(df, output_dir):
    """ Fichier 2 : Boxplot de la fiabilité """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='statut', y='fiabilite', palette='Set2', hue='statut', legend=False)
    sns.stripplot(data=df, x='statut', y='fiabilite', color='black', alpha=0.3, jitter=0.2, s=6)
    plt.ylabel('Indice de Fiabilité (0 à 1)', fontweight='bold')
    plt.xlabel('Groupe de Patients', fontweight='bold')
    plt.title('🎯 Distribution de la Fiabilité Diagnostique par Statut', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(-0.05, 1.05)
    
    path = os.path.join(output_dir, "macro_distribution_fiabilite.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 🖼️  Image sauvegardée : {path}")

def plot_performance_phases(df, output_dir):
    """ Fichier 3 : Barplot comparatif du taux d'incertitude des phases """
    phase_data = []
    for phase in dp.PHASES_STANDARD:
        col = f'status_{phase}'
        total_p = df[df[col] != 'ABSENTE']
        nb_conclu = len(total_p[total_p[col] == 'COCONCLUSIVE'])
        nb_non_conclu = len(total_p[total_p[col] == 'NON_CONCLUANTE'])
        
        tot = nb_conclu + nb_non_conclu
        pct_conclu = (nb_conclu / tot * 100) if tot > 0 else 0
        
        phase_data.append({'Phase': phase, 'Statut': 'Conclusive ✓', 'Pourcentage': pct_conclu})
        phase_data.append({'Phase': phase, 'Statut': 'Incertitude ⚠️', 'Pourcentage': 100 - pct_conclu})
        
    df_phase = pd.DataFrame(phase_data)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_phase, x='Phase', y='Pourcentage', hue='Statut', palette=['#2ca02c', '#ff7f0e'])
    plt.ylabel('% des Patients Présents', fontweight='bold')
    plt.xlabel('Phase d\'Injection', fontweight='bold')
    plt.title('⚡ Pouvoir Décisif et Clarté par Phase d\'Injection', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 110)
    
    path = os.path.join(output_dir, "macro_performance_phases.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 🖼️  Image sauvegardée : {path}")

def plot_volume_vs_fiabilite(df, output_dir):
    """ Fichier 4 : Scatter plot volume de données vs fiabilité """
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x='n_slices', y='fiabilite', hue='verite', style='statut', s=120, palette='Set1')
    plt.xlabel('Nombre Total de Slices Extraites', fontweight='bold')
    plt.ylabel('Score de Fiabilité Global', fontweight='bold')
    plt.title('📈 Impact du Volume de Données (Slices) sur la Certitude', fontsize=13, fontweight='bold', pad=15)
    plt.legend(title="Légende", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    path = os.path.join(output_dir, "macro_volume_vs_fiabilite.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 🖼️  Image sauvegardée : {path}")

# =====================================================================
# POINT D'ENTRÉE PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 LANCEMENT DU SCRIPT DE GÉNÉRATION DES VISUALISATIONS MACRO")
    print("=" * 70)

    # Extraction des données une seule fois
    df_global = collecter_donnees_cohortes()
    
    # Définition et création du dossier de sortie
    output_dir = os.path.join(dp.root_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Dossier cible pour les images : {output_dir}\n")

    # Génération séquentielle des 4 fichiers graphiques indépendants
    print("🎬 Génération des fichiers graphiques séparés...")
    plot_matrice_confusion(df_global, output_dir)
    plot_distribution_fiabilite(df_global, output_dir)
    plot_performance_phases(df_global, output_dir)
    plot_volume_vs_fiabilite(df_global, output_dir)

    print("\n" + "=" * 70)
    print("✅ TOUT EST TERMINÉ ! Les 4 fichiers .png distincts sont prêts.")
    print("=" * 70)