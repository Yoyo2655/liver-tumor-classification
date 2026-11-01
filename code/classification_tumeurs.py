import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# 1. Gestion des chemins et création du dossier plot
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

# On descend dans le dossier 'dataset' pour trouver le fichier Excel
data_path = os.path.join(root_dir, "dataset", "Relectures_imageries.xlsx")
plot_dir = os.path.join(root_dir, "plot")
os.makedirs(plot_dir, exist_ok=True)

# Chargement du fichier Excel (.xlsx)
df = pd.read_excel(data_path)

# 2. Sélection des descripteurs de contraste (niveaux de gris / morphologie)
features = [
    "Rim_APHE",
    "nonrim_APHE",
    "APHE_heterogeneous",
    "Nonperiph_washout",
    "Late_enhancement",
    "Necrosis",
    "Capsule",
    "Intratumoral_fat",
    "Size_mm",
]

# Nettoyage rapide des données
df = df.dropna(subset=["Type_tumeur"])
X = df[features].fillna(0)
y = df["Type_tumeur"]

# Mappage des noms de classes pour l'affichage médical personnalisé
class_names = {1: "CCK (Type 1)", 2: "CHC (Type 2)", 3: "Mixte (Type 3)"}
target_names = [class_names[i] for i in sorted(y.unique())]

# 3. Séparation et entraînement de l'Arbre de Décision
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Profondeur limitée à 3 pour générer des règles simples et un graphique lisible
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 4. Sortie Console : Génération des arguments logiques écrits
print("=" * 60)
print("  LOGIQUE D'AIDE À LA DÉCISION MÉDICALE EXTRAITE DES IMAGES")
print("=" * 60)
rules = export_text(clf, feature_names=features)
print(rules)
print("=" * 60)

# 5. GÉNÉRATION DES GRAPHIQUES (Sauvegardés dans ../plot/)

# Graphique A : Visualisation de l'arbre de décision (Logique clinique)
plt.figure(figsize=(15, 8))
plot_tree(
    clf,
    feature_names=features,
    class_names=target_names,
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title(
    "Arbre de Décision Médical : Algorithme de classification des tumeurs hépatiques",
    fontsize=14,
    fontweight="bold",
)
plot_tree_path = os.path.join(plot_dir, "arbre_decision_clinique.png")
plt.savefig(plot_tree_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"-> Graphique de l'arbre décisionnel sauvegardé dans : {plot_tree_path}")

# Graphique B : Importance des critères sémantiques (Quels gris comptent le plus ?)
importances = pd.Series(clf.feature_importances_, index=features).sort_values(
    ascending=True
)

plt.figure(figsize=(10, 6))
importances.plot(kind="barh", color="skyblue", edgecolor="black")
plt.title(
    "Poids des critères d'imagerie dans le diagnostic différentiel",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Importance relative de la feature")
plt.ylabel("Critères (Signaux d'imagerie)")
plt.grid(axis="x", linestyle="--", alpha=0.7)

importance_path = os.path.join(plot_dir, "importance_features_imagerie.png")
plt.savefig(importance_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"-> Graphique de l'importance des features sauvegardé dans : {importance_path}")