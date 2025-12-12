# run_all_regression.py
# Script pour exécuter automatiquement tous les modèles de régression sur tous les datasets

import subprocess
import pandas as pd
import os
import sys

# Ajouter le répertoire parent et Data Preparation au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data Preparation"))

# Imports depuis run_total_preparation
from run_total_preparation import (
    find_csv,
    auto_detect_target,
    repair_csv_if_needed
)

# ============================
# LISTE DES DATASETS DE RÉGRESSION
# ============================

REG_DATASETS = [
    "auto-mpg",
    "concrete_data",
    "HousingData",
    "ozone",
    "winequality-red",
    "winequality-white",
    "dataset_scenario1",
    "dataset_scenario2",
    "dataset_scenario3",
    "dataset_scenario4",
]

# Chemin vers le dossier Data (depuis Modeling)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")


# ============================
# FONCTION POUR EXÉCUTER UN DATASET
# ============================

def run_regression_on_dataset(dataset_name):
    """
    Exécute run_single_regression.py sur un dataset donné.
    Retourne le DataFrame des résultats ou None en cas d'erreur.
    """
    print(f"\n{'='*60}")
    print(f"  TRAITEMENT: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # 1. Trouver le fichier CSV
        csv_file = find_csv(dataset_name)
        csv_path = os.path.join(DATA_DIR, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"❌ Fichier non trouvé: {csv_path}")
            return None
        
        # 2. Charger et réparer le CSV si nécessaire
        df = repair_csv_if_needed(csv_path)
        
        # 3. Détecter automatiquement la colonne cible
        target_column = auto_detect_target(df, dataset_name)
        print(f"   Colonne cible détectée: {target_column}")
        
        # 4. Exécuter le script de régression
        print(f"   Exécution de run_single_regression.py...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_single_regression.py")
        result = subprocess.run(
            [sys.executable, script_path, dataset_name, target_column],
            check=True,
            capture_output=False  # Afficher la sortie en temps réel
        )
        
        # 5. Charger les résultats
        result_file = os.path.join(BASE_DIR, f"{dataset_name}_results.csv")
        if os.path.exists(result_file):
            results_df = pd.read_csv(result_file)
            print(f"   ✓ Résultats sauvegardés: {result_file}")
            print(f"   ✓ {len(results_df)} modèles évalués")
            return results_df
        else:
            print(f"   ⚠ Fichier de résultats non trouvé: {result_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erreur lors de l'exécution: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None


# ============================
# FONCTION PRINCIPALE
# ============================

def main():
    """
    Exécute tous les modèles de régression sur tous les datasets.
    """
    print("\n" + "="*60)
    print("  EXÉCUTION AUTOMATIQUE - RÉGRESSION")
    print("  Tous les modèles sur tous les datasets")
    print("="*60)
    
    all_results = []
    successful = []
    failed = []
    
    # Parcourir tous les datasets
    for i, dataset_name in enumerate(REG_DATASETS, 1):
        print(f"\n[{i}/{len(REG_DATASETS)}] Traitement de {dataset_name}...")
        
        results_df = run_regression_on_dataset(dataset_name)
        
        if results_df is not None:
            all_results.append(results_df)
            successful.append(dataset_name)
        else:
            failed.append(dataset_name)
    
    # ============================
    # SAUVEGARDE DES RÉSULTATS COMBINÉS
    # ============================
    
    print("\n" + "="*60)
    print("  RÉSUMÉ")
    print("="*60)
    print(f"  ✓ Datasets réussis: {len(successful)}/{len(REG_DATASETS)}")
    print(f"  ❌ Datasets échoués: {len(failed)}/{len(REG_DATASETS)}")
    
    if successful:
        print(f"\n  Datasets réussis: {', '.join(successful)}")
    if failed:
        print(f"\n  Datasets échoués: {', '.join(failed)}")
    
    if len(all_results) > 0:
        # Combiner tous les résultats
        df_final = pd.concat(all_results, ignore_index=True)
        
        # Sauvegarder le fichier final dans le répertoire parent
        output_file = os.path.join(BASE_DIR, "ALL_REGRESSION_RESULTS.csv")
        df_final.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"  ✓ RÉSULTATS COMBINÉS SAUVEGARDÉS")
        print(f"  → {output_file}")
        print(f"{'='*60}")
        print(f"\n  Total: {len(df_final)} résultats")
        print(f"  Datasets: {df_final['dataset'].nunique()}")
        print(f"  Modèles: {df_final['model'].nunique()}")
        
        # Afficher un aperçu
        print(f"\n  Aperçu des résultats:")
        print(df_final.to_string(index=False))
        
        # Statistiques par dataset
        print(f"\n  Meilleur R² par dataset:")
        for dataset in df_final['dataset'].unique():
            dataset_results = df_final[df_final['dataset'] == dataset]
            best = dataset_results.loc[dataset_results['r2'].idxmax()]
            print(f"    {dataset:20s} → {best['model']:15s} (R² = {best['r2']:.4f})")
    else:
        print("\n  ❌ Aucun résultat n'a pu être généré.")
        sys.exit(1)


if __name__ == "__main__":
    main()

