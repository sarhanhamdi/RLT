#!/usr/bin/env python
# coding: utf-8

"""
pipeline_model_classification.py
Pipeline de modeling pour la classification utilisant les datasets préparés.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Models"))

# Importer RLT pour classification
from Models.rlt_python import RLTClassifier

# ============================================
# CONFIGURATION DES CHEMINS
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PREPARED_DIR = os.path.join(BASE_DIR, "data_prepared")
OUTPUT_DIR = BASE_DIR


# ============================================
# LISTE DES DATASETS DE CLASSIFICATION
# ============================================

CLASS_DATASETS = [
    "BreastCanDT",
    "sonar",
    "parkinsons",
    "ReplicatedAcousticFeatures-ParkinsonDatabase"
]


# ============================================
# FONCTION POUR OBTENIR LE MODÈLE RLT CLASSIFICATION
# ============================================

def get_rlt_classifier(dataset_name=None, n_samples=None):
    """
    Retourne un modèle RLT Classifier avec paramètres adaptés.
    """
    # Paramètres par défaut
    default_params = {
        'min_samples_leaf': 5,
        'combsplit': 3,
        'reinforcement': True,
        'muting_percent': 0.4,
        'random_state': 42
    }
    
    # Ajuster le nombre d'arbres selon le dataset
    if dataset_name and n_samples and n_samples > 400:
        n_trees = 20  # Medium trees for larger datasets
    else:
        n_trees = 50  # Standard for smaller datasets
    
    default_params['ntrees'] = n_trees
    
    return RLTClassifier(**default_params)


# ============================================
# FONCTION DE MODELING POUR UN DATASET
# ============================================

def modeling_classification_dataset(dataset_name):
    """
    Entraîne tous les modèles de classification sur un dataset préparé.
    
    Args:
        dataset_name: Nom du dataset (ex: 'BreastCanDT')
        
    Returns:
        DataFrame avec les résultats pour ce dataset
    """
    # Charger le dataset préparé
    prepared_file = os.path.join(DATA_PREPARED_DIR, f"{dataset_name}_prepared.csv")
    
    if not os.path.exists(prepared_file):
        print(f"❌ Fichier préparé non trouvé: {prepared_file}")
        return None
    
    print(f"\n{'='*60}")
    print(f"  TRAITEMENT: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Charger le dataset
        df = pd.read_csv(prepared_file)
        
        # Identifier la colonne target (dernière colonne)
        target_col = df.columns[-1]
        
        # Séparer X et y
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        print(f"   Dataset shape: {df.shape}")
        print(f"   Target: {target_col}")
        print(f"   Classes: {sorted(np.unique(y))}")
        
        # Vérifier qu'il y a au moins 2 classes
        if len(np.unique(y)) < 2:
            print(f"   ❌ Le dataset doit contenir au moins 2 classes")
            return None
        
        # Train/test split avec stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        
        results = []
        
        # ============================================
        # LISTE DES MODÈLES À ENTRÂINER
        # ============================================
        models_to_train = [
            {
                'name': 'RF',
                'model': RandomForestClassifier(random_state=42),
                'use_all_features': True
            },
            {
                'name': 'RF_sqrtp',
                'model': RandomForestClassifier(max_features="sqrt", random_state=42),
                'use_all_features': True
            },
            {
                'name': 'RF_logp',
                'model': RandomForestClassifier(
                    max_features=max(1, int(np.log(n_features))), 
                    random_state=42
                ),
                'use_all_features': False  # Utilise log(n_features) features
            },
            {
                'name': 'ET',
                'model': ExtraTreesClassifier(random_state=42),
                'use_all_features': True
            },
            {
                'name': 'Boosting',
                'model': GradientBoostingClassifier(random_state=42),
                'use_all_features': True
            },
            {
                'name': 'Lasso',
                'model': LogisticRegression(
                    penalty='l1', 
                    solver='liblinear', 
                    C=0.01, 
                    random_state=42,
                    max_iter=1000
                ),
                'use_all_features': False  # Lasso fait de la sélection de features
            },
            {
                'name': 'RLT',
                'model': get_rlt_classifier(dataset_name=dataset_name, n_samples=n_samples),
                'use_all_features': False  # RLT peut faire de la sélection
            }
        ]
        
        # ============================================
        # ENTRÂINER TOUS LES MODÈLES
        # ============================================
        for model_config in models_to_train:
            model_name = model_config['name']
            model = model_config['model']
            use_all = model_config['use_all_features']
            
            print(f"   Running {model_name}...", end=" ")
            
            try:
                # Entraîner le modèle
                model.fit(X_train, y_train)
                
                # Prédire
                y_pred = model.predict(X_test)
                
                # Calculer les métriques
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                
                # ROC-AUC (seulement si binaire)
                auc = None
                if len(np.unique(y)) == 2:
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_prob = model.predict_proba(X_test)[:, 1]
                            auc = roc_auc_score(y_test, y_prob)
                        else:
                            # Pour RLT qui n'a pas predict_proba standard
                            auc = None
                    except:
                        auc = None
                
                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "accuracy": round(acc, 6),
                    "f1_score": round(f1, 6),
                    "roc_auc": round(auc, 6) if auc is not None else np.nan,
                    "n_features_used": n_features if use_all else "subset"
                })
                
                print(f"✓ (Acc: {acc:.4f}, F1: {f1:.4f})")
                
            except Exception as e:
                print(f"❌ FAILED: {e}")
                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "accuracy": np.nan,
                    "f1_score": np.nan,
                    "roc_auc": np.nan,
                    "n_features_used": "error"
                })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"❌ ERREUR lors du traitement de {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# FONCTION PRINCIPALE
# ============================================

def run_all_classification():
    """
    Exécute tous les modèles de classification sur tous les datasets préparés.
    Sauvegarde les résultats dans un fichier CSV.
    """
    print("\n" + "="*60)
    print("  PIPELINE DE MODELING - CLASSIFICATION")
    print("  Tous les modèles sur tous les datasets préparés")
    print("="*60)
    
    all_results = []
    successful = []
    failed = []
    
    # Parcourir tous les datasets
    for i, dataset_name in enumerate(CLASS_DATASETS, 1):
        print(f"\n[{i}/{len(CLASS_DATASETS)}] {dataset_name}...")
        
        results_df = modeling_classification_dataset(dataset_name)
        
        if results_df is not None and not results_df.empty:
            all_results.append(results_df)
            successful.append(dataset_name)
        else:
            failed.append(dataset_name)
    
    # ============================================
    # SAUVEGARDE DES RÉSULTATS COMBINÉS
    # ============================================
    
    print("\n" + "="*60)
    print("  RÉSUMÉ")
    print("="*60)
    print(f"  ✓ Datasets réussis: {len(successful)}/{len(CLASS_DATASETS)}")
    print(f"  ❌ Datasets échoués: {len(failed)}/{len(CLASS_DATASETS)}")
    
    if successful:
        print(f"\n  Datasets réussis: {', '.join(successful)}")
    if failed:
        print(f"\n  Datasets échoués: {', '.join(failed)}")
    
    if len(all_results) > 0:
        # Combiner tous les résultats
        df_final = pd.concat(all_results, ignore_index=True)
        
        # Sauvegarder le fichier final
        output_file = os.path.join(OUTPUT_DIR, "classification_results.csv")
        df_final.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"  ✓ RÉSULTATS SAUVEGARDÉS")
        print(f"  → {output_file}")
        print(f"{'='*60}")
        print(f"\n  Total: {len(df_final)} résultats")
        print(f"  Datasets: {df_final['dataset'].nunique()}")
        print(f"  Modèles: {df_final['model'].nunique()}")
        
        # Afficher un aperçu
        print(f"\n  Aperçu des résultats (premières lignes):")
        print(df_final.head(20).to_string(index=False))
        
        # Statistiques par dataset
        print(f"\n  Meilleure Accuracy par dataset:")
        for dataset in sorted(df_final['dataset'].unique()):
            dataset_results = df_final[df_final['dataset'] == dataset]
            best = dataset_results.loc[dataset_results['accuracy'].idxmax()]
            print(f"    {dataset:40s} → {best['model']:12s} (Acc: {best['accuracy']:.4f})")
        
        return df_final
    else:
        print("\n  ❌ Aucun résultat n'a pu être généré.")
        return None


# ============================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================

if __name__ == "__main__":
    run_all_classification()
