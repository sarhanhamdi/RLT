#!/usr/bin/env python
# coding: utf-8

"""
Pipeline.py - Data Preparation Pipeline
Ce script prépare tous les datasets (régression et classification) 
et sauvegarde les fichiers CSV préparés dans le dossier data_prepared.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ============================================
# CONFIGURATION DES CHEMINS
# ============================================

# Chemin vers le dossier Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_prepared")

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# LISTE DES DATASETS
# ============================================

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

CLASS_DATASETS = [
    "BreastCanDT",
    "sonar",
    "parkinsons",
    "ReplicatedAcousticFeatures-ParkinsonDatabase"
]


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def normalize(name):
    """Normalise le nom d'un dataset pour la recherche de fichiers."""
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def find_csv(name):
    """
    Trouve le fichier CSV correspondant au nom du dataset.
    """
    target = normalize(name)

    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith(".csv"):
            continue

        base = normalize(f.replace(".csv", ""))

        if base == target:
            return f

        if target in base:
            return f

    raise FileNotFoundError(f"❌ CSV file for '{name}' not found in Data/")


def repair_csv_if_needed(path):
    """
    Répare les CSV malformés en détectant automatiquement le séparateur.
    """
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_csv(path, engine="python")

    # Si le fichier a plus d'une colonne, c'est OK
    if df.shape[1] > 1:
        return df

    print(f"⚠ Repairing malformed CSV: {os.path.basename(path)}")

    # Essayer le point-virgule
    try:
        df2 = pd.read_csv(path, sep=";")
        if df2.shape[1] > 1:
            df2.to_csv(path, index=False)
            return df2
    except:
        pass

    # Essayer la virgule
    try:
        df2 = pd.read_csv(path, sep=",")
        if df2.shape[1] > 1:
            df2.to_csv(path, index=False)
            return df2
    except:
        pass

    # Dernier recours : division manuelle
    df2 = df.iloc[:, 0].str.split("[;,]", expand=True)
    df2.columns = [f"col{i}" for i in range(df2.shape[1])]
    df2.to_csv(path, index=False)
    return df2


def fix_winequality(path: str) -> pd.DataFrame:
    """
    Charge et nettoie les datasets winequality (red & white).
    Détecte automatiquement le séparateur et convertit les colonnes.
    """
    # Essayer le point-virgule d'abord
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] > 1:
            df.columns = [c.strip().replace('"', "") for c in df.columns]
            df = df.apply(pd.to_numeric, errors="ignore")
            return df
    except Exception:
        pass

    # Essayer la virgule
    try:
        df = pd.read_csv(path, sep=",")
        if df.shape[1] > 1:
            df.columns = [c.strip().replace('"', "") for c in df.columns]
            df = df.apply(pd.to_numeric, errors="ignore")
            return df
    except Exception:
        pass

    # Si le fichier est malformé (une seule colonne)
    raw = pd.read_csv(path, header=None)
    lines = raw.iloc[:, 0]

    header = lines.iloc[0].replace('"', "").split(";")
    data = [row.replace('"', "").split(";") for row in lines.iloc[1:]]

    df = pd.DataFrame(data, columns=header)

    # Convertir les colonnes numériques
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def auto_detect_target(df, dataset_name):
    """
    Détecte automatiquement la colonne cible pour chaque dataset.
    """
    name = dataset_name.lower()

    # BREAST CANCER
    if "breast" in name:
        return "diagnosis"

    # SONAR
    if "sonar" in name:
        return df.columns[-1]

    # PARKINSONS
    if "parkinson" in name:
        for col in df.columns:
            if col.lower() == "status":
                return col

    # Replicated Acoustic Parkinson Database
    if "replicatedacoustic" in name:
        for col in df.columns:
            if col.lower() == "status":
                return col

    # WINEQUALITY
    if "winequality" in name:
        return "quality"

    # AUTO MPG
    if "auto" in name and "mpg" in name:
        return "mpg"

    # CONCRETE
    if "concrete" in name:
        for c in df.columns:
            if "compressive" in c.lower():
                return c
            if "strength" in c.lower():
                return c

    # HOUSING DATASET (Boston)
    if "housing" in name:
        for col in df.columns:
            if col.lower() == "medv":
                return col
        raise ValueError("❌ HousingData target 'MEDV' not found in columns.")

    # OZONE
    if "ozone" in name:
        for col in df.columns:
            if "maxo3" in col.lower():
                return col
            if "ozone" in col.lower():
                return col

    # SCENARIOS
    if "scenario" in name:
        return "Y"

    raise ValueError(f"❌ Could not detect target for dataset '{dataset_name}'.")


# ============================================
# FONCTIONS DE PRÉPARATION
# ============================================

def prepare_regression(df, target, dataset_name):
    """
    Prépare les données pour la régression.
    Returns:
        X_skl : numpy array pour les modèles sklearn (scaled)
        y     : numpy array target
        X_rlt : numpy array pour RLT python (non-scaled)
        y_rlt : numpy array target pour RLT python
    """
    # Fix winequality datasets BEFORE anything
    if "winequality" in dataset_name.lower():
        data_dir = os.path.join(BASE_DIR, "Data")
        csv_path = os.path.join(data_dir, dataset_name + ".csv")
        df = fix_winequality(csv_path)

    # Supprimer les lignes où la cible est manquante
    df = df.dropna(subset=[target])

    # Séparer les features et la cible
    y = df[target].astype(float).values
    X = df.drop(columns=[target])

    # Convertir les colonnes en numériques si possible
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Supprimer les colonnes complètement NaN
    X = X.dropna(axis=1, how="all")

    # Imputer les valeurs manquantes (median)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scaler les features pour les modèles sklearn
    scaler = StandardScaler()
    X_skl = scaler.fit_transform(X_imputed)

    # Préparer les données pour RLT (SANS scaling)
    X_rlt = X_imputed.copy()
    y_rlt = y.copy()

    return X_skl, y, X_rlt, y_rlt


def prepare_classification(df, target, dataset_name):
    """
    Prépare les données pour la classification.
    Returns:
        X_skl : numpy array pour les modèles sklearn (scaled)
        y_skl : numpy array target (encoded)
        X_rlt : numpy array pour RLT python (non-scaled)
        y_rlt : numpy array target pour RLT python (encoded)
    """
    # Supprimer les lignes où la cible est manquante
    df = df.dropna(subset=[target])

    # Séparer les features et la cible
    y = df[target]
    X = df.drop(columns=[target])

    # Convertir les colonnes en numériques si possible
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Supprimer les colonnes complètement NaN
    X = X.dropna(axis=1, how="all")

    # Imputer les valeurs manquantes (median)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Encoder la cible si nécessaire (catégorielle -> numérique)
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y.values.astype(int)

    # Scaler les features pour les modèles sklearn
    scaler = StandardScaler()
    X_skl = scaler.fit_transform(X_imputed)

    # Préparer les données pour RLT (SANS scaling)
    X_rlt = X_imputed.copy()
    y_rlt = y_encoded.copy()

    return X_skl, y_encoded, X_rlt, y_rlt


# ============================================
# FONCTION PRINCIPALE DE PRÉPARATION
# ============================================

def prepare_all_datasets():
    """
    Prépare tous les datasets (régression et classification) 
    et sauvegarde les CSV dans data_prepared.
    """
    print("\n" + "="*60)
    print("  PRÉPARATION DES DATASETS")
    print("="*60 + "\n")

    all_errors = []

    # ============================================
    # PRÉPARATION DES DATASETS DE RÉGRESSION
    # ============================================
    print("\n" + "-"*60)
    print("  DATASETS DE RÉGRESSION")
    print("-"*60 + "\n")

    for name in REG_DATASETS:
        try:
            # Gérer les scénarios (scenario1_p1000, scenario1_p200, etc.)
            if "scenario" in name.lower():
                # Pour les scénarios, on prépare tous les fichiers correspondants
                # "dataset_scenario1" -> "scenario1"
                scenario_num = name.replace("dataset_", "").replace("dataset", "")
                # Normaliser pour trouver les fichiers (scenario1 -> trouve scenario1_p1000, etc.)
                scenario_normalized = normalize(scenario_num)
                matching_files = [f for f in os.listdir(DATA_DIR) 
                                if normalize(f.replace(".csv", "")).startswith(scenario_normalized) 
                                and f.endswith(".csv")]
                
                if not matching_files:
                    print(f"⚠ Aucun fichier trouvé pour {name}")
                    continue
                
                for csv_file in matching_files:
                    dataset_key = csv_file.replace(".csv", "")
                    full_path = os.path.join(DATA_DIR, csv_file)
                    
                    df = repair_csv_if_needed(full_path)
                    target = auto_detect_target(df, dataset_key)
                    
                    X_skl, y_skl, X_rlt, y_rlt = prepare_regression(df, target, dataset_key)
                    
                    # Créer un DataFrame avec les features et la cible
                    df_out = pd.DataFrame(X_skl, columns=[f"feature_{i}" for i in range(X_skl.shape[1])])
                    df_out[target] = y_skl
                    
                    # Sauvegarder
                    out_path = os.path.join(OUTPUT_DIR, f"{dataset_key}_prepared.csv")
                    df_out.to_csv(out_path, index=False)
                    
                    print(f"✔ Préparé : {dataset_key}")
                    print(f"   Shape: {df_out.shape}")
            else:
                # Pour les autres datasets
                csv_file = find_csv(name)
                full_path = os.path.join(DATA_DIR, csv_file)
                
                df = repair_csv_if_needed(full_path)
                target = auto_detect_target(df, name)
                
                X_skl, y_skl, X_rlt, y_rlt = prepare_regression(df, target, name)
                
                # Créer un DataFrame avec les features et la cible
                df_out = pd.DataFrame(X_skl, columns=[f"feature_{i}" for i in range(X_skl.shape[1])])
                df_out[target] = y_skl
                
                # Sauvegarder
                out_path = os.path.join(OUTPUT_DIR, f"{name}_prepared.csv")
                df_out.to_csv(out_path, index=False)
                
                print(f"✔ Préparé : {name}")
                print(f"   Shape: {df_out.shape}")

        except Exception as e:
            error_msg = f"❌ ERREUR lors de la préparation de {name}: {e}"
            print(error_msg)
            all_errors.append(error_msg)

    # ============================================
    # PRÉPARATION DES DATASETS DE CLASSIFICATION
    # ============================================
    print("\n" + "-"*60)
    print("  DATASETS DE CLASSIFICATION")
    print("-"*60 + "\n")

    for name in CLASS_DATASETS:
        try:
            csv_file = find_csv(name)
            full_path = os.path.join(DATA_DIR, csv_file)
            
            df = repair_csv_if_needed(full_path)
            target = auto_detect_target(df, name)
            
            X_skl, y_skl, X_rlt, y_rlt = prepare_classification(df, target, name)
            
            # Créer un DataFrame avec les features et la cible
            df_out = pd.DataFrame(X_skl, columns=[f"feature_{i}" for i in range(X_skl.shape[1])])
            df_out[target] = y_skl
            
            # Sauvegarder
            out_path = os.path.join(OUTPUT_DIR, f"{name}_prepared.csv")
            df_out.to_csv(out_path, index=False)
            
            print(f"✔ Préparé : {name}")
            print(f"   Shape: {df_out.shape}")

        except Exception as e:
            error_msg = f"❌ ERREUR lors de la préparation de {name}: {e}"
            print(error_msg)
            all_errors.append(error_msg)

    # ============================================
    # RÉSUMÉ FINAL
    # ============================================
    print("\n" + "="*60)
    print("  RÉSUMÉ")
    print("="*60 + "\n")
    
    if all_errors:
        print(f"⚠ {len(all_errors)} erreur(s) rencontrée(s):")
        for err in all_errors:
            print(f"   {err}")
    else:
        print("✔ Tous les datasets ont été préparés avec succès !")
    
    print(f"\n📁 Fichiers sauvegardés dans : {OUTPUT_DIR}")
    
    # Lister les fichiers créés
    prepared_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_prepared.csv")]
    print(f"\n📊 Nombre de fichiers préparés : {len(prepared_files)}")
    print("\nFichiers créés :")
    for f in sorted(prepared_files):
        file_path = os.path.join(OUTPUT_DIR, f)
        file_size = os.path.getsize(file_path) / 1024  # Taille en KB
        print(f"   - {f} ({file_size:.2f} KB)")


# ============================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================

if __name__ == "__main__":
    prepare_all_datasets()
