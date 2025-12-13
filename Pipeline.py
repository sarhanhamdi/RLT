#!/usr/bin/env python
# coding: utf-8

# ## 1. Importation :

# In[48]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor, GradientBoostingClassifier


# ## 2. Load Data : 

# In[50]:


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier {path} n'existe pas.")

    if not path.endswith(".csv"):
        raise ValueError("Format non supporté : seuls les fichiers .csv sont acceptés.")

    print(f"Chargement du dataset : {path}")
    return pd.read_csv(path)


# ## 2. Data Understanding : 

# In[52]:


def data_understanding(df, target_column=None):

    print("\n🔹 Shape:", df.shape)

    print("\n🔹 First 5 rows:")
    display(df.head())

    print("\n🔹 Last 5 rows:")
    display(df.tail())

    print("\n🔹 Info:")
    print(df.info())

    print("\n🔹 Data types:")
    print(df.dtypes)

    print("\n🔹 Missing values per column:")
    missing_vals = df.isnull().sum()
    display(missing_vals[missing_vals > 0])

    print("\n🔹 Percentage of missing values per column:")
    missing_percent = (df.isnull().mean() * 100).round(2)
    display(missing_percent[missing_percent > 0])

    print("\n🔹 Duplicate rows count:", df.duplicated().sum())
    if df.duplicated().sum() > 0:
        print("🔹 Duplicate rows:")
        display(df[df.duplicated(keep=False)])

    print("\n🔹 Target variable preview:")
    if target_column and target_column in df.columns:
        display(df[[target_column]].head())
    else:
        print(" Target column not found or not provided.")


    # -------------------------------------------------
    # Détection des colonnes numériques (une seule fois)
    # -------------------------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns

    print("\n🔹 Numeric columns:", list(numeric_cols))



    # -------------------------------------------------
    # Détection des outliers
    # -------------------------------------------------
    outlier_counts = {}

    if len(numeric_cols) > 0:
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
    else:
        print("\n Aucun champ numérique → impossible de détecter les outliers.")

    print("\n🔹 Number of outliers per numeric column:")
    display(outlier_counts)



    # -------------------------------------------------
    # Histogrammes des variables numériques
    # -------------------------------------------------
    if len(numeric_cols) > 0:
        print("\n Distribution des variables numériques :")
        df[numeric_cols].hist(bins=30, figsize=(12, 8))
        plt.tight_layout()
        plt.show()
    else:
        print("\n Aucun champ numérique → pas d’histogrammes.")



    # -------------------------------------------------
    # Heatmap de corrélation
    # -------------------------------------------------
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        mask = corr < 0.5

        plt.figure(figsize=(18, 14))
        ax = sns.heatmap(
            corr, mask=mask, cmap="coolwarm", annot=False,
            linewidths=0.5, 
            cbar_kws={'label': 'Force de corrélation'}
        )

        plt.title("Heatmap des corrélations (seulement |corr| > 0.5)", fontsize=16)

        # Légende explicative
        plt.text(
            x=0.02, y=1.12,
            s=(
                "Légende des couleurs :\n"
                "Rouge foncé → Corrélation très positive (≈ 0.8 à 1.0)\n"
                "Bleu foncé → Corrélation très négative (≈ -0.8 à -1.0)\n"
                "Blanc → Corrélation faible (< 0.5) ou masquée"
            ),
            fontsize=12,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.8)
        )

        plt.show()
    else:
        print("\n Pas assez de colonnes numériques pour une heatmap.")



    # -------------------------------------------------
    # Barplot des valeurs manquantes
    # -------------------------------------------------
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        plt.figure(figsize=(10, 5))
        missing.sort_values().plot(kind='barh')
        plt.title("Valeurs manquantes par colonne")
        plt.xlabel("Nombre de valeurs manquantes")
        plt.show()
    else:
        print("\n Aucune valeur manquante.")



    # -------------------------------------------------
    # Boxplots pour visualiser les outliers
    # -------------------------------------------------
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if df[col].dropna().nunique() > 1:
                plt.figure(figsize=(6, 3))
                sns.boxplot(x=df[col])
                plt.title(f"Boxplot – {col}")
                plt.show()
            else:
                print(f"Impossible de tracer un boxplot pour {col} (pas assez de valeurs).")
    else:
        print("\n Aucun champ numérique → pas de boxplots.")


    return outlier_counts


# ## 3. Data Preperation :

# In[54]:


def cap_iqr(df, numeric_columns, factor=1.5):
    df = df.copy()

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0 or np.isnan(IQR):
            continue

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df[col] = np.clip(df[col], lower, upper)

    return df


def data_preparation(df, target_column, apply_capping=True):
    df = df.copy()
    original_shape = df.shape

    # =============================
    # 1) Supprimer colonnes 100% NaN
    # =============================
    nan_cols = [
        col for col in df.columns
        if col != target_column and df[col].isna().all()
    ]
    df.drop(columns=nan_cols, inplace=True)

    # =============================
    # 2) Supprimer variance nulle
    # =============================
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_column]

    zero_var_cols = []
    if numeric_cols:
        selector = VarianceThreshold(0.0)
        selector.fit(df[numeric_cols])
        zero_var_cols = [
            col for col, keep in zip(numeric_cols, selector.get_support())
            if not keep
        ]
        df.drop(columns=zero_var_cols, inplace=True)

    # =============================
    # 3) Séparation X / y
    # =============================
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' supprimée par erreur")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # =============================
    # 4) Détection types
    # =============================
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # =============================
    # 5) Capping IQR
    # =============================
    if apply_capping and numeric_cols:
        X = cap_iqr(X, numeric_cols)

    # =============================
    # 6) Pipeline sklearn
    # =============================
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    X_prepared = preprocessor.fit_transform(X)

    # =============================
    # 7) Dataset nettoyé final
    # =============================
    df_clean = pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1
    )

    # =============================
    # 8) Résumé
    # =============================
    print("\n=== DATA PREPARATION SUMMARY ===")
    print(f"Shape initiale : {original_shape}")
    print(f"Shape finale   : {df_clean.shape}")
    print(f"Colonnes supprimées : {nan_cols + zero_var_cols}")
    print(f"Features numériques : {len(numeric_cols)}")
    print(f"Features catégorielles : {len(categorical_cols)}")
    print(f"X_prepared shape : {X_prepared.shape}")

    return X, X_prepared, y, preprocessor, nan_cols + zero_var_cols, df_clean


# In[55]:


def save_clean_dataset(name, df_clean, folder="clean_datasets"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_clean.csv")
    df_clean.to_csv(path, index=False)
    print(f" Dataset sauvegardé : {path}")


# ## 4. Modeling :

# In[69]:





# ## 5. Evaluation :

# In[ ]:





# ## 6. Deploiment : 

# In[ ]:




