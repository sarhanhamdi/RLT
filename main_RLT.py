from Pipelines.data_understanding import summarize_all_real_datasets

def main():
    """
    Étape 2 de CRISP-DM : Data Understanding
    - Passe sur tous les jeux de données réels définis dans REAL_DATASETS
    - Affiche un résumé pour chacun
    - Produit un fichier results/data_summary.csv
    """
    summarize_all_real_datasets()

if __name__ == "__main__":
    main()