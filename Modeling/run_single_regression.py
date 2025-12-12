import sys
import pandas as pd
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data Preparation"))

from prep_regression import prepare_regression
from modeling_regression import modeling_regression
from run_total_preparation import find_csv   # <== Import essentiel pour winequality


# ============================================
#  CLI ARGUMENTS
# ============================================
dataset_name = sys.argv[1]        # ex: auto-mpg
target_column = sys.argv[2]       # ex: mpg

# ============================================
#  FIND THE CORRECT CSV FILE IN /Data
# ============================================
csv_file = find_csv(dataset_name)    # detects winequality-red.csv, etc.
# Chemin relatif depuis Modeling vers Data
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
csv_path = os.path.join(data_dir, csv_file)

print(f"\n=== LOADING REGR DATASET: {csv_path} ===")

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"❌ ERROR loading dataset {csv_path}: {e}")
    sys.exit(1)

# ============================================
#  PREPARATION
# ============================================
X_skl, y, X_rlt, y_rlt = prepare_regression(df, target_column, dataset_name)

# ============================================
#  MODELING
# ============================================
print("Running regression modeling...")
results_df = modeling_regression(X_skl, y, X_rlt, y_rlt, dataset_name)

print("\n===== REGRESSION COMPLETE =====")
print(results_df)

# ============================================
#  SAVE RESULTS
# ============================================
# Sauvegarder dans le répertoire parent (racine du projet)
output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(output_dir, f"{dataset_name}_results.csv")
results_df.to_csv(out_path, index=False)

print("Saved:", out_path)
