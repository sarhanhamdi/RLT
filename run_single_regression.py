import sys
import pandas as pd
import os

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
csv_path = os.path.join("Data", csv_file)

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
out_path = f"{dataset_name}_results.csv"
results_df.to_csv(out_path, index=False)

print("Saved:", out_path)
