# run_single_regression.py
import sys
import pandas as pd
from modeling_regression import regression_modeling


# =====================================
# 1. READ INPUT ARGUMENTS
# =====================================
if len(sys.argv) != 3:
    print("Usage: python run_single_regression.py <dataset_name> <target_column>")
    sys.exit()

dataset_name = sys.argv[1]
target_column = sys.argv[2]
file_path = f"{dataset_name}.csv"

print(f"\n=== LOADING REGR DATASET: {dataset_name} ===")


# =====================================
# 2. SPECIAL LOADER FOR WINEQUALITY
# =====================================
def load_winequality(path):
    """
    WineQuality datasets are improperly formatted.
    Each row is a single string -> must be split manually.
    """
    df_raw = pd.read_csv(path, header=None)
    header = df_raw.iloc[0, 0].split(",")
    rows = df_raw.iloc[1:, 0].apply(lambda x: x.split(","))
    df = pd.DataFrame(rows.to_list(), columns=header)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


# =====================================
# 3. LOAD DATASET
# =====================================
if dataset_name in ["winequality-red", "winequality-white"]:
    df = load_winequality(file_path)
else:
    df = pd.read_csv(file_path)


# =====================================
# 4. FIX WRONG OR MISSING TARGET COLUMNS
# =====================================

# ----- Concrete dataset -----
if dataset_name == "concrete_data":
    if target_column not in df.columns:

        # The most common names for target:
        candidates = ["Concrete_compressive_strength", "csMPa", "strength"]
        found = False

        for c in candidates:
            if c in df.columns:
                target_column = c
                found = True
                print(f"✔ Concrete target corrected → {c}")
                break

        if not found:
            print("❌ ERROR: No valid target column found for concrete_data.")
            sys.exit()

# ----- Ozone dataset -----
if dataset_name == "ozone":
    if target_column not in df.columns:
        print("✔ Target corrected → maxO3")
        target_column = "maxO3"


# =====================================
# 5. RUN MODELING
# =====================================
print("Running regression modeling...")

results = regression_modeling(df, target_column, dataset_name)

print("\n===== REGRESSION COMPLETE =====")
print(results)


# =====================================
# 6. SAVE RESULTS
# =====================================
output_file = f"{dataset_name}_results.csv"
results.to_csv(output_file, index=False)
print(f"Saved → {output_file}")
