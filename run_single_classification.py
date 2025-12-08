# run_single_classification.py
import sys
import pandas as pd
from modeling_classification import classification_modeling

# ---------------------------------
# Input arguments: dataset name + target column
# ---------------------------------
if len(sys.argv) != 3:
    print("Usage: python run_single_classification.py <dataset_name> <target_column>")
    sys.exit()

dataset_name = sys.argv[1]
target_column = sys.argv[2]
file_path = f"{dataset_name}.csv"

print(f"\n=== LOADING DATASET: {dataset_name} ===")

df = pd.read_csv(file_path)

# ---------------------------------
# Fix dataset-specific issues
# ---------------------------------

# SONAR correction (column name)
if dataset_name.lower() == "sonar":
    if "Class" not in df.columns:
        # last column is the label in most sonar datasets
        df.rename(columns={df.columns[-1]: "Class"}, inplace=True)
    target_column = "Class"

# PARKINSONS : convert status column to int (some datasets store it incorrectly)
if dataset_name.lower() == "parkinsons_clean":
    if df[target_column].dtype != int:
        df[target_column] = df[target_column].astype(int)

# ---------------------------------
# Run modeling
# ---------------------------------
print("Running classification...")
results = classification_modeling(df, target_column, dataset_name)

print("\n===== CLASSIFICATION COMPLETE =====")
print(results)

output_file = f"{dataset_name}_results.csv"
results.to_csv(output_file, index=False)
print(f"Saved → {output_file}")
