import os
import sys
import pandas as pd

# Ajouter le répertoire courant au path pour les imports locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prep_regression import prepare_regression

# prep_classification peut être à la racine ou dans Data Preparation
try:
    from prep_classification import prepare_classification
except ImportError:
    # Si prep_classification est à la racine
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prep_classification import prepare_classification


# Chemins relatifs depuis Data Preparation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "Prepared")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------
# 1. Normalize names
# -----------------------------------------------------------

def normalize(name):
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def find_csv(name):
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


# -----------------------------------------------------------
# 2. Fix malformed CSV (separator , or ;)
# -----------------------------------------------------------

def repair_csv_if_needed(path):
    """
    If CSV loads into 1 column → repair by detecting separator.
    """
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_csv(path, engine="python")

    # If it has more than one column → OK
    if df.shape[1] > 1:
        return df

    print(f"⚠ Repairing malformed CSV: {os.path.basename(path)}")

    # Try semicolon
    try:
        df2 = pd.read_csv(path, sep=";")
        if df2.shape[1] > 1:
            df2.to_csv(path, index=False)
            return df2
    except:
        pass

    # Try comma
    try:
        df2 = pd.read_csv(path, sep=",")
        if df2.shape[1] > 1:
            df2.to_csv(path, index=False)
            return df2
    except:
        pass

    # Last fallback: manual split
    df2 = df.iloc[:, 0].str.split("[;,]", expand=True)
    df2.columns = [f"col{i}" for i in range(df2.shape[1])]
    df2.to_csv(path, index=False)
    return df2


# -----------------------------------------------------------
# 3. Auto Target Detection
# -----------------------------------------------------------

def auto_detect_target(df, dataset_name):
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
    if "auto" in name:
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



# -----------------------------------------------------------
# 4. CLASSIFICATION DATASETS
# -----------------------------------------------------------

CLASS_DATASETS = [
    "BreastCanDT",
    "sonar",
    "parkinsons",
    "ReplicatedAcousticFeatures-ParkinsonDatabase"
]


def prepare_all_classification():
    print("\n==========================")
    print(" PREPARING CLASSIFICATION ")
    print("==========================\n")

    for name in CLASS_DATASETS:
        try:
            csv_file = find_csv(name)
            full_path = os.path.join(DATA_DIR, csv_file)

            df = repair_csv_if_needed(full_path)
            target = auto_detect_target(df, name)

            X_skl, y_skl, X_rlt, y_rlt = prepare_classification(df, target, name)

            # Save prepared dataset
            out = os.path.join(OUTPUT_DIR, f"{name}_prepared.csv")
            df_out = pd.DataFrame(X_skl)
            df_out[target] = y_skl
            df_out.to_csv(out, index=False)

            print(f"✔ Prepared {name}")
            print(f"   SKL shape: {df_out.shape}")
            print(f"   RLT shape: {X_rlt.shape}\n")

        except Exception as e:
            print(f"❌ ERROR preparing {name}: {e}\n")


# -----------------------------------------------------------
# 5. REGRESSION DATASETS
# -----------------------------------------------------------

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



def prepare_all_regression():
    print("\n==========================")
    print("   PREPARING REGRESSION   ")
    print("==========================\n")

    for name in REG_DATASETS:
        try:
            csv_file = find_csv(name)
            full_path = os.path.join(DATA_DIR, csv_file)

            df = repair_csv_if_needed(full_path)
            target = auto_detect_target(df, name)

            X_skl, y_skl, X_rlt, y_rlt = prepare_regression(df, target, name)

            out = os.path.join(OUTPUT_DIR, f"{name}_prepared.csv")
            df_out = pd.DataFrame(X_skl)
            df_out[target] = y_skl
            df_out.to_csv(out, index=False)

            print(f"✔ Prepared {name}")
            print(f"   SKL shape: {df_out.shape}")
            print(f"   RLT shape: {X_rlt.shape}\n")

        except Exception as e:
            print(f"❌ ERROR preparing {name}: {e}\n")


# -----------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------

if __name__ == "__main__":
    prepare_all_classification()
    prepare_all_regression()
    print("\n✔ ALL DATASETS PREPARED SUCCESSFULLY ✔")
