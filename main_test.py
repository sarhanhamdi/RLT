# RLT/main_experiments.py

from Pipelines.data_preparation import load_and_prepare_data
from Pipelines.modeling import run_experiments

def main():
    config = {
        "dataset_name": "HousingData",  # ou un autre de ta liste
        "task_type": None,             # laisser None pour inférence auto
        "scenario": "real_HousingData",
        "random_state": 42,
    }

    X_train, X_test, y_train, y_test, meta = load_and_prepare_data(config)
    # tu peux imprimer quelques infos
    print("Task type:", meta["task_type"])
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    run_experiments(X_train, X_test, y_train, y_test, config)

if __name__ == "__main__":
    main()
