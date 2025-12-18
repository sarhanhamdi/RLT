# RLT/main_benchmark_all.py

from Pipelines.data_preparation import load_and_prepare_data
from Pipelines.modeling import run_experiments

# mêmes noms que dans data_understanding.py
REAL_DATASETS = [
    "HousingData",
    "parkinsons",
    "sonar",
    "winequality-red",
    "winequality-white",
    "ReplicatedAcousticFeatures-ParkinsonDatabase",
    "ozone",
    "concrete_data",
    "BreastCanDT",
    "auto-mpg",
]

SIMU_SCENARIOS = ["scenario1", "scenario2", "scenario3", "scenario4"]
SIMU_P = [200, 500, 1000]


def run_real_datasets():
    for name in REAL_DATASETS:
        config = {
            "source": "real",
            "dataset_name": name,
            "scenario": f"real_{name}",   # label pour le CSV
            "random_state": 42,
        }

        print(f"\n##### REAL dataset = {name} #####")
        X_train, X_test, y_train, y_test, meta = load_and_prepare_data(config)
        config["task_type"] = meta["task_type"]

        run_experiments(X_train, X_test, y_train, y_test, config)


def run_simulated_scenarios():
    for scenario in SIMU_SCENARIOS:
        for p in SIMU_P:
            config = {
                "source": "simulated",
                "scenario": scenario,
                "simulation_params": {
                    "p": p,
                    # n sera choisi automatiquement selon le scénario
                },
                "dataset_name": f"{scenario}_p{p}",  # pour identifier dans le CSV
                "random_state": 42,
            }

            print(f"\n##### SIMU scenario = {scenario}, p = {p} #####")
            X_train, X_test, y_train, y_test, meta = load_and_prepare_data(config)
            config["task_type"] = meta["task_type"]

            run_experiments(X_train, X_test, y_train, y_test, config)


def main():
    # 1) Tous les datasets réels
    run_real_datasets()

    # 2) Tous les scénarios simulés
    run_simulated_scenarios()


if __name__ == "__main__":
    main()
