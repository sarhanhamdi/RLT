# RLT/Models/registry.py

from Models.rf import make_rf_regressor, make_rf_classifier
from Models.et import make_et_regressor, make_et_classifier
from Models.linear_models import make_lasso, make_elasticnet, make_logreg_l2
from Models.gb import make_gbr, make_gbc
from Models.rlt import RLTRegressor, RLTClassifier
import numpy as np


def compute_p0(p, p1_expected=None):
    """
    Calcule p_0 (nombre de variables protégées) selon l'article Zhu et al. (2015).
    
    Selon Section 3 et Table 3 de l'article :
    - p_0 doit être ≥ p_1 (nombre de variables fortes)
    - Recommandation : p_0 = log(p)
    
    Paramètres:
    -----------
    p : int
        Nombre total de features
    p1_expected : int, optional
        Nombre attendu de variables fortes (si connu)
        Dans les scénarios de l'article : p_1 = 2-4
        
    Retourne:
    ---------
    int : p_0 optimal
    
    Exemples:
    ---------
    >>> compute_p0(50)
    3
    >>> compute_p0(200)
    5
    >>> compute_p0(1000)
    6
    >>> compute_p0(200, p1_expected=4)
    5
    
    Notes:
    ------
    Valeurs typiques selon l'article (Table 3) :
    - p = 200  → p_0 = 5  (log(200) ≈ 5.3)
    - p = 500  → p_0 = 6  (log(500) ≈ 6.2)
    - p = 1000 → p_0 = 6  (log(1000) ≈ 6.9)
    """
    # Calcul de base : log naturel (ln)
    p0_log = int(np.log(p)) if p > 1 else 1
    
    # Si on connaît p_1, garantir p_0 ≥ p_1 + 1
    if p1_expected is not None:
        p0 = max(p1_expected + 1, p0_log)
    else:
        p0 = p0_log
    
    # Bornes de sécurité
    p0 = max(1, min(p0, p // 2))  # Entre 1 et p/2
    
    return p0


def get_benchmark_models(task_type: str, p: int = None, random_state: int = 42):
    """
    Retourne un dict {nom_modele: instance} pour la tâche donnée.
    
    ⚠️ IMPORTANT : Passez le paramètre 'p' (nombre de features) pour que p_0
    soit calculé automatiquement selon l'article : p_0 = log(p)
    
    Pour la régression :
      - Modèles de référence : RF, ET, GBR, Lasso, ElasticNet
      - RLT : 9 variantes (3 muting × 3 k)
    
    Pour la classification :
      - Modèles de référence : RF, ET, GBC, LogRegL2
      - RLT : version simplifiée
    
    Paramètres selon l'article Zhu et al. (2015), Table 3 :
    ----------------------------------------------------------
      - n_estimators : 50-100 arbres
      - n_min : n^(1/3) ou 5 par défaut
      - p_0 : log(p) variables protégées [ADAPTATIF selon p]
      - muting : none (0%), moderate (50%), aggressive (80%)
      - k : 1 (single variable), 2, 5 (linear combination)
    
    Paramètres:
    -----------
    task_type : str
        "regression" ou "classification"
    p : int, optional
        Nombre de features dans vos données
        Si fourni : p_0 = log(p) [RECOMMANDÉ]
        Si None : p_0 = 10 par défaut [NON OPTIMAL]
    random_state : int, default=42
        Seed pour reproductibilité
    
    Retourne:
    ---------
    dict : {nom_modèle: instance_sklearn}
    
    Exemples:
    ---------
    >>> # Avec p spécifié (recommandé)
    >>> models = get_benchmark_models("regression", p=200, random_state=42)
    [INFO] p=200 → p_0=5 (selon article : log(200) ≈ 5.30)
    
    >>> # Sans p (pas recommandé)
    >>> models = get_benchmark_models("regression", random_state=42)
    [WARN] p non spécifié, utilisation de p_0=10 par défaut
    """
    models = {}
    
    # ========================================
    # CALCUL DE p_0 SELON L'ARTICLE
    # ========================================
    if p is not None:
        p_0 = compute_p0(p)
        print(f"[INFO] p={p} → p_0={p_0} (selon article : log({p}) ≈ {np.log(p):.2f})")
    else:
        p_0 = 10
        print(f"[WARN] p non spécifié, utilisation de p_0={p_0} par défaut")
        print(f"       Recommandé : passer p pour p_0 = log(p) adaptatif")
        print(f"       Exemple : get_benchmark_models('regression', p=200)")

    if task_type == "regression":
        # =============================================
        # BASELINES : Modèles de référence
        # =============================================
        models["RF"] = make_rf_regressor(random_state=random_state)
        models["ET"] = make_et_regressor(random_state=random_state)
        models["GBR"] = make_gbr(random_state=random_state)
        models["Lasso"] = make_lasso(random_state=random_state)
        models["ElasticNet"] = make_elasticnet(random_state=random_state)

        # =============================================
        # RLT : 9 variantes (3 muting × 3 k)
        # =============================================
        # Selon Table 3 de l'article :
        # - M = 100 arbres (on utilise 50 pour rapidité)
        # - n_min = n^(1/3) (on utilise 5 par défaut)
        # - p_0 = log(p) [MAINTENANT ADAPTATIF]
        # - muting : none (0%), moderate (50%), aggressive (80%)
        # - k : 1 (single), 2, 5 (linear combination)
        
        for mut in ["none", "moderate", "aggressive"]:
            for k in [1, 2, 5]:
                name = f"RLT_{mut}_k{k}"
                models[name] = RLTRegressor(
                    n_estimators=20,       # 100 dans l'article, 50 pour rapidité
                    n_min=5,               # n^(1/3) recommandé, 5 par défaut
                    muting=mut,            # none / moderate / aggressive
                    k=k,                   # Nb de variables pour combinaison linéaire
                    p_0=p_0,               # ✅ MAINTENANT ADAPTATIF : log(p)
                    max_depth=None,        # Pas de limite (défaut article)
                    random_state=random_state,
                )

    else:  # classification
        # =============================================
        # BASELINES
        # =============================================
        models["RF"] = make_rf_classifier(random_state=random_state)
        models["ET"] = make_et_classifier(random_state=random_state)
        models["GBC"] = make_gbc(random_state=random_state)
        models["LogRegL2"] = make_logreg_l2(random_state=random_state)
        
        # =============================================
        # RLT Classification (version simplifiée)
        # =============================================
        # Note : Le RLTClassifier actuel est très simplifié
        # Pour une vraie implémentation, il faudrait adapter
        # tout le mécanisme de muting à la classification
        for mut in ["none", "moderate", "aggressive"]:
            for k in [1,2,5]:
                name = f"RLT_{mut}_k{k}"
                models[name] = RLTClassifier(
                    n_estimators=20,
                    n_min=5,
                    muting=mut,
                    k=k,
                    p_0=p_0,  # ✅ Adaptatif aussi
                    random_state=random_state,
        )

    return models


def get_rlt_only(task_type: str, p: int = None, random_state: int = 42):
    """
    Retourne uniquement les modèles RLT (utile pour analyse spécifique).
    
    Paramètres:
    -----------
    task_type : str
        "regression" ou "classification"
    p : int, optional
        Nombre de features (pour p_0 adaptatif)
    random_state : int
        Seed
        
    Retourne:
    ---------
    dict : {nom_rlt: instance}
    
    Exemple:
    --------
    >>> rlt_models = get_rlt_only("regression", p=200, random_state=42)
    >>> print(list(rlt_models.keys()))
    ['RLT_none_k1', 'RLT_none_k2', 'RLT_none_k5',
     'RLT_moderate_k1', 'RLT_moderate_k2', 'RLT_moderate_k5',
     'RLT_aggressive_k1', 'RLT_aggressive_k2', 'RLT_aggressive_k5']
    """
    all_models = get_benchmark_models(task_type, p=p, random_state=random_state)
    return {k: v for k, v in all_models.items() if k.startswith("RLT_")}


def get_best_rlt_config(
    task_type: str, 
    p: int = None,
    p1_expected: int = None,
    scenario: str = "general", 
    random_state: int = 42
):
    """
    Retourne la configuration RLT recommandée selon le scénario.
    
    Paramètres:
    -----------
    task_type : str
        "regression" ou "classification"
    p : int, optional
        Nombre de features (pour calculer p_0 automatiquement)
    p1_expected : int, optional
        Nombre attendu de variables fortes (si connu)
        Scénarios de l'article : 2-4 variables fortes
    scenario : str, default="general"
        Type de problème :
        - "general"   : configuration par défaut (moderate, k=2)
        - "sparse"    : données sparse high-dim (aggressive, k=1)
        - "linear"    : structure linéaire suspectée (moderate, k=5)
        - "nonlinear" : interactions complexes (moderate, k=2)
    random_state : int
        Seed
        
    Retourne:
    ---------
    estimator : Instance RLT configurée optimalement
    
    Exemples:
    ---------
    >>> # Pour des données sparse (p >> n, peu de variables fortes)
    >>> model = get_best_rlt_config("regression", p=500, scenario="sparse")
    
    >>> # Pour un problème linéaire
    >>> model = get_best_rlt_config("regression", p=200, scenario="linear")
    
    >>> # Config générale
    >>> model = get_best_rlt_config("regression", p=200)
    """
    estimator_class = RLTRegressor if task_type == "regression" else RLTClassifier
    
    # Calcul de p_0 adaptatif
    if p is not None:
        p_0 = compute_p0(p, p1_expected)
    else:
        p_0 = 10
        print(f"[WARN] p non spécifié pour get_best_rlt_config, p_0={p_0} par défaut")
    
    # ========================================
    # CONFIGURATIONS PAR SCÉNARIO
    # ========================================
    configs = {
        "general": {
            "n_estimators": 100,
            "muting": "moderate",
            "k": 2,
            "p_0": p_0,
            "n_min": 5,
            "max_depth": None,
        },
        "sparse": {
            "n_estimators": 100,
            "muting": "aggressive",  # Plus agressif pour sparse
            "k": 1,                  # Une seule variable à la fois
            "p_0": max(3, p_0 // 2), # Moins de protection
            "n_min": 10,             # Nœuds plus grands
            "max_depth": None,
        },
        "linear": {
            "n_estimators": 50,
            "muting": "moderate",
            "k": 5,                  # Combinaison linéaire de 5 variables
            "p_0": min(p_0 * 2, p // 5) if p else p_0 * 2,  # Plus de protection
            "n_min": 5,
            "max_depth": None,
        },
        "nonlinear": {
            "n_estimators": 100,
            "muting": "moderate",
            "k": 2,
            "p_0": p_0,
            "n_min": 3,              # Nœuds plus petits pour capturer interactions
            "max_depth": None,
        },
    }
    
    config = configs.get(scenario, configs["general"])
    
    print(f"[INFO] Config '{scenario}' : muting={config['muting']}, k={config['k']}, p_0={config['p_0']}")
    
    return estimator_class(**config, random_state=random_state)


def analyze_rlt_results(results_dict, verbose=True):
    """
    Analyse les résultats des différentes variantes RLT.
    
    Affiche :
    - Meilleure configuration RLT
    - Impact du muting (moyenne par niveau)
    - Impact de k (moyenne par valeur)
    - Comparaison avec les baselines
    
    Paramètres:
    -----------
    results_dict : dict
        {nom_modèle: score} pour tous les modèles testés
        Pour la régression : score = MSE (plus petit = meilleur)
        Pour la classification : score = accuracy (plus grand = meilleur)
    verbose : bool, default=True
        Afficher les détails
        
    Retourne:
    ---------
    dict : Statistiques d'analyse
        - best_rlt : (nom, score) du meilleur RLT
        - muting_stats : stats par niveau de muting
        - k_stats : stats par valeur de k
        - baseline_comparison : comparaison avec baselines
    
    Exemple:
    --------
    >>> results = {
    ...     "RF": 8.35,
    ...     "RLT_none_k2": 4.93,
    ...     "RLT_moderate_k2": 3.43,
    ...     "RLT_aggressive_k2": 2.66,
    ... }
    >>> stats = analyze_rlt_results(results)
    """
    rlt_results = {k: v for k, v in results_dict.items() if k.startswith("RLT_")}
    baseline_results = {k: v for k, v in results_dict.items() if not k.startswith("RLT_")}
    
    if not rlt_results:
        print("⚠️  Aucun résultat RLT trouvé.")
        return {}
    
    # Meilleure config RLT (minimiser le score pour régression)
    best_rlt = min(rlt_results.items(), key=lambda x: x[1])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f" ANALYSE DES RÉSULTATS RLT ".center(70))
        print(f"{'='*70}")
        print(f"\n🏆 MEILLEURE CONFIG RLT : {best_rlt[0]}")
        print(f"   Score : {best_rlt[1]:.4f}")
        print(f"{'='*70}")
    
    # ========================================
    # ANALYSE PAR MUTING
    # ========================================
    muting_stats = {}
    if verbose:
        print("\n📊 IMPACT DU MUTING (moyenne par niveau)")
        print("-" * 70)
    
    for mut in ["none", "moderate", "aggressive"]:
        scores = [v for k, v in rlt_results.items() if f"_{mut}_" in k]
        if scores:
            muting_stats[mut] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
            }
            if verbose:
                print(f"  {mut:12s} : {np.mean(scores):.4f} ± {np.std(scores):.4f}  "
                      f"[{np.min(scores):.4f} - {np.max(scores):.4f}]")
    
    # Amélioration du muting
    if verbose and "none" in muting_stats:
        print("\n  Amélioration vs 'none' :")
        for mut in ["moderate", "aggressive"]:
            if mut in muting_stats:
                improvement = ((muting_stats["none"]["mean"] - muting_stats[mut]["mean"]) 
                             / muting_stats["none"]["mean"] * 100)
                print(f"    {mut:12s} : {improvement:+.1f}%")
    
    # ========================================
    # ANALYSE PAR K
    # ========================================
    k_stats = {}
    if verbose:
        print("\n🔢 IMPACT DE K (moyenne par valeur)")
        print("-" * 70)
    
    for k in [1, 2, 5]:
        scores = [v for key, v in rlt_results.items() if f"_k{k}" in key]
        if scores:
            k_stats[k] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
            }
            if verbose:
                print(f"  k={k} : {np.mean(scores):.4f} ± {np.std(scores):.4f}  "
                      f"[{np.min(scores):.4f} - {np.max(scores):.4f}]")
    
    # ========================================
    # COMPARAISON AVEC BASELINES
    # ========================================
    baseline_comparison = {}
    if baseline_results and verbose:
        print("\n⚖️  COMPARAISON AVEC BASELINES")
        print("-" * 70)
        best_baseline = min(baseline_results.items(), key=lambda x: x[1])
        baseline_comparison = {
            "best_baseline_name": best_baseline[0],
            "best_baseline_score": best_baseline[1],
            "best_rlt_name": best_rlt[0],
            "best_rlt_score": best_rlt[1],
        }
        
        print(f"  Meilleure baseline : {best_baseline[0]:20s} → {best_baseline[1]:.4f}")
        print(f"  Meilleure RLT      : {best_rlt[0]:20s} → {best_rlt[1]:.4f}")
        
        if best_rlt[1] < best_baseline[1]:
            improvement = ((best_baseline[1] - best_rlt[1]) / abs(best_baseline[1])) * 100
            print(f"\n  ✅ RLT GAGNE avec {improvement:.2f}% d'amélioration !")
            baseline_comparison["improvement"] = improvement
        else:
            degradation = ((best_rlt[1] - best_baseline[1]) / abs(best_baseline[1])) * 100
            print(f"\n  ⚠️  Baseline gagne ({degradation:.2f}% meilleure)")
            baseline_comparison["improvement"] = -degradation
    
    if verbose:
        print(f"\n{'='*70}\n")
    
    return {
        "best_rlt": best_rlt,
        "muting_stats": muting_stats,
        "k_stats": k_stats,
        "baseline_comparison": baseline_comparison,
    }