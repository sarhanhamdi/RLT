# Models/registry_updated.py
"""
Updated Registry with RLTClassifierRAdvanced & RLTRegressorRAdvanced
Integrates 3 muting strategies x 3 linear combination splits = 9 RLT variants
"""

from Models.rf import make_rf_regressor, make_rf_classifier
from Models.et import make_et_regressor, make_et_classifier
from Models.linear_models import make_lasso, make_elasticnet, make_logreg_l2
from Models.gb import make_gbr, make_gbc

# NEW: Import advanced RLT classes
from Models.rlt_classifier_advanced import RLTClassifierRAdvanced
from Models.rlt_regressor_advanced import RLTRegressorRAdvanced

import numpy as np


def compute_p0(p, p1_expected=None):
    """
    Calcule p0 (nombre de variables protégées) selon l'article Zhu et al. 2015.
    
    Selon Section 3 et Table 3 de l'article:
    - p0 doit être >= p1 (nombre de variables fortes)
    - Recommandation: p0 = log(p)
    
    Paramètres
    ----------
    p : int
        Nombre total de features
    p1_expected : int, optional
        Nombre attendu de variables fortes (si connu)
        Dans les scénarios de l'article: p1 ∈ {2-4}
    
    Retourne
    --------
    int
        p0 optimal
    
    Exemples
    --------
    >>> compute_p0(50)
    3
    >>> compute_p0(200)
    5
    >>> compute_p0(1000)
    6
    >>> compute_p0(200, p1_expected=4)
    5
    
    Notes
    -----
    Valeurs typiques selon l'article (Table 3):
    - p=200:  p0=5  (log(200)≈5.30)
    - p=500:  p0=6  (log(500)≈6.21)
    - p=1000: p0=6  (log(1000)≈6.91)
    """
    p0_log = int(np.log(p)) if p > 1 else 1
    
    if p1_expected is not None:
        p0 = max(p1_expected + 1, p0_log)
    else:
        p0 = p0_log
    
    p0 = max(1, min(p0, p // 2))  # Entre 1 et p/2
    return p0


def get_benchmark_models(task_type: str, p: int = None, random_state: int = 42):
    """
    Retourne un dict {nom: model} pour la tâche donnée.
    
    IMPORTANT: Passez le paramètre p (nombre de features) pour que p0 soit 
    calculé automatiquement selon l'article: p0 = log(p)
    
    Pour la RÉGRESSION:
    - Modèles de référence: RF, ET, GBR, Lasso, ElasticNet
    - RLT AVANCÉ: 9 variantes (3 muting × 3 combsplit)
        - muting: none (0), moderate (0.3), aggressive (0.7)
        - combsplit: 1 (binary), 2 (linear 2-var), 5 (linear 5-var)
    
    Pour la CLASSIFICATION:
    - Modèles de référence: RF, ET, GBC, LogRegL2
    - RLT AVANCÉ: 9 variantes (même structure)
    
    Paramètres selon l'article Zhu et al. 2015 (Table 3):
    - n_estimators: 50-100 arbres
    - n_min: n^(1/3) ou 5 par défaut
    - p0: log(p) variables protégées (ADAPTATIF selon p)
    - muting: none (0), moderate (30%), aggressive (70%)
    - combsplit: 1, 2, 5 (linear combination splits)
    
    Paramètres
    ----------
    task_type : str
        'regression' ou 'classification'
    p : int, optional
        Nombre de features dans vos données
        Si fourni: p0 = log(p) (RECOMMANDÉ)
        Si None: p0 = 10 par défaut (NON OPTIMAL)
    random_state : int, default=42
        Seed pour reproductibilité
    
    Retourne
    --------
    dict
        {nom_modele: instance_sklearn}
    
    Exemples
    --------
    Avec p spécifié (RECOMMANDÉ):
    >>> models = get_benchmark_models('regression', p=200, random_state=42)
    INFO: p=200 → p0=5 selon article log(200)=5.30
    
    Sans p (pas recommandé):
    >>> models = get_benchmark_models('regression', random_state=42)
    WARN: p non spécifié, utilisation de p0=10 par défaut
    """
    
    # Bornes de sécurité
    if p is not None:
        p0 = compute_p0(p)
        print(f"INFO: p={p} → p0={p0} selon article (log({p})={np.log(p):.2f})")
    else:
        p0 = 10
        print(f"WARN: p non spécifié, utilisation de p0={p0} par défaut")
        print(f"      Recommandé: passer p pour p0=log(p) adaptatif")
        print(f"      Exemple: get_benchmark_models('regression', p=200)")
    
    models = {}
    
    if task_type == 'regression':
        # =====================================================================
        # MODÈLES DE RÉFÉRENCE (Baselines)
        # =====================================================================
        models['RF'] = make_rf_regressor(random_state=random_state)
        models['ET'] = make_et_regressor(random_state=random_state)
        models['GBR'] = make_gbr(random_state=random_state)
        models['Lasso'] = make_lasso(random_state=random_state)
        models['ElasticNet'] = make_elasticnet(random_state=random_state)
        
        # =====================================================================
        # RLT AVANCÉ - 9 VARIANTES (3 muting × 3 combsplit)
        # =====================================================================
        # Paramètres des stratégies:
        # - muting: 'none' (0%), 'moderate' (30%), 'aggressive' (70%)
        # - combsplit: 1 (binary), 2 (linear 2-var), 5 (linear 5-var)
        
        muting_strategies = {
            'none': {
                'muting': -1,
                'muting_percent': 0.0,
                'description': 'No muting, random splits'
            },
            'moderate': {
                'muting': -1,
                'muting_percent': 0.3,
                'description': 'Mute 30% of variables'
            },
            'aggressive': {
                'muting': -1,
                'muting_percent': 0.7,
                'description': 'Mute 70% of variables'
            }
        }
        
        combsplit_values = [1, 2, 5]  # Binary, 2-var combo, 5-var combo
        
        # Créer toutes les 9 combinaisons
        for muting_key, muting_params in muting_strategies.items():
            for combsplit in combsplit_values:
                name = f'RLT_{muting_key}_combsplit{combsplit}'
                
                models[name] = RLTRegressorRAdvanced(
                    n_estimators=100,           # 50-100 selon article
                    mtry=None,                  # auto = p/3
                    n_min=5,                    # n^(1/3) ou 5 par défaut
                    replace=True,               # Bootstrap avec remplacement
                    split_gen='random',         # Génération aléatoire des splits
                    
                    # AVANCÉ: Paramètres clés
                    combsplit=combsplit,        # 1, 2, 5
                    muting=muting_params['muting'],
                    muting_percent=muting_params['muting_percent'],
                    reinforcement=True,         # Apprentissage renforcé
                    importance=True,            # Calculer importance
                    protect=p0,                 # Protéger p0 variables
                    
                    random_state=random_state,
                    verbose=False
                )
    
    elif task_type == 'classification':
        # =====================================================================
        # MODÈLES DE RÉFÉRENCE (Baselines)
        # =====================================================================
        models['RF'] = make_rf_classifier(random_state=random_state)
        models['ET'] = make_et_classifier(random_state=random_state)
        models['GBC'] = make_gbc(random_state=random_state)
        models['LogRegL2'] = make_logreg_l2(random_state=random_state)
        
        # =====================================================================
        # RLT AVANCÉ - 9 VARIANTES (3 muting × 3 combsplit)
        # =====================================================================
        muting_strategies = {
            'none': {
                'muting': -1,
                'muting_percent': 0.0,
                'description': 'No muting'
            },
            'moderate': {
                'muting': -1,
                'muting_percent': 0.3,
                'description': 'Mute 30%'
            },
            'aggressive': {
                'muting': -1,
                'muting_percent': 0.7,
                'description': 'Mute 70%'
            }
        }
        
        combsplit_values = [1, 2, 5]
        
        # Créer toutes les 9 combinaisons
        for muting_key, muting_params in muting_strategies.items():
            for combsplit in combsplit_values:
                name = f'RLT_{muting_key}_combsplit{combsplit}'
                
                models[name] = RLTClassifierRAdvanced(
                    n_estimators=100,
                    mtry=None,
                    n_min=5,
                    replace=True,
                    split_gen='random',
                    
                    # AVANCÉ: Paramètres clés
                    combsplit=combsplit,
                    muting=muting_params['muting'],
                    muting_percent=muting_params['muting_percent'],
                    reinforcement=True,
                    importance=True,
                    protect=p0,
                    
                    random_state=random_state,
                    verbose=False
                )
    
    return models


def get_rlt_only(task_type: str, p: int = None, random_state: int = 42):
    """
    Retourne uniquement les modèles RLT (utile pour analyse spécifique).
    
    Paramètres
    ----------
    task_type : str
        'regression' ou 'classification'
    p : int, optional
        Nombre de features pour p0 adaptatif
    random_state : int
        Seed
    
    Retourne
    --------
    dict
        {nom_rlt: instance}
    
    Exemple
    -------
    >>> rlt_models = get_rlt_only('regression', p=200, random_state=42)
    >>> print(list(rlt_models.keys()))
    ['RLT_none_combsplit1', 'RLT_none_combsplit2', 'RLT_none_combsplit5',
     'RLT_moderate_combsplit1', 'RLT_moderate_combsplit2', 'RLT_moderate_combsplit5',
     'RLT_aggressive_combsplit1', 'RLT_aggressive_combsplit2', 'RLT_aggressive_combsplit5']
    """
    all_models = get_benchmark_models(task_type, p=p, random_state=random_state)
    return {k: v for k, v in all_models.items() if k.startswith('RLT')}


def get_best_rlt_config(task_type: str, p: int = None, p1_expected: int = None, 
                        scenario: str = 'general', random_state: int = 42):
    """
    Retourne la configuration RLT recommandée selon le scénario.
    
    Paramètres
    ----------
    task_type : str
        'regression' ou 'classification'
    p : int, optional
        Nombre de features pour calculer p0 automatiquement
    p1_expected : int, optional
        Nombre attendu de variables fortes (si connu)
        Scénarios de l'article: 2-4 variables fortes
    scenario : str, default='general'
        Type de problème:
        - 'general': Configuration par défaut (moderate, combsplit=2)
        - 'sparse': Données sparse/high-dim (aggressive, combsplit=1)
        - 'linear': Structure linéaire suspecte (moderate, combsplit=5)
        - 'nonlinear': Interactions complexes (moderate, combsplit=2)
    random_state : int
        Seed
    
    Retourne
    --------
    estimator
        Instance RLT configurée optimalement
    
    Exemples
    --------
    Pour des données sparse (p >> n, peu de variables fortes):
    >>> model = get_best_rlt_config('regression', p=500, scenario='sparse')
    
    Pour un problème linéaire:
    >>> model = get_best_rlt_config('regression', p=200, scenario='linear')
    
    Config générale:
    >>> model = get_best_rlt_config('regression', p=200)
    """
    
    if p is not None:
        p0 = compute_p0(p, p1_expected)
    else:
        p0 = 10
        print(f"WARN: p non spécifié pour get_best_rlt_config, p0={p0} par défaut")
    
    estimator_class = RLTRegressorRAdvanced if task_type == 'regression' else RLTClassifierRAdvanced
    
    # Configurations recommandées selon l'article
    configs = {
        'general': {
            'n_estimators': 100,
            'muting': -1,
            'muting_percent': 0.3,     # Moderate
            'combsplit': 2,
            'p0': p0,
            'n_min': 5,
        },
        'sparse': {
            'n_estimators': 100,
            'muting': -1,
            'muting_percent': 0.7,     # Aggressive (pour sparse)
            'combsplit': 1,             # Une seule variable à la fois
            'p0': max(3, p0 - 2),       # Moins de protection
            'n_min': 10,
        },
        'linear': {
            'n_estimators': 50,
            'muting': -1,
            'muting_percent': 0.3,
            'combsplit': 5,             # Combinaison linéaire de 5 variables
            'p0': min(p0 + 2, p // 5) if p else p0 + 2,  # Plus de protection
            'n_min': 5,
        },
        'nonlinear': {
            'n_estimators': 100,
            'muting': -1,
            'muting_percent': 0.3,
            'combsplit': 2,
            'p0': p0,
            'n_min': 3,                 # Noeuds plus petits pour capturer interactions
        }
    }
    
    config = configs.get(scenario, configs['general'])
    
    print(f"INFO: Scénario '{scenario}' → muting={config['muting_percent']}, "
          f"combsplit={config['combsplit']}, p0={config['p0']}")
    
    return estimator_class(
        n_estimators=config['n_estimators'],
        muting=config['muting'],
        muting_percent=config['muting_percent'],
        combsplit=config['combsplit'],
        protect=config['p0'],
        n_min=config['n_min'],
        reinforcement=True,
        importance=True,
        random_state=random_state
    )


def analyze_rlt_results(results_dict: dict, verbose=True):
    """
    Analyse les résultats des différentes variantes RLT.
    
    Affiche:
    - Meilleure configuration RLT
    - Impact du muting (moyenne par niveau)
    - Impact de combsplit (moyenne par valeur)
    - Comparaison avec les baselines
    
    Paramètres
    ----------
    results_dict : dict
        {nom_modele: score} pour tous les modèles testés
        Pour régression: score = MSE (plus petit meilleur)
        Pour classification: score = accuracy (plus grand meilleur)
    verbose : bool, default=True
        Afficher les détails
    
    Retourne
    --------
    dict
        Statistiques d'analyse:
        - best_rlt: (nom, score) du meilleur RLT
        - muting_stats: stats par niveau de muting
        - combsplit_stats: stats par valeur de combsplit
        - baseline_comparison: comparaison avec baselines
    
    Exemple
    -------
    >>> results = {
    ...     'RF': 8.35,
    ...     'RLT_none_combsplit2': 4.93,
    ...     'RLT_moderate_combsplit2': 3.43,
    ...     'RLT_aggressive_combsplit2': 2.66,
    ... }
    >>> stats = analyze_rlt_results(results)
    """
    
    rlt_results = {k: v for k, v in results_dict.items() if k.startswith('RLT')}
    baseline_results = {k: v for k, v in results_dict.items() if not k.startswith('RLT')}
    
    if not rlt_results:
        print("Aucun résultat RLT trouvé.")
        return
    
    # Meilleur RLT (minimiser le score pour régression, maximiser pour classification)
    best_rlt = min(rlt_results.items(), key=lambda x: x[1])
    
    if verbose:
        print("=" * 70)
        print("ANALYSE DES RÉSULTATS RLT".center(70))
        print("=" * 70)
        print(f"\nMeilleure config RLT: {best_rlt[0]}")
        print(f"Score: {best_rlt[1]:.4f}")
        print("=" * 70)
    
    # Impact du muting
    muting_stats = {}
    if verbose:
        print("\nIMPACT DU MUTING (moyenne par niveau):")
        print("-" * 70)
    
    for mut in ['none', 'moderate', 'aggressive']:
        scores = [v for k, v in rlt_results.items() if f'_{mut}_' in k]
        if scores:
            muting_stats[mut] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
            }
            if verbose:
                print(f"  {mut:12s}: μ={np.mean(scores):7.4f} σ={np.std(scores):7.4f} "
                      f"[{np.min(scores):.4f} - {np.max(scores):.4f}]")
    
    # Améliorations vs none
    if verbose and 'none' in muting_stats:
        print("\n  Améliorations vs 'none':")
        for mut in ['moderate', 'aggressive']:
            if mut in muting_stats:
                improvement = ((muting_stats['none']['mean'] - muting_stats[mut]['mean']) / 
                              muting_stats['none']['mean'] * 100)
                print(f"    {mut:12s}: +{improvement:.1f}%")
    
    # Impact de combsplit
    combsplit_stats = {}
    if verbose:
        print("\nIMPACT DE COMBSPLIT (moyenne par valeur):")
        print("-" * 70)
    
    for cs in [1, 2, 5]:
        scores = [v for key, v in rlt_results.items() if f'combsplit{cs}' in key]
        if scores:
            combsplit_stats[cs] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
            }
            if verbose:
                print(f"  combsplit{cs}: μ={np.mean(scores):7.4f} σ={np.std(scores):7.4f} "
                      f"[{np.min(scores):.4f} - {np.max(scores):.4f}]")
    
    # Comparaison avec baselines
    baseline_comparison = {}
    if baseline_results and verbose:
        print("\nCOMPARAISON AVEC BASELINES:")
        print("-" * 70)
        
        best_baseline = min(baseline_results.items(), key=lambda x: x[1])
        baseline_comparison = {
            'best_baseline_name': best_baseline[0],
            'best_baseline_score': best_baseline[1],
            'best_rlt_name': best_rlt[0],
            'best_rlt_score': best_rlt[1],
        }
        
        print(f"  Meilleure baseline: {best_baseline[0]:20s} = {best_baseline[1]:.4f}")
        print(f"  Meilleur RLT:       {best_rlt[0]:20s} = {best_rlt[1]:.4f}")
        
        if best_rlt[1] < best_baseline[1]:
            improvement = ((best_baseline[1] - best_rlt[1]) / best_baseline[1] * 100)
            print(f"\n  ✓ RLT GAGNE avec {improvement:.2f}% d'amélioration!")
            baseline_comparison['improvement'] = improvement
        else:
            degradation = ((best_rlt[1] - best_baseline[1]) / best_baseline[1] * 100)
            print(f"\n  ✗ Baseline gagne avec {degradation:.2f}% meilleure")
            baseline_comparison['improvement'] = -degradation
    
    if verbose:
        print("=" * 70)
    
    return {
        'best_rlt': best_rlt,
        'muting_stats': muting_stats,
        'combsplit_stats': combsplit_stats,
        'baseline_comparison': baseline_comparison,
    }
