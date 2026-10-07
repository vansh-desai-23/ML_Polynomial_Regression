"""Refine var1 models using the same grouped folds as the broad search.

Run: python3 var1_narrow_search.py --data-dir ./data
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from explore import (
    DATA_DIR, OUTPUT_DIR, candidate_id, configuration, file_hash,
    load_data, run_search, validate_splits,
)

SOURCE_DIR = OUTPUT_DIR / 'broad'
RUN_DIR = OUTPUT_DIR / 'var1_narrow'
TOL = 1e-7
MAX_ITER = 400000
MAX_FEATURES = 20000
MAX_MATRIX_MB = 512

SEARCH_GRIDS = {
    5: {
        "alphas": [0.00225, 0.0025, 0.00275, 0.003, 0.003125,
                   0.00325, 0.0035, 0.00375, 0.004],
        "l1_ratios": [0.75, 0.85, 0.90, 0.95, 0.975, 1.0],
    },
    6: {
        "alphas": [0.0025, 0.003, 0.00325, 0.0035, 0.00375, 0.004, 0.0045],
        "l1_ratios": [0.85, 0.90, 0.95, 0.975, 1.0],
    },
}

EXTRA_CONFIGS = [
    {"degree": 5, "method": "elastic_net", "alpha": 0.006, "l1_ratio": 0.50},
    {"degree": 6, "method": "elastic_net", "alpha": 0.0035, "l1_ratio": 0.95},
]


def validate_configuration(config):
    degree = config['degree']
    if isinstance(degree, bool) or int(degree) != degree or not 1 <= degree <= 10:
        raise ValueError('Var1 degrees must be integers from 1 through 10.')
    alpha, ratio, method = config['alpha'], config['l1_ratio'], config['method']
    if method == 'ols':
        if alpha != 0 or ratio is not None:
            raise ValueError('OLS requires alpha=0 and l1_ratio=None.')
    elif method == 'elastic_net':
        if not np.isfinite(alpha) or alpha <= 0 or ratio is None or not np.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError('Elastic Net requires finite alpha > 0 and 0 <= l1_ratio <= 1.')
    else:
        raise ValueError(f'Unsupported regression method: {method}')


def make_candidates(source, scores):
    var1 = scores[scores.problem == 1].copy()
    eligible = var1[(var1.status == 'ok') & (var1.completed_folds == 5)
                    & (var1.failed_folds == 0) & (var1.unconverged_folds == 0)
                    & np.isfinite(var1.pooled_oof_mse)].sort_values('pooled_oof_mse')
    if eligible.empty:
        raise ValueError('No complete, converged var1 broad candidates.')
    exact = {c['id']: c for c in source['candidates'] if c['problem'] == 1}
    candidates = {}

    def add(config):
        validate_configuration(config)
        config = configuration(config)
        key = candidate_id(1, config)
        candidates[key] = dict(id=key, problem=1, **config)

    def retain(row):
        original = exact[row.id]
        if configuration(original) != configuration(row):
            raise ValueError('Broad experiment and screening configurations disagree.')
        add(original)

    for _, row in eligible.drop_duplicates('degree').iterrows():
        retain(row)
    families = eligible.assign(family=eligible.apply(
        lambda r: 'ols' if r.method == 'ols' else 'ridge' if r.l1_ratio == 0 else 'lasso' if r.l1_ratio == 1 else 'mixed', axis=1))
    for _, row in families.drop_duplicates('family').iterrows():
        retain(row)
    for config in EXTRA_CONFIGS:
        add(config)
    if not SEARCH_GRIDS:
        raise ValueError('SEARCH_GRIDS must contain at least one local grid.')
    for degree, grid in SEARCH_GRIDS.items():
        if not grid['alphas'] or not grid['l1_ratios']:
            raise ValueError(f'Degree {degree}: alpha and ratio lists must be nonempty.')
        for alpha in grid['alphas']:
            for ratio in grid['l1_ratios']:
                add(dict(degree=degree, method='elastic_net', alpha=alpha, l1_ratio=ratio))
    return sorted(candidates.values(), key=lambda c: (c['degree'], c['id']))


def prepare_experiment(args):
    if not np.isfinite(TOL) or TOL <= 0 or MAX_ITER < 1:
        raise ValueError('Invalid tolerance or iteration limit.')
    if MAX_FEATURES < 1 or not np.isfinite(MAX_MATRIX_MB) or MAX_MATRIX_MB <= 0:
        raise ValueError('Invalid memory limits.')
    source = json.loads((args.source / 'experiment.json').read_text())
    selected = json.loads((args.source / 'selected_models.json').read_text())
    if selected.get('status') != 'complete':
        raise ValueError('The source broad search must be complete.')
    if '1' not in source['data']:
        raise ValueError('The source search does not contain var1.')
    scores = pd.read_csv(args.source / 'screening.csv', float_precision='round_trip')
    x, y, features = load_data(args.data_dir, 1)
    prior = source['data']['1']
    if file_hash(args.data_dir / 'BT2024134_train_var1.csv') != prior['sha256']:
        raise ValueError('Var1 training CSV differs from the broad-search input.')
    if features != prior['features']:
        raise ValueError('Var1 feature order differs from the broad search.')
    validate_splits(prior['split'], x)
    candidates = make_candidates(source, scores)
    return dict(stage='refine', problems=[1],
                data={'1': {**prior, 'development_variance': float(np.var(y[prior['split']['development']]))}},
                source_directory=str(args.source),
                source_sha256={name: file_hash(args.source / name)
                               for name in ('experiment.json', 'screening.csv', 'selected_models.json')},
                solver=dict(tol=TOL, max_iter=MAX_ITER,
                            max_features=MAX_FEATURES, max_matrix_mb=MAX_MATRIX_MB),
                candidates=candidates)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE_DIR)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--run-dir', type=Path, default=RUN_DIR)
    args = parser.parse_args()
    if args.run_dir.resolve() == args.source.resolve():
        parser.error('Use a new output directory, separate from the broad results.')
    return args


if __name__ == '__main__':
    try:
        args = parse_args()
        raise SystemExit(run_search(args, prepare_experiment(args)))
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(str(exc))
