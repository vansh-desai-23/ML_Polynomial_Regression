"""Broad search of polynomial regression models using grouped five-fold validation.

Run from a terminal: python3 explore.py --data-dir ./data
Search results are saved after all candidates have been evaluated.
The narrow-search scripts import the shared fitting and scoring functions here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'local_results'
TOL = 1e-5
MAX_ITER = 30000
MAX_FEATURES = 20000
MAX_MATRIX_MB = 512
CV_KEYS = ('pooled_oof_mse', 'pooled_oof_r2', 'mean_fold_mse', 'std_fold_mse')
SVD_RCOND = 1e-12

BROAD_DEGREES = {1: [4, 5, 6], 2: [10, 11, 12, 13]}
BROAD_ALPHAS = {
    1: np.array([3e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3,
                 3.5e-3, 4e-3, 6e-3, 1e-2, 3e-2, 1e-1]),
    2: np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4,
                 8e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 1e-2]),
}
BROAD_L1_RATIOS = {
    1: [0.0, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 1.0],
    2: [0.0, 0.25, 0.40, 0.50, 0.56, 0.60, 0.75, 0.90, 1.0],
}
RIDGE_ALPHAS = np.logspace(-6, 2, 9)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def split_data(x):
    """Reserve 20% of original-X groups; the row fraction can differ slightly."""
    _, groups = np.unique(x, axis=0, return_inverse=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=134)
    return next(splitter.split(x, groups=groups))


def development_folds(x):
    """Create one set of group-safe folds shared by every configuration."""
    _, groups = np.unique(x, axis=0, return_inverse=True)
    if len(np.unique(groups)) < 5:
        raise ValueError("Need at least 5 development input groups for GroupKFold.")
    return list(GroupKFold(n_splits=5).split(x, groups=groups))


def fit_preprocessing(x, degree):
    """Fit raw scaler -> polynomial expansion -> polynomial scaler on train only."""
    raw_scaler = StandardScaler().fit(x)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    z = poly.fit_transform(raw_scaler.transform(x))
    scaler = StandardScaler().fit(z)
    a = scaler.transform(z)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite polynomial training features.")
    return (raw_scaler, poly, scaler), a


def transform_preprocessing(x, preprocessing):
    raw_scaler, poly, scaler = preprocessing
    z = poly.transform(raw_scaler.transform(x))
    a = scaler.transform(z)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite polynomial prediction features.")
    return a


def standardize_target(y):
    mean, scale = float(np.mean(y)), float(np.std(y)) or 1.0
    return (y - mean) / scale, mean, scale


def configuration(row):
    return {"degree": int(row["degree"]), "method": str(row["method"]),
            "alpha": float(row["alpha"]),
            "l1_ratio": None if row["method"] == "ols" else float(row["l1_ratio"])}


def candidate_id(problem, config):
    payload = json.dumps([int(problem), configuration(config)], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def best_candidate(rows):
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No completed, converged candidates.")
    candidates['pooled_oof_mse'] = pd.to_numeric(candidates.pooled_oof_mse, errors='coerce')
    candidates = candidates[
        np.isfinite(candidates.pooled_oof_mse)
        & (candidates.unconverged_folds == 0)
    ]
    if 'completed_folds' in candidates:
        candidates = candidates[candidates.completed_folds == 5]
    if 'failed_folds' in candidates:
        candidates = candidates[candidates.failed_folds == 0]
    if candidates.empty:
        raise ValueError("No completed, converged candidates.")
    return candidates.sort_values(
        ['pooled_oof_mse', 'degree', 'alpha'], ascending=[True, True, False]
    ).iloc[0]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_data(data_dir, problem):
    frame = pd.read_csv(Path(data_dir) / f'BT2024134_train_var{problem}.csv')
    if 'y' not in frame or len(frame.columns) < 2:
        raise ValueError(f'var{problem}: require y and at least one feature.')
    features = frame.columns.drop('y').tolist()
    x, y = frame[features].to_numpy(dtype=float), frame.y.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f'var{problem}: non-finite training values.')
    return x, y, features


def validate_splits(split, x):
    """Validate saved fold membership without generating any replacement split."""
    def indices(values, size):
        a = np.asarray(values)
        if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or len(a) == 0:
            raise ValueError('Saved split indices must be nonempty integer arrays.')
        if len(np.unique(a)) != len(a) or np.any(a < 0) or np.any(a >= size):
            raise ValueError('Saved split contains duplicate or out-of-range indices.')
        return a
    dev, holdout = (indices(split[name], len(x)) for name in ('development', 'holdout'))
    if np.intersect1d(dev, holdout).size or len(dev)+len(holdout) != len(x):
        raise ValueError('Saved development/holdout split is not a partition.')
    _, groups = np.unique(x, axis=0, return_inverse=True)
    if np.intersect1d(groups[dev], groups[holdout]).size:
        raise ValueError('Identical feature rows cross the saved development/holdout split.')
    if len(split['folds']) != 5:
        raise ValueError('Expected five saved development folds.')
    validations = []
    for fold in split['folds']:
        tr, va = (indices(fold[name], len(dev)) for name in ('train', 'valid'))
        if np.intersect1d(tr, va).size or len(tr)+len(va) != len(dev):
            raise ValueError('Saved CV fold is not a partition of development rows.')
        if np.intersect1d(groups[dev[tr]], groups[dev[va]]).size:
            raise ValueError('Identical feature rows cross a saved CV fold.')
        validations.extend(va)
    if sorted(validations) != list(range(len(dev))):
        raise ValueError('Each development row must appear in validation exactly once.')


def fit_coefficients(a, target, config, solver):
    """Fit OLS, Ridge, Lasso or Elastic Net to standardized features and target."""
    alpha, ratio = config['alpha'], config['l1_ratio']
    if config['method'] == 'ols' or ratio == 0:
        u, singular_values, vt = svd(a, full_matrices=False)
        if config['method'] == 'ols':
            cutoff = singular_values.max() * SVD_RCOND
            weights = np.divide(1.0, singular_values, out=np.zeros_like(singular_values),
                                where=singular_values > cutoff)
        else:
            weights = singular_values / (singular_values**2 + len(a) * alpha)
        coefficients = vt.T @ (weights * (u.T @ target))
        return coefficients, dict(converged=True, n_iter=0, dual_gap=0.0)

    model = ElasticNet(alpha=alpha, l1_ratio=ratio, fit_intercept=False,
                       max_iter=solver['max_iter'], tol=solver['tol'])
    model.fit(a, target)
    gap = float(model.dual_gap_)
    threshold = solver['tol'] * float(np.mean(target**2))
    converged = model.n_iter_ < solver['max_iter'] and np.isfinite(gap) and gap <= threshold
    return model.coef_, dict(converged=bool(converged), n_iter=int(model.n_iter_), dual_gap=gap)


def evaluate_candidate(x, y, candidate, split, solver):
    """Fit each fold independently; fit preprocessing only on its training rows."""
    development = np.asarray(split['development'])
    fold_results = []
    for number, fold in enumerate(split['folds']):
        train = development[np.asarray(fold['train'])]
        valid = development[np.asarray(fold['valid'])]
        result = dict(id=candidate['id'], fold=number, n_valid=len(valid))
        try:
            width = math.comb(x.shape[1] + candidate['degree'], candidate['degree']) - 1
            if width > solver['max_features'] or 4 * (len(train) + len(valid)) * width * 8 > solver['max_matrix_mb'] * 1024**2:
                raise ValueError(f'Polynomial expansion exceeds memory limit ({width} features).')
            preprocessing, a = fit_preprocessing(x[train], candidate['degree'])
            b = transform_preprocessing(x[valid], preprocessing)
            target, mean, scale = standardize_target(y[train])
            coefficients, diagnostics = fit_coefficients(a, target, candidate, solver)
            predictions = (b @ coefficients) * scale + mean
            mse = float(np.mean((y[valid] - predictions)**2))
            if not np.isfinite(mse) or not np.isfinite(diagnostics['dual_gap']):
                raise ValueError('Non-finite loss or dual gap.')
            result.update(status='ok' if diagnostics['converged'] else 'unconverged',
                          mse=mse, n_iter=diagnostics['n_iter'], dual_gap=diagnostics['dual_gap'])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            result.update(status='failed', error=str(exc))
        fold_results.append(result)

    failed = sum(f['status'] == 'failed' for f in fold_results)
    unconverged = sum(f['status'] == 'unconverged' for f in fold_results)
    row = dict(candidate, completed_folds=len(fold_results), failed_folds=failed,
               unconverged_folds=unconverged,
               status='failed' if failed else 'unconverged' if unconverged else 'ok',
               error=next((f['error'] for f in fold_results if 'error' in f), ''),
               **dict.fromkeys(CV_KEYS))
    row.update({f'fold{i+1}_mse': f.get('mse') for i, f in enumerate(fold_results)})
    if not failed:
        losses = np.array([f['mse'] for f in fold_results])
        mse = float(np.average(losses, weights=[f['n_valid'] for f in fold_results]))
        variance = float(np.var(y[development]))
        row.update(pooled_oof_mse=mse, pooled_oof_r2=1-mse/variance if variance else float(mse == 0),
                   mean_fold_mse=float(losses.mean()), std_fold_mse=float(losses.std()))
    return row, fold_results


def run_search(args, experiment):
    """Evaluate configurations sequentially and write the final result tables once."""
    args.run_dir.mkdir(parents=True, exist_ok=False)
    rows, folds = [], []
    data = {problem: load_data(args.data_dir, problem) for problem in experiment['problems']}
    total = len(experiment['candidates'])
    for number, candidate in enumerate(experiment['candidates'], start=1):
        problem = candidate['problem']
        x, y, _ = data[problem]
        row, fold_results = evaluate_candidate(
            x, y, candidate, experiment['data'][str(problem)]['split'], experiment['solver'])
        rows.append(row)
        folds.extend(fold_results)
        score = f"{row['pooled_oof_mse']:.8g}" if row['pooled_oof_mse'] is not None else 'unavailable'
        print(f"[{number}/{total}] var{problem} degree={candidate['degree']} "
              f"alpha={candidate['alpha']:g} l1_ratio={candidate['l1_ratio']} "
              f"CV MSE={score} ({row['status']})")

    models = {}
    for problem in experiment['problems']:
        try:
            best = best_candidate([row for row in rows if row['problem'] == problem])
        except ValueError:
            print(f'var{problem}: no fully converged candidate.')
            continue
        models[f'var{problem}'] = {
            **configuration(best), **{key: float(best[key]) for key in CV_KEYS},
            'selection_metric': 'pooled_oof_mse', 'unconverged_folds': 0,
        }
    pd.DataFrame(rows).to_csv(args.run_dir / 'screening.csv', index=False, float_format='%.17g')
    pd.DataFrame(folds).to_csv(args.run_dir / 'fold_scores.csv', index=False, float_format='%.17g')
    write_json(args.run_dir / 'experiment.json', experiment)
    write_json(args.run_dir / 'selected_models.json',
               dict(status='complete', stage=experiment['stage'], models=models))
    print(f'Saved results to {args.run_dir}')
    if len(models) != len(experiment['problems']):
        raise ValueError('Search finished, but at least one problem has no converged model.')
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=['1', '2', 'both'], default='both')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--run-dir', type=Path, default=OUTPUT_DIR / 'broad')
    return parser.parse_args()


def build_candidates(problems):
    """Use the grids above, adding OLS and the Ridge/Lasso endpoints."""
    candidates = {}

    def add(problem, degree, method='ols', alpha=0.0, ratio=None):
        config = dict(degree=int(degree), method=method, alpha=float(alpha), l1_ratio=ratio)
        key = candidate_id(problem, config)
        candidates[key] = dict(id=key, problem=int(problem), **config)

    for problem in problems:
        degrees = BROAD_DEGREES[problem]
        alphas = BROAD_ALPHAS[problem]
        ratios = BROAD_L1_RATIOS[problem]
        limit = 10 if problem == 1 else 20
        if not degrees or any(int(d) != d or not 1 <= d <= limit for d in degrees):
            raise ValueError(f'var{problem}: degrees must be integers from 1 through {limit}.')
        for grid in (alphas, RIDGE_ALPHAS):
            if len(grid) == 0 or any(not np.isfinite(a) or a <= 0 for a in grid):
                raise ValueError('Alpha grids must contain positive, finite values.')
        if not ratios or any(not np.isfinite(r) or not 0 <= r <= 1 for r in ratios):
            raise ValueError('L1 ratios must be between 0 and 1.')
        for degree in degrees:
            add(problem, degree)
            for ratio in ratios:
                grid = RIDGE_ALPHAS if ratio == 0 else alphas
                for alpha in grid:
                    add(problem, degree, 'elastic_net', alpha, float(ratio))
    return sorted(candidates.values(), key=lambda c: (c['degree'], c['problem'], c['id']))


def prepare_experiment(args):
    if not np.isfinite(TOL) or TOL <= 0 or MAX_ITER < 1:
        raise ValueError('Invalid tolerance or iteration limit.')
    if MAX_FEATURES < 1 or not np.isfinite(MAX_MATRIX_MB) or MAX_MATRIX_MB <= 0:
        raise ValueError('Invalid memory limits.')
    problems = [1, 2] if args.problem == 'both' else [int(args.problem)]
    data = {}
    for problem in problems:
        x, y, features = load_data(args.data_dir, problem)
        dev, holdout = split_data(x)
        split = dict(development=dev.tolist(), holdout=holdout.tolist(),
                     folds=[dict(train=tr.tolist(), valid=va.tolist())
                            for tr, va in development_folds(x[dev])])
        data[str(problem)] = dict(
            sha256=file_hash(args.data_dir / f'BT2024134_train_var{problem}.csv'),
            features=features, split=split, development_variance=float(np.var(y[dev])))
    return dict(stage='broad', problems=problems, data=data,
                solver=dict(tol=TOL, max_iter=MAX_ITER,
                            max_features=MAX_FEATURES, max_matrix_mb=MAX_MATRIX_MB),
                candidates=build_candidates(problems))


if __name__ == '__main__':
    try:
        args = parse_args()
        raise SystemExit(run_search(args, prepare_experiment(args)))
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(str(exc))
