"""Fit the selected polynomial models, assess the holdout, and predict test targets.

Run: python3 train.py --data-dir ./data
The configurations and CV scores below come from the completed narrow searches.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'local_results' / 'final'
TOL, MAX_ITER = 1e-7, 400000

FROZEN_MODELS = {
    1: {
        'config': {'degree': 5, 'method': 'elastic_net', 'alpha': 0.003125, 'l1_ratio': 1.0},
        'cv': {'pooled_oof_mse': 0.3715757098389658, 'pooled_oof_r2': 0.9625652659714968},
        'training_sha256': '83bbe4ed2e20076ae57604eb190aaf9033532130414bbc63418049c3a255cda1',
        'features': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
        'training_rows': 1000,
        'holdout': [
            2, 9, 20, 23, 26, 27, 28, 29, 31, 32, 51, 52, 58, 62, 67, 73,
            77, 91, 101, 104, 117, 119, 146, 148, 150, 151, 157, 160, 163, 164, 170, 178,
            180, 184, 185, 186, 188, 192, 194, 196, 199, 200, 209, 214, 215, 217, 224, 226,
            239, 240, 243, 244, 246, 253, 254, 260, 262, 268, 278, 283, 296, 301, 307, 330,
            331, 332, 336, 337, 341, 356, 361, 365, 371, 379, 383, 386, 387, 397, 404, 405,
            408, 410, 413, 416, 419, 420, 421, 422, 426, 434, 443, 446, 450, 460, 480, 481,
            482, 487, 497, 499, 500, 507, 511, 517, 534, 541, 550, 555, 557, 558, 559, 564,
            570, 578, 579, 582, 584, 590, 607, 608, 611, 623, 624, 632, 633, 634, 640, 646,
            650, 654, 656, 660, 661, 669, 673, 686, 688, 703, 712, 718, 721, 728, 730, 736,
            738, 740, 745, 750, 753, 755, 756, 757, 762, 774, 777, 788, 792, 794, 795, 800,
            803, 804, 807, 809, 820, 822, 824, 828, 831, 832, 841, 843, 848, 850, 851, 859,
            861, 870, 874, 875, 880, 882, 889, 890, 893, 895, 904, 913, 918, 929, 932, 950,
            955, 958, 969, 970, 976, 991, 992, 994,
        ],
    },
    2: {
        'config': {'degree': 11, 'method': 'elastic_net', 'alpha': 0.0002, 'l1_ratio': 0.9},
        'cv': {'pooled_oof_mse': 0.3306391141222827, 'pooled_oof_r2': 0.9924206152412208},
        'training_sha256': '8be2ec0acf9a5c1df5850ce7f386ad57f3ff10b14e2681a6ff00dc94561f4bb0',
        'features': ['x1', 'x2', 'x3'],
        'training_rows': 1000,
        'holdout': [
            2, 5, 7, 16, 22, 27, 29, 32, 33, 34, 35, 57, 58, 59, 63, 70,
            74, 78, 86, 88, 91, 93, 110, 111, 113, 121, 123, 133, 141, 154, 156, 157,
            162, 179, 180, 185, 193, 207, 208, 210, 211, 223, 225, 232, 235, 241, 244, 246,
            252, 255, 261, 264, 266, 267, 268, 281, 283, 284, 291, 301, 302, 303, 306, 315,
            322, 324, 343, 346, 347, 349, 353, 358, 372, 374, 376, 379, 382, 384, 387, 397,
            399, 408, 413, 414, 421, 425, 428, 429, 430, 431, 437, 443, 445, 449, 453, 456,
            482, 487, 490, 507, 510, 513, 514, 519, 530, 531, 546, 556, 559, 566, 567, 568,
            574, 580, 585, 597, 599, 604, 607, 608, 612, 619, 629, 632, 635, 639, 640, 642,
            647, 654, 658, 659, 664, 666, 672, 681, 691, 692, 697, 698, 708, 709, 718, 723,
            729, 731, 732, 735, 736, 738, 740, 752, 754, 757, 773, 775, 779, 783, 789, 797,
            800, 811, 820, 830, 835, 838, 843, 844, 850, 856, 860, 867, 872, 876, 877, 878,
            884, 891, 903, 908, 909, 910, 913, 918, 920, 922, 924, 926, 929, 930, 940, 942,
            944, 948, 955, 964, 967, 969, 970, 971, 985, 988, 997, 999,
        ],
    },
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PolynomialRegressor:
    """Scale inputs, expand polynomial features, and fit Elastic Net/Lasso."""

    def __init__(self, degree, method, alpha, l1_ratio, tol=TOL, max_iter=MAX_ITER):
        if method != 'elastic_net' or not np.isfinite(alpha) or alpha <= 0 or not 0 < l1_ratio <= 1:
            raise ValueError('Expected a positive-alpha Elastic Net/Lasso configuration.')
        self.degree, self.alpha, self.l1_ratio = int(degree), float(alpha), float(l1_ratio)
        self.tol, self.max_iter = tol, max_iter

    def fit(self, x, y):
        self.raw_scaler = StandardScaler().fit(x)
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        expanded = self.poly.fit_transform(self.raw_scaler.transform(x))
        self.feature_scaler = StandardScaler().fit(expanded)
        a = self.feature_scaler.transform(expanded)
        if not np.isfinite(a).all():
            raise ValueError('Non-finite polynomial training features.')
        self.ymean, self.yscale = float(np.mean(y)), float(np.std(y)) or 1.0
        target = (y - self.ymean) / self.yscale
        self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=False,
                                tol=self.tol, max_iter=self.max_iter)
        self.model.fit(a, target)
        gap = float(self.model.dual_gap_)
        threshold = self.tol * float(np.mean(target**2))
        if self.model.n_iter_ >= self.max_iter or not np.isfinite(gap) or gap > threshold:
            raise RuntimeError('The selected model did not converge; increase MAX_ITER.')
        self.diagnostics = dict(converged=True, n_iter=int(self.model.n_iter_), dual_gap=gap)
        return self

    def predict(self, x):
        expanded = self.poly.transform(self.raw_scaler.transform(x))
        a = self.feature_scaler.transform(expanded)
        if not np.isfinite(a).all():
            raise ValueError('Non-finite polynomial prediction features.')
        prediction = self.model.predict(a) * self.yscale + self.ymean
        if not np.isfinite(prediction).all():
            raise ValueError('Non-finite predictions.')
        return prediction


def load_inputs(data_dir, problem):
    frozen = FROZEN_MODELS[problem]
    train_path = data_dir / f'BT2024134_train_var{problem}.csv'
    test_path = data_dir / f'BT2024134_test_var{problem}.csv'
    if file_hash(train_path) != frozen['training_sha256']:
        raise ValueError(f'var{problem}: training CSV differs from the reviewed narrow search.')
    data, test = pd.read_csv(train_path), pd.read_csv(test_path)
    features = frozen['features']
    if 'y' not in data or data.columns.drop('y').tolist() != features or set(test.columns) != set(features):
        raise ValueError(f'var{problem}: training/test features differ from the saved schema.')
    if len(data) != frozen['training_rows'] or len(test) == 0:
        raise ValueError(f'var{problem}: unexpected training row count or empty test data.')
    x, y = data[features].to_numpy(dtype=float), data.y.to_numpy(dtype=float)
    xtest = test[features].to_numpy(dtype=float)
    if not all(np.isfinite(a).all() for a in (x, y, xtest)):
        raise ValueError(f'var{problem}: non-finite training/test values.')
    holdout = np.asarray(frozen['holdout'], dtype=int)
    if len(holdout) < 2 or len(set(holdout)) != len(holdout) or np.any((holdout < 0) | (holdout >= len(y))):
        raise ValueError(f'var{problem}: invalid frozen holdout indices.')
    development = np.setdiff1d(np.arange(len(y)), holdout)
    if len(development) < 2:
        raise ValueError(f'var{problem}: insufficient development rows.')
    return x, y, xtest, development, holdout, file_hash(test_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--problem', type=int, choices=(1, 2), help='Fit only this problem; default: both.')
    args = parser.parse_args()
    problems = [args.problem] if args.problem else [1, 2]
    metrics_path = args.output_dir / 'metrics.json'
    prediction_paths = {p: args.output_dir / f'BT2024134_pred_var{p}.csv' for p in problems}
    if any(path.exists() for path in [metrics_path, *prediction_paths.values()]):
        raise ValueError('Outputs already exist. Use a new --output-dir to preserve them.')
    inputs = {p: load_inputs(args.data_dir, p) for p in problems}
    report = dict(status='complete', solver=dict(tol=TOL, max_iter=MAX_ITER), problems={})
    outputs = {}
    for problem in problems:
        key, frozen = f'var{problem}', FROZEN_MODELS[problem]
        x, y, xtest, dev, holdout, test_hash = inputs[problem]
        print(f'{key}: fitting {frozen["config"]}')
        assessment = PolynomialRegressor(**frozen['config']).fit(x[dev], y[dev])
        prediction = assessment.predict(x[holdout])
        holdout_scores = dict(mse=float(mean_squared_error(y[holdout], prediction)),
                              r2=float(r2_score(y[holdout], prediction)))
        print(f'{key}: holdout MSE={holdout_scores["mse"]:.8g}, R2={holdout_scores["r2"]:.8g}')
        final = PolynomialRegressor(**frozen['config']).fit(x, y)
        outputs[problem] = final.predict(xtest)
        report['problems'][key] = dict(
            status='complete', config=frozen['config'], cv=frozen['cv'],
            training_sha256=frozen['training_sha256'], test_sha256=test_hash,
            development_rows=len(dev), holdout_rows=len(holdout), holdout=holdout_scores,
            development_fit=assessment.diagnostics, final_fit=final.diagnostics,
            prediction_rows=len(xtest),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for problem, predictions in outputs.items():
        path = prediction_paths[problem]
        pd.DataFrame({'y': predictions}).to_csv(path, index=False, float_format='%.17g')
        report['problems'][f'var{problem}']['prediction_sha256'] = file_hash(path)
        print(f'var{problem}: saved {path}')
    metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {metrics_path}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(str(exc))
