"""Training-only model selection and held-out evaluation for anomaly detection."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from cluster import scaled_block, write_csv


HERE = Path(__file__).resolve().parent
CATEGORIES = ("hazelnut", "zipper")
SEED = 2026
FOLDS = 5
PER_IMAGE = 64
PATCH_FRACTIONS = (.01, .05, .15)
METRIC_NAMES = ("AUROC", "AP", "Accuracy", "Balanced_accuracy", "Precision", "Recall", "F1")


def coordinate_plan(keys, category):
    training = np.flatnonzero(np.char.startswith(keys, category + "/train/"))
    test = np.flatnonzero(np.char.startswith(keys, category + "/test/"))
    labels = np.array(["/bad/" in keys[i] for i in training], dtype=int)
    if np.bincount(labels).tolist() != [200, 50]:
        raise ValueError(f"Unexpected training counts for {category}")
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(training, labels))
    return training, test, labels, splits


def patch_distances(patch, references, *, cache, name):
    """One distance map per bank, with a fixed random patch sample per normal image."""
    digest = hashlib.sha256(patch.tobytes())
    digest.update(np.stack(references).astype("<i8").tobytes())
    digest.update(f"v1:{SEED}:{PER_IMAGE}:{patch.shape}".encode())
    filename = cache / f"{name}-{digest.hexdigest()[:16]}.npy"
    if filename.exists():
        return np.load(filename)
    generator = np.random.default_rng(SEED)
    n, height, width, channels = patch.shape
    flat = normalize(patch.reshape(n * height * width, channels).astype(np.float32))
    result = np.empty((FOLDS, len(flat)), dtype=np.float32)
    for fold, indices in enumerate(references):
        reference_rows = np.concatenate([
            index * height * width + generator.choice(height * width, PER_IMAGE, replace=False)
            for index in indices
        ])
        bank = np.ascontiguousarray(flat[reference_rows].T)
        for start in range(0, len(flat), 768):
            batch = flat[start : start + 768]
            result[fold, start : start + len(batch)] = 1 - (batch @ bank).max(axis=1)
    result = np.maximum(result, 0).reshape(FOLDS, n, height, width)
    np.save(filename, result)
    return result


def top_mean(maps, fraction):
    flat = maps.reshape(*maps.shape[:-2], -1)
    count = int(np.ceil(fraction * flat.shape[-1]))
    return np.partition(flat, -count, axis=-1)[..., -count:].mean(axis=-1)


def robust_parameters(scores):
    lo, midpoint, hi = np.percentile(scores, [25, 50, 75])
    return float(midpoint), float(max(hi - lo, 1e-8))


def standardize(values, parameters):
    return (values - parameters[0]) / parameters[1]


def svm_scores(values, training, test, labels, splits, c, gamma):
    oof = np.empty(len(training))
    held_out = []
    for fit, val in splits:
        svm = SVC(C=c, gamma=gamma, class_weight="balanced")
        svm.fit(values[training[fit]], labels[fit])
        oof[val] = svm.decision_function(values[training[val]])
        held_out.append(svm.decision_function(values[test]))
    return oof, np.mean(held_out, axis=0)


def validation_key(y, scores, splits):
    return [
        float(np.mean([metric(y[val], scores[val]) for _, val in splits]))
        for metric in (roc_auc_score, average_precision_score)
    ]


def get_threshold(labels, score):
    false_rate, true_rate, thresholds = roc_curve(labels, score, drop_intermediate=False)
    criterion = true_rate - false_rate
    candidates = np.flatnonzero(np.isclose(criterion, criterion.max(), rtol=0, atol=1e-12))
    finite = candidates[np.isfinite(thresholds[candidates])]
    return float(thresholds[finite[-1]] if len(finite) else np.nextafter(max(score), np.inf))


def fit(cache, out):
    data = np.load(cache / "anomaly.npz")
    keys = data["keys"]
    deep = np.hstack([
        np.sqrt(weight) * scaled_block(data[name])
        for name, weight in (
            ("avg2", .18), ("std2", .12), ("avg3", .22),
            ("std3", .13), ("avg4", .25), ("colour", .10),
        )
    ])
    shallow = np.hstack([np.sqrt(.85) * scaled_block(data["colour"]),
                         np.sqrt(.15) * normalize(data["texture"])])
    configurations = {}
    saved = {}
    for category in CATEGORIES:
        training, test, labels, splits = coordinate_plan(keys, category)
        selected = np.r_[training, test]
        bank_groups = [val[labels[val] == 0] for _, val in splits]
        patch_maps = []
        for layer in (1, 2):
            distances = patch_distances(
                data[f"patch{layer}"][selected], bank_groups,
                cache=cache, name=f"{category}_patch{layer}_distances",
            )
            # Each fold's patch bank is the union of the other four normal groups.
            fold_maps = np.stack([np.delete(distances, fold, axis=0).min(axis=0)
                                  for fold in range(FOLDS)])
            patch_maps.append(fold_maps)
            print(f"{category}: patch branch {layer} ready", flush=True)

        patch_candidates = []
        for fraction in PATCH_FRACTIONS:
            fold_scores = [top_mean(maps, fraction) for maps in patch_maps]
            oof_levels = np.zeros((2, len(training)))
            for fold, (_, val) in enumerate(splits):
                for branch in range(2):
                    oof_levels[branch, val] = fold_scores[branch][fold, val]
            test_levels = np.array([values[:, len(training):].mean(axis=0)
                                    for values in fold_scores])
            levels_calibration = [robust_parameters(level[labels == 0]) for level in oof_levels]
            oof = np.mean([standardize(level, cal)
                           for level, cal in zip(oof_levels, levels_calibration)], axis=0)
            score = np.mean([standardize(level, cal)
                             for level, cal in zip(test_levels, levels_calibration)], axis=0)
            patch_candidates.append((validation_key(labels, oof, splits), fraction,
                                     levels_calibration, oof, score))
        validation_patch, fraction, calibration, patch_oof, patch_test = max(
            patch_candidates, key=lambda row: row[0]
        )

        svm_candidates = []
        for c in (.5, 2., 8., 32.):
            for gamma in (.25, 1., 4.):
                oof, score = svm_scores(deep, training, test, labels, splits, c, gamma)
                svm_candidates.append((validation_key(labels, oof, splits), c, gamma, oof, score))
        validation_svm, c, gamma, svm_oof, svm_test = max(svm_candidates, key=lambda row: row[0])
        simple_oof, simple_test = svm_scores(shallow, training, test, labels, splits, 2., 1.)
        patch_scale = robust_parameters(patch_oof[labels == 0])
        svm_scale = robust_parameters(svm_oof[labels == 0])
        candidates = []
        for weight in (.5, .25, .75, 0., 1.):
            combined = weight * standardize(patch_oof, patch_scale) + (
                1 - weight
            ) * standardize(svm_oof, svm_scale)
            candidates.append((validation_key(labels, combined, splits), weight, combined))
        validation_fused, weight, combined = max(candidates, key=lambda row: row[0])
        combined_test = weight * standardize(patch_test, patch_scale) + (
            1 - weight
        ) * standardize(svm_test, svm_scale)
        methods = {
            "Handcrafted SVM": (simple_oof, simple_test),
            "Global SVM": (svm_oof, svm_test),
            "Patch memory": (patch_oof, patch_test),
            "Selected fusion": (combined, combined_test),
        }
        thresholds = {method: get_threshold(labels, oof) for method, (oof, _) in methods.items()}
        settings = {
            "patch_top_fraction": fraction,
            "patch_normal_median_iqr": calibration,
            "patch_fusion_median_iqr": patch_scale,
            "svm_C": c,
            "svm_gamma": gamma,
            "svm_fusion_median_iqr": svm_scale,
            "patch_weight": weight,
            "thresholds": thresholds,
            "mean_fold_AUROC_AP": {
                "Handcrafted SVM": validation_key(labels, simple_oof, splits),
                "Global SVM": validation_svm,
                "Patch memory": validation_patch,
                "Selected fusion": validation_fused,
            },
            "patch_search": [[param, value] for value, param, *_ in patch_candidates],
            "svm_search": [[c, gamma, value] for value, c, gamma, *_ in svm_candidates],
            "fusion_search": [[param, value] for value, param, *_ in candidates],
        }
        configurations[category] = settings
        print(category, json.dumps(settings), flush=True)
        saved[f"{category}_methods"] = np.array(list(methods))
        saved[f"{category}_scores"] = np.stack([test_score for _, test_score in methods.values()])
        saved[f"{category}_oof"] = np.stack([oof for oof, _ in methods.values()])
        saved[f"{category}_train_keys"] = keys[training]
        saved[f"{category}_train_labels"] = labels
        saved[f"{category}_test_keys"] = keys[test]
        saved[f"{category}_patch_map"] = np.maximum.reduce([
            standardize(maps[:, len(training):].mean(axis=0), cal)
            for maps, cal in zip(patch_maps, calibration)
        ])
    np.savez(out / "anomaly_unlabeled.npz", **saved)
    (out / "anomaly_model_selection.json").write_text(json.dumps(configurations, indent=2), encoding="utf-8")


def binary_metrics(y, score, prediction):
    return (
        roc_auc_score(y, score),
        average_precision_score(y, score),
        accuracy_score(y, prediction),
        balanced_accuracy_score(y, prediction),
        precision_score(y, prediction, zero_division=0),
        recall_score(y, prediction),
        f1_score(y, prediction, zero_division=0),
    )


def bootstrap(y, score, seed):
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    samples = []
    for _ in range(2000):
        index = np.r_[rng.choice(positive, size=len(positive)),
                      rng.choice(negative, size=len(negative))]
        samples.append((roc_auc_score(y[index], score[index]),
                        average_precision_score(y[index], score[index])))
    interval = np.percentile(samples, [2.5, 97.5], axis=0).T
    return interval.tolist()


def evaluate(root, out):
    data = np.load(out / "anomaly_unlabeled.npz")
    settings = json.loads((out / "anomaly_model_selection.json").read_text(encoding="utf-8"))
    label_path = root / "Image_Anomaly_Detection" / "Image_Anomaly_Detection" / "image_anomaly_labels.json"
    lookup = json.loads(label_path.read_text(encoding="utf-8"))
    expected = {key for category in CATEGORIES for key in data[f"{category}_test_keys"]}
    if expected != {key for key in lookup if "/test/" in key}:
        raise ValueError("Test labels and images do not match")
    comparisons, predictions, intervals = [], [], {}
    macro = defaultdict(list)
    for index, category in enumerate(CATEGORIES):
        keys = data[f"{category}_test_keys"]
        entries = [lookup[key] for key in keys]
        if any(entry["original_category"] != category or entry["label"] not in ("good", "bad")
               for entry in entries):
            raise ValueError(f"Invalid evaluation labels for {category}")
        y = np.array([entry["label"] == "bad" for entry in entries], dtype=int)
        for method, score in zip(data[f"{category}_methods"], data[f"{category}_scores"]):
            threshold = settings[category]["thresholds"][method]
            prediction = (score >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
            metrics = binary_metrics(y, score, prediction)
            macro[method].append(metrics)
            comparisons.append((category, method, *metrics, tn, fp, fn, tp, threshold))
            predictions.extend((category, key, method, float(prob), thres, int(pred), int(actual))
                               for key, prob, thres, pred, actual in
                               zip(keys, score, np.full(len(keys), threshold), prediction, y))
            if method == "Selected fusion":
                intervals[category] = bootstrap(y, score, SEED + index)
    for method, values in macro.items():
        comparisons.append(("macro", method, *np.mean(values, axis=0), "", "", "", "", ""))
    write_csv(out / "anomaly_metrics.csv",
              ("category", "method", *METRIC_NAMES, "TN", "FP", "FN", "TP", "threshold"), comparisons)
    write_csv(out / "anomaly_predictions.csv",
              ("category", "image", "method", "score", "threshold", "prediction", "actual"), predictions)
    (out / "anomaly_bootstrap_ci.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    for row in comparisons:
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parent)
    parser.add_argument("--cache", type=Path, default=HERE / "_work")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--stage", choices=("fit", "evaluate", "both"), default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=6):
        if args.stage != "evaluate":
            fit(args.cache, args.output)
        if args.stage != "fit":
            evaluate(args.root, args.output)


if __name__ == "__main__":
    main()
