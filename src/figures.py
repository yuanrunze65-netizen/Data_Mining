"""Rebuild report figures from the frozen predictions and evaluation files."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve


from image_common import DATA_ROOT, PROJECT_ROOT, RESULTS_DIR
FIGURE_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#724E91", "#D55E00"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, filename):
    fig.savefig(filename, bbox_inches="tight", pad_inches=.045)
    plt.close(fig)


def clustering(root, results, output):
    rows = read_csv(results / "cluster_confusion.csv")
    labels = [row["actual/predicted"] for row in rows]
    confusion = np.array([[int(row[label]) for label in labels] for row in rows])
    data = np.load(results / "cluster_unlabeled.npz")
    rows = read_csv(results / "cluster_predictions.csv")
    labels_mapped = np.array([row["mapped_class"] for row in rows])
    points = data["plot_points"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.7, 2.55), layout="constrained",
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    ax0.imshow(confusion, cmap="Blues", vmin=0, vmax=100)
    ax0.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels,
            xlabel="Matched cluster", ylabel="True class", title="(a) Confusion matrix")
    ax0.tick_params(axis="x", rotation=35)
    for (row, col), value in np.ndenumerate(confusion):
        ax0.text(col, row, value, ha="center", va="center",
                 color="white" if value > 59 else "#142436")
    for label, colour in zip(labels, FIGURE_COLOURS):
        keep = labels_mapped == label
        ax1.scatter(*points[keep].T, s=7, color=colour, alpha=.65, lw=0, label=label)
    ax1.set(xlabel="PC 1", ylabel="PC 2", title="(b) Hybrid features")
    ax1.legend(loc="upper left", frameon=True, fontsize=6.5, ncol=2)
    ax1.grid(alpha=.2)
    save(fig, output / "cluster_diagnostics.pdf")

    fig, axs = plt.subplots(2, 6, figsize=(6.7, 2.12), layout="constrained")
    base = root / "Cluster" / "Cluster"
    for col, name in enumerate(labels):
        first_member = np.flatnonzero(labels_mapped == name)[0]
        group = data["Hybrid spectral"][first_member]
        nearest = data["representatives"][group]
        for row, index in enumerate(nearest):
            ax = axs[row, col]
            with Image.open(base / data["keys"][index]) as image:
                ax.imshow(image.convert("RGB").resize((130, 130)))
            ax.set_axis_off()
        axs[0, col].set_title(name, pad=1)
    save(fig, output / "cluster_exemplars.pdf")


def anomaly(root, results, output):
    rows = read_csv(results / "anomaly_predictions.csv")
    data = np.load(results / "anomaly_unlabeled.npz")
    categories = ("hazelnut", "zipper")
    methods = ("Handcrafted SVM", "Global SVM", "Patch memory", "Selected fusion")
    style = ("--", "-.", ":", "-")
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.8), layout="constrained")
    for col, category in enumerate(categories):
        for method, line, colour in zip(methods, style, FIGURE_COLOURS):
            items = [row for row in rows if row["category"] == category and row["method"] == method]
            y = [int(row["actual"]) for row in items]
            scores = [float(row["score"]) for row in items]
            fpr, tpr, _ = roc_curve(y, scores)
            precision, recall, _ = precision_recall_curve(y, scores)
            axes[0, col].plot(fpr, tpr, line, color=colour, label=method)
            axes[1, col].step(recall, precision, where="post", ls=line, color=colour)
        axes[0, col].plot([0, 1], [0, 1], color=".65", lw=.6)
        axes[1, col].axhline(float(np.mean(y)), color=".65", lw=.6)
        axes[0, col].set(title=category, xlabel="False positive rate", ylabel="True positive rate")
        axes[1, col].set(xlabel="Recall", ylabel="Precision")
        for ax in axes[:, col]:
            ax.set(xlim=(-.023, 1.023), ylim=(-.023, 1.023))
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.grid(alpha=.15)
    axes[0, 0].legend(loc="lower right", fontsize=6.2)
    save(fig, output / "anomaly_curves.pdf")

    base = root / "Image_Anomaly_Detection" / "Image_Anomaly_Detection"
    fig, axes = plt.subplots(2, 4, figsize=(6.7, 3.08), layout="constrained")
    for row_num, category in enumerate(categories):
        entries = [row for row in rows if row["category"] == category and row["method"] == methods[-1]]
        errors = [entry for entry in entries if entry["actual"] != entry["prediction"]]
        examples = []
        for label in ("0", "1"):
            pool = [entry for entry in errors if entry["actual"] == label]
            if not pool:
                pool = [entry for entry in entries if entry["actual"] == label]
            examples.append(min(pool, key=lambda e: abs(float(e["score"]) - float(e["threshold"]))))
        for col, entry in enumerate(examples):
            key = entry["image"]
            keys = data[f"{category}_test_keys"].tolist()
            spatial_map = data[f"{category}_patch_map"][keys.index(key)]
            level = np.maximum(spatial_map, 0)
            level /= max(np.percentile(level, 99), .001)
            with Image.open(base / key) as image:
                picture = image.convert("RGB").resize((240, 240), Image.Resampling.BILINEAR)
            layer = Image.fromarray(level.astype(np.float32)).resize(picture.size, Image.Resampling.BILINEAR)
            left, right = axes[row_num, 2 * col:2 * col + 2]
            left.imshow(picture)
            right.imshow(picture)
            values = np.clip(np.array(layer), 0, 1)
            right.imshow(values, cmap="inferno", vmin=0, vmax=1, alpha=.63 * values)
            outcome = {("0", "0"): "TN", ("0", "1"): "FP", ("1", "0"): "FN", ("1", "1"): "TP"}
            left.set_title(f"{category} {Path(key).stem} ({outcome[entry['actual'], entry['prediction']]})", pad=1)
            right.set_title("Patch score", pad=1)
            left.set_axis_off()
            right.set_axis_off()
    save(fig, output / "anomaly_examples.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "report" / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font_path = DATA_ROOT / "course-report-template" / "fonts" / "times.ttf"
    font_manager.fontManager.addfont(font_path)
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": font, "font.size": 8, "pdf.fonttype": 42,
                         "axes.linewidth": .6, "lines.linewidth": 1.1,
                         "xtick.major.width": .6, "ytick.major.width": .6})
    clustering(args.root, args.results, args.output)
    anomaly(args.root, args.results, args.output)


if __name__ == "__main__":
    main()
