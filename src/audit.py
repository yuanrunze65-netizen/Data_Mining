"""Check image inventory and exact decoded-pixel overlap between train and test."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image


from image_common import DATA_ROOT, RESULTS_DIR


def scan(paths):
    dimensions = Counter()
    contents = {}
    for path in sorted(paths):
        with Image.open(path) as image:
            pixels = image.convert("RGB")
            dimensions[f"{image.width}x{image.height}"] += 1
            digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
        contents[path] = digest
    return dict(dimensions), contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    cluster = args.root / "Cluster" / "Cluster"
    cluster_sizes, cluster_content = scan((cluster / "dataset").glob("*.png"))
    result = {"cluster": {"n": len(cluster_content), "dimensions": cluster_sizes,
                          "pixel_duplicates": len(cluster_content) - len(set(cluster_content.values()))}}
    anomaly = args.root / "Image_Anomaly_Detection" / "Image_Anomaly_Detection"
    for name in ("hazelnut", "zipper"):
        training_sizes, training = scan((anomaly / name / "train").glob("*/*.png"))
        test_sizes, test = scan((anomaly / name / "test").glob("*.png"))
        counts = Counter(path.parent.name for path in training)
        hashes = set(training.values())
        overlap = [path.relative_to(anomaly).as_posix() for path, digest in test.items() if digest in hashes]
        result[name] = {"training": dict(counts), "test": len(test),
                        "training_dimensions": training_sizes, "test_dimensions": test_sizes,
                        "train_test_pixel_overlap": overlap,
                        "train_pixel_duplicates": len(training) - len(hashes),
                        "test_pixel_duplicates": len(test) - len(set(test.values()))}
    args.output.mkdir(parents=True, exist_ok=True)
    output = json.dumps(result, indent=2)
    (args.output / "data_audit.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
