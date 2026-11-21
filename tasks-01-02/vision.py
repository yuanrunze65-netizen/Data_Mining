"""Frozen ResNet-50 features for the two course experiments."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


WEIGHT_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
WEIGHT_SHA = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
CATEGORIES = ("hazelnut", "zipper")
HERE = Path(__file__).resolve().parent


class Bottleneck(nn.Module):
    def __init__(self, channels, width, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.downsample = None
        if stride != 1 or channels != width * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels, width * 4, 1, stride, bias=False),
                nn.BatchNorm2d(width * 4),
            )

    def forward(self, image):
        residual = image if self.downsample is None else self.downsample(image)
        features = F.relu(self.bn1(self.conv1(image)), inplace=True)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)
        features = self.bn3(self.conv3(features))
        return F.relu(features + residual, inplace=True)


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.stage(64, 64, 3, 1)
        self.layer2 = self.stage(256, 128, 4, 2)
        self.layer3 = self.stage(512, 256, 6, 2)
        self.layer4 = self.stage(1024, 512, 3, 2)
        self.fc = nn.Linear(2048, 1000)

    @staticmethod
    def stage(channels, width, blocks, stride):
        return nn.Sequential(
            Bottleneck(channels, width, stride),
            *(Bottleneck(width * 4, width, 1) for _ in range(blocks - 1)),
        )

    def forward(self, image):
        features = F.relu(self.bn1(self.conv1(image)), inplace=True)
        level1 = self.layer1(F.max_pool2d(features, 3, 2, 1))
        level2 = self.layer2(level1)
        level3 = self.layer3(level2)
        return level1, level2, level3, self.layer4(level3)


def inputs(root, dataset):
    if dataset == "cluster":
        base = root / "Cluster" / "Cluster"
        return base, sorted((base / "dataset").glob("*.png"))
    base = root / "Image_Anomaly_Detection" / "Image_Anomaly_Detection"
    paths = []
    for category in CATEGORIES:
        paths.extend(sorted((base / category).glob("train/*/*.png")))
        paths.extend(sorted((base / category).glob("test/*.png")))
    return base, paths


def image_array(path, size):
    with Image.open(path) as image:
        return np.array(
            image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )


def hand_features(images):
    # Global colour, coarse appearance, and simple local contrast statistics.
    rgb = images.astype(np.float32) / 255
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    histogram = [
        np.concatenate([
            np.histogram(image[:, :, channel], bins=16, range=(0, 1))[0] / image.shape[0] ** 2
            for channel in range(3)
        ])
        for image in rgb
    ]
    colour = F.adaptive_avg_pool2d(torch.from_numpy(rgb).permute(0, 3, 1, 2), 8)
    texture = []
    for lag in (1, 2, 4, 8, 16):
        diff_x = np.abs(gray[:, :, lag:] - gray[:, :, :-lag])
        diff_y = np.abs(gray[:, lag:, :] - gray[:, :-lag, :])
        for diff in (diff_x, diff_y):
            texture.extend([diff.mean(axis=(1, 2)), diff.std(axis=(1, 2))])
    return {
        "colour": np.concatenate([np.asarray(histogram, np.float32), colour.flatten(1).numpy()], axis=1),
        "texture": np.asarray(texture, np.float32).T,
    }


@torch.inference_mode()
def extract(model, paths, size, batch_size, device, patches):
    features = {}
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
    generator = np.random.default_rng(2026)
    projection = [
        torch.tensor(
            generator.standard_normal((channels, 48), dtype=np.float32) / np.sqrt(48),
            device=device,
        ).float()
        for channels in (768, 1536)
    ]
    for start in range(0, len(paths), batch_size):
        images = np.stack([image_array(path, size) for path in paths[start : start + batch_size]])
        image = torch.from_numpy(images).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        levels = model(((image / 255 - mean) / std).contiguous(memory_format=torch.channels_last))
        batch = hand_features(images)
        for level, values in enumerate(levels[1:], start=2):
            batch[f"avg{level}"] = values.mean(dim=(2, 3)).cpu().numpy()
            batch[f"std{level}"] = values.std(dim=(2, 3), correction=0).cpu().numpy()
        if patches:
            grid = size // 16
            grids = []
            for values in levels[:3]:
                local = F.avg_pool2d(values, 3, 1, 1, count_include_pad=False)
                local = F.adaptive_avg_pool2d(local, grid).permute(0, 2, 3, 1)
                grids.append(F.normalize(local, dim=-1))
            for index, levels_pair in enumerate((grids[:2], grids[1:]), start=1):
                projected = torch.cat(levels_pair, dim=-1) @ projection[index - 1]
                batch[f"patch{index}"] = F.normalize(projected, dim=-1).cpu().numpy().astype(np.float16)
        for name, values in batch.items():
            features.setdefault(name, []).append(values)
        if start % (batch_size * 8) == 0:
            print(f"extracted {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return {name: np.concatenate(values) for name, values in features.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parent)
    parser.add_argument("--cache", type=Path, default=HERE / "_work")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--task", choices=("cluster", "anomaly", "both"), default="both")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.weights or Path(torch.hub.get_dir()) / "checkpoints" / WEIGHT_URL.rsplit("/", 1)[-1]
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHT_URL, str(weights), hash_prefix=WEIGHT_SHA)
    with weights.open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if digest != WEIGHT_SHA:
        raise RuntimeError("Unexpected ResNet-50 checkpoint checksum")
    model = ResNet50()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model = model.eval().to(device=device, memory_format=torch.channels_last)
    args.cache.mkdir(parents=True, exist_ok=True)
    selected = ("cluster", "anomaly") if args.task == "both" else (args.task,)
    for dataset in selected:
        base, paths = inputs(args.root, dataset)
        expected = 600 if dataset == "cluster" else 602
        if len(paths) != expected:
            raise RuntimeError(f"{dataset}: expected {expected} images, found {len(paths)}")
        size = 224 if dataset == "cluster" else 384
        print(f"{dataset}: {len(paths)} images on {device} ({size} px)", flush=True)
        values = extract(model, paths, size, 12, device, patches=dataset == "anomaly")
        keys = np.array([path.relative_to(base).as_posix() for path in paths])
        np.savez(args.cache / f"{dataset}.npz", keys=keys, **values)


if __name__ == "__main__":
    main()
