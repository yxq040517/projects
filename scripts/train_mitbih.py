"""Train, independently evaluate, and export the frozen patient-disjoint MIT-BIH CNN.

Prepare the contracted cache first. This command deliberately does not resume or
reuse prior training outputs: model selection sees validation subjects only, and
the held-out test partition is evaluated in one pass after selection is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from backend.model_runtime import _normalize, validate_model
from scripts.training_metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[1]

CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_SYMBOLS = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "F": ["F"],
    "Q": ["/", "f", "Q"],
}
SEED = 20260930
INPUT_SAMPLES = 720
MARGIN = 16
JITTER = 12
PATIENCE = 8
DEFAULT_OUTPUT = ROOT / "training" / "mitbih_cnn_v1"
ARCHITECTURE = {
    "name": "mitbih_residual1d_v1",
    "input_shape": [None, 1, 720],
    "stem": {"channels": 16, "kernel": 9, "stride": 2, "padding": 4},
    "residual_blocks": [
        {"channels": 32, "kernels": [7, 5], "stride": 2},
        {"channels": 64, "kernels": [5, 3], "stride": 2},
    ],
    "skip": "Conv1d(kernel=1,stride=2,bias=False), BatchNorm1d",
    "convolution_bias": False,
    "normalization": "BatchNorm1d",
    "activation": "ReLU",
    "pool": {"kind": "AvgPool1d", "kernel": 6, "stride": 6},
    "flatten_features": 960,
    "hidden_features": 128,
    "dropout": 0.25,
    "output_features": 5,
}
CAVEATS = [
    "Research evaluation, not medical or clinical validation; no diagnostic use claim.",
    "Patient-disjoint fixed split includes 201/202 as one person; no test output was used for selection.",
    "Evaluation uses original reference-annotation R-peak anchors, not automatically detected pipeline peaks; deployment peak errors may reduce performance.",
    "Q combines paced '/', paced-normal fusion 'f', and unclassifiable 'Q', and is predominantly pacing-related; it is not an AF or generic disease label.",
    "Actual MLII is prioritized; records 102/104 use their actual V5 channel, so the single-selected-lead cohort mixes lead types without renaming them.",
    "The five mapped heartbeat classes are not five record-level disease diagnoses; nonbeat annotations and incomplete boundary windows are excluded.",
    "Class imbalance and rare fusion beats make per-class support and F1 important; overall accuracy alone is inadequate.",
    "All selected leads are filtered once before cropping, using the production bandpass helper at 360 Hz with notch disabled; each float32 window uses production float64 z-score statistics.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    # A file object avoids legacy Windows filename encoding problems.
    with temporary.open("wb") as stream:
        torch.save(checkpoint, stream)
    temporary.replace(path)


def software_versions() -> dict:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda_build": torch.version.cuda,
    }
    for package in ("scipy", "onnx", "onnxruntime", "wfdb", "matplotlib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


class ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, kernel1: int, kernel2: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(incoming, outgoing, kernel1, stride=2, padding=kernel1 // 2, bias=False),
            nn.BatchNorm1d(outgoing),
            nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, kernel2, padding=kernel2 // 2, bias=False),
            nn.BatchNorm1d(outgoing),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 1, stride=2, bias=False),
            nn.BatchNorm1d(outgoing),
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(values) + self.skip(values))


class MitBihCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            ResidualBlock(16, 32, 7, 5),
            ResidualBlock(32, 64, 5, 3),
            nn.AvgPool1d(6, stride=6),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    result = classification_metrics(matrix, CLASSES)
    result.update(
        {
            "sample_count": result["total"],
            "support": {item["label"]: item["support"] for item in result["per_class"]},
            "confusion_axes": {"rows": "true", "columns": "predicted", "class_order": CLASSES},
            "zero_division": 0,
        }
    )
    return result


class TrainingCache:
    def __init__(self, folder: Path):
        self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self.split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
        self.windows = np.load(folder / "windows.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(folder / "labels.npy", mmap_mode="r", allow_pickle=False)
        self.records = np.load(folder / "record_indices.npy", mmap_mode="r", allow_pickle=False)
        self.peaks = np.load(folder / "peak_indices.npy", mmap_mode="r", allow_pickle=False)
        self.symbols = np.load(folder / "symbols.npy", mmap_mode="r", allow_pickle=False)
        self.indices = {
            part: np.load(folder / f"{part}_indices.npy", mmap_mode="r", allow_pickle=False)
            for part in ("train", "val", "test")
        }
        self.counts: dict[str, np.ndarray] = {}
        self.audit()

    def audit(self) -> None:
        if (
            self.metadata.get("classes") != CLASSES
            or self.metadata.get("class_mapping") != CLASS_SYMBOLS
        ):
            raise ValueError("Cache class mapping/order differs from the frozen training contract.")
        preprocessing = self.metadata.get("preprocessing", {})
        expected = {
            "sample_rate": 360,
            "input_samples": 720,
            "pre_samples": 360,
            "margin": 16,
            "extended_samples": 752,
            "normalization": "zscore",
            "input_filter": "bandpass",
            "notch": 0,
            "units": "mV",
            "windows_normalized": False,
        }
        if any(preprocessing.get(key) != value for key, value in expected.items()):
            raise ValueError("Cache preprocessing differs from the frozen production contract.")
        fingerprint = self.metadata.get("fingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or self.split.get("fingerprint") != fingerprint
        ):
            raise ValueError("Cache metadata/split fingerprints are missing or inconsistent.")
        if self.split.get("seed") != SEED:
            raise ValueError("The frozen split seed differs from the training contract.")
        count = len(self.labels)
        if count != self.metadata.get("sample_count") or self.windows.shape != (count, 1, 752):
            raise ValueError("Cache sample count or extended window shape is inconsistent.")
        if self.windows.dtype != np.float32 or self.labels.dtype != np.int64:
            raise ValueError("Expected float32 windows and int64 labels.")
        for array, dtype in (
            (self.records, np.int16),
            (self.peaks, np.int64),
            (self.symbols, np.dtype("U1")),
        ):
            if array.shape != (count,) or array.dtype != dtype:
                raise ValueError("Cache record/peak/symbol array shape or dtype is inconsistent.")
        record_ids = self.metadata.get("record_ids", [])
        if len(record_ids) != 48 or len(set(record_ids)) != 48:
            raise ValueError("Expected all 48 original MIT-BIH records.")
        if (
            np.any(self.labels < 0)
            or np.any(self.labels >= 5)
            or np.any(self.records < 0)
            or np.any(self.records >= 48)
        ):
            raise ValueError("Cache labels or record indices are out of range.")
        if np.any(self.peaks < 0):
            raise ValueError("Cache annotation peak indices must be original nonnegative anchors.")
        for number, label in enumerate(CLASSES):
            if not np.isin(self.symbols[self.labels == number], CLASS_SYMBOLS[label]).all():
                raise ValueError(f"Cache contains a symbol inconsistent with class {label}.")
        joined = np.concatenate(list(self.indices.values()))
        if any(indices.ndim != 1 or indices.dtype != np.int64 for indices in self.indices.values()):
            raise ValueError("Partition indices must be one-dimensional int64 arrays.")
        if (
            len(joined) != count
            or len(np.unique(joined)) != count
            or np.any(joined < 0)
            or np.any(joined >= count)
        ):
            raise ValueError(
                "Partitions must cover every usable beat exactly once, without overlap."
            )
        seen_groups: set[str] = set()
        seen_records: set[str] = set()
        minimum = {
            "train": [2000, 300, 500, 100, 2000],
            "val": [1500, 100, 300, 20, 1000],
            "test": [1500, 100, 300, 20, 1000],
        }
        for part, group_count in (("train", 30), ("val", 7), ("test", 10)):
            indices = self.indices[part]
            actual_records = {record_ids[int(i)] for i in np.unique(self.records[indices])}
            actual_groups = {
                "201_202" if name in ("201", "202") else name for name in actual_records
            }
            details = self.split[part]
            if actual_records != set(details["record_ids"]) or actual_groups != set(
                details["subject_groups"]
            ):
                raise ValueError(f"{part} arrays disagree with the frozen patient/record lists.")
            if (
                len(actual_groups) != group_count
                or seen_groups & actual_groups
                or seen_records & actual_records
            ):
                raise ValueError(f"{part} has incorrect patient counts or leaks patients/records.")
            seen_groups.update(actual_groups)
            seen_records.update(actual_records)
            counts = np.bincount(self.labels[indices], minlength=5)
            self.counts[part] = counts
            if np.any(counts < minimum[part]):
                raise ValueError(
                    f"{part} has insufficient contracted class support: {counts.tolist()}"
                )
            expected_counts = {label: int(counts[i]) for i, label in enumerate(CLASSES)}
            if (
                details["sample_count"] != len(indices)
                or details["class_counts"] != expected_counts
            ):
                raise ValueError(f"{part} support differs from split.json.")
        if len(seen_groups) != 47 or len(seen_records) != 48:
            raise ValueError("All 47 subjects and 48 records must be represented exactly once.")
        total_counts = {
            label: int(np.count_nonzero(self.labels == i)) for i, label in enumerate(CLASSES)
        }
        if self.metadata.get("class_counts") != total_counts:
            raise ValueError("Cache total class counts are inconsistent.")

    def batch(
        self, indices: np.ndarray, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng is None:
            batch = np.ascontiguousarray(
                self.windows[indices, :, MARGIN : MARGIN + INPUT_SAMPLES], dtype=np.float32
            )
        else:
            starts = MARGIN + rng.integers(-JITTER, JITTER + 1, size=len(indices))
            positions = starts[:, None] + np.arange(INPUT_SAMPLES)[None, :]
            batch = np.ascontiguousarray(
                self.windows[indices[:, None], 0, positions][:, None, :], dtype=np.float32
            )
        _normalize(batch, "zscore")
        if rng is not None:
            noise = rng.standard_normal(batch.shape, dtype=np.float32)
            noise *= np.float32(0.02)
            batch += noise
        return batch, np.asarray(self.labels[indices], dtype=np.int64)


def evaluate(
    model: nn.Module,
    cache: TrainingCache,
    partition: str,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
    by_record: bool = False,
) -> dict:
    model.eval()
    confusion = np.zeros((5, 5), dtype=np.int64)
    record_confusion = np.zeros((48, 5, 5), dtype=np.int64) if by_record else None
    loss_sum = 0.0
    indices = cache.indices[partition]
    began = perf_counter()
    with torch.inference_mode():
        for first in range(0, len(indices), batch_size):
            selected = indices[first : first + batch_size]
            values, labels = cache.batch(selected)
            targets = torch.from_numpy(labels).to(device)
            logits = model(torch.from_numpy(values).to(device))
            loss = criterion(logits, targets)
            if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Nonfinite {partition} logits/loss; refusing invalid model metrics."
                )
            predicted = logits.argmax(dim=1).cpu().numpy()
            loss_sum += float(loss.item()) * len(selected)
            encoded = labels * 5 + predicted
            confusion += np.bincount(encoded, minlength=25).reshape(5, 5)
            if record_confusion is not None:
                record_confusion += np.bincount(
                    np.asarray(cache.records[selected], dtype=np.int64) * 25 + encoded,
                    minlength=48 * 25,
                ).reshape(48, 5, 5)
    result = metrics_from_confusion(confusion)
    result.update({"loss": loss_sum / len(indices), "seconds": perf_counter() - began})
    if record_confusion is not None:
        result["by_record"] = {
            record: metrics_from_confusion(record_confusion[i])
            for i, record in enumerate(cache.metadata["record_ids"])
            if int(record_confusion[i].sum()) > 0
        }
        subject_matrices: dict[str, np.ndarray] = {}
        for i, record in enumerate(cache.metadata["record_ids"]):
            if int(record_confusion[i].sum()) > 0:
                subject = "201_202" if record in ("201", "202") else record
                subject_matrices.setdefault(subject, np.zeros((5, 5), dtype=np.int64))[:] += (
                    record_confusion[i]
                )
        result["by_subject"] = {
            subject: metrics_from_confusion(matrix) for subject, matrix in subject_matrices.items()
        }
        result["subgroup_metric_note"] = (
            "Per-record/subject macro scores include all five labels (absent-class recall/F1=0); interpret with support, not as global balanced accuracy."
        )
    return result


def platform_manifest() -> dict:
    return {
        "schema_version": 1,
        "name": "MIT-BIH 1D-CNN 心搏分类",
        "version": "20260930.v1",
        "task": "beat_classification",
        "classes": CLASSES,
        "sample_rate": 360,
        "input_samples": 720,
        "lead_names": ["selected"],
        "pre_samples": 360,
        "normalization": "zscore",
        "input_filter": "bandpass",
        "notch": 0,
        "output_type": "logits",
        "threshold": 0.5,
        "description": "真实 MIT-BIH 数据训练的单选导联心搏 CNN；按患者独立划分，201/202 同组。优先实际 MLII，102/104 使用实际 V5，训练导联混合但未重命名。N/S/V/F/Q 为心搏类别，Q 为起搏、起搏融合及不可分类，主要起搏相关，不是其他疾病诊断。独立测试使用参考注释峰锚点，平台自动检测峰存在额外误差；未作临床或医疗诊断有效性验证。",
    }


def export_and_verify(
    model: nn.Module, cache: TrainingCache, output: Path, best_epoch: int
) -> dict:
    began = perf_counter()
    model.cpu().float().eval()
    actual, _ = cache.batch(cache.indices["val"][:32])
    if len(actual) != 32:
        raise ValueError(
            "At least 32 validation beats are required for actual export parity checks."
        )
    temporary = output / "model.onnx.tmp"
    with torch.inference_mode(), temporary.open("wb") as stream:
        torch.onnx.export(
            model,
            torch.from_numpy(actual[:1]),
            stream,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["waveform"],
            output_names=["logits"],
            dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
            external_data=False,
        )
    manifest = platform_manifest()
    _, session, signature = validate_model(temporary.read_bytes(), manifest)
    if (
        not signature["dynamic_batch"]
        or signature["input_shape"][1:] != [1, 720]
        or signature["output_shape"][1:] != [5]
    ):
        raise RuntimeError("Export does not implement dynamic [B,1,720] -> [B,5].")
    checks = []
    with torch.inference_mode():
        for size in (1, 8, 32):
            values = actual[:size]
            expected = model(torch.from_numpy(values)).numpy()
            observed = session.run([signature["output_name"]], {signature["input_name"]: values})[0]
            valid = (
                observed.shape == (size, 5)
                and observed.dtype == np.float32
                and np.isfinite(observed).all()
                and np.isfinite(expected).all()
            )
            close = bool(valid and np.allclose(observed, expected, rtol=1e-4, atol=1e-5))
            predictions_equal = bool(
                valid and np.array_equal(observed.argmax(axis=1), expected.argmax(axis=1))
            )
            difference = (
                np.abs(observed.astype(np.float64) - expected.astype(np.float64)) if valid else None
            )
            checks.append(
                {
                    "batch_size": size,
                    "shape": list(observed.shape),
                    "dtype": str(observed.dtype),
                    "allclose": close,
                    "argmax_equal": predictions_equal,
                    "max_absolute_error": float(difference.max())
                    if difference is not None
                    else None,
                    "mean_absolute_error": float(difference.mean())
                    if difference is not None
                    else None,
                }
            )
    result = {
        "passed": all(item["allclose"] and item["argmax_equal"] for item in checks),
        "rtol": 1e-4,
        "atol": 1e-5,
        "opset": 17,
        "dynamo": False,
        "best_epoch": best_epoch,
        "input_source": "First 32 actual centrally cropped, production-normalized validation beats; no test data or synthetic signal",
        "provider": session.get_providers(),
        "signature": signature,
        "checks": checks,
        "seconds": perf_counter() - began,
    }
    write_json(output / "export_parity.json", result)
    if not result["passed"]:
        raise RuntimeError(
            "Torch/ONNX Runtime parity failed; see export_parity.json. Model is not deployable."
        )
    temporary.replace(output / "model.onnx")
    write_json(output / "manifest.json", manifest)
    return result


def save_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *CLASSES])
        for label, row in zip(CLASSES, matrix, strict=True):
            writer.writerow([label, *row])
    temporary.replace(path)


def save_plots(output: Path, history: list[dict], test: dict) -> dict:
    # Plotting is optional; failures are recorded, not disguised as plot success.
    result = {"learning_curves": False, "confusion_matrix": False, "error": None}
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as error:
        result["error"] = f"Matplotlib unavailable: {type(error).__name__}: {error}"
        return result
    try:
        epochs = [item["epoch"] for item in history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        axes[0].plot(
            epochs, [item["train_loss"] for item in history], label="Balanced augmented train"
        )
        axes[0].plot(
            epochs,
            [item["validation"]["loss"] for item in history],
            label="Full untouched validation",
        )
        axes[0].set(xlabel="Epoch", ylabel="Cross entropy (label smoothing 0.03)")
        axes[0].legend()
        axes[1].plot(
            epochs,
            [item["validation"]["macro_f1"] for item in history],
            label="Validation macro F1",
        )
        axes[1].plot(
            epochs,
            [item["validation"]["balanced_accuracy"] for item in history],
            label="Validation balanced accuracy",
        )
        axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1))
        axes[1].legend()
        fig.savefig(output / "learning-curves.png", dpi=160)
        plt.close(fig)
        result["learning_curves"] = True
        matrix = np.asarray(test["confusion_matrix"], dtype=np.int64)
        fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        image = axis.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=axis, label="Beat count")
        axis.set(
            xticks=range(5),
            yticks=range(5),
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            xlabel="Predicted",
            ylabel="True",
            title="Patient-disjoint held-out test confusion",
        )
        threshold = float(matrix.max()) / 2
        for row in range(5):
            for column in range(5):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        fig.savefig(output / "confusion-matrix.png", dpi=160)
        plt.close(fig)
        result["confusion_matrix"] = True
    except Exception as error:
        plt.close("all")
        result["error"] = f"Plot creation failed: {type(error).__name__}: {error}"
    return result


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory containing cache/ prepared by prepare_mitbih_training.py",
    )
    parser.add_argument(
        "--epochs",
        type=positive_integer,
        default=40,
        help="Maximum epochs, at most the frozen limit of 40",
    )
    parser.add_argument("--batch-size", type=positive_integer, default=128)
    parser.add_argument("--samples-per-epoch", type=positive_integer, default=40000)
    parser.add_argument("--threads", type=positive_integer, default=8)
    args = parser.parse_args()
    if args.epochs > 40:
        parser.error("--epochs cannot exceed the frozen 40-epoch maximum.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = (
        "best.pt",
        "learning_history.jsonl",
        "training_config.json",
        "evaluation.json",
        "confusion_matrix.csv",
        "learning-curves.png",
        "confusion-matrix.png",
        "model.onnx",
        "manifest.json",
        "export_parity.json",
    )
    existing = [name for name in artifacts if (output / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite prior training/test artifacts or silently resume: "
            + ", ".join(existing)
        )
    began = perf_counter()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    cache = TrainingCache(output / "cache")
    model = MitBihCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    train_indices = cache.indices["train"]
    train_counts = cache.counts["train"]
    weights = 1.0 / train_counts[cache.labels[train_indices]]
    sampling_generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        args.samples_per_epoch,
        replacement=True,
        generator=sampling_generator,
    )
    augmentation_rng = np.random.default_rng(SEED)
    config = {
        "status": "training",
        "started_at": utc_now(),
        "output": str(output),
        "seed": SEED,
        "classes": CLASSES,
        "class_mapping": CLASS_SYMBOLS,
        "architecture": ARCHITECTURE,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "preprocessing": cache.metadata["preprocessing"],
        "cache_fingerprint": cache.metadata["fingerprint"],
        "split": cache.split,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else platform.processor(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": True,
        "software_versions": software_versions(),
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.001},
        "loss": {"name": "CrossEntropyLoss", "label_smoothing": 0.03, "class_weights": None},
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "metric": "validation macro_f1",
            "mode": "max",
            "factor": 0.5,
            "patience": 3,
        },
        "sampler": {
            "name": "WeightedRandomSampler",
            "replacement": True,
            "samples_per_epoch": args.samples_per_epoch,
            "uses": "training indices only",
            "train_class_counts": {label: int(train_counts[i]) for i, label in enumerate(CLASSES)},
            "per_beat_weight_by_class": {
                label: float(1.0 / train_counts[i]) for i, label in enumerate(CLASSES)
            },
            "expected_sampled_class_probability": {label: 0.2 for label in CLASSES},
        },
        "augmentation": {
            "partition": "train only",
            "jitter_samples": [-12, 12],
            "jitter_source": "actual 752-sample extended windows, never rolled/padded",
            "gaussian_noise_std_after_normalization": 0.02,
        },
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "selection": {
            "metric": "full validation macro_f1",
            "rule": "strict improvement; ties retain earlier epoch",
            "patience": PATIENCE,
            "test_evaluations_for_selection": 0,
        },
        "caveats": CAVEATS,
    }
    write_json(output / "training_config.json", config)
    print(
        json.dumps(
            {
                "event": "training_started",
                "device": str(device),
                "threads": args.threads,
                "classes": CLASSES,
                "train_support": config["sampler"]["train_class_counts"],
                "val_support": cache.split["val"]["class_counts"],
                "test_support": cache.split["test"]["class_counts"],
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )
    history: list[dict] = []
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    training_began = perf_counter()
    with (output / "learning_history.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for epoch in range(1, args.epochs + 1):
            epoch_began = perf_counter()
            model.train()
            sampled_counts = np.zeros(5, dtype=np.int64)
            loss_sum = 0.0
            samples_seen = 0
            iterator = iter(sampler)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            while True:
                local = np.fromiter(islice(iterator, args.batch_size), dtype=np.int64)
                if len(local) == 0:
                    break
                selected = train_indices[local]
                values, labels = cache.batch(selected, augmentation_rng)
                targets = torch.from_numpy(labels).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(torch.from_numpy(values).to(device))
                loss = criterion(logits, targets)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        "Nonfinite training loss; refusing to continue/export invalid weights."
                    )
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * len(local)
                samples_seen += len(local)
                sampled_counts += np.bincount(labels, minlength=5)
            if samples_seen != args.samples_per_epoch:
                raise RuntimeError(
                    "Sampler failed to produce the requested number of training beats."
                )
            validation = evaluate(model, cache, "val", args.batch_size, device, criterion)
            score = validation["macro_f1"]
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(
                    output / "best.pt",
                    {
                        "state_dict": {
                            name: tensor.detach().cpu()
                            for name, tensor in model.state_dict().items()
                        },
                        "architecture": ARCHITECTURE,
                        "config": config,
                        "epoch": epoch,
                        "validation": validation,
                        "seed": SEED,
                        "cache_fingerprint": cache.metadata["fingerprint"],
                    },
                )
            else:
                stale_epochs += 1
            scheduler.step(score)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / samples_seen,
                "train_sample_count": samples_seen,
                "train_sampled_support": {
                    label: int(sampled_counts[i]) for i, label in enumerate(CLASSES)
                },
                "validation": validation,
                "learning_rate": learning_rate,
                "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "new_best": improved,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_score,
                "epochs_without_improvement": stale_epochs,
                "epoch_seconds": perf_counter() - epoch_began,
                "total_seconds": perf_counter() - began,
            }
            history.append(row)
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            print(
                json.dumps({"event": "epoch", **row}, ensure_ascii=False, allow_nan=False),
                flush=True,
            )
            if stale_epochs >= PATIENCE:
                break
    training_seconds = perf_counter() - training_began
    # Selection is now immutable. No training, scheduler, or model choice follows.
    with (output / "best.pt").open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cpu().float().eval()
    selection = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "epochs_completed": len(history),
        "early_stopped": stale_epochs >= PATIENCE,
        "training_seconds": training_seconds,
        "selection_frozen_at": utc_now(),
    }
    checkpoint["completed_training"] = selection
    checkpoint["config"] = {**config, "status": "selection_frozen", **selection}
    save_checkpoint(output / "best.pt", checkpoint)
    config.update({"status": "selection_frozen", **selection})
    write_json(output / "training_config.json", config)
    # Verify export using validation inputs before spending the one held-out pass.
    parity = export_and_verify(model, cache, output, best_epoch)
    test = evaluate(
        model,
        cache,
        "test",
        args.batch_size,
        torch.device("cpu"),
        nn.CrossEntropyLoss(),
        by_record=True,
    )
    evaluation = {
        "schema_version": 1,
        "evaluated_at": utc_now(),
        "partition": "test",
        "test_passes": 1,
        "selection_frozen_before_test": True,
        "threshold_selection": "None: multiclass logits argmax; no threshold tuning",
        "classes": CLASSES,
        "class_definitions": CLASS_SYMBOLS,
        "class_descriptions": json.loads(
            (ROOT / "training" / "training-contract.json").read_text(encoding="utf-8")
        )["descriptions"],
        "split": cache.split,
        "cache_fingerprint": cache.metadata["fingerprint"],
        "preprocessing": cache.metadata["preprocessing"],
        "selected_lead_records": cache.metadata["records"],
        "skipped_counts": cache.metadata["skipped_counts"],
        "source_provenance": cache.metadata["source_provenance"],
        "selection": selection,
        "best_validation": checkpoint["validation"],
        "test": test,
        "test_loss_definition": "Unsmoothed, unweighted mean cross entropy over every held-out usable beat",
        "caveats": CAVEATS,
        "export_parity": parity,
    }
    write_json(output / "evaluation.json", evaluation)
    save_confusion_csv(output / "confusion_matrix.csv", test["confusion_matrix"])
    plots = save_plots(output, history, test)
    evaluation["plots"] = plots
    config.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "total_seconds": perf_counter() - began,
            "test_seconds": test["seconds"],
            "export_seconds": parity["seconds"],
            "plots": plots,
            "test_passes": 1,
            "artifact_sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ("best.pt", "model.onnx", "manifest.json")
            },
        }
    )
    evaluation["training_metadata"] = config
    write_json(output / "evaluation.json", evaluation)
    write_json(output / "training_config.json", config)
    print(
        json.dumps(
            {
                "event": "completed",
                "output": str(output),
                **selection,
                "total_seconds": config["total_seconds"],
                "test": {
                    key: test[key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "balanced_accuracy",
                        "per_class",
                        "confusion_matrix",
                    )
                },
                "export_parity_passed": parity["passed"],
                "plots": plots,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
