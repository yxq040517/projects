"""Evaluate exported classifier behind the actual detector on frozen held-out subjects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.ecg import analyze_signal
from backend.model_runtime import predict, validate_model
from scripts.training_metrics import classification_metrics


def evaluate(output: Path):
    cache = output / "cache"
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    split = json.loads((cache / "split.json").read_text(encoding="utf-8"))
    inventory = json.loads(
        (ROOT / "training" / "dataset_inventory.json").read_text(encoding="utf-8")
    )
    data = Path(inventory["dataset_path"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest, session, signature = validate_model((output / "model.onnx").read_bytes(), manifest)
    classes = manifest["classes"]
    count = len(classes)
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    record_indices = np.load(cache / "record_indices.npy", mmap_mode="r")
    peaks = np.load(cache / "peak_indices.npy", mmap_mode="r")
    test_indices = np.load(cache / "test_indices.npy", mmap_mode="r")
    record_ids = metadata["record_ids"]
    inventory_records = {record["record"]: record for record in inventory["records"]}
    matrix = np.zeros((count, count), dtype=np.int64)
    support = np.zeros(count, dtype=np.int64)
    matched_classes = np.zeros(count, dtype=np.int64)
    correct = np.zeros(count, dtype=np.int64)
    detections = 0
    matches = 0
    blocked = []
    by_record = []
    boundary_excluded = 0
    started = perf_counter()
    frozen_test_records = {
        record_ids[int(index)] for index in np.unique(record_indices[test_indices])
    }
    for record_id in sorted(frozen_test_records):
        description = inventory_records[record_id]
        source = wfdb.rdrecord(str(data / record_id))
        lead = description["selected_lead"]
        channel = source.sig_name.index(lead)
        raw = source.p_signal[:, channel]
        rate = float(source.fs)
        if source.units[channel] != "mV":
            raise ValueError(f"Unexpected source units in {record_id}: {source.units[channel]}")
        rows = test_indices[record_indices[test_indices] == record_ids.index(record_id)]
        reference = peaks[rows]
        truth = labels[rows]
        local_support = np.zeros(count, dtype=np.int64)
        local_correct = np.zeros(count, dtype=np.int64)
        local_matches = 0
        local_detections = 0
        step = int(rate * 300)
        for begin in range(0, len(raw), step):
            end = min(len(raw), begin + step)
            if end - begin < 5 * rate:
                continue
            valid = (reference >= begin + manifest["pre_samples"]) & (
                reference + manifest["input_samples"] - manifest["pre_samples"] <= end
            )
            ref = reference[valid]
            actual = truth[valid]
            if not len(ref):
                continue
            counts = np.bincount(actual, minlength=count)
            support += counts
            local_support += counts
            record = analyze_signal({lead: raw[begin:end]}, rate, lead, notch=0)
            record["has_original"] = True
            if not record["quality"]["reliable"]:
                blocked.append(
                    {
                        "record": record_id,
                        "start": begin / rate,
                        "end": end / rate,
                        "reason": record["quality"]["label"],
                        "reference_count": len(ref),
                        "per_class": dict(zip(classes, counts.tolist(), strict=True)),
                    }
                )
                continue
            result = predict(
                session,
                manifest,
                signature,
                record,
                {"sample_rate": rate, "units": "mV", "leads": {lead: raw[begin:end]}},
            )
            predicted = result["predictions"]
            candidate = np.asarray([item["index"] + begin for item in predicted], dtype=np.int64)
            predicted_classes = np.asarray(
                [item["class_index"] for item in predicted], dtype=np.int64
            )
            detections += len(candidate)
            local_detections += len(candidate)
            used = set()
            for sample, label in zip(ref, actual, strict=True):
                position = int(np.searchsorted(candidate, sample))
                choices = [
                    index
                    for index in (position - 1, position, position + 1)
                    if 0 <= index < len(candidate) and index not in used
                ]
                if not choices:
                    continue
                closest = min(choices, key=lambda index: abs(int(candidate[index]) - int(sample)))
                if abs(int(candidate[closest]) - int(sample)) > round(rate * 0.15):
                    continue
                used.add(closest)
                matches += 1
                local_matches += 1
                matched_classes[int(label)] += 1
                predicted_label = int(predicted_classes[closest])
                matrix[int(label), predicted_label] += 1
                if predicted_label == int(label):
                    correct[int(label)] += 1
                    local_correct[int(label)] += 1
        boundary_excluded += len(reference) - int(local_support.sum())
        summary = {
            "record": record_id,
            "subject_group": description["subject_group"],
            "lead": lead,
            "reference_support": local_support.tolist(),
            "detected_windows": local_detections,
            "matched": local_matches,
            "correct_by_class": local_correct.tolist(),
        }
        by_record.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    result = {
        "scope": "untouched held-out subjects, actual platform detector -> original-window preprocessing -> exported ONNX; no tuning",
        "test_records": sorted(frozen_test_records),
        "sample_rate": 360,
        "chunk_seconds": 300,
        "match_tolerance_ms": 150,
        "boundary_reference_excluded": boundary_excluded,
        "eligible_reference_beats": int(support.sum()),
        "predicted_complete_windows": detections,
        "matched_windows": matches,
        "detection_precision": matches / detections if detections else 0.0,
        "detection_recall": matches / int(support.sum()) if support.sum() else 0.0,
        "classification_on_matched_windows": classification_metrics(matrix, classes),
        "end_to_end_per_class": [
            {
                "label": label,
                "support": int(support[i]),
                "detected_matched": int(matched_classes[i]),
                "correct": int(correct[i]),
                "detection_recall": float(matched_classes[i] / support[i]) if support[i] else 0.0,
                "end_to_end_recall": float(correct[i] / support[i]) if support[i] else 0.0,
            }
            for i, label in enumerate(classes)
        ],
        "blocked_segments": blocked,
        "records": by_record,
        "elapsed_seconds": perf_counter() - started,
        "caveat": "Reference-centred classification scores do not measure detector misses. Unmatched detections are not assigned fabricated ground-truth classes. Q includes paced/pace fusion/unknown, not a generic disease.",
    }
    (output / "pipeline_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "detection_precision": result["detection_precision"],
                "detection_recall": result["detection_recall"],
                "matched_macro_f1": result["classification_on_matched_windows"]["macro_f1"],
                "end_to_end_per_class": result["end_to_end_per_class"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "mitbih_cnn_v1")
    args = parser.parse_args()
    evaluate(args.output)
