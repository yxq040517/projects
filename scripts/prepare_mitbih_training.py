"""Prepare physical MIT-BIH beat windows and a frozen subject-disjoint split.

Run with the isolated training environment; this script never trains a model or
changes the downloaded source. --output is the experiment directory (cache/ is
created inside it). Existing incompatible or incomplete caches are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import wfdb

from backend.ecg import _filter_waveform

ROOT = Path(__file__).resolve().parents[1]

CLASSES = ["N", "S", "V", "F", "Q"]
MAPPING = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "F": ["F"],
    "Q": ["/", "f", "Q"],
}
SCHEMA_VERSION = 1
SPLIT_POLICY = {
    "candidate_permutations": 15000,
    "rng": "numpy.random.Generator(PCG64)",
    "subject_order": "sorted subject group IDs",
    "partition_subject_counts": {"train": 30, "val": 7, "test": 10},
    "target_fractions": [0.65, 0.15, 0.20],
    "minimum_class_counts": {
        "train": {"N": 2000, "S": 300, "V": 500, "F": 100, "Q": 2000},
        "val": {"N": 1500, "S": 100, "V": 300, "F": 20, "Q": 1000},
        "test": {"N": 1500, "S": 100, "V": 300, "F": 20, "Q": 1000},
    },
    "paced_groups": {"V5": ["102", "104"], "MLII": ["107", "217"]},
    "paced_constraint": "train one V5 and one MLII paced subject; val and test each one remaining paced subject",
    "objective": "mean(((partition_class_counts/global_class_counts-target_fraction)/target_fraction)**2) + 0.1*mean(((partition_total/global_total-target_fraction)/target_fraction)**2)",
    "total_fraction_weight": 0.1,
    "ties": "first feasible minimum in seeded candidate order",
    "labels_used": "eligible original reference annotations only, before model fitting",
}
CACHE_FILES = (
    "windows.npy",
    "labels.npy",
    "record_indices.npy",
    "peak_indices.npy",
    "symbols.npy",
    "train_indices.npy",
    "val_indices.npy",
    "test_indices.npy",
    "metadata.json",
    "split.json",
)
CAVEATS = [
    "F is rare and concentrated in a few subjects: subject-level coverage does not imply broad clinical validation.",
    "Q combines paced '/', paced-normal fusion 'f', and unclassifiable 'Q'; it is mostly paced, not a generic disease or atrial-fibrillation label.",
    "Selected physical leads are MLII where available, otherwise V5, otherwise the first physical channel; 102/104 use V5. No lead is renamed or silently mixed.",
    "These are expert-reference-anchored beats; evaluation of them is not evaluation of automatic peak detection or the complete platform pipeline.",
    "Patient-disjoint MIT-BIH partitions are research evaluation, not medical validation.",
]


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def class_counts(values: np.ndarray) -> dict[str, int]:
    counts = np.bincount(values, minlength=len(CLASSES))
    return {name: int(counts[index]) for index, name in enumerate(CLASSES)}


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def read_configuration() -> tuple[dict, dict]:
    contract = json.loads(
        (ROOT / "training" / "training-contract.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (ROOT / "training" / "dataset_inventory.json").read_text(encoding="utf-8")
    )
    if (
        contract["classes"] != CLASSES
        or contract["mapping"] != MAPPING
        or inventory["class_mapping"] != MAPPING
    ):
        raise ValueError(
            "Frozen class mapping differs from the implementation; no implicit mapping migration is allowed."
        )
    expected = {
        "sample_rate": 360,
        "input_samples": 720,
        "pre_samples": 360,
        "margin": 16,
        "normalization": "zscore",
        "input_filter": "bandpass",
        "notch": 0,
    }
    if any(contract["preprocessing"].get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Frozen preprocessing differs from this implementation; refusing to prepare an incompatible cache."
        )
    return contract, inventory


def verify_sources(data: Path, inventory: dict) -> tuple[list[str], dict[str, dict], dict]:
    records_path = data / "RECORDS"
    record_ids = [
        line.strip()
        for line in records_path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if (
        len(record_ids) != 48
        or len(set(record_ids)) != 48
        or any(not item.isdigit() for item in record_ids)
    ):
        raise ValueError(
            "Original RECORDS must contain exactly 48 unique numeric MIT-BIH record IDs."
        )
    records = {item["record"]: item for item in inventory["records"]}
    if len(inventory["records"]) != 48 or set(record_ids) != set(records):
        raise ValueError("Original RECORDS does not match the verified 48-record inventory.")
    for record_id, item in records.items():
        expected_group = "201_202" if record_id in {"201", "202"} else record_id
        if item["subject_group"] != expected_group:
            raise ValueError(f"Inventory subject grouping mismatch for {record_id}.")
    if inventory["record_count"] != 48 or inventory["subject_count"] != 47:
        raise ValueError("Inventory does not identify 48 records from 47 subjects.")
    expected_files = {
        f"{record}.{extension}" for record in record_ids for extension in ("hea", "dat", "atr")
    }
    verified = inventory["verified_source_files"]
    if (
        len(verified) != 144
        or inventory["files_verified"] != 144
        or {item["file"] for item in verified} != expected_files
    ):
        raise ValueError(
            "Inventory must have exactly the 144 original .hea/.dat/.atr source checksums."
        )
    checksums = []
    for index, item in enumerate(verified, 1):
        actual = sha256_file(data / item["file"])
        if actual != item["sha256"]:
            raise ValueError(
                f"Source checksum mismatch: {item['file']}; originals or inventory changed. Refusing to proceed."
            )
        checksums.append({"file": item["file"], "sha256": actual})
        if index % 3 == 0:
            print(f"Verified source {index // 3}/48: {Path(item['file']).stem}", flush=True)
    provenance = {
        "source_url": inventory["source"],
        "license": inventory["license"],
        "dataset_path": str(data),
        "records_file": "RECORDS",
        "records_sha256": sha256_file(records_path),
        "verified_file_count": len(checksums),
        "files": sorted(checksums, key=lambda item: item["file"]),
        "inventory_sha256": hashlib.sha256(canonical_json(inventory)).hexdigest(),
    }
    return record_ids, records, provenance


def inspect_annotations(
    data: Path, record_id: str, expected: dict, symbol_to_label: dict[str, int]
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    header = wfdb.rdheader(str(data / record_id))
    names = list(header.sig_name)
    units = list(header.units)
    if (
        header.fs != 360
        or header.sig_len != expected["samples"]
        or names != expected["lead_names"]
        or units != expected["units"]
    ):
        raise ValueError(
            f"Record {record_id}: actual sample rate, samples, leads, or units differ from inventory."
        )
    selected = names.index("MLII") if "MLII" in names else names.index("V5") if "V5" in names else 0
    if names[selected] != expected["selected_lead"] or units[selected] != "mV":
        raise ValueError(
            f"Record {record_id}: selected physical lead or mV units differ from inventory."
        )
    ann = wfdb.rdann(str(data / record_id), "atr")
    peaks = np.asarray(ann.sample, dtype=np.int64)
    symbols = np.asarray(ann.symbol, dtype="U1")
    if len(peaks) != len(symbols) or np.any(peaks < 0) or np.any(peaks >= header.sig_len):
        raise ValueError(f"Record {record_id}: annotation samples are invalid.")
    actual_symbols = dict(Counter(ann.symbol))
    if actual_symbols != expected["annotation_symbols"]:
        raise ValueError(f"Record {record_id}: annotation symbols differ from verified inventory.")
    labels = np.fromiter(
        (symbol_to_label.get(symbol, -1) for symbol in ann.symbol), dtype=np.int64, count=len(peaks)
    )
    mapped = labels >= 0
    mapped_counts = class_counts(labels[mapped])
    if mapped_counts != expected["class_counts"]:
        raise ValueError(f"Record {record_id}: mapped label counts differ from verified inventory.")
    starts = peaks - 376
    complete = (starts >= 0) & (starts + 752 <= header.sig_len)
    eligible = mapped & complete
    boundary = mapped & ~complete
    info = {
        "record_id": record_id,
        "subject_group": expected["subject_group"],
        "sample_rate": int(header.fs),
        "samples": int(header.sig_len),
        "lead_names": names,
        "units": units,
        "selected_lead": names[selected],
        "selected_channel_index": selected,
        "annotation_symbols": actual_symbols,
        "mapped_class_counts": mapped_counts,
        "class_counts": class_counts(labels[eligible]),
        "sample_count": int(np.count_nonzero(eligible)),
        "skipped_counts": {
            "nonbeat_annotations": int(np.count_nonzero(~mapped)),
            "boundary_beats": int(np.count_nonzero(boundary)),
            "boundary_by_class": class_counts(labels[boundary]),
        },
    }
    return info, peaks[eligible], labels[eligible], symbols[eligible]


def choose_split(records: list[dict], seed: int, fingerprint: str) -> dict:
    subjects = sorted({item["subject_group"] for item in records})
    if len(subjects) != 47:
        raise ValueError("Exactly 47 subject groups are required; 201 and 202 must be grouped.")
    counts = np.zeros((47, len(CLASSES)), dtype=np.int64)
    for item in records:
        counts[subjects.index(item["subject_group"])] += [
            item["class_counts"][name] for name in CLASSES
        ]
    totals = counts.sum(axis=0)
    if np.any(totals == 0):
        raise ValueError(
            f"No valid support for all classes: {dict(zip(CLASSES, totals.tolist(), strict=True))}"
        )
    minimum = np.array(
        [
            [SPLIT_POLICY["minimum_class_counts"][part][name] for name in CLASSES]
            for part in ("train", "val", "test")
        ],
        dtype=np.int64,
    )
    target = np.array(SPLIT_POLICY["target_fractions"], dtype=np.float64)
    v5 = {subjects.index(name) for name in ("102", "104")}
    mlii = {subjects.index(name) for name in ("107", "217")}
    paced = v5 | mlii
    rng = np.random.Generator(np.random.PCG64(seed))
    best = None
    best_score = float("inf")
    feasible = 0
    for candidate in range(SPLIT_POLICY["candidate_permutations"]):
        permutation = rng.permutation(len(subjects))
        partitions = (permutation[:30], permutation[30:37], permutation[37:])
        train = set(partitions[0].tolist())
        if len(train & v5) != 1 or len(train & mlii) != 1:
            continue
        if any(len(set(indices.tolist()) & paced) != 1 for indices in partitions[1:]):
            continue
        distribution = np.stack([counts[indices].sum(axis=0) for indices in partitions])
        if np.any(distribution < minimum):
            continue
        feasible += 1
        class_deviation = (distribution / totals - target[:, None]) / target[:, None]
        total_deviation = (distribution.sum(axis=1) / totals.sum() - target) / target
        score = float(np.mean(class_deviation**2) + 0.1 * np.mean(total_deviation**2))
        if score < best_score:
            best_score = score
            best = ([indices.copy() for indices in partitions], distribution.copy(), candidate)
    if best is None:
        actual = {
            subject: {name: int(counts[index, label]) for label, name in enumerate(CLASSES)}
            for index, subject in enumerate(subjects)
        }
        raise ValueError(
            "No feasible split among exactly 15000 seeded permutations; constraints were NOT relaxed. "
            + json.dumps(
                {
                    "global_class_counts": dict(zip(CLASSES, totals.tolist(), strict=True)),
                    "subject_class_counts": actual,
                    "policy": SPLIT_POLICY,
                },
                ensure_ascii=False,
            )
        )
    partitions, distribution, candidate = best
    result = {
        "seed": seed,
        "strategy": "annotation-count-only deterministic group permutation search",
        "fingerprint": fingerprint,
        "score": best_score,
        "candidate_permutations": SPLIT_POLICY["candidate_permutations"],
        "feasible_candidates": feasible,
        "selected_candidate_zero_based": candidate,
        "policy": SPLIT_POLICY,
    }
    for index, name in enumerate(("train", "val", "test")):
        groups = sorted(subjects[position] for position in partitions[index])
        result[name] = {
            "subject_groups": groups,
            "record_ids": [
                item["record_id"] for item in records if item["subject_group"] in groups
            ],
            "sample_count": int(distribution[index].sum()),
            "class_counts": {
                label: int(distribution[index, position]) for position, label in enumerate(CLASSES)
            },
        }
    result["leakage_checks"] = {
        "subject_overlap": False,
        "record_overlap": False,
        "all_47_subjects_used": True,
        "all_48_records_used": True,
        "201_202_same_partition": True,
    }
    return result


def cache_arrays(cache: Path, count: int, mode: str) -> dict[str, np.memmap]:
    specs = {
        "windows": (np.dtype("float32"), (count, 1, 752)),
        "labels": (np.dtype("int64"), (count,)),
        "record_indices": (np.dtype("int16"), (count,)),
        "peak_indices": (np.dtype("int64"), (count,)),
        "symbols": (np.dtype("U1"), (count,)),
    }
    result = {}
    try:
        for name, (dtype, shape) in specs.items():
            array = (
                np.lib.format.open_memmap(
                    cache / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
                )
                if mode == "w+"
                else np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            )
            result[name] = array
            if array.dtype != dtype or array.shape != shape:
                raise ValueError(f"Cache {name} has incompatible dtype/shape.")
    except BaseException:
        close_arrays(result)
        raise
    return result


def close_arrays(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        if array.flags.writeable:
            array.flush()
        array._mmap.close()
    arrays.clear()


def verify_cache(cache: Path, metadata: dict, split: dict) -> None:
    count = metadata["sample_count"]
    arrays = cache_arrays(cache, count, "r")
    try:
        labels = arrays["labels"]
        record_indices = arrays["record_indices"]
        if (
            np.any(labels < 0)
            or np.any(labels >= 5)
            or class_counts(labels) != metadata["class_counts"]
        ):
            raise ValueError("Cache labels are invalid or disagree with metadata.")
        if np.any(record_indices < 0) or np.any(record_indices >= 48):
            raise ValueError("Cache record indices are invalid.")
        seen_samples = np.zeros(count, dtype=bool)
        seen_groups = set()
        seen_records = set()
        for part in ("train", "val", "test"):
            details = split[part]
            groups = set(details["subject_groups"])
            records = set(details["record_ids"])
            if (
                len(groups) != SPLIT_POLICY["partition_subject_counts"][part]
                or seen_groups & groups
                or seen_records & records
            ):
                raise ValueError(
                    "Split does not have the required disjoint subject/record partitions."
                )
            actual_records = {
                item["record_id"] for item in metadata["records"] if item["subject_group"] in groups
            }
            if actual_records != records:
                raise ValueError("Split record membership disagrees with actual subject groups.")
            indices = np.load(cache / f"{part}_indices.npy", mmap_mode="r", allow_pickle=False)
            try:
                if (
                    indices.dtype != np.dtype("int64")
                    or indices.ndim != 1
                    or len(indices) != details["sample_count"]
                ):
                    raise ValueError(f"{part} indices have incompatible dtype/shape.")
                if (
                    np.any(indices < 0)
                    or np.any(indices >= count)
                    or len(np.unique(indices)) != len(indices)
                    or np.any(seen_samples[indices])
                ):
                    raise ValueError("Split indices are out of bounds, duplicated, or overlapping.")
                actual_counts = class_counts(labels[indices])
                if actual_counts != details["class_counts"] or any(
                    actual_counts[name] < SPLIT_POLICY["minimum_class_counts"][part][name]
                    for name in CLASSES
                ):
                    raise ValueError(
                        "Partition class counts violate the frozen support constraints."
                    )
                allowed = [metadata["record_ids"].index(record) for record in records]
                if not np.all(np.isin(record_indices[indices], allowed)):
                    raise ValueError("Partition samples cross their assigned subject boundary.")
                seen_samples[indices] = True
                seen_groups.update(groups)
                seen_records.update(records)
            finally:
                indices._mmap.close()
        if (
            not seen_samples.all()
            or len(seen_groups) != 47
            or seen_records != set(metadata["record_ids"])
        ):
            raise ValueError("Partitions do not preserve every eligible sample/subject/record.")
        train_groups = set(split["train"]["subject_groups"])
        paced_v5 = set(SPLIT_POLICY["paced_groups"]["V5"])
        paced_mlii = set(SPLIT_POLICY["paced_groups"]["MLII"])
        if (
            len(train_groups & paced_v5) != 1
            or len(train_groups & paced_mlii) != 1
            or any(
                len(set(split[part]["subject_groups"]) & (paced_v5 | paced_mlii)) != 1
                for part in ("val", "test")
            )
        ):
            raise ValueError(
                "Split violates physical-lead paced training/held-out subject coverage."
            )
        if split.get("policy") != SPLIT_POLICY or split.get("candidate_permutations") != 15000:
            raise ValueError("Split policy differs from the frozen search contract.")
    finally:
        close_arrays(arrays)


def print_summary(cache: Path, metadata: dict, split: dict, reused: bool) -> None:
    print(f"{'Reused verified' if reused else 'Prepared'} cache: {cache}", flush=True)
    print(f"Fingerprint: {metadata['fingerprint']}", flush=True)
    print(
        f"Split search: {split['candidate_permutations']} seeded candidates, {split['feasible_candidates']} feasible; "
        f"frozen annotation-only objective={split['score']:.8f}",
        flush=True,
    )
    print(
        f"Actual samples: {metadata['sample_count']}; classes: {metadata['class_counts']}; skips: {metadata['skipped_counts']}",
        flush=True,
    )
    for part in ("train", "val", "test"):
        details = split[part]
        print(
            f"{part}: subjects={details['subject_groups']}; records={details['record_ids']}; "
            f"samples={details['sample_count']}; classes={details['class_counts']}",
            flush=True,
        )
    print(
        "Checked: no subject/record/sample overlap; 201/202 grouped; all 48 records and 47 subjects used.",
        flush=True,
    )
    for warning in CAVEATS:
        print(f"Caveat: {warning}", flush=True)


def prepare(data: Path, output: Path, contract: dict, inventory: dict) -> None:
    data = data.resolve()
    output = output.resolve()
    if output == data or data in output.parents:
        raise ValueError("Cache output must not be inside the original downloaded dataset.")
    record_ids, expected_records, provenance = verify_sources(data, inventory)
    preprocessing = dict(
        contract["preprocessing"], extended_samples=752, units="mV", windows_normalized=False
    )
    filter_source_sha = hashlib.sha256(
        inspect.getsource(_filter_waveform).encode("utf-8")
    ).hexdigest()
    versions = {"numpy": np.__version__, "scipy": scipy.__version__, "wfdb": wfdb.__version__}
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source": provenance,
        "record_ids": record_ids,
        "subject_groups": {key: value["subject_group"] for key, value in expected_records.items()},
        "classes": CLASSES,
        "mapping": MAPPING,
        "preprocessing": preprocessing,
        "filter_helper_sha256": filter_source_sha,
        "versions": versions,
        "split_policy": SPLIT_POLICY,
        "seed": contract["seed"],
    }
    fingerprint = hashlib.sha256(canonical_json(identity)).hexdigest()
    cache = output / "cache"
    if cache.exists():
        if not cache.is_dir() or any(not (cache / name).is_file() for name in CACHE_FILES):
            raise ValueError(
                f"Incomplete existing cache at {cache}; refusing to overwrite it. Move it aside explicitly."
            )
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        split = json.loads((cache / "split.json").read_text(encoding="utf-8"))
        if (
            metadata.get("fingerprint") != fingerprint
            or split.get("fingerprint") != fingerprint
            or metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("classes") != CLASSES
            or metadata.get("preprocessing") != preprocessing
            or metadata.get("record_ids") != record_ids
            or metadata.get("class_mapping") != MAPPING
            or split.get("seed") != contract["seed"]
        ):
            raise ValueError(
                f"Stale mismatching cache at {cache}; sources/preprocessing/mapping/split policy changed. Move it aside explicitly."
            )
        verify_cache(cache, metadata, split)
        print_summary(cache, metadata, split, reused=True)
        return
    symbol_to_label = {
        symbol: index for index, name in enumerate(CLASSES) for symbol in MAPPING[name]
    }
    records = []
    for index, record_id in enumerate(record_ids, 1):
        info, _, _, _ = inspect_annotations(
            data, record_id, expected_records[record_id], symbol_to_label
        )
        records.append(info)
        print(
            f"Counted record {index}/48: {record_id}, physical {info['selected_lead']}, "
            f"eligible={info['sample_count']}, classes={info['class_counts']}, skips={info['skipped_counts']}",
            flush=True,
        )
    original_counts = {
        name: sum(item["mapped_class_counts"][name] for item in records) for name in CLASSES
    }
    if original_counts != inventory["class_counts"]:
        raise ValueError("Actual total mapped annotations disagree with inventory.")
    count = sum(item["sample_count"] for item in records)
    split = choose_split(records, int(contract["seed"]), fingerprint)
    skipped = {
        "nonbeat_annotations": sum(
            item["skipped_counts"]["nonbeat_annotations"] for item in records
        ),
        "boundary_beats": sum(item["skipped_counts"]["boundary_beats"] for item in records),
        "boundary_by_class": {
            name: sum(item["skipped_counts"]["boundary_by_class"][name] for item in records)
            for name in CLASSES
        },
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(data),
        "record_ids": record_ids,
        "classes": CLASSES,
        "class_mapping": MAPPING,
        "preprocessing": preprocessing,
        "sample_count": count,
        "class_counts": {
            name: sum(item["class_counts"][name] for item in records) for name in CLASSES
        },
        "mapped_annotation_class_counts": original_counts,
        "skipped_counts": skipped,
        "records": records,
        "source_provenance": provenance,
        "filter_helper_sha256": filter_source_sha,
        "versions": versions,
        "split_policy": SPLIT_POLICY,
        "caveats": CAVEATS,
    }
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cache-preparing-", dir=output))
    arrays = {}
    try:
        arrays = cache_arrays(staging, count, "w+")
        cursor = 0
        for record_index, info in enumerate(records):
            record_id = info["record_id"]
            checked, peaks, labels, symbols = inspect_annotations(
                data, record_id, expected_records[record_id], symbol_to_label
            )
            if checked != info:
                raise ValueError(f"Record {record_id} changed during preparation.")
            # Only one physical channel from one record is materialized at a time.
            waveform = wfdb.rdrecord(
                str(data / record_id),
                channels=[info["selected_channel_index"]],
                physical=True,
                return_res=64,
            )
            if (
                waveform.sig_name != [info["selected_lead"]]
                or waveform.units != ["mV"]
                or waveform.fs != 360
                or waveform.sig_len != info["samples"]
            ):
                raise ValueError(
                    f"Record {record_id}: physical waveform disagrees with inspected header."
                )
            raw = waveform.p_signal[:, 0]
            if not np.isfinite(raw).all():
                raise ValueError(
                    f"Record {record_id}: nonfinite physical waveform; no synthetic replacement allowed."
                )
            filtered = _filter_waveform(raw, 360, 0)
            if not np.isfinite(filtered).all():
                raise ValueError(f"Record {record_id}: nonfinite filtered waveform.")
            end = cursor + len(peaks)
            arrays["labels"][cursor:end] = labels
            arrays["record_indices"][cursor:end] = record_index
            arrays["peak_indices"][cursor:end] = peaks
            arrays["symbols"][cursor:end] = symbols
            for offset, peak in enumerate(peaks):
                start = int(peak) - 376
                arrays["windows"][cursor + offset, 0, :] = filtered[start : start + 752]
            cursor = end
            del filtered, raw, waveform
            print(
                f"Wrote record {record_index + 1}/48: {record_id}, {len(peaks)} real windows, cumulative {cursor}/{count}",
                flush=True,
            )
        if cursor != count:
            raise ValueError("Eligible sample count changed during writing.")
        for part in ("train", "val", "test"):
            allowed = [record_ids.index(record_id) for record_id in split[part]["record_ids"]]
            indices = np.flatnonzero(np.isin(arrays["record_indices"], allowed)).astype(
                np.int64, copy=False
            )
            np.save(staging / f"{part}_indices.npy", indices, allow_pickle=False)
        close_arrays(arrays)
        write_json(staging / "metadata.json", metadata)
        write_json(staging / "split.json", split)
        verify_cache(staging, metadata, split)
        # Detect edits concurrent with preparation rather than publishing stale provenance.
        for item in provenance["files"]:
            if sha256_file(data / item["file"]) != item["sha256"]:
                raise ValueError(f"Source changed during preparation: {item['file']}")
        if sha256_file(data / "RECORDS") != provenance["records_sha256"]:
            raise ValueError("Original RECORDS changed during preparation.")
        if cache.exists():
            raise ValueError(
                "Another process created cache during preparation; refusing to overwrite it."
            )
        os.replace(staging, cache)
    finally:
        close_arrays(arrays)
        if staging.exists():
            shutil.rmtree(staging)
    print_summary(cache, metadata, split, reused=False)


def main() -> None:
    contract, inventory = read_configuration()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(contract["dataset"]),
        help="Original MIT-BIH directory containing RECORDS and all 48 .hea/.dat/.atr triples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / contract["output"],
        help="Experiment output directory; files are finalized atomically in its cache/ subdirectory.",
    )
    args = parser.parse_args()
    try:
        prepare(args.data, args.output, contract, inventory)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
