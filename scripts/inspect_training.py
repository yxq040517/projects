"""Inspect local MIT-BIH source files and training hardware without touching user records."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "mit-bih-arrhythmia-database" / "mit-bih-arrhythmia-database-1.0.0"
CLASS_SYMBOLS = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "F": ["F"],
    "Q": ["/", "f", "Q"],
}
SYMBOL_CLASS = {symbol: label for label, symbols in CLASS_SYMBOLS.items() for symbol in symbols}


def inspect(data: Path):
    record_ids = data.joinpath("RECORDS").read_text(encoding="ascii").split()
    if len(record_ids) != 48 or len(set(record_ids)) != 48:
        raise ValueError("Expected the complete original 48-record MIT-BIH dataset.")
    checksums = {}
    for line in data.joinpath("SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            checksums[parts[1].strip().lstrip("*").removeprefix("./")] = parts[0]
    records = []
    totals = Counter()
    verified = []
    for record_id in record_ids:
        for suffix in (".hea", ".dat", ".atr"):
            filename = record_id + suffix
            path = data / filename
            if not path.is_file():
                raise ValueError(f"Missing required source file: {filename}")
            expected = checksums.get(filename)
            if expected is None:
                raise ValueError(f"Official checksum missing for {filename}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"Checksum mismatch: {filename}")
            verified.append({"file": filename, "sha256": digest})
        header = wfdb.rdheader(str(data / record_id))
        annotation = wfdb.rdann(str(data / record_id), "atr")
        symbols = Counter(annotation.symbol)
        classes = Counter(
            SYMBOL_CLASS[symbol] for symbol in annotation.symbol if symbol in SYMBOL_CLASS
        )
        totals.update(classes)
        preferred = (
            "MLII"
            if "MLII" in header.sig_name
            else "V5"
            if "V5" in header.sig_name
            else header.sig_name[0]
        )
        records.append(
            {
                "record": record_id,
                "subject_group": "201_202" if record_id in ("201", "202") else record_id,
                "sample_rate": header.fs,
                "samples": header.sig_len,
                "lead_names": header.sig_name,
                "selected_lead": preferred,
                "units": header.units,
                "annotation_symbols": dict(symbols),
                "class_counts": {label: classes[label] for label in CLASS_SYMBOLS},
            }
        )
        print(f"{record_id}: {preferred}, {dict(classes)}", flush=True)
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count(),
        "cpu": platform.processor(),
        "threads": torch.get_num_threads(),
        "numpy_version": np.__version__,
        "wfdb_version": wfdb.__version__,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    result = {
        "dataset_path": str(data),
        "records": records,
        "record_count": len(records),
        "subject_count": len({record["subject_group"] for record in records}),
        "class_mapping": CLASS_SYMBOLS,
        "class_counts": {label: totals[label] for label in CLASS_SYMBOLS},
        "files_verified": len(verified),
        "verified_source_files": verified,
        "environment": info,
        "q_definition": "paced /, fusion of paced+normal f, unclassifiable Q; not AF or a generic disease diagnosis",
        "source": "https://physionet.org/content/mitdb/1.0.0/",
        "license": "ODC Attribution 1.0",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    result = inspect(args.data)
    output = ROOT / "training"
    output.mkdir(exist_ok=True)
    output.joinpath("dataset_inventory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "class_counts": result["class_counts"],
                "record_count": result["record_count"],
                "subject_count": result["subject_count"],
                "files_verified": result["files_verified"],
                "environment": result["environment"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
