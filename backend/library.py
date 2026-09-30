"""Local record library and original-sample persistence, independent of display decimation."""

from __future__ import annotations

import gzip
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from backend.ecg import analyze_signal, parse_upload

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PULSELAB_DATA_DIR", str(ROOT / "storage")))
DB = DATA / "records.sqlite3"
DEMOS = ROOT / "data" / "demos"


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def demo_catalog() -> list[dict]:
    metadata = DEMOS / "catalog.json"
    if not metadata.exists():
        return []
    entries = json.loads(metadata.read_text(encoding="utf-8"))
    return [entry for entry in entries if (DEMOS / f"{entry['id']}.csv").is_file()]


def normalize_record(record: dict, has_original: bool) -> dict:
    record["schema_version"] = 2
    record["has_original"] = has_original
    record.setdefault("review_status", "unreviewed")
    record.setdefault("beat_reviews", [])
    record.setdefault("segment", None)
    record.setdefault("source_offset", 0.0)
    return record


def summary(record: dict) -> dict:
    keys = (
        "id",
        "name",
        "created_at",
        "source",
        "filename",
        "sample_rate",
        "duration",
        "lead_names",
        "selected_lead",
        "metrics",
        "quality",
        "findings",
        "note",
        "has_original",
        "review_status",
        "segment",
        "source_offset",
    )
    result = {key: record[key] for key in keys}
    result["reviewed_beats"] = sum(
        review["lead"] == record["selected_lead"] for review in record["beat_reviews"]
    )
    return result


def encode_original(original: dict) -> bytes:
    names = list(original["leads"])
    count = len(original["leads"][names[0]])
    # Float64 preserves imported values; one contiguous matrix avoids object/pickle storage.
    samples = np.empty((len(names), count), dtype=np.float64)
    for index, name in enumerate(names):
        samples[index] = original["leads"][name]
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        names=np.asarray(names, dtype=str),
        samples=samples,
        sample_rate=np.asarray(original["sample_rate"], dtype=np.float64),
    )
    return buffer.getvalue()


def decode_original(payload: bytes) -> dict:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        names = archive["names"].tolist()
        samples = archive["samples"]
        rate = float(archive["sample_rate"])
    return {
        "sample_rate": rate,
        "units": "mV",
        "leads": {name: samples[index] for index, name in enumerate(names)},
    }


def save_record(record: dict, original: dict | None = None) -> dict:
    original_bytes = encode_original(original) if original is not None else None
    with connect() as connection:
        has_original = (
            original is not None
            or connection.execute(
                "SELECT 1 FROM original_signals WHERE record_id=?", (record["id"],)
            ).fetchone()
            is not None
        )
        normalize_record(record, has_original)
        encoded = gzip.compress(
            json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        compact = json.dumps(summary(record), ensure_ascii=False, allow_nan=False)
        # REPLACE would delete the existing row and cascade-delete its source samples.
        connection.execute(
            """INSERT INTO records(id,created_at,summary,payload) VALUES (?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET created_at=excluded.created_at,
            summary=excluded.summary,payload=excluded.payload""",
            (record["id"], record["created_at"], compact, encoded),
        )
        if original_bytes is not None:
            connection.execute(
                """INSERT INTO original_signals(record_id,payload) VALUES (?,?)
                ON CONFLICT(record_id) DO UPDATE SET payload=excluded.payload""",
                (record["id"], original_bytes),
            )
    return record


def load_record(record_id: str) -> dict:
    with connect() as connection:
        row = connection.execute(
            """SELECT r.payload, o.record_id IS NOT NULL AS has_original
            FROM records r LEFT JOIN original_signals o ON o.record_id=r.id WHERE r.id=?""",
            (record_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "分析记录不存在，可能已被删除。")
    return normalize_record(json.loads(gzip.decompress(row["payload"])), bool(row["has_original"]))


def load_original(record_id: str) -> dict:
    with connect() as connection:
        row = connection.execute(
            """SELECT o.payload FROM records r
            LEFT JOIN original_signals o ON o.record_id=r.id WHERE r.id=?""",
            (record_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "分析记录不存在。")
    if row["payload"] is None:
        raise HTTPException(
            409,
            "这条旧记录没有保留原始采样，不能重新分析或截取；请重新导入原文件。现有测量结果仍可查看。",
        )
    try:
        return decode_original(row["payload"])
    except (ValueError, KeyError, OSError) as error:
        raise HTTPException(409, "原始波形存储损坏，请从原文件重新导入。") from error


def list_records() -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT summary FROM records ORDER BY created_at DESC,id DESC"
        ).fetchall()
    return [json.loads(row["summary"]) for row in rows]


def recover_public_original(record: dict, catalog: list[dict]) -> dict | None:
    if record.get("segment") or record.get("source_offset", 0):
        return None
    entry = next(
        (
            entry
            for entry in catalog
            if f"{entry['id']}.csv" == record["filename"] and entry["source"] == record["source"]
        ),
        None,
    )
    if entry is None:
        return None
    parsed = parse_upload(
        (DEMOS / record["filename"]).read_bytes(), record["filename"], entry["sample_rate"], "mV"
    )
    names = list(parsed["leads"])
    rate = parsed["sample_rate"]
    duration = len(parsed["leads"][names[0]]) / rate
    if (
        names != record["lead_names"]
        or record["selected_lead"] not in parsed["leads"]
        or abs(rate - record["sample_rate"]) > 1e-4
        or abs(duration - record["duration"]) > 1e-4
    ):
        return None
    wave = record["waveform"]
    indices = np.rint(np.asarray(wave["time"]) * rate).astype(int)
    if (
        len(indices) != len(wave["raw"])
        or np.any(indices < 0)
        or np.any(indices >= int(round(duration * rate)))
    ):
        return None
    original = np.asarray(parsed["leads"][record["selected_lead"]])
    if not np.allclose(original[indices], np.asarray(wave["raw"]), rtol=1e-6, atol=1e-8):
        return None
    for name in names:
        displayed = wave.get("leads", {}).get(name)
        if displayed is None or len(displayed) != len(indices):
            return None
        if not np.allclose(
            np.asarray(parsed["leads"][name])[indices], displayed, rtol=1e-6, atol=1e-8
        ):
            return None
    return parsed


def migrate_records() -> None:
    catalog = demo_catalog()
    with connect() as connection:
        rows = connection.execute("SELECT id,payload FROM records").fetchall()
    for row in rows:
        record = json.loads(gzip.decompress(row["payload"]))
        if record.get("schema_version", 1) >= 2:
            continue
        try:
            original = load_original(row["id"])
        except HTTPException as error:
            if error.status_code != 409:
                raise
            original = recover_public_original(record, catalog)
        if original is not None:
            # Original source verified, so upgrade outputs without changing identity or human metadata.
            updated = analyze_signal(
                original["leads"],
                original["sample_rate"],
                record["selected_lead"],
                record.get("notch", 50),
            )
            record.update(updated)
        else:
            record["tools"] = None
            record["findings"].append(
                {
                    "level": "warning",
                    "title": "缺少原始采样",
                    "detail": "旧版本只保存了显示波形，不能据此生成原始数据。请重新导入原文件以使用导联重分析、选段和模板工具；现有结果保持不变。",
                }
            )
        record["engine"] = "PulseLab DSP 2.0 · 工程启发式，非医学诊断"
        save_record(record, original)


def init_db() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        existing = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "records" in existing and "original_signals" not in existing:
            count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            if count:
                backups = DATA / "backups"
                backups.mkdir(exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                with sqlite3.connect(backups / f"before-v2-{stamp}.sqlite3") as backup:
                    connection.backup(backup)
        connection.execute("""CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, summary TEXT NOT NULL, payload BLOB NOT NULL
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS original_signals (
            record_id TEXT PRIMARY KEY REFERENCES records(id) ON DELETE CASCADE, payload BLOB NOT NULL
        )""")
    migrate_records()
