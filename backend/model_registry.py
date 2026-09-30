"""Local ONNX model lifecycle and immutable, model-specific prediction snapshots."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import OrderedDict
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from fastapi import HTTPException

from backend.ecg import _json_object
from backend.library import connect, load_original, load_record
from backend.model_runtime import compatibility, predict, validate_model

MAX_MODEL_BYTES = 100 * 1024 * 1024
MAX_CONFIG_BYTES = 512 * 1024
_CACHE = OrderedDict()
_CACHE_LOCK = RLock()

TEMPLATE = {
    "schema_version": 1,
    "name": "MIT-BIH 心搏分类",
    "version": "1.0",
    "task": "beat_classification",
    "classes": ["N", "S", "V", "F", "Q"],
    "sample_rate": 360,
    "input_samples": 256,
    "lead_names": ["MLII"],
    "pre_samples": 90,
    "normalization": "none",
    "input_filter": "raw",
    "notch": 0,
    "output_type": "logits",
    "threshold": 0.5,
    "description": "",
}


def init_models() -> None:
    with connect() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL,
            metadata TEXT NOT NULL, signature TEXT NOT NULL, onnx BLOB NOT NULL,
            sha256 TEXT NOT NULL, filename TEXT NOT NULL, size_bytes INTEGER NOT NULL,
            imported_at TEXT NOT NULL, UNIQUE(name,version)
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS model_settings (
            key TEXT PRIMARY KEY, value TEXT
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS model_predictions (
            id TEXT PRIMARY KEY, record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL, created_at TEXT NOT NULL, payload BLOB NOT NULL
        )""")
        connection.execute("""CREATE INDEX IF NOT EXISTS predictions_record_time
            ON model_predictions(record_id,created_at DESC)""")


def _default(connection) -> str | None:
    row = connection.execute(
        "SELECT value FROM model_settings WHERE key='default_model_id'"
    ).fetchone()
    return row["value"] if row is not None else None


def _metadata(row, default_id) -> dict:
    manifest = json.loads(row["metadata"])
    return {
        **manifest,
        "id": row["id"],
        "signature": json.loads(row["signature"]),
        "sha256": row["sha256"],
        "filename": row["filename"],
        "size_bytes": row["size_bytes"],
        "imported_at": row["imported_at"],
        "is_default": row["id"] == default_id,
    }


def list_models() -> dict:
    with connect() as connection:
        rows = connection.execute("""SELECT id,metadata,signature,sha256,filename,size_bytes,imported_at
            FROM models ORDER BY imported_at DESC,id DESC""").fetchall()
        default_id = _default(connection)
    return {"models": [_metadata(row, default_id) for row in rows], "default_model_id": default_id}


def get_model(model_id: str) -> dict:
    with connect() as connection:
        row = connection.execute(
            """SELECT id,metadata,signature,sha256,filename,size_bytes,imported_at
            FROM models WHERE id=?""",
            (model_id,),
        ).fetchone()
        default_id = _default(connection)
    if row is None:
        raise HTTPException(404, "模型不存在，可能已被删除。")
    return _metadata(row, default_id)


def manifest_of(model: dict) -> dict:
    return {key: model[key] for key in TEMPLATE}


def import_model(content: bytes, filename: str, config: str) -> dict:
    if not filename.lower().endswith(".onnx"):
        raise HTTPException(400, "支持 ONNX 模型；.pt/.pth 等训练权重请先从训练框架导出为 .onnx。")
    if not content:
        raise HTTPException(400, "模型文件为空。")
    if len(content) > MAX_MODEL_BYTES:
        raise HTTPException(413, "模型文件不能超过 100 MB。")
    if len(config.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise HTTPException(413, "模型配置不能超过 512 KB。")
    try:
        metadata = json.loads(config, object_pairs_hook=_json_object)
        if not isinstance(metadata, dict):
            raise ValueError("模型配置必须为 JSON 对象。")
        manifest, session, signature = validate_model(content, metadata)
    except (ValueError, UnicodeError) as error:
        raise HTTPException(400, str(error)) from error
    model_id = uuid4().hex
    digest = hashlib.sha256(content).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with connect() as connection:
            connection.execute(
                """INSERT INTO models
                (id,name,version,metadata,signature,onnx,sha256,filename,size_bytes,imported_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    model_id,
                    manifest["name"],
                    manifest["version"],
                    json.dumps(manifest, ensure_ascii=False, allow_nan=False),
                    json.dumps(signature, ensure_ascii=False, allow_nan=False),
                    content,
                    digest,
                    filename,
                    len(content),
                    timestamp,
                ),
            )
            if _default(connection) is None:
                connection.execute(
                    """INSERT INTO model_settings(key,value) VALUES('default_model_id',?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                    (model_id,),
                )
    except sqlite3.IntegrityError as error:
        raise HTTPException(409, "同名同版本模型已经存在，请使用新的名称或版本。") from error
    with _CACHE_LOCK:
        _CACHE[model_id] = session
        _CACHE.move_to_end(model_id)
        while len(_CACHE) > 2:
            _CACHE.popitem(last=False)
    return get_model(model_id)


def edit_model(model_id: str, name: str | None = None, description: str | None = None) -> dict:
    current = get_model(model_id)
    metadata = manifest_of(current)
    if name is not None:
        if not name.strip():
            raise HTTPException(400, "模型名称不能为空。")
        metadata["name"] = name.strip()
    if description is not None:
        metadata["description"] = description.strip()
    try:
        with connect() as connection:
            cursor = connection.execute(
                "UPDATE models SET name=?,metadata=? WHERE id=?",
                (
                    metadata["name"],
                    json.dumps(metadata, ensure_ascii=False, allow_nan=False),
                    model_id,
                ),
            )
            if cursor.rowcount == 0:
                raise HTTPException(404, "模型不存在。")
    except sqlite3.IntegrityError as error:
        raise HTTPException(409, "修改后的名称与已有同版本模型重复。") from error
    return get_model(model_id)


def set_default(model_id: str) -> dict:
    with connect() as connection:
        if connection.execute("SELECT 1 FROM models WHERE id=?", (model_id,)).fetchone() is None:
            raise HTTPException(404, "模型不存在。")
        connection.execute(
            """INSERT INTO model_settings(key,value) VALUES('default_model_id',?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (model_id,),
        )
    return {"default_model_id": model_id}


def delete_model(model_id: str) -> dict:
    with connect() as connection:
        cursor = connection.execute("DELETE FROM models WHERE id=?", (model_id,))
        if cursor.rowcount == 0:
            raise HTTPException(404, "模型不存在。")
        if _default(connection) == model_id:
            connection.execute("UPDATE model_settings SET value=NULL WHERE key='default_model_id'")
        default_id = _default(connection)
    with _CACHE_LOCK:
        _CACHE.pop(model_id, None)
    return {"ok": True, "default_model_id": default_id}


def compatible_models(record_id: str) -> dict:
    record = load_record(record_id)
    result = list_models()
    for model in result["models"]:
        compatible, reason = compatibility(manifest_of(model), record)
        model.update({"compatible": compatible, "reason": reason})
    return result


def _runtime(model_id: str):
    model = get_model(model_id)
    with _CACHE_LOCK:
        session = _CACHE.get(model_id)
        if session is not None:
            _CACHE.move_to_end(model_id)
            return model, session
        with connect() as connection:
            row = connection.execute("SELECT onnx FROM models WHERE id=?", (model_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "模型已被删除。")
        try:
            _, session, _ = validate_model(row["onnx"], manifest_of(model))
        except ValueError as error:
            raise HTTPException(409, f"已保存模型无法加载，请重新导入：{error}") from error
        _CACHE[model_id] = session
        while len(_CACHE) > 2:
            _CACHE.popitem(last=False)
    return model, session


def record_context(record: dict) -> dict:
    # Review notes/names are intentionally excluded: they do not change numerical model inputs.
    indices = [peak["index"] for peak in record["peaks"]]
    digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
    return {
        "selected_lead": record["selected_lead"],
        "notch": record["notch"],
        "sample_rate": record["sample_rate"],
        "duration": record["duration"],
        "peaks_hash": digest,
    }


def run_prediction(record_id: str, model_id: str, start: float = 0) -> dict:
    record = load_record(record_id)
    original = load_original(record_id)
    model, session = _runtime(model_id)
    manifest = manifest_of(model)
    allowed, reason = compatibility(manifest, record)
    if not allowed:
        raise HTTPException(400, reason)
    try:
        result = predict(session, manifest, model["signature"], record, original, start)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    snapshot = {
        **result,
        "id": uuid4().hex,
        "record_id": record_id,
        "model_id": model_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": model["name"],
            "version": model["version"],
            "sha256": model["sha256"],
            "manifest": manifest,
        },
        "record_context": record_context(record),
    }
    payload = gzip.compress(
        json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    try:
        with connect() as connection:
            connection.execute(
                "INSERT INTO model_predictions VALUES(?,?,?,?,?)",
                (snapshot["id"], record_id, model_id, snapshot["created_at"], payload),
            )
    except sqlite3.IntegrityError as error:
        raise HTTPException(409, "记录在推理期间被删除，结果未保存。") from error
    return get_prediction(snapshot["id"])


def _status(prediction: dict, current_context: dict, existing_models: set[str]) -> dict:
    stale = prediction["record_context"] != current_context
    prediction.update(
        {
            "stale": stale,
            "stale_reason": "记录的分析导联或检测设置已改变，请重新运行模型。" if stale else "",
            "model_available": prediction["model_id"] in existing_models,
        }
    )
    return prediction


def list_predictions(record_id: str, model_id: str | None = None) -> dict:
    record = load_record(record_id)
    parameters = [record_id]
    clause = ""
    if model_id is not None:
        clause = " AND model_id=?"
        parameters.append(model_id)
    with connect() as connection:
        rows = connection.execute(
            "SELECT payload FROM model_predictions WHERE record_id=?"
            + clause
            + " ORDER BY created_at DESC,id DESC LIMIT 20",
            parameters,
        ).fetchall()
        existing = {row["id"] for row in connection.execute("SELECT id FROM models")}
    context = record_context(record)
    predictions = [
        _status(json.loads(gzip.decompress(row["payload"])), context, existing) for row in rows
    ]
    return {"predictions": predictions}


def get_prediction(prediction_id: str) -> dict:
    with connect() as connection:
        row = connection.execute(
            "SELECT payload FROM model_predictions WHERE id=?", (prediction_id,)
        ).fetchone()
        existing = {row["id"] for row in connection.execute("SELECT id FROM models")}
    if row is None:
        raise HTTPException(404, "模型结果不存在，可能已随记录删除。")
    prediction = json.loads(gzip.decompress(row["payload"]))
    return _status(prediction, record_context(load_record(prediction["record_id"])), existing)


def report_prediction(record_id: str, prediction_id: str | None = None) -> dict | None:
    if prediction_id is not None:
        prediction = get_prediction(prediction_id)
        if prediction["record_id"] != record_id:
            raise HTTPException(400, "模型结果不属于当前记录。")
        return prediction
    return next(
        (
            prediction
            for prediction in list_predictions(record_id)["predictions"]
            if not prediction["stale"]
        ),
        None,
    )
