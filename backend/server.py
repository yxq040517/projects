"""PulseLab: a loopback-only, offline ECG research workstation."""

from __future__ import annotations

import csv
import io
import json
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend import model_registry
from backend.ecg import analyze_signal, parse_upload
from backend.library import (
    connect,
    demo_catalog,
    init_db,
    list_records,
    load_original,
    load_record,
    save_record,
)
from backend.reporting import render_report

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "data" / "demos"
MAX_UPLOAD = 20 * 1024 * 1024
DISCLAIMER = "仅用于科研、教学与信号复核，不用于诊断、治疗或急救决策。信号质量评分是未经临床验证的工程启发式指标，不是健康评分或诊断置信度。"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    model_registry.init_models()
    yield


app = FastAPI(
    title="PulseLab 心电智能分析",
    version="2.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_security(request, call_next):
    # A malicious web page must not mutate a localhost user's record library.
    if request.method in {"POST", "PATCH", "DELETE", "PUT"}:
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.url.netloc}"
        if origin and origin != expected:
            return JSONResponse({"detail": "只允许本机同源页面修改数据。"}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "拒绝跨站数据操作。"}, status_code=403)
        length = request.headers.get("content-length", "0")
        try:
            model_import = request.url.path.rstrip("/") == "/api/models/import"
            limit = (
                model_registry.MAX_MODEL_BYTES + model_registry.MAX_CONFIG_BYTES
                if model_import
                else MAX_UPLOAD
            )
            if int(length) > limit + 65536:
                message = (
                    "模型文件不能超过 100 MB，配置不能超过 512 KB。"
                    if model_import
                    else "上传文件不能超过 20 MB。"
                )
                return JSONResponse({"detail": message}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "请求长度格式错误。"}, status_code=400)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(_request, error: RequestValidationError):
    labels = {
        "file": "数据文件",
        "sample_rate": "采样率（100–1000 Hz）",
        "units": "幅值单位（mV/uV/V）",
        "notch": "工频陷波（0/50/60 Hz）",
        "name": "名称（1–80 字）",
        "note": "备注（最多 2000 字）",
        "lead": "导联名称",
        "body": "请求内容",
        "start": "起始时间（秒）",
        "end": "结束时间（秒）",
        "status": "人工复核状态",
        "left": "左侧记录",
        "right": "右侧记录",
        "config": "模型 JSON 配置",
        "model_id": "所选模型",
        "description": "模型备注（最多 1000 字）",
    }
    fields = []
    for item in error.errors():
        field = str(item["loc"][-1])
        label = labels.get(field, field)
        if field == "file" and _request.url.path.rstrip("/") == "/api/models/import":
            label = "ONNX 模型文件"
        if field == "note" and "/beats/" in _request.url.path:
            label = "心搏备注（最多 300 字）"
        if label not in fields:
            fields.append(label)
    return JSONResponse(
        {"detail": "参数缺失或格式不正确，请检查：" + "、".join(fields) + "。"},
        status_code=422,
    )


class DemoOptions(BaseModel):
    name: str = Field(default="", max_length=80)
    lead: str = Field(default="", max_length=64)
    notch: Literal[0, 50, 60] = 50


class RecordEdit(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=2000)


class ReanalysisOptions(BaseModel):
    lead: str = Field(min_length=1, max_length=80)
    notch: Literal[0, 50, 60] | None = None


class SegmentOptions(BaseModel):
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)
    lead: str = Field(default="", max_length=80)
    save: bool = False
    name: str = Field(default="", max_length=80)


class ReviewOptions(BaseModel):
    status: Literal["unreviewed", "in_progress", "reviewed"]


class BeatReviewOptions(BaseModel):
    status: Literal["confirmed", "artifact", "uncertain"]
    note: str = Field(default="", max_length=300)


class ModelEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=1000)


class PredictionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=80)
    start: float = Field(default=0, ge=0, allow_inf_nan=False)


def build_record(
    content: bytes,
    filename: str,
    sample_rate: float,
    units: str,
    lead: str,
    name: str,
    note: str,
    notch: int,
    source: str,
) -> dict:
    try:
        parsed = parse_upload(content, filename, sample_rate, units)
        selected = lead.strip() or next(iter(parsed["leads"]))
        result = analyze_signal(parsed["leads"], parsed["sample_rate"], selected, notch)
    except (ValueError, UnicodeError) as error:
        raise HTTPException(400, str(error)) from error
    record = {
        **result,
        "id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": name.strip() or Path(filename).stem[:80] or "未命名记录",
        "source": source,
        "filename": filename,
        "note": note.strip(),
        "parser_warnings": parsed["warnings"],
        "notch": notch,
        "engine": "PulseLab DSP 2.0 · 工程启发式，非医学诊断",
        "disclaimer": DISCLAIMER,
    }
    return save_record(record, parsed)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "engine": "DSP · 非诊断",
        "local_only": True,
        "model_runtime": "ONNX Runtime CPU",
    }


@app.get("/api/records")
def records():
    return list_records()


@app.get("/api/stats")
def stats():
    items = records()
    return {
        "total_records": len(items),
        "total_duration": round(sum(item["duration"] for item in items), 2),
        "review_records": sum(
            any(f["level"] != "info" for f in item["findings"]) for item in items
        ),
        "last_created_at": items[0]["created_at"] if items else None,
    }


@app.get("/api/demos")
def demos():
    return demo_catalog()


@app.post("/api/demos/{demo_id}/analyze")
def analyze_demo(demo_id: str, options: DemoOptions):
    entry = next((entry for entry in demo_catalog() if entry["id"] == demo_id), None)
    if entry is None:
        raise HTTPException(404, "示例不存在。")
    filename = f"{entry['id']}.csv"
    return build_record(
        (DEMOS / filename).read_bytes(),
        filename,
        entry["sample_rate"],
        "mV",
        options.lead,
        options.name or entry["name"],
        "",
        options.notch,
        entry["source"],
    )


@app.post("/api/analyze")
async def analyze_upload(
    file: Annotated[UploadFile, File()],
    sample_rate: Annotated[float, Form(ge=100, le=1000)] = 250,
    units: Annotated[Literal["mV", "uV", "V"], Form()] = "mV",
    lead: Annotated[str, Form(max_length=64)] = "",
    name: Annotated[str, Form(max_length=80)] = "",
    note: Annotated[str, Form(max_length=2000)] = "",
    notch: Annotated[int, Form()] = 50,
):
    if notch not in {0, 50, 60}:
        raise HTTPException(400, "工频陷波只能选择 0、50 或 60 Hz。")
    content = await file.read(MAX_UPLOAD + 1)
    await file.close()
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "上传文件不能超过 20 MB。")
    filename = (file.filename or "signal.csv").replace("\\", "/").rsplit("/", 1)[-1]
    if len(filename) > 200:
        raise HTTPException(400, "文件名过长，请缩短至 200 个字符以内。")
    if Path(filename).suffix.lower() not in {".csv", ".txt", ".json"}:
        raise HTTPException(400, "支持 CSV、TXT 和 JSON 数值波形，不支持图片或 PDF 心电图。")
    return await run_in_threadpool(
        build_record,
        content,
        filename,
        sample_rate,
        units,
        lead,
        name,
        note,
        notch,
        "用户导入 · 本地文件（来源未经核验）",
    )


@app.get("/api/records/{record_id}")
def get_record(record_id: str):
    return load_record(record_id)


@app.patch("/api/records/{record_id}")
def edit_record(record_id: str, edit: RecordEdit):
    record = load_record(record_id)
    if edit.name is not None:
        if not edit.name.strip():
            raise HTTPException(400, "记录名称不能为空。")
        record["name"] = edit.name.strip()
    if edit.note is not None:
        record["note"] = edit.note.strip()
    return save_record(record)


@app.delete("/api/records/{record_id}")
def delete_record(record_id: str):
    with connect() as connection:
        cursor = connection.execute("DELETE FROM records WHERE id = ?", (record_id,))
        if cursor.rowcount == 0:
            raise HTTPException(404, "分析记录不存在。")
    return {"ok": True}


@app.get("/api/records/{record_id}/export")
def export_record(record_id: str):
    record = load_record(record_id)
    body = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    return Response(
        body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="pulselab-{record_id[:8]}.json"'},
    )


@app.post("/api/records/{record_id}/reanalyze")
def reanalyze_record(record_id: str, options: ReanalysisOptions):
    record = load_record(record_id)
    original = load_original(record_id)
    notch = record["notch"] if options.notch is None else options.notch
    changed = options.lead != record["selected_lead"] or notch != record["notch"]
    try:
        result = analyze_signal(original["leads"], original["sample_rate"], options.lead, notch)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    record.update(result)
    record["notch"] = notch
    if changed:
        record["review_status"] = "in_progress"
    return save_record(record)


@app.post("/api/records/{record_id}/segment")
def analyze_segment(record_id: str, options: SegmentOptions):
    parent = load_record(record_id)
    original = load_original(record_id)
    rate = original["sample_rate"]
    count = len(next(iter(original["leads"].values())))
    duration = count / rate
    if options.end <= options.start or options.end > duration + 1e-6:
        raise HTTPException(400, f"选段范围必须递增且位于 0–{duration:g} 秒内。")
    start_index = int(round(options.start * rate))
    end_index = min(count, int(round(options.end * rate)))
    if end_index - start_index < 5 * rate - 1e-8:
        raise HTTPException(400, "所选原始采样不足 5 秒，请扩大时间范围。")
    actual_start = start_index / rate
    actual_end = end_index / rate
    cropped = {name: values[start_index:end_index] for name, values in original["leads"].items()}
    lead = options.lead.strip() or parent["selected_lead"]
    try:
        result = analyze_signal(cropped, rate, lead, parent["notch"])
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    record = {
        **result,
        "id": uuid4().hex if options.save else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": options.name.strip() or f"{parent['name']} · {actual_start:g}–{actual_end:g}s"[:80],
        "source": parent["source"],
        "filename": parent["filename"],
        "note": "",
        "parser_warnings": [],
        "notch": parent["notch"],
        "engine": "PulseLab DSP 2.0 · 工程启发式，非医学诊断",
        "disclaimer": DISCLAIMER,
        "schema_version": 2,
        "has_original": True,
        "review_status": "unreviewed",
        "beat_reviews": [],
        "segment": {"parent_id": record_id, "start": actual_start, "end": actual_end},
        "source_offset": parent.get("source_offset", 0.0) + actual_start,
    }
    if options.save:
        return save_record(record, {"sample_rate": rate, "leads": cropped})
    return record


@app.patch("/api/records/{record_id}/review")
def review_record(record_id: str, options: ReviewOptions):
    record = load_record(record_id)
    record["review_status"] = options.status
    return save_record(record)


@app.put("/api/records/{record_id}/beats/{index}")
def review_beat(record_id: str, index: int, options: BeatReviewOptions):
    record = load_record(record_id)
    peak = next((peak for peak in record["peaks"] if peak["index"] == index), None)
    if peak is None:
        raise HTTPException(400, "该采样点不是当前分析导联检出的 R 峰，请刷新记录后选择。")
    lead = record["selected_lead"]
    record["beat_reviews"] = [
        review
        for review in record["beat_reviews"]
        if not (review["lead"] == lead and review["index"] == index)
    ]
    record["beat_reviews"].append(
        {
            "lead": lead,
            "index": index,
            "time": peak["time"],
            "status": options.status,
            "note": options.note.strip(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    record["beat_reviews"].sort(key=lambda item: (item["lead"], item["index"]))
    if record["review_status"] != "reviewed":
        record["review_status"] = "in_progress"
    return save_record(record)


@app.delete("/api/records/{record_id}/beats/{index}")
def clear_beat_review(record_id: str, index: int):
    record = load_record(record_id)
    lead = record["selected_lead"]
    record["beat_reviews"] = [
        review
        for review in record["beat_reviews"]
        if not (review["lead"] == lead and review["index"] == index)
    ]
    return save_record(record)


@app.get("/api/compare")
def compare_records(
    left: Annotated[str, Query(min_length=1)], right: Annotated[str, Query(min_length=1)]
):
    if left == right:
        raise HTTPException(400, "请选择两条不同的记录。")
    a, b = load_record(left), load_record(right)
    fields = [
        ("heart_rate", "平均心率", "bpm"),
        ("mean_rr", "平均 RR", "ms"),
        ("sdnn", "SDRR", "ms"),
        ("rmssd", "相邻 RR 差均方根", "ms"),
        ("pnn50", "pRR50", "%"),
        ("beat_count", "检出峰数", "个"),
        ("quality", "信号质量（工程评分）", "/100"),
    ]
    differences = []
    for key, label, unit in fields:
        av = a["quality"]["score"] if key == "quality" else a["metrics"].get(key)
        bv = b["quality"]["score"] if key == "quality" else b["metrics"].get(key)
        delta = (
            round(bv - av, 3)
            if isinstance(av, (int, float))
            and isinstance(bv, (int, float))
            and math.isfinite(av)
            and math.isfinite(bv)
            else None
        )
        differences.append(
            {"key": key, "label": label, "unit": unit, "left": av, "right": bv, "delta": delta}
        )
    return {
        "left": a,
        "right": b,
        "same_lead": a["selected_lead"] == b["selected_lead"],
        "differences": differences,
    }


@app.get("/api/records/{record_id}/signal.csv")
def export_original(record_id: str):
    original = load_original(record_id)
    rate = original["sample_rate"]
    names = list(original["leads"])

    def chunks():
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        # Prevent a user-supplied lead label being executed by spreadsheet software.
        headers = [
            "'" + name if name.lstrip().startswith(("=", "+", "-", "@")) else name for name in names
        ]
        writer.writerow(["time_s", *headers])
        count = len(original["leads"][names[0]])
        for index in range(count):
            writer.writerow(
                [
                    f"{index / rate:.10f}",
                    *(format(original["leads"][name][index], ".17g") for name in names),
                ]
            )
            if index % 1000 == 999:
                yield buffer.getvalue()
                buffer.seek(0)
                buffer.truncate(0)
        if buffer.tell():
            yield buffer.getvalue()

    return StreamingResponse(
        chunks(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="pulselab-original-{record_id[:8]}.csv"'
        },
    )


@app.get("/api/models")
def list_models():
    return model_registry.list_models()


@app.get("/api/models/template")
def model_template():
    body = json.dumps(model_registry.TEMPLATE, ensure_ascii=False, indent=2)
    return Response(
        body,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="pulselab-model-config.json"'},
    )


@app.post("/api/models/import")
async def import_model(file: Annotated[UploadFile, File()], config: Annotated[str, Form()]):
    content = await file.read(model_registry.MAX_MODEL_BYTES + 1)
    await file.close()
    filename = (file.filename or "model.onnx").replace("\\", "/").rsplit("/", 1)[-1]
    if len(filename) > 200:
        raise HTTPException(400, "模型文件名过长，请缩短至 200 字以内。")
    return await run_in_threadpool(model_registry.import_model, content, filename, config)


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    return model_registry.get_model(model_id)


@app.patch("/api/models/{model_id}")
def edit_model(model_id: str, options: ModelEdit):
    return model_registry.edit_model(model_id, options.name, options.description)


@app.post("/api/models/{model_id}/default")
def default_model(model_id: str):
    return model_registry.set_default(model_id)


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    return model_registry.delete_model(model_id)


@app.get("/api/models/{model_id}/manifest")
def export_model_manifest(model_id: str):
    body = json.dumps(
        model_registry.manifest_of(model_registry.get_model(model_id)), ensure_ascii=False, indent=2
    )
    return Response(
        body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="model-config-{model_id[:8]}.json"'},
    )


@app.get("/api/records/{record_id}/models")
def record_models(record_id: str):
    return model_registry.compatible_models(record_id)


@app.post("/api/records/{record_id}/predict")
def predict_record(record_id: str, options: PredictionOptions):
    return model_registry.run_prediction(record_id, options.model_id, options.start)


@app.get("/api/records/{record_id}/predictions")
def record_predictions(record_id: str, model_id: str | None = None):
    return model_registry.list_predictions(record_id, model_id)


@app.get("/api/predictions/{prediction_id}/export")
def export_prediction(prediction_id: str):
    body = json.dumps(
        model_registry.get_prediction(prediction_id), ensure_ascii=False, allow_nan=False, indent=2
    )
    return Response(
        body,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="model-prediction-{prediction_id[:8]}.json"'
        },
    )


@app.get("/api/template.csv")
def template():
    path = DEMOS / "synthetic-regular.csv"
    if not path.exists():
        raise HTTPException(404, "模板尚未生成，请运行 scripts/prepare_data.py。")
    # Take the first 10 seconds; timestamp provides an unambiguous sampling calibration.
    rows = path.read_text(encoding="utf-8").splitlines()[:2501]
    return Response(
        "\n".join(rows) + "\n",
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="synthetic-template-fs250.csv"'},
    )


@app.get("/api/records/{record_id}/report", response_class=HTMLResponse)
def report(record_id: str, prediction_id: str | None = None):
    record = load_record(record_id)
    prediction = model_registry.report_prediction(record_id, prediction_id)
    return HTMLResponse(render_report(record, prediction))


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


# Mount only assets, never source code, uploaded data or the SQLite database.
(ROOT / "web").mkdir(exist_ok=True)
app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")
