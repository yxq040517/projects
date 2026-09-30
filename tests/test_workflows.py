"""Observable contracts for original-sample workflows and safe legacy migration."""

from __future__ import annotations

import csv
import gzip
import io
import json
import sqlite3
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import library
from backend.ecg import analyze_signal
from backend.server import app


def waveform(rate=250, seconds=24, bpm=75):
    time = np.arange(rate * seconds) / rate
    values = 0.03 * np.sin(2 * np.pi * 0.25 * time)
    for beat in np.arange(0.7, seconds - 0.3, 60 / bpm):
        values += np.exp(-0.5 * ((time - beat) / 0.012) ** 2)
        values -= 0.2 * np.exp(-0.5 * ((time - beat - 0.035) / 0.012) ** 2)
        values += 0.25 * np.exp(-0.5 * ((time - beat - 0.24) / 0.06) ** 2)
    return {"II": values.tolist(), "V1": (-0.7 * values).tolist()}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "DATA", tmp_path)
    monkeypatch.setattr(library, "DB", tmp_path / "records.sqlite3")
    return tmp_path


@pytest.fixture
def client(data_dir):
    with TestClient(app) as client:
        yield client


def upload(client, leads=None, name="研究记录"):
    payload = {"sample_rate": 250, "units": "mV", "leads": leads or waveform()}
    response = client.post(
        "/api/analyze",
        files={"file": ("study.json", json.dumps(payload).encode(), "application/json")},
        data={"name": name},
    )
    assert response.status_code == 200, response.text
    return response.json()


def csv_rows(client, record_id):
    response = client.get(f"/api/records/{record_id}/signal.csv")
    assert response.status_code == 200
    return list(csv.reader(io.StringIO(response.text)))


def test_metadata_edits_and_lead_reanalysis_preserve_exact_original_samples(client):
    leads = waveform()
    record = upload(client, leads)
    record_id = record["id"]
    original = csv_rows(client, record_id)
    assert len(original) == 6001
    assert original[0] == ["time_s", "II", "V1"]
    np.testing.assert_array_equal([float(row[1]) for row in original[1:]], leads["II"])
    response = client.patch(
        f"/api/records/{record_id}", json={"name": "编辑后", "note": "电极条件已记录"}
    )
    assert response.status_code == 200
    updated = client.post(f"/api/records/{record_id}/reanalyze", json={"lead": "V1", "notch": 0})
    assert updated.status_code == 200
    body = updated.json()
    assert body["id"] == record_id and body["selected_lead"] == "V1"
    assert body["name"] == "编辑后" and body["note"] == "电极条件已记录"
    assert body["review_status"] == "in_progress"
    assert abs(body["metrics"]["heart_rate"] - 75) < 1
    assert body["tools"]["template"]["reliable"] is True
    assert csv_rows(client, record_id) == original


def test_segment_preview_is_nonmutating_and_saved_crop_survives_parent_deletion(client):
    leads = waveform()
    parent = upload(client, leads)
    record_id = parent["id"]
    options = {"start": 5, "end": 15, "lead": "II", "save": False}
    preview = client.post(f"/api/records/{record_id}/segment", json=options)
    assert preview.status_code == 200
    result = preview.json()
    assert result["id"] is None and result["duration"] == 10
    assert result["segment"] == {"parent_id": record_id, "start": 5, "end": 15}
    assert len(client.get("/api/records").json()) == 1
    assert client.get(f"/api/records/{record_id}").json()["metrics"] == parent["metrics"]
    saved = client.post(
        f"/api/records/{record_id}/segment", json={**options, "save": True, "name": "独立片段"}
    ).json()
    assert saved["id"] != record_id and saved["source_offset"] == 5
    rows = csv_rows(client, saved["id"])
    assert len(rows) == 2501
    np.testing.assert_array_equal([float(row[1]) for row in rows[1:]], leads["II"][1250:3750])
    assert client.delete(f"/api/records/{record_id}").status_code == 200
    assert csv_rows(client, saved["id"]) == rows
    child = client.post(f"/api/records/{saved['id']}/reanalyze", json={"lead": "V1"})
    assert child.status_code == 200 and abs(child.json()["metrics"]["heart_rate"] - 75) < 1


def test_segment_bounds_reject_short_reversed_outside_and_nonfinite_times(client):
    record = upload(client)
    url = f"/api/records/{record['id']}/segment"
    for options in ({"start": 10, "end": 9}, {"start": 0, "end": 4.9}, {"start": 20, "end": 30}):
        response = client.post(url, json=options)
        assert response.status_code == 400
    response = client.post(
        url, content='{"start":0,"end":NaN}', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422 and isinstance(response.json()["detail"], str)
    assert len(client.get("/api/records").json()) == 1


def test_manual_reviews_are_lead_scoped_persistent_and_do_not_change_algorithm_metrics(client):
    record = upload(client)
    record_id = record["id"]
    index = record["rr"][0]["peak_index"]
    url = f"/api/records/{record_id}/beats/{index}"
    response = client.put(url, json={"status": "artifact", "note": "<script>bad()</script>"})
    assert response.status_code == 200
    marked = response.json()
    assert marked["metrics"] == record["metrics"]
    assert marked["review_status"] == "in_progress"
    assert marked["beat_reviews"][0]["lead"] == "II"
    report = client.get(f"/api/records/{record_id}/report")
    assert report.status_code == 200
    assert (
        "<script>bad()</script>" not in report.text
        and "&lt;script&gt;bad()&lt;/script&gt;" in report.text
    )
    assert (
        client.put(f"/api/records/{record_id}/beats/1", json={"status": "confirmed"}).status_code
        == 400
    )
    assert (
        client.patch(f"/api/records/{record_id}/review", json={"status": "reviewed"}).json()[
            "review_status"
        ]
        == "reviewed"
    )
    assert client.get(f"/api/records/{record_id}").json()["beat_reviews"] == marked["beat_reviews"]
    switched = client.post(f"/api/records/{record_id}/reanalyze", json={"lead": "V1"}).json()
    assert switched["review_status"] == "in_progress"
    assert switched["beat_reviews"] == marked["beat_reviews"]
    # Clearing an index on a different lead must not delete the original lead's review.
    assert client.delete(url).json()["beat_reviews"] == marked["beat_reviews"]
    client.post(f"/api/records/{record_id}/reanalyze", json={"lead": "II"})
    assert client.delete(url).json()["beat_reviews"] == []


def test_comparison_reports_numerical_delta_without_inventing_missing_metrics(client):
    slow = upload(client, waveform(bpm=50), "较慢记录")
    fast = upload(client, waveform(bpm=120), "较快记录")
    result = client.get("/api/compare", params={"left": slow["id"], "right": fast["id"]}).json()
    assert result["same_lead"] is True
    heart_rate = next(row for row in result["differences"] if row["key"] == "heart_rate")
    assert abs(heart_rate["delta"] - 70) < 1
    assert (
        client.get("/api/compare", params={"left": slow["id"], "right": slow["id"]}).status_code
        == 400
    )
    flat = upload(client, {"II": [0.0] * 6000}, "无法测量记录")
    result = client.get("/api/compare", params={"left": flat["id"], "right": fast["id"]}).json()
    heart_rate = next(row for row in result["differences"] if row["key"] == "heart_rate")
    assert heart_rate["left"] is None and heart_rate["delta"] is None


def legacy_record(parsed, source, filename):
    result = analyze_signal(parsed["leads"], parsed["sample_rate"], next(iter(parsed["leads"])))
    result.pop("tools", None)
    for rr in result["rr"]:
        rr.pop("peak_index", None)
    return {
        **result,
        "id": uuid4().hex,
        "created_at": "2026-09-30T00:00:00+00:00",
        "name": "原有名称",
        "note": "用户不能丢失的备注",
        "source": source,
        "filename": filename,
        "parser_warnings": [],
        "notch": 50,
        "engine": "1.0",
        "disclaimer": "研究用途",
    }


def write_legacy(path, record):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE records(id TEXT PRIMARY KEY,created_at TEXT,summary TEXT,payload BLOB)"
        )
        connection.execute(
            "INSERT INTO records VALUES(?,?,?,?)",
            (record["id"], record["created_at"], "{}", gzip.compress(json.dumps(record).encode())),
        )


def test_migration_backs_up_and_recovers_verified_public_original_without_losing_human_data(
    data_dir,
):
    entry = library.demo_catalog()[0]
    filename = f"{entry['id']}.csv"
    parsed = library.parse_upload(
        (library.DEMOS / filename).read_bytes(), filename, entry["sample_rate"], "mV"
    )
    record = legacy_record(parsed, entry["source"], filename)
    write_legacy(library.DB, record)
    with TestClient(app) as client:
        migrated = client.get(f"/api/records/{record['id']}").json()
        for field in ("id", "name", "note", "created_at"):
            assert migrated[field] == record[field]
        assert migrated["has_original"] and migrated["tools"]["template"]["used_count"] > 3
        rows = csv_rows(client, record["id"])
        assert len(rows) == len(next(iter(parsed["leads"].values()))) + 1
    backups = list((data_dir / "backups").glob("*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as connection:
        old_payload = connection.execute(
            "SELECT payload FROM records WHERE id=?", (record["id"],)
        ).fetchone()[0]
    assert json.loads(gzip.decompress(old_payload))["note"] == record["note"]


def test_legacy_upload_never_reconstructs_original_from_display_decimation(data_dir):
    parsed = {"leads": waveform(seconds=60), "sample_rate": 250}
    record = legacy_record(parsed, "用户导入 · 本地文件（来源未经核验）", "mitbih-100.csv")
    write_legacy(library.DB, record)
    with TestClient(app) as client:
        legacy = client.get(f"/api/records/{record['id']}").json()
        assert legacy["has_original"] is False and legacy["tools"] is None
        assert legacy["metrics"] == record["metrics"] and legacy["note"] == record["note"]
        for endpoint, options in (
            ("reanalyze", {"lead": "II"}),
            ("segment", {"start": 0, "end": 10}),
        ):
            response = client.post(f"/api/records/{record['id']}/{endpoint}", json=options)
            assert response.status_code == 409 and "重新导入" in response.json()["detail"]
        assert client.get(f"/api/records/{record['id']}/signal.csv").status_code == 409
