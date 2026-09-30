"""Real synthetic ONNX graphs verify model lifecycle and immutable prediction audit."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

from backend import library, model_registry
from backend.server import app


def synthetic_model(samples=128, bias=0.3):
    weights = np.column_stack(
        (np.full(samples, 1 / samples), np.full(samples, -1 / samples))
    ).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["signal"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weights"], ["weighted"]),
            helper.make_node("Add", ["weighted", "bias"], ["scores"]),
        ],
        "SyntheticIntegrationClassifier",
        [helper.make_tensor_value_info("signal", TensorProto.FLOAT, ["batch", 1, samples])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", 2])],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(np.asarray([bias, -bias], dtype=np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    return model.SerializeToString()


def config(name="合成集成测试模型", version="1", **changes):
    return {
        **model_registry.TEMPLATE,
        "name": name,
        "version": version,
        "classes": ["类别 A", "类别 B"],
        "lead_names": ["II"],
        "sample_rate": 250,
        "input_samples": 128,
        "pre_samples": 40,
        **changes,
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "DATA", tmp_path)
    monkeypatch.setattr(library, "DB", tmp_path / "records.sqlite3")
    with TestClient(app) as client:
        yield client


def import_model(client, metadata=None, content=None, filename="synthetic-test.onnx"):
    return client.post(
        "/api/models/import",
        files={"file": (filename, content or synthetic_model(), "application/octet-stream")},
        data={"config": json.dumps(metadata or config(), ensure_ascii=False)},
    )


def record(client):
    rate = 250
    time = np.arange(rate * 12) / rate
    signal = 0.02 * np.sin(2 * np.pi * 0.25 * time)
    for beat in np.arange(0.7, 11.7, 0.8):
        signal += np.exp(-0.5 * ((time - beat) / 0.012) ** 2)
        signal -= 0.2 * np.exp(-0.5 * ((time - beat - 0.03) / 0.012) ** 2)
        signal += 0.25 * np.exp(-0.5 * ((time - beat - 0.24) / 0.06) ** 2)
    payload = {
        "sample_rate": rate,
        "leads": {"II": signal.tolist(), "V1": (-0.7 * signal).tolist()},
    }
    response = client.post(
        "/api/analyze",
        files={"file": ("signal.json", json.dumps(payload).encode(), "application/json")},
    )
    assert response.status_code == 200, response.text
    return response.json(), signal


def test_empty_registry_never_produces_placeholder_predictions(client):
    data, _ = record(client)
    assert client.get("/api/models").json() == {"models": [], "default_model_id": None}
    assert client.get(f"/api/records/{data['id']}/models").json()["models"] == []
    assert client.get(f"/api/records/{data['id']}/predictions").json()["predictions"] == []
    response = client.post(f"/api/records/{data['id']}/predict", json={"model_id": "missing"})
    assert response.status_code == 404
    assert "模型输出 ·" not in client.get(f"/api/records/{data['id']}/report").text


def test_import_default_version_conflicts_and_immutable_configuration(client):
    first = import_model(client)
    assert first.status_code == 200, first.text
    a = first.json()
    assert a["is_default"] is True and a["signature"]["provider"] == "CPUExecutionProvider"
    duplicate = import_model(client, content=synthetic_model(bias=0.7))
    assert duplicate.status_code == 409
    b = import_model(client, config(version="2"), synthetic_model(bias=-0.3)).json()
    assert b["is_default"] is False
    client.post(f"/api/models/{b['id']}/default").raise_for_status()
    assert client.get("/api/models").json()["default_model_id"] == b["id"]
    edited = client.patch(
        f"/api/models/{b['id']}", json={"name": "新名称", "description": "用户配置备注"}
    ).json()
    assert edited["version"] == "2" and edited["description"] == "用户配置备注"
    forbidden = client.patch(f"/api/models/{b['id']}", json={"classes": ["互换标签", "另一个标签"]})
    assert forbidden.status_code == 422
    manifest = client.get(f"/api/models/{b['id']}/manifest").json()
    assert manifest["name"] == "新名称" and manifest["classes"] == config()["classes"]
    assert client.delete(f"/api/models/{b['id']}").json()["default_model_id"] is None
    assert client.get("/api/models").json()["default_model_id"] is None
    assert client.get(f"/api/models/{a['id']}").status_code == 200


def test_real_prediction_math_persists_without_rewriting_dsp_or_human_state(client):
    data, signal = record(client)
    model = import_model(client).json()
    response = client.post(f"/api/records/{data['id']}/predict", json={"model_id": model["id"]})
    assert response.status_code == 200, response.text
    result = response.json()
    first = result["predictions"][0]
    start = first["index"] - 40
    mean = np.asarray(signal[start : start + 128], dtype=np.float32).mean()
    logits = np.asarray([mean + 0.3, -mean - 0.3])
    expected = np.exp(logits - np.max(logits))
    expected /= expected.sum()
    np.testing.assert_allclose(first["scores"], expected, rtol=1e-5)
    assert result["model"]["manifest"]["classes"] == ["类别 A", "类别 B"]
    current = client.get(f"/api/records/{data['id']}").json()
    assert current["metrics"] == data["metrics"] and current["review_status"] == "unreviewed"
    listed = client.get(
        f"/api/records/{data['id']}/predictions", params={"model_id": model["id"]}
    ).json()["predictions"]
    assert listed[0]["id"] == result["id"] and listed[0]["stale"] is False
    client.put(
        f"/api/records/{data['id']}/beats/{first['index']}",
        json={"status": "confirmed", "note": "人工意见"},
    ).raise_for_status()
    assert (
        client.get(f"/api/records/{data['id']}/predictions").json()["predictions"][0]["stale"]
        is False
    )


def test_selected_models_keep_distinct_scores_and_historical_snapshots(client):
    data, _ = record(client)
    a = import_model(client, config(name="模型 A")).json()
    b = import_model(
        client, config(name="模型 B", classes=["标签 X", "标签 Y"]), synthetic_model(bias=-0.6)
    ).json()
    result_a = client.post(f"/api/records/{data['id']}/predict", json={"model_id": a["id"]}).json()
    result_b = client.post(f"/api/records/{data['id']}/predict", json={"model_id": b["id"]}).json()
    assert result_a["predictions"][0]["class_label"] == "类别 A"
    assert result_b["predictions"][0]["class_label"] == "标签 Y"
    listed_a = client.get(
        f"/api/records/{data['id']}/predictions", params={"model_id": a["id"]}
    ).json()["predictions"]
    assert [item["id"] for item in listed_a] == [result_a["id"]]
    client.patch(f"/api/models/{a['id']}", json={"name": "模型 A 重命名"}).raise_for_status()
    exported = client.get(f"/api/predictions/{result_a['id']}/export").json()
    assert (
        exported["model"]["name"] == "模型 A" and exported["predictions"] == result_a["predictions"]
    )
    selected_report = client.get(
        f"/api/records/{data['id']}/report", params={"prediction_id": result_a["id"]}
    )
    assert selected_report.status_code == 200 and "模型输出 · 模型 A /" in selected_report.text
    assert "模型输出 · 模型 B /" not in selected_report.text


def test_reanalysis_marks_old_predictions_stale_but_keeps_exports(client):
    data, _ = record(client)
    model = import_model(client).json()
    result = client.post(
        f"/api/records/{data['id']}/predict", json={"model_id": model["id"]}
    ).json()
    client.post(f"/api/records/{data['id']}/reanalyze", json={"lead": "V1"}).raise_for_status()
    old = client.get(f"/api/records/{data['id']}/predictions").json()["predictions"][0]
    assert old["stale"] is True and old["predictions"] == result["predictions"]
    assert "模型输出 ·" not in client.get(f"/api/records/{data['id']}/report").text
    explicit = client.get(
        f"/api/records/{data['id']}/report", params={"prediction_id": result["id"]}
    )
    assert "历史结果" in explicit.text
    assert client.get(f"/api/predictions/{result['id']}/export").json()["stale"] is True


def test_deleted_model_retains_results_and_deleted_record_cascades_them(client):
    data, _ = record(client)
    model = import_model(client).json()
    result = client.post(
        f"/api/records/{data['id']}/predict", json={"model_id": model["id"]}
    ).json()
    assert client.delete(f"/api/models/{model['id']}").status_code == 200
    preserved = client.get(f"/api/predictions/{result['id']}/export").json()
    assert preserved["model_available"] is False and preserved["model"]["sha256"] == model["sha256"]
    assert preserved["predictions"] == result["predictions"]
    assert "模型文件已删除" in client.get(f"/api/records/{data['id']}/report").text
    assert client.delete(f"/api/records/{data['id']}").status_code == 200
    assert client.get(f"/api/predictions/{result['id']}/export").status_code == 404


def test_incompatible_leads_bad_files_and_cross_origin_mutation_are_rejected(client):
    data, _ = record(client)
    unavailable = import_model(client, config(name="导联不匹配", lead_names=["不存在"])).json()
    candidates = client.get(f"/api/records/{data['id']}/models").json()["models"]
    assert candidates[0]["compatible"] is False and candidates[0]["reason"]
    assert (
        client.post(
            f"/api/records/{data['id']}/predict", json={"model_id": unavailable["id"]}
        ).status_code
        == 400
    )
    assert import_model(client, filename="checkpoint.pt").status_code == 400
    assert import_model(client, content=b"not an onnx graph").status_code == 400
    wrong = import_model(client, config(name="配置维度错误", input_samples=256))
    assert wrong.status_code == 400
    cross_origin = client.post(
        f"/api/models/{unavailable['id']}/default", headers={"Origin": "https://other.example"}
    )
    assert cross_origin.status_code == 403
    assert client.get(f"/api/records/{data['id']}/predictions").json()["predictions"] == []
