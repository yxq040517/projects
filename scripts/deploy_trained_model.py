"""Import a completed, parity-verified trained model into the running local platform."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def request_json(base, path, method="GET", value=None, body=None, content_type=None):
    if value is not None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        content_type = "application/json"
    request = Request(
        base + path,
        data=body,
        method=method,
        headers={"Content-Type": content_type or "application/json"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.load(response)
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Platform request failed ({error.code}): {details}") from error


def deploy(output: Path, base: str):
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    parity = json.loads((output / "export_parity.json").read_text(encoding="utf-8"))
    if not parity.get("passed") or not evaluation.get("test"):
        raise ValueError(
            "Completed independent test evaluation and passing export parity are required before deployment."
        )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    content = (output / "model.onnx").read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    registry = request_json(base, "/api/models")
    config_keys = [key for key in manifest if key not in {"name", "version", "description"}]
    same = next(
        (
            model
            for model in registry["models"]
            if model["sha256"] == digest
            and all(model.get(key) == manifest[key] for key in config_keys)
        ),
        None,
    )
    if same is None:
        if any(
            model["name"] == manifest["name"] and model["version"] == manifest["version"]
            for model in registry["models"]
        ):
            manifest["version"] = (manifest["version"] + "." + digest[:8])[:32]
        boundary = "PulseLabTrained_" + uuid4().hex
        prefix = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="config"\r\n\r\n'
            + json.dumps(manifest, ensure_ascii=False)
            + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="model.onnx"\r\nContent-Type: application/octet-stream\r\n\r\n'
        ).encode("utf-8")
        body = prefix + content + f"\r\n--{boundary}--\r\n".encode("ascii")
        model = request_json(
            base,
            "/api/models/import",
            "POST",
            body=body,
            content_type="multipart/form-data; boundary=" + boundary,
        )
    else:
        model = same
    request_json(base, f"/api/models/{model['id']}/default", "POST", value={})
    result = {
        "model_id": model["id"],
        "name": model["name"],
        "version": model["version"],
        "sha256": digest,
        "platform": base,
        "default_model_id": model["id"],
        "test_metrics": evaluation["test"],
        "export_parity_passed": True,
        "source": "actual trained weights from the complete MIT-BIH dataset",
    }
    (output / "deployment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "deployed_manifest.json").write_text(
        json.dumps({key: model[key] for key in manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model_id": model["id"],
                "name": model["name"],
                "version": model["version"],
                "default_model_id": model["id"],
                "sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "mitbih_cnn_v1")
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    deploy(args.output, args.base.rstrip("/"))
