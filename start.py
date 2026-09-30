"""Windows-friendly bootstrap. No administrator privileges and no global installs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def ready_browser(url: str) -> None:
    for _ in range(80):
        try:
            with urlopen(url + "/api/health", timeout=1) as response:
                if json.load(response).get("status") == "ok":
                    webbrowser.open(url)
                    return
        except Exception:
            threading.Event().wait(0.25)
    print("Browser did not open automatically. Open manually:", url)


def main() -> None:
    parser = argparse.ArgumentParser(description="PulseLab local ECG research workstation")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required. Install from https://www.python.org/")
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    venv_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        if not venv_python.exists():
            print("Creating isolated Python environment...")
            subprocess.check_call([sys.executable, "-m", "venv", str(ROOT / ".venv")])
        result = subprocess.run(
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT
        )
        raise SystemExit(result.returncode)
    modules = ("fastapi", "uvicorn", "numpy", "scipy", "python_multipart", "onnx", "onnxruntime")
    if not all(importlib.util.find_spec(module) for module in modules):
        print("Installing project dependencies. First launch requires internet access...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(ROOT / "requirements.txt"),
            ],
            cwd=ROOT,
        )
    if not (ROOT / "data" / "demos" / "catalog.json").exists():
        print("Preparing public ECG data. First launch requires internet access...")
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "prepare_data.py")], cwd=ROOT)
    url = f"http://127.0.0.1:{args.port}"
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            try:
                with urlopen(url + "/api/health", timeout=2) as response:
                    health = json.load(response)
                if health.get("engine") == "DSP · 非诊断" and health.get("local_only"):
                    print("PulseLab already running:", url)
                    if not args.no_browser:
                        webbrowser.open(url)
                    return
            except Exception:
                pass
            raise RuntimeError(f"Port {args.port} is occupied by another program. Use --port 8766.")
    print("\nPulseLab ECG research workstation")
    print("Local address:", url)
    print("Data stays on this computer. Research only; not medical diagnosis.")
    print("Keep this window open. Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Thread(target=ready_browser, args=(url,), daemon=True).start()
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host="127.0.0.1",
        port=args.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print("Startup failed:", error, file=sys.stderr)
        raise SystemExit(1) from error
