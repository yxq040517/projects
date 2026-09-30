"""Prepare attributed public ECG excerpts and explicitly synthetic engineering examples."""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source"
DEMOS = ROOT / "data" / "demos"
BASE = "https://physionet.org/files/mitdb/1.0.0/"


def fetch(filename: str) -> Path:
    path = SOURCE / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    error = None
    for attempt in range(3):
        try:
            request = Request(BASE + filename, headers={"User-Agent": "PulseLab-Research/1.0"})
            with urlopen(request, timeout=40, context=ssl.create_default_context()) as response:
                content = response.read()
            if not content:
                raise ValueError("Empty response")
            path.write_bytes(content)
            return path
        except Exception as exc:
            error = exc
            time.sleep(attempt + 1)
    raise RuntimeError(f"Cannot download {filename}: {error}")


def read_format212(record: str, duration: float = 60) -> tuple[float, dict[str, np.ndarray]]:
    header = fetch(f"{record}.hea").read_text(encoding="ascii").splitlines()
    first = header[0].split()
    if int(first[1]) != 2:
        raise ValueError("This extractor only supports two-channel format-212 records")
    fs = float(first[2])
    count = min(int(fs * duration), int(first[3]))
    raw = np.frombuffer(fetch(f"{record}.dat").read_bytes(), dtype=np.uint8)
    triplets = raw[: count * 3].reshape(-1, 3).astype(np.int32)
    a = triplets[:, 0] | ((triplets[:, 1] & 15) << 8)
    b = triplets[:, 2] | ((triplets[:, 1] >> 4) << 8)
    a = np.where(a >= 2048, a - 4096, a)
    b = np.where(b >= 2048, b - 4096, b)
    signals = {}
    for line, digital in zip(header[1:3], (a, b), strict=True):
        fields = line.split()
        if fields[1] != "212":
            raise ValueError("Unexpected source encoding")
        gain = float(fields[2])
        baseline = float(fields[4])
        signals[fields[8]] = (digital - baseline) / gain
    return fs, signals


def synthetic_signal(
    fs: int = 250, duration: int = 60, bpm: float = 72, noise: bool = False
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260930)
    t = np.arange(fs * duration) / fs
    signal = np.zeros_like(t)
    rr = 60 / bpm
    beats = np.arange(0.7, duration - 0.3, rr)
    for beat in beats:
        signal += 0.12 * np.exp(-0.5 * ((t - (beat - 0.18)) / 0.035) ** 2)
        signal -= 0.12 * np.exp(-0.5 * ((t - (beat - 0.025)) / 0.010) ** 2)
        signal += 1.12 * np.exp(-0.5 * ((t - beat) / 0.012) ** 2)
        signal -= 0.22 * np.exp(-0.5 * ((t - (beat + 0.03)) / 0.012) ** 2)
        signal += 0.30 * np.exp(-0.5 * ((t - (beat + 0.24)) / 0.055) ** 2)
    signal += 0.04 * np.sin(2 * np.pi * 0.20 * t)
    signal += rng.normal(0, 0.005, len(t))
    if noise:
        signal += 0.5 * np.sin(2 * np.pi * 0.22 * t)
        signal += 0.3 * np.sin(2 * np.pi * 50 * t)
        signal += rng.normal(0, 0.25, len(t))
        signal[int(26 * fs) : int(31 * fs)] = 0.0
    return {"II": signal, "V1": -0.55 * signal + 0.02 * np.sin(2 * np.pi * 0.15 * t)}


def write_csv(path: Path, fs: float, signals: dict[str, np.ndarray]) -> None:
    names = list(signals)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["time_s", *names])
        for index in range(len(signals[names[0]])):
            writer.writerow(
                [
                    f"{index / fs:.8f}",
                    *(f"{signals[name][index]:.6f}" for name in names),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip public dataset network access",
    )
    args = parser.parse_args()
    SOURCE.mkdir(parents=True, exist_ok=True)
    DEMOS.mkdir(parents=True, exist_ok=True)
    entries = []
    if not args.synthetic_only:
        for record, title, description in (
            (
                "100",
                "公开心电 · MIT-BIH 100",
                "真实动态心电前 60 秒，MLII / V5；用于波形与检测标记复核。",
            ),
            (
                "119",
                "公开心电 · MIT-BIH 119",
                "真实动态心电前 60 秒，MLII / V1；可观察形态与 RR 间期变化。",
            ),
        ):
            fs, signals = read_format212(record)
            fetch(f"{record}.atr")
            demo_id = f"mitbih-{record}"
            write_csv(DEMOS / f"{demo_id}.csv", fs, signals)
            entries.append(
                {
                    "id": demo_id,
                    "name": title,
                    "description": description,
                    "source": f"MIT-BIH Arrhythmia Database · record {record} · 前60秒 · https://physionet.org/content/mitdb/1.0.0/ · ODC Attribution 1.0",
                    "sample_rate": fs,
                    "duration": 60,
                    "lead_names": list(signals),
                    "kind": "public",
                }
            )
            print(f"Public example prepared: {record}, {fs:g} Hz, {list(signals)}")
    for demo_id, name, description, bpm, noise in (
        (
            "synthetic-regular",
            "规则节律 · 合成演示",
            "72 bpm 的合成信号；用于检查导入、采样率与峰定位，不代表患者。",
            72,
            False,
        ),
        (
            "synthetic-fast",
            "较快节律 · 合成演示",
            "120 bpm 的合成信号；用于展示较快心率的规则提示，不代表患者。",
            120,
            False,
        ),
        (
            "synthetic-noisy",
            "噪声与平坦片段 · 合成演示",
            "加入基线漂移、工频、随机噪声与平坦片段，观察质量提示。",
            72,
            True,
        ),
    ):
        signals = synthetic_signal(bpm=bpm, noise=noise)
        write_csv(DEMOS / f"{demo_id}.csv", 250, signals)
        entries.append(
            {
                "id": demo_id,
                "name": name,
                "description": description,
                "source": "PulseLab 数学合成信号 · 非患者数据 · 非临床验证数据",
                "sample_rate": 250,
                "duration": 60,
                "lead_names": list(signals),
                "kind": "synthetic",
            }
        )
    (DEMOS / "catalog.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "data" / "数据来源与许可.txt").write_text(
        "MIT-BIH Arrhythmia Database v1.0.0\n"
        "Source: https://physionet.org/content/mitdb/1.0.0/\n"
        "DOI: https://doi.org/10.13026/C2F305\n"
        "License: Open Data Commons Attribution License v1.0\n"
        "License terms: https://opendatacommons.org/licenses/by/1-0/\n"
        "Citation: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. "
        "IEEE Engineering in Medicine and Biology 20(3):45-50 (2001).\n"
        "Modifications: decode original two-channel WFDB format 212 to millivolts; "
        "export only the first 60 seconds to CSV; no annotation labels assigned to application output.\n"
        "Raw originals are cached in data/source for reproducibility.\n"
        "Synthetic examples: mathematically generated by scripts/prepare_data.py with fixed seed, "
        "not patient records or evidence of clinical efficacy.\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(entries)} usable examples in {DEMOS}")


if __name__ == "__main__":
    main()
