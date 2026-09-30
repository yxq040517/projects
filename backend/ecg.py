"""Local ECG DSP and strict waveform import; research measurements, not diagnosis."""

import csv
import io
import json
import math
from collections.abc import Mapping

import numpy as np
from scipy import signal

__all__ = ["parse_upload", "analyze_signal"]

_TIME_NAMES = {"time", "time_s", "timestamp"}
_MAX_SAMPLES = 300000
_MAX_MV = 100.0


def _rate(value):
    if isinstance(value, bool):
        raise ValueError("采样率必须为 100–1000 Hz 的数字。")
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("采样率必须为 100–1000 Hz 的数字。") from exc
    if not math.isfinite(rate) or not 100 <= rate <= 1000:
        raise ValueError("采样率必须在 100–1000 Hz 之间。")
    return rate


def _unit_factor(units):
    if not isinstance(units, str):
        raise ValueError("信号单位仅支持 mV、uV（微伏）或 V。")
    key = units.strip().replace("μ", "u").replace("µ", "u").lower()
    factors = {"mv": 1.0, "uv": 0.001, "v": 1000.0}
    if key not in factors:
        raise ValueError("信号单位仅支持 mV、uV（微伏）或 V。")
    return factors[key]


def _number(value, location):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{location}必须为数字，不支持布尔值、空值或嵌套数据。")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{location}缺少数值，不能跳过缺失数据。")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{location}不是有效数字。") from exc
    if not math.isfinite(result):
        raise ValueError(f"{location}包含 NaN 或无穷值，请先修复原始数据。")
    return result


def _validate_leads(leads, rate, factor=1.0):
    if not isinstance(leads, Mapping) or not leads:
        raise ValueError("请提供至少一个导联的数字波形。")
    if len(leads) > 12:
        raise ValueError("最多支持 12 个导联，请减少导联列数。")
    arrays = {}
    length = None
    for name, values in leads.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("导联名称不能为空，且不得超过 80 个字符。")
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError(f"导联 {name} 必须是一维数字数组。")
        if len(values) > _MAX_SAMPLES:
            raise ValueError(f"导联 {name} 超过每导联 300000 个采样点的限制。")
        if length is not None and len(values) != length:
            raise ValueError("所有导联必须具有相同的采样点数，不能截断或补齐。")
        length = len(values)
        if isinstance(values, np.ndarray) and values.ndim != 1:
            raise ValueError(f"导联 {name} 必须是一维数字数组。")
        # JSON/CSV validation is element-wise so missing and boolean cells are never coerced.
        array = np.asarray(
            [_number(v, f"导联 {name} 第 {i + 1} 个采样点") for i, v in enumerate(values)],
            dtype=float,
        )
        array *= factor
        if np.any(np.abs(array) > _MAX_MV):
            raise ValueError(f"导联 {name} 转换后幅值超过 ±100 mV，请检查单位或数据。")
        arrays[name] = array
    duration = length / rate
    if not 5.0 - 1e-9 <= duration <= 300.0 + 1e-9:
        raise ValueError("波形时长必须在 5–300 秒之间，请检查采样率和数据长度。")
    return arrays


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 存在重复字段或导联名称：{key}。")
        result[key] = value
    return result


def _numeric_text(value):
    try:
        float(value)
        return bool(value.strip())
    except ValueError:
        return False


def _parse_csv(text, supplied_rate):
    if not text.strip():
        raise ValueError("文件为空，请选择包含数字波形的文件。")
    first_line = text.splitlines()[0]
    delimiters = [delimiter for delimiter in (",", ";", "\t") if delimiter in first_line]
    if len(delimiters) > 1:
        raise ValueError("表格分隔符混杂，请统一使用逗号、分号或制表符。")
    delimiter = delimiters[0] if delimiters else ","
    try:
        rows = csv.reader(io.StringIO(text), delimiter=delimiter, strict=True)
        first = next(rows)
        if not first or any(not cell.strip() for cell in first):
            raise ValueError("表格第一行含空单元格，请补全数据或列名。")
        numeric = [_numeric_text(cell) for cell in first]
        header = not all(numeric)
        if header and any(numeric):
            raise ValueError("首行混有数字和非数字，无法作为完整表头；请检查损坏的采样点。")
        names = (
            [cell.strip() for cell in first]
            if header
            else [f"导联 {i + 1}" for i in range(len(first))]
        )
        if len(set(names)) != len(names):
            raise ValueError("表格包含重复的导联列名，请使用唯一列名。")
        time_columns = [i for i, name in enumerate(names) if header and name.lower() in _TIME_NAMES]
        if len(time_columns) > 1:
            raise ValueError("仅支持一个时间列（time、time_s 或 timestamp）。")
        time_column = time_columns[0] if time_columns else None
        lead_columns = [i for i in range(len(names)) if i != time_column]
        if not 1 <= len(lead_columns) <= 12:
            raise ValueError("表格必须包含 1–12 个数字信号列（时间列不计入导联数）。")
        columns = [[] for _ in names]

        def append_row(row, line):
            if len(row) != len(names):
                raise ValueError(f"表格第 {line} 行列数不一致或为空行，不能跳过数据。")
            if len(columns[0]) >= _MAX_SAMPLES:
                raise ValueError("超过每导联 300000 个采样点的限制。")
            for i, cell in enumerate(row):
                columns[i].append(_number(cell, f"表格第 {line} 行第 {i + 1} 列"))

        if not header:
            append_row(first, 1)
        for row in rows:
            append_row(row, rows.line_num)
    except csv.Error as exc:
        raise ValueError("CSV 引号或分隔格式错误，请导出标准数值表格。") from exc
    warnings = []
    rate = supplied_rate
    if time_column is not None:
        times = np.asarray(columns[time_column], dtype=float)
        if len(times) < 2:
            raise ValueError("时间列至少需要两个采样点。")
        differences = np.diff(times)
        if np.any(differences <= 0):
            raise ValueError("时间列必须以秒为单位严格递增，不能重复或倒序。")
        step = float(np.median(differences))
        if np.any(np.abs(differences - step) > max(1e-7, step * 0.01)):
            raise ValueError("时间列采样间隔不均匀（允许 1% 的舍入误差）；请先重采样。")
        # Endpoint calibration avoids accumulating median-step rounding error,
        # especially when timestamp values contain a large absolute offset.
        calibrated_step = float((times[-1] - times[0]) / (len(times) - 1))
        rate = _rate(round(1.0 / calibrated_step, 6))
        # Rounded timestamps are accepted, but accumulated clock drift is not.
        ideal = times[0] + np.arange(len(times)) / rate
        if np.max(np.abs(times - ideal)) > max(1e-6, step * 0.02):
            raise ValueError("时间列存在累计时钟偏移，不能作为均匀采样波形分析。")
        if abs(rate - supplied_rate) > max(1e-6, rate * 1e-6):
            warnings.append(
                f"依据秒制时间列校准采样率为 {rate:.6g} Hz；输入的 {supplied_rate:.6g} Hz 未采用。"
            )
    return {names[i]: columns[i] for i in lead_columns}, rate, warnings


def parse_upload(content: bytes, filename: str, sample_rate: float, units: str) -> dict:
    """Parse strict numeric CSV/TXT or JSON, returning mV and calibrated sampling rate."""
    if not isinstance(content, bytes) or not content:
        raise ValueError("上传文件为空或不是有效字节数据。")
    if len(content) > 100 * 1024 * 1024:
        raise ValueError("文件超过 100 MB，请缩短记录后上传。")
    if not isinstance(filename, str):
        raise ValueError("请提供有效文件名。")
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in {"csv", "txt", "json"}:
        raise ValueError("仅支持 .csv、.txt 或 .json 数字波形文件。")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("文件编码无法读取，请另存为 UTF-8 编码的 CSV 或 JSON。") from exc
    warnings = []
    if suffix == "json":
        try:
            payload = json.loads(text, object_pairs_hook=_json_object)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(
                "JSON 格式错误，请提供 sample_rate、units 和 leads 或 signal。"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON 顶层必须是包含 leads 或 signal 的对象。")
        rate = _rate(payload.get("sample_rate", sample_rate))
        factor = _unit_factor(payload.get("units", units))
        if "leads" in payload and "signal" in payload:
            raise ValueError("JSON 不能同时提供 leads 和 signal，请保留一种波形格式。")
        leads = payload.get("leads") if "leads" in payload else {"II": payload.get("signal")}
    else:
        rate = _rate(sample_rate)
        factor = _unit_factor(units)
        leads, rate, warnings = _parse_csv(text, rate)
    arrays = _validate_leads(leads, rate, factor)
    return {
        "sample_rate": float(rate),
        "units": "mV",
        "leads": {name: array.tolist() for name, array in arrays.items()},
        "warnings": warnings,
    }


def _filter_waveform(raw, rate, notch):
    upper = min(40.0, rate * 0.45)
    sos = signal.butter(3, (0.5, upper), btype="bandpass", fs=rate, output="sos")
    filtered = signal.sosfiltfilt(sos, raw)
    if notch and notch < rate / 2:
        b, a = signal.iirnotch(notch, 30.0, fs=rate)
        filtered = signal.filtfilt(b, a, filtered)
    return filtered


def _quality(raw, rate):
    centered = raw - np.mean(raw)
    frequencies, power = signal.welch(centered, fs=rate, nperseg=min(len(raw), int(4 * rate)))
    total = float(np.sum(power))
    baseline = float(np.sum(power[frequencies < 0.5]) / total) if total > 1e-18 else 0.0
    high = float(np.sum(power[frequencies > 40]) / total) if total > 1e-18 else 0.0
    window = max(1, int(rate * 0.5))
    segments = [raw[start : start + window] for start in range(0, len(raw), window)]
    flat = sum(len(part) for part in segments if np.ptp(part) < 0.005) / len(raw)
    essentially_flat = np.ptp(raw) < 0.015 or np.std(centered) < 0.003
    if essentially_flat:
        flat = 1.0
    # Repeated rails indicate digital saturation; isolated maxima are not clipping.
    rail = np.isclose(raw, np.min(raw), atol=1e-9, rtol=0) | np.isclose(
        raw, np.max(raw), atol=1e-9, rtol=0
    )
    repeated = rail[1:] & rail[:-1] & (np.abs(np.diff(raw)) < 1e-9)
    clipping = float(np.count_nonzero(repeated) / len(raw)) if not essentially_flat else 0.0
    score = max(0.0, min(100.0, 100 - 45 * baseline - 65 * high - 80 * flat - 60 * clipping))
    return (
        {
            "score": int(round(score)),
            "baseline_ratio": round(baseline, 5),
            "high_frequency_ratio": round(high, 5),
            "flatline_ratio": round(float(flat), 5),
            "clipping_ratio": round(clipping, 5),
        },
        essentially_flat,
        frequencies,
        power,
    )


def _detect_qrs(raw, filtered, rate):
    """Polarity-independent QRS energy; reject broad T waves and diffuse noise."""
    qrs = signal.sosfiltfilt(
        signal.butter(3, (5.0, min(25.0, rate * 0.4)), btype="bandpass", fs=rate, output="sos"),
        raw,
    )
    derivative = np.gradient(qrs)
    energy = derivative * derivative
    integration = max(3, int(round(rate * 0.08)))
    envelope = np.convolve(energy, np.ones(integration) / integration, mode="same")
    floor = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - floor)))
    amplitude_floor = float(np.median(np.abs(qrs)))
    candidates, properties = signal.find_peaks(
        envelope,
        distance=max(1, int(rate * 0.20)),
        prominence=max(floor * 2.0, mad * 3.0, 1e-14),
    )
    if not len(candidates):
        return np.asarray([], dtype=int), 0.0
    strengths = envelope[candidates]
    signal_level = float(np.percentile(strengths, 85))
    # Clean recordings may contain only real QRS candidates: do not initialize
    # the noise estimate to the same height as the signal estimate.
    noise_level = max(floor, min(float(np.percentile(strengths, 20)), signal_level * 0.15))
    accepted = []
    refractory = int(round(rate * 0.28))
    edge = int(round(rate * 0.18))
    for candidate, prominence in zip(candidates, properties["prominences"], strict=True):
        if candidate < edge or candidate >= len(raw) - edge:
            continue
        value = float(envelope[candidate])
        threshold = max(floor + 3 * mad, noise_level + 0.25 * (signal_level - noise_level), 1e-14)
        left, right = (
            max(0, candidate - int(rate * 0.08)),
            min(len(raw), candidate + int(rate * 0.08) + 1),
        )
        refined = left + int(np.argmax(np.abs(qrs[left:right])))
        # Compact derivative energy is characteristic of QRS, unlike broad T waves.
        narrow = energy[
            max(0, refined - int(rate * 0.05)) : min(len(raw), refined + int(rate * 0.05) + 1)
        ]
        wide = energy[
            max(0, refined - int(rate * 0.14)) : min(len(raw), refined + int(rate * 0.14) + 1)
        ]
        concentration = float(np.sum(narrow) / max(float(np.sum(wide)), 1e-20))
        amplitude = float(abs(qrs[refined]))
        is_qrs = (
            value >= threshold
            and prominence >= max(floor * 2, 1e-14)
            and concentration >= 0.50
            and amplitude >= max(0.01, amplitude_floor * 4.0)
        )
        if not is_qrs:
            noise_level = 0.875 * noise_level + 0.125 * value
            continue
        signal_level = 0.875 * signal_level + 0.125 * value
        # Refine to the local extremum of the visible 0.5–40 Hz waveform.
        radius = max(1, int(round(rate * 0.012)))
        start, end = max(0, refined - radius), min(len(raw), refined + radius + 1)
        baseline = float(
            np.median(
                filtered[
                    max(0, refined - int(rate * 0.15)) : min(
                        len(raw), refined + int(rate * 0.15) + 1
                    )
                ]
            )
        )
        refined = start + int(np.argmax(np.abs(filtered[start:end] - baseline)))
        if accepted and refined - accepted[-1][0] < refractory:
            if value > accepted[-1][1]:
                accepted[-1] = (refined, value)
        else:
            accepted.append((refined, value))
    indices = np.asarray([item[0] for item in accepted], dtype=int)
    contrast = (
        float(np.median(np.abs(qrs[indices])) / max(amplitude_floor, 1e-12))
        if len(indices)
        else 0.0
    )
    return indices, contrast


def _display_indices(arrays, filtered, limit=12000):
    length = len(filtered)
    if length <= limit:
        return np.arange(length, dtype=int)
    # Each bucket retains every lead's extrema, plus filtered selected extrema.
    # A shared union keeps leads, raw, filtered and time perfectly aligned.
    channels = list(arrays.values()) + [filtered]
    per_bucket = 2 * len(channels)
    buckets = max(1, (limit - 2) // per_bucket)
    boundaries = np.linspace(1, length - 1, buckets + 1, dtype=int)
    indices = {0, length - 1}
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end <= start:
            continue
        for channel in channels:
            part = channel[start:end]
            indices.add(int(start + np.argmin(part)))
            indices.add(int(start + np.argmax(part)))
    return np.asarray(sorted(indices), dtype=int)


def _beat_template(filtered, rate, indices, reliable):
    left, right = int(math.ceil(0.2 * rate)), int(math.ceil(0.4 * rate))
    complete = indices[(indices >= left) & (indices + right < len(filtered))]
    available = len(complete)
    template = {
        "time_ms": [],
        "median": [],
        "p25": [],
        "p75": [],
        "traces": [],
        "peak_times": [],
        "available_count": int(available),
        "used_count": 0,
        "reliable": False,
        "reason": "",
    }
    if not reliable:
        template["reason"] = "信号质量不足或有效搏动不足，未生成心搏模板。"
        return template
    if available < 3:
        template["reason"] = "完整心搏窗口不足 3 个；每个窗口需覆盖 R 峰前 200 ms 至后 400 ms。"
        return template
    selected = complete[np.linspace(0, available - 1, min(200, available), dtype=int)]
    time_ms = np.linspace(-200.0, 400.0, 201)
    source_ms = np.arange(-left, right + 1) * (1000.0 / rate)
    traces = np.empty((len(selected), len(time_ms)))
    baseline_left = int(math.ceil(0.12 * rate))
    baseline_right = int(math.floor(0.06 * rate))
    for row, peak in enumerate(selected):
        # Local pre-QRS baseline only: keep mV scale, polarity and morphology.
        baseline = np.median(filtered[peak - baseline_left : peak - baseline_right + 1])
        traces[row] = np.interp(time_ms, source_ms, filtered[peak - left : peak + right + 1])
        traces[row] -= baseline
    p25, median, p75 = np.percentile(traces, [25, 50, 75], axis=0)
    preview = np.linspace(0, len(selected) - 1, min(30, len(selected)), dtype=int)
    template.update(
        {
            "time_ms": time_ms.tolist(),
            "median": median.tolist(),
            "p25": p25.tolist(),
            "p75": p75.tolist(),
            "traces": traces[preview].tolist(),
            "peak_times": (selected[preview] / rate).tolist(),
            "used_count": int(len(selected)),
            "reliable": True,
        }
    )
    return template


def _spectrum(filtered, rate, frequencies, raw_power):
    # The quality calculation already measured raw Welch PSD. Its global
    # centering does not change PSD: Welch detrends each segment by its mean.
    _, filtered_power = signal.welch(filtered, fs=rate, nperseg=min(len(filtered), int(4 * rate)))
    selected = np.flatnonzero(frequencies <= min(100.0, rate / 2.0))
    if len(selected) > 512:
        selected = selected[np.linspace(0, len(selected) - 1, 512, dtype=int)]
    return {
        "frequency_hz": frequencies[selected].tolist(),
        "raw_power": raw_power[selected].tolist(),
        "filtered_power": filtered_power[selected].tolist(),
    }


def _rr_tools(intervals, plausible, reliable):
    histogram = {"edges_ms": [], "counts": []}
    poincare = {"x_ms": [], "y_ms": []}
    if not reliable or np.count_nonzero(plausible) < 2:
        return histogram, poincare
    rr_ms = intervals * 1000.0
    values = rr_ms[plausible]
    lower, upper = float(np.min(values)), float(np.max(values))
    if upper - lower < 1e-7:
        # A constant RR needs a readable interval range, not ten zero-width bins.
        margin = max(25.0, lower * 0.05)
        lower, upper = max(300.0, lower - margin), min(2000.0, upper + margin)
    counts, edges = np.histogram(values, bins=10, range=(lower, upper))
    histogram = {"edges_ms": edges.tolist(), "counts": counts.tolist()}
    adjacent = plausible[:-1] & plausible[1:]
    poincare = {
        "x_ms": rr_ms[:-1][adjacent].tolist(),
        "y_ms": rr_ms[1:][adjacent].tolist(),
    }
    return histogram, poincare


def analyze_signal(
    leads: dict[str, list[float]],
    sample_rate: float,
    selected_lead: str,
    notch: int = 50,
) -> dict:
    """Return transparent spectral quality, detected peaks and exploratory RR metrics."""
    rate = _rate(sample_rate)
    arrays = _validate_leads(leads, rate)
    if not isinstance(selected_lead, str) or selected_lead not in arrays:
        raise ValueError("所选导联不存在，请选择文件中的导联。")
    if isinstance(notch, bool) or notch not in (0, 50, 60):
        raise ValueError("工频陷波仅支持关闭（0）、50 Hz 或 60 Hz。")
    raw = arrays[selected_lead]
    duration = len(raw) / rate
    filtered = _filter_waveform(raw, rate, notch)
    quality, flat, frequencies, raw_power = _quality(raw, rate)
    indices, contrast = (
        _detect_qrs(raw, filtered, rate) if not flat else (np.asarray([], dtype=int), 0.0)
    )
    intervals = np.diff(indices) / rate
    plausible = (intervals >= 0.30) & (intervals <= 2.0)
    good_intervals = intervals[plausible]
    reliable = bool(
        not flat
        and len(indices) >= 4
        and len(good_intervals) >= 3
        and np.mean(plausible) >= 0.70
        and contrast >= 4.0
        and quality["high_frequency_ratio"] < 0.55
        and quality["flatline_ratio"] < 0.50
        and quality["clipping_ratio"] < 0.15
        and quality["score"] >= 40
    )
    if not reliable:
        quality["score"] = min(
            quality["score"],
            20 if flat or quality["high_frequency_ratio"] >= 0.55 else 35,
        )
    quality.update(
        {
            "grade": "good"
            if quality["score"] >= 75
            else "fair"
            if quality["score"] >= 45
            else "poor",
            "label": "信号质量较好"
            if quality["score"] >= 75
            else "信号质量一般"
            if quality["score"] >= 45
            else "信号质量较差或有效搏动不足",
            "reliable": reliable,
        }
    )
    metrics = {
        "heart_rate": None,
        "min_heart_rate": None,
        "max_heart_rate": None,
        "beat_count": int(len(indices)),
        "mean_rr": None,
        "sdnn": None,
        "rmssd": None,
        "pnn50": None,
        "rr_cv": None,
    }
    if reliable:
        rr_ms = good_intervals * 1000.0
        mean = float(np.mean(rr_ms))
        std = float(np.std(rr_ms, ddof=1))
        # Do not bridge over excluded intervals when measuring successive differences.
        adjacent = plausible[:-1] & plausible[1:]
        differences = np.diff(intervals)[adjacent] * 1000.0
        metrics.update(
            {
                "heart_rate": round(60000.0 / mean, 1),
                "min_heart_rate": round(float(60.0 / np.max(good_intervals)), 1),
                "max_heart_rate": round(float(60.0 / np.min(good_intervals)), 1),
                "mean_rr": round(mean, 2),
                "sdnn": round(std, 2),
                "rr_cv": round(std / mean, 5),
            }
        )
        if len(differences) >= 2:
            metrics["rmssd"] = round(float(np.sqrt(np.mean(differences**2))), 2)
            metrics["pnn50"] = round(float(np.mean(np.abs(differences) > 50.0 + 1e-7) * 100), 2)
    median_rr = float(np.median(good_intervals)) if len(good_intervals) else 0.0
    flags = (
        (~plausible) | (np.abs(intervals - median_rr) > max(0.12, median_rr * 0.20))
        if len(intervals)
        else np.asarray([], dtype=bool)
    )
    findings = [
        {
            "level": "info",
            "title": "研究用途的信号测量",
            "detail": "采用带通、导数能量和自适应阈值检测 QRS 候选；结果不是训练模型输出，不用于诊断。质量分数是频谱与伪迹启发式评分，不是准确率、置信度或健康评分。",
        }
    ]
    if reliable and not 60 <= metrics["heart_rate"] <= 100:
        direction = "较快" if metrics["heart_rate"] > 100 else "较慢"
        findings.append(
            {
                "level": "warning",
                "title": f"检出平均心率{direction}",
                "detail": f"当前检出平均心率 {metrics['heart_rate']:g} bpm，位于成人静息常用参考范围 60–100 bpm 之外。未记录年龄、活动状态或用药，不能据此诊断；请核对波形、采样率和记录背景。",
            }
        )
    if duration < 60:
        findings.append(
            {
                "level": "warning",
                "title": "记录较短",
                "detail": "记录不足 60 秒，RR 统计容易受样本量影响，仅适合探索；sdnn 字段表示检出 RR 的样本标准差，不是经正常搏动筛选的 NN 指标。",
            }
        )
    else:
        findings.append(
            {
                "level": "info",
                "title": "RR 统计范围",
                "detail": "均值与标准差仅使用 300–2000 ms 的 RR；相邻差值不跨越被排除的间隔。sdnn 字段为 RR 标准差，不是临床 NN/HRV 指标。",
            }
        )
    if flat:
        findings.append(
            {
                "level": "warning",
                "title": "波形近乎平坦",
                "detail": "有效振幅不足，未发布心率或 RR 统计。请核对电极、设备连接、导联和信号单位；这不是对心脏状态的判断。",
            }
        )
    elif not reliable:
        findings.append(
            {
                "level": "warning",
                "title": "无法可靠测量心率",
                "detail": "噪声、伪迹或可检出的 QRS 数量不足。保留候选标记供人工查看，但不发布心率及 RR 汇总；请检查原始波形后重新采集。",
            }
        )
    if np.any(flags):
        findings.append(
            {
                "level": "warning",
                "title": "RR 间隔需要人工复核",
                "detail": "部分间隔超出 300–2000 ms，或偏离中位间隔超过 20%（至少 120 ms）。可能与真实节律变化、漏检或重复检出有关；请逐搏查看，不据此判断房颤或其他疾病。",
            }
        )
    if quality["baseline_ratio"] > 0.40:
        findings.append(
            {
                "level": "warning",
                "title": "低频漂移较明显",
                "detail": "低于 0.5 Hz 的功率占比较高。滤波可改善显示，但不能恢复被伪迹覆盖的真实信号。",
            }
        )
    if quality["high_frequency_ratio"] > 0.30 or quality["clipping_ratio"] > 0.02:
        findings.append(
            {
                "level": "warning",
                "title": "高频噪声或饱和迹象",
                "detail": "请检查肌电干扰、工频干扰与量程。频谱及重复极值只提供伪迹线索，不能替代人工审阅。",
            }
        )
    if notch and notch >= rate / 2:
        findings.append(
            {
                "level": "info",
                "title": "未应用工频陷波",
                "detail": f"{notch} Hz 不低于当前奈奎斯特频率，无法应用该陷波。显示仍采用 0.5–{min(40.0, rate * 0.45):g} Hz 带通滤波。",
            }
        )
    display = _display_indices(arrays, filtered)
    histogram, poincare = _rr_tools(intervals, plausible, reliable)
    return {
        "selected_lead": selected_lead,
        "sample_rate": float(rate),
        "duration": float(duration),
        "lead_names": list(arrays),
        "units": "mV",
        "metrics": metrics,
        "quality": quality,
        "findings": findings,
        "peaks": [
            {
                "index": int(index),
                "time": round(float(index / rate), 6),
                "amplitude": round(float(filtered[index]), 6),
            }
            for index in indices
        ],
        "rr": [
            {
                "time": round(float(indices[i + 1] / rate), 6),
                "rr_ms": round(float(interval * 1000), 3),
                "heart_rate": round(float(60 / interval), 2) if reliable and plausible[i] else None,
                "flag": bool(flags[i] or not reliable),
                "peak_index": int(indices[i + 1]),
            }
            for i, interval in enumerate(intervals)
        ],
        "tools": {
            "template": _beat_template(filtered, rate, indices, reliable),
            "spectrum": _spectrum(filtered, rate, frequencies, raw_power),
            "rr_histogram": histogram,
            "poincare": poincare,
        },
        "waveform": {
            "time": (display / rate).tolist(),
            "raw": raw[display].tolist(),
            "filtered": filtered[display].tolist(),
            "leads": {name: array[display].tolist() for name, array in arrays.items()},
        },
    }
