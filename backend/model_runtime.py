"""Strict ONNX CPU inference on original ECG samples, without model file access."""

import math
import os
from fractions import Fraction
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
from scipy.signal import resample_poly
from scipy.special import expit

from . import ecg

__all__ = ["validate_model", "compatibility", "predict"]

_TASKS = {"beat_classification", "record_classification", "record_multilabel"}
_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}
_DEFAULTS = {
    "pre_samples": None,
    "normalization": "none",
    "input_filter": "raw",
    "notch": 0,
    "output_type": "logits",
    "threshold": 0.5,
    "description": "",
}
_REQUIRED = {
    "schema_version",
    "name",
    "version",
    "task",
    "classes",
    "sample_rate",
    "input_samples",
    "lead_names",
}


def _finite_number(value, name, low, high, exclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}必须是有限数字。")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name}必须是有限数字。") from error
    valid = low < number < high if exclusive else low <= number <= high
    if not math.isfinite(number) or not valid:
        raise ValueError(
            f"{name}必须{'大于' if exclusive else '不小于'} {low} 且{'小于' if exclusive else '不大于'} {high}。"
        )
    return number


def _text(value, name, maximum, empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise ValueError(f"{name}必须是{'0' if empty else '1'}–{maximum} 个字符的文本。")
    return value


def _names(value, name, minimum, maximum):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{name}必须是包含 {minimum}–{maximum} 项的有序列表。")
    result = [_text(item, name, 80) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name}不能包含重复名称。")
    return result


def _manifest(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("模型配置必须是 JSON 对象。")
    unknown = set(metadata) - _REQUIRED - set(_DEFAULTS)
    missing = _REQUIRED - set(metadata)
    if unknown:
        raise ValueError("模型配置包含不支持的字段：" + "、".join(sorted(map(str, unknown))) + "。")
    if missing:
        raise ValueError("模型配置缺少字段：" + "、".join(sorted(missing)) + "。")
    result = {**_DEFAULTS, **metadata}
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("模型配置 schema_version 仅支持整数 1。")
    result["name"] = _text(result["name"], "模型名称", 80)
    result["version"] = _text(result["version"], "模型版本", 32)
    result["description"] = _text(result["description"], "模型说明", 1000, empty=True)
    if not isinstance(result["task"], str) or result["task"] not in _TASKS:
        raise ValueError(
            "模型任务仅支持 beat_classification、record_classification 或 record_multilabel。"
        )
    result["classes"] = _names(result["classes"], "类别", 2, 64)
    result["lead_names"] = _names(result["lead_names"], "导联", 1, 12)
    if "selected" in result["lead_names"] and result["lead_names"] != ["selected"]:
        raise ValueError("selected 只能单独使用，表示记录当前分析导联。")
    result["sample_rate"] = _finite_number(result["sample_rate"], "模型采样率", 100, 1000)
    size = result["input_samples"]
    if type(size) is not int or not 64 <= size <= 20000:
        raise ValueError("input_samples 必须是 64–20000 的整数。")
    pre = result["pre_samples"]
    if result["task"] == "beat_classification" and pre is None:
        raise ValueError("心搏模型必须配置 R 峰之前的 pre_samples。")
    if pre is not None and (type(pre) is not int or not 0 < pre < size):
        raise ValueError(
            "pre_samples 必须是大于 0 且小于 input_samples 的整数，记录任务也可设为 null。"
        )
    for key, choices in (
        ("normalization", {"none", "zscore", "minmax"}),
        ("input_filter", {"raw", "bandpass"}),
        ("output_type", {"logits", "probabilities"}),
    ):
        if not isinstance(result[key], str) or result[key] not in choices:
            raise ValueError(f"{key} 仅支持 " + "、".join(sorted(choices)) + "。")
    if type(result["notch"]) is not int or result["notch"] not in (0, 50, 60):
        raise ValueError("notch 仅支持整数 0、50 或 60。")
    if result["input_filter"] == "raw" and result["notch"] != 0:
        raise ValueError("原始信号 raw 不能启用陷波，请将 notch 设为 0 或选择 bandpass。")
    result["threshold"] = _finite_number(result["threshold"], "多标签阈值", 0, 1, exclusive=True)
    return result


def _safe_protobuf(message):
    # Reflection traverses graphs, attributes, sparse tensors and local functions alike.
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
            raise ValueError("模型含外部张量数据，请重新导出为权重内嵌的单个 ONNX 文件。")
    if message.DESCRIPTOR.full_name in {
        "onnx.NodeProto",
        "onnx.OperatorSetIdProto",
        "onnx.FunctionProto",
    }:
        if message.domain not in _DOMAINS:
            raise ValueError(f"模型包含不支持的算子域 {message.domain}，请导出标准 ONNX 算子。")
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            if field.is_repeated:
                for child in value:
                    _safe_protobuf(child)
            else:
                _safe_protobuf(value)


def _dynamic(dimension):
    return dimension is None or isinstance(dimension, str)


def _dimension(dimension, expected, name):
    if not _dynamic(dimension) and (type(dimension) is not int or dimension != expected):
        raise ValueError(
            f"模型{name}维度为 {dimension}，配置要求 {expected}，请检查导出形状和配置。"
        )


def _protobuf_shape(value):
    if (
        not value.type.HasField("tensor_type")
        or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise ValueError("模型输入和输出必须是 float32 张量，请重新导出。")
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        raise ValueError("模型必须明确声明输入和输出张量的维度布局。")
    return [
        dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param or None
        for dimension in tensor.shape.dim
    ]


def _tensor_contract(shape, out_shape, manifest):
    if not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3):
        raise ValueError("模型输入仅支持 [B,N] 或 [B,C,N]，不支持其他维度布局。")
    channels = len(manifest["lead_names"])
    if len(shape) == 2 and channels != 1:
        raise ValueError("二维输入 [B,N] 仅支持一个导联，多导联请导出 [B,C,N]。")
    _dimension(shape[0], 1, "输入批次（仅动态或固定 1）")
    if len(shape) == 3:
        _dimension(shape[1], channels, "输入导联")
    _dimension(shape[-1], manifest["input_samples"], "输入采样点")
    if not isinstance(out_shape, (list, tuple)) or len(out_shape) != 2:
        raise ValueError("模型输出仅支持二维 [B,K] 分类分数。")
    _dimension(out_shape[0], 1, "输出批次（仅动态或固定 1）")
    _dimension(out_shape[1], len(manifest["classes"]), "输出类别")


def _scores(output, rows, manifest):
    if not isinstance(output, np.ndarray) or output.shape != (rows, len(manifest["classes"])):
        raise ValueError(
            f"模型输出必须为 [批次, {len(manifest['classes'])}]，且每个输入均返回一行。"
        )
    if output.dtype != np.float32 or not np.isfinite(output).all():
        raise ValueError("模型输出必须是有限 float32 数值，不能包含 NaN 或无穷值。")
    values = output.astype(np.float64)
    if manifest["output_type"] == "probabilities":
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("模型声明输出概率，但值不在 0–1 范围；请修正模型或 output_type。")
        if manifest["task"] != "record_multilabel" and not np.all(
            np.abs(values.sum(axis=1) - 1) <= 1e-3
        ):
            raise ValueError(
                "单标签概率每行之和必须为 1（误差不超过 0.001），不会自动归一化错误概率。"
            )
        return values
    if manifest["task"] == "record_multilabel":
        return expit(values)
    values -= values.max(axis=1, keepdims=True)
    np.exp(values, out=values)
    values /= values.sum(axis=1, keepdims=True)
    return values


def _run(session, signature, batch, manifest):
    try:
        outputs = session.run([signature["output_name"]], {signature["input_name"]: batch})
    except Exception as error:
        raise ValueError("ONNX CPU 推理失败，请检查算子支持及输入形状：" + str(error)) from error
    if len(outputs) != 1:
        raise ValueError("模型必须返回唯一的分类输出张量。")
    return _scores(outputs[0], len(batch), manifest)


def validate_model(content: bytes, metadata: dict):
    """Validate a self-contained model and exercise its actual CPU output contract."""
    manifest = _manifest(metadata)
    if not isinstance(content, bytes) or not content:
        raise ValueError(
            "请提供非空 ONNX 文件；不支持 .pt、.pth 或 pickle，请从训练框架导出 ONNX。"
        )
    try:
        model = onnx.load_model_from_string(content)
    except Exception as error:
        raise ValueError("ONNX 文件解析失败，请重新导出有效模型：" + str(error)) from error
    _safe_protobuf(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("模型必须恰有一个输入和一个输出，且不能有可覆盖的权重输入。")
    _tensor_contract(
        _protobuf_shape(model.graph.input[0]), _protobuf_shape(model.graph.output[0]), manifest
    )
    try:
        onnx.checker.check_model(model)
    except Exception as error:
        raise ValueError("ONNX 结构检查失败，请重新导出有效模型：" + str(error)) from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        session = ort.InferenceSession(
            content, sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        raise ValueError(
            "无法创建 ONNX CPU 会话，请检查模型版本及支持的算子：" + str(error)
        ) from error
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1 or session.get_overridable_initializers():
        raise ValueError("模型必须恰有一个输入和一个输出，且不能有可覆盖的权重输入。")
    source, destination = inputs[0], outputs[0]
    if source.type != "tensor(float)" or destination.type != "tensor(float)":
        raise ValueError("模型输入和输出必须是 float32 张量，请重新导出。")
    shape, out_shape = source.shape, destination.shape
    _tensor_contract(shape, out_shape, manifest)
    channels = len(manifest["lead_names"])
    signature = {
        "input_name": source.name,
        "output_name": destination.name,
        "input_shape": list(shape),
        "output_shape": list(out_shape),
        "layout": "NL" if len(shape) == 2 else "NCL",
        "dynamic_batch": _dynamic(shape[0]),
        "runtime_version": ort.__version__,
        "provider": "CPUExecutionProvider",
    }
    for rows in (1, 2) if signature["dynamic_batch"] else (1,):
        dimensions = (
            (rows, manifest["input_samples"])
            if len(shape) == 2
            else (rows, channels, manifest["input_samples"])
        )
        _run(session, signature, np.zeros(dimensions, dtype=np.float32), manifest)
    return manifest, session, signature


def _lead_names(manifest, record):
    names = manifest["lead_names"]
    if names == ["selected"]:
        selected = record.get("selected_lead")
        if not isinstance(selected, str) or not selected:
            raise ValueError("记录未设置分析导联，请先选择导联并重新分析。")
        return [selected]
    return list(names)


def _source_filter_error(manifest, source_rate):
    if manifest["input_filter"] != "bandpass" or not manifest["notch"]:
        return ""
    try:
        rate = ecg._rate(source_rate)
    except ValueError as error:
        return str(error)
    if manifest["notch"] >= rate / 2:
        return f"源采样率 {rate:g} Hz 无法执行配置中的 {manifest['notch']} Hz 陷波；陷波频率必须低于源采样率的一半。请选择训练预处理匹配的模型或更高采样率的原始数据。"
    return ""


def compatibility(manifest: dict, record: dict):
    """Report availability only; compatibility does not measure model accuracy."""
    if not record.get("has_original"):
        return False, "记录未保留原始采样，请重新导入原始文件后推理。"
    try:
        names = _lead_names(manifest, record)
    except ValueError as error:
        return False, str(error)
    missing = [name for name in names if name not in record.get("lead_names", [])]
    if missing:
        return False, "记录缺少模型要求的导联：" + "、".join(
            missing
        ) + "；请选择匹配的模型或导入对应导联。"
    filter_error = _source_filter_error(manifest, record.get("sample_rate"))
    if filter_error:
        return False, filter_error
    if manifest["task"] == "beat_classification":
        if not record.get("quality", {}).get("reliable"):
            return False, "当前记录的 R 峰检测不可靠，不能进行心搏模型推理；请检查信号并重新分析。"
        if not record.get("peaks"):
            return False, "记录没有检出的 R 峰，请先重新分析或选择记录分类模型。"
    else:
        duration = record.get("duration")
        required = manifest["input_samples"] / manifest["sample_rate"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration + 1e-9 < required
        ):
            return (
                False,
                f"模型需要至少 {required:g} 秒的完整原始窗口，请导入更长记录或选择较短窗口模型。",
            )
    return True, ""


def _normalize(batch, method):
    if method == "zscore":
        mean = batch.mean(axis=-1, keepdims=True, dtype=np.float64)
        scale = np.maximum(batch.std(axis=-1, keepdims=True, dtype=np.float64), 1e-6)
        np.subtract(batch, mean, out=batch)
        np.divide(batch, scale, out=batch)
    elif method == "minmax":
        low = batch.min(axis=-1, keepdims=True)
        scale = np.maximum(batch.max(axis=-1, keepdims=True) - low, 1e-6)
        batch -= low
        batch /= scale
    if not np.isfinite(batch).all():
        raise ValueError("预处理产生非有限 float32 输入，请检查原始幅值和归一化配置。")


def predict(
    session, manifest: dict, signature: dict, record: dict, original: dict, start: float = 0
):
    """Infer complete original-anchored windows, filtering/resampling each lead once."""
    began = perf_counter()
    allowed, reason = compatibility(manifest, record)
    if not allowed:
        raise ValueError(reason)
    start = _finite_number(start, "窗口起点", 0, float("inf"))
    names = _lead_names(manifest, record)
    if not isinstance(original, dict) or original.get("units") != "mV":
        raise ValueError("模型推理需要以 mV 保存的原始采样，不能使用显示波形或其他单位。")
    source_rate = ecg._rate(original.get("sample_rate"))
    filter_error = _source_filter_error(manifest, source_rate)
    if filter_error:
        raise ValueError(filter_error)
    target_rate = manifest["sample_rate"]
    ratio = Fraction(target_rate / source_rate).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    arrays = []
    source_length = None
    leads = original.get("leads")
    if not isinstance(leads, dict):
        raise ValueError("原始导联采样不可用，请重新导入原始文件。")
    for name in names:
        if name not in leads:
            raise ValueError(f"原始采样缺少导联 {name}，请重新导入相应数据。")
        raw = np.asarray(leads[name])
        if (
            raw.ndim != 1
            or raw.dtype.kind not in "fiu"
            or not len(raw)
            or not np.isfinite(raw).all()
        ):
            raise ValueError(f"原始导联 {name} 必须是非空的一维有限数字数组。")
        if source_length is not None and len(raw) != source_length:
            raise ValueError("原始导联长度不一致，不能截断或补齐。")
        source_length = len(raw)
        raw = raw.astype(np.float64, copy=False)
        if manifest["input_filter"] == "bandpass":
            try:
                raw = ecg._filter_waveform(raw, source_rate, manifest["notch"])
            except Exception as error:
                raise ValueError(
                    f"导联 {name} 无法按模型配置滤波，请检查原始信号长度：{error}"
                ) from error
        if up != down:
            raw = resample_poly(raw, up, down)
        if not np.isfinite(raw).all():
            raise ValueError(f"导联 {name} 滤波或重采样后包含非有限值。")
        arrays.append(raw)
    size = manifest["input_samples"]
    duration = source_length / source_rate
    length = min(map(len, arrays))
    windows = []
    skipped = 0
    if manifest["task"] == "beat_classification":
        for peak in record["peaks"]:
            index = peak.get("index") if isinstance(peak, dict) else None
            if type(index) is not int or not 0 <= index < source_length:
                raise ValueError("记录 R 峰索引超出原始采样或不是整数，请重新分析。")
            time = peak.get("time", index / source_rate)
            if (
                isinstance(time, bool)
                or not isinstance(time, (int, float))
                or not math.isfinite(time)
            ):
                raise ValueError("记录 R 峰时间不是有限数字，请重新分析。")
            offset = int(round(index * target_rate / source_rate)) - manifest["pre_samples"]
            if (
                offset < 0
                or offset + size > length
                or (offset + size) / target_rate > duration + 1e-9
            ):
                skipped += 1
                continue
            windows.append((offset, index, float(time)))
        if not windows:
            raise ValueError(
                "所有 R 峰的模型窗口均超出原始采样边界；不会填充样本，请选择更短窗口或更长记录。"
            )
    else:
        if start > duration:
            raise ValueError("所选起点超出原始记录，请选择记录内的完整模型窗口。")
        offset = int(round(start * target_rate))
        if offset + size > length or (offset + size) / target_rate > duration + 1e-9:
            raise ValueError(
                "所选起点没有足够的完整模型窗口，不能越界或填充；请选择更早起点或更长记录。"
            )
        windows.append((offset, None, offset / target_rate))
    predictions = []
    counts = [0] * len(manifest["classes"])
    batch_size = 32 if signature["dynamic_batch"] else 1
    for first in range(0, len(windows), batch_size):
        chunk = windows[first : first + batch_size]
        dimensions = (
            (len(chunk), size) if signature["layout"] == "NL" else (len(chunk), len(names), size)
        )
        batch = np.empty(dimensions, dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            for row, (offset, _, _) in enumerate(chunk):
                if signature["layout"] == "NL":
                    batch[row] = arrays[0][offset : offset + size]
                else:
                    for channel, values in enumerate(arrays):
                        batch[row, channel] = values[offset : offset + size]
            _normalize(batch, manifest["normalization"])
        scores = _run(session, signature, batch, manifest)
        for (offset, index, time), values in zip(chunk, scores, strict=True):
            if manifest["task"] == "record_multilabel":
                indices = np.flatnonzero(values >= manifest["threshold"]).tolist()
                labels = [manifest["classes"][item] for item in indices]
                class_index = None
                class_label = "、".join(labels) if labels else "未超过阈值"
                score = float(values.max())
            else:
                class_index = int(values.argmax())
                class_label = manifest["classes"][class_index]
                score = float(values[class_index])
                indices = [class_index]
                labels = [class_label]
            for item in indices:
                counts[item] += 1
            predictions.append(
                {
                    "index": index,
                    "time": time,
                    "start": offset / target_rate,
                    "end": (offset + size) / target_rate,
                    "class_index": class_index,
                    "class_label": class_label,
                    "score": score,
                    "scores": values.tolist(),
                    "labels": labels,
                }
            )
    return {
        "task": manifest["task"],
        "input_leads": names,
        "predictions": predictions,
        "summary": {
            "evaluated": len(predictions),
            "skipped": skipped,
            "counts": [
                {"label": label, "count": count}
                for label, count in zip(manifest["classes"], counts, strict=True)
            ],
        },
        "elapsed_ms": (perf_counter() - began) * 1000,
        "preprocessing": {
            "source_sample_rate": source_rate,
            "sample_rate": target_rate,
            "resample_up": up,
            "resample_down": down,
            "normalization": manifest["normalization"],
            "input_filter": manifest["input_filter"],
            "notch": manifest["notch"],
            "input_samples": size,
            "pre_samples": manifest["pre_samples"],
        },
        "warnings": [f"已跳过 {skipped} 个原始采样边界处的不完整心搏窗口（未填充）。"]
        if skipped
        else [],
    }
