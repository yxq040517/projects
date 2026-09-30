"""Printable research report: actual measurements, provenance and manual review audit."""

from __future__ import annotations

import html

REVIEW_LABELS = {"unreviewed": "待复核", "in_progress": "复核中", "reviewed": "已复核（人工设定）"}
BEAT_LABELS = {"confirmed": "已确认峰", "artifact": "伪迹 / 误检", "uncertain": "待确认"}


def escape(value) -> str:
    return html.escape(str(value))


def display(value, suffix="") -> str:
    if value is None:
        return "无法可靠计算"
    return f"{value:g}{suffix}" if isinstance(value, (float, int)) else escape(value)


def waveform_svg(record: dict) -> str:
    wave = record["waveform"]
    shown = [
        (time, value)
        for time, value in zip(wave["time"], wave["filtered"], strict=True)
        if time <= 10
    ]
    if len(shown) < 2:
        return ""
    span = max(shown[-1][0], 0.01)
    low, high = min(value for _, value in shown), max(value for _, value in shown)
    amplitude = max(high - low, 0.2)
    points = " ".join(
        f"{40 + time / span * 1020:.2f},{30 + (high - value) / amplitude * 155:.2f}"
        for time, value in shown
    )
    markers = "".join(
        f'<line x1="{40 + peak["time"] / span * 1020:.2f}" y1="20" x2="{40 + peak["time"] / span * 1020:.2f}" y2="190" stroke="#cbd5e1" stroke-dasharray="3 3"/>'
        for peak in record["peaks"]
        if peak["time"] <= span
    )
    return f'''<svg viewBox="0 0 1100 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="滤波心电片段">
    <defs><pattern id="ecg-grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="#f0dede" stroke-width="0.6"/></pattern></defs>
    <rect width="1100" height="200" fill="url(#ecg-grid)"/>{markers}<polyline fill="none" stroke="#0d766e" stroke-width="1.4" points="{points}"/>
    <text x="40" y="220" font-size="12" fill="#64748b">0 s</text><text x="1030" y="220" font-size="12" fill="#64748b">{span:.1f} s</text></svg>'''


def template_svg(template: dict) -> str:
    if not template.get("reliable") or not template.get("median"):
        return f'<p class="hint">{escape(template.get("reason") or "原始采样不可用，未生成模板。")}</p>'
    times = template["time_ms"]
    median = template["median"]
    lower, upper = template["p25"], template["p75"]
    low, high = min(lower), max(upper)
    span = max(high - low, 0.2)

    def points(values):
        return " ".join(
            f"{60 + (time + 200) / 600 * 970:.2f},{25 + (high - value) / span * 145:.2f}"
            for time, value in zip(times, values, strict=True)
        )

    # The band is descriptive quartiles across measured beats, not diagnostic confidence.
    band = (
        points(upper)
        + " "
        + " ".join(
            f"{60 + (time + 200) / 600 * 970:.2f},{25 + (high - value) / span * 145:.2f}"
            for time, value in zip(reversed(times), reversed(lower), strict=True)
        )
    )
    return f'''<svg viewBox="0 0 1100 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="心搏模板中位数与四分位范围">
    <rect x="60" y="20" width="970" height="155" fill="#f5faf9"/><polygon points="{band}" fill="#cce7e2"/>
    <polyline points="{points(median)}" stroke="#0d766e" stroke-width="2" fill="none"/>
    <text x="10" y="30" font-size="12" fill="#64748b">{high:.2f} mV</text><text x="10" y="170" font-size="12" fill="#64748b">{low:.2f} mV</text>
    <text x="60" y="200" font-size="12" fill="#64748b">−200 ms</text><text x="370" y="200" font-size="12" fill="#64748b">R · 0 ms</text><text x="975" y="200" font-size="12" fill="#64748b">400 ms</text></svg>'''


def prediction_section(prediction: dict | None) -> str:
    if prediction is None:
        return ""
    model = prediction["model"]
    manifest = model["manifest"]
    classes = manifest["classes"]
    preprocessing = prediction["preprocessing"]
    status = "历史结果（当前检测设置已变化）" if prediction.get("stale") else "已保存的模型输出"
    availability = (
        "" if prediction.get("model_available", True) else "模型文件已删除，运行快照仍保留。"
    )
    rows = []
    for item in prediction["predictions"][:200]:
        scores = " · ".join(
            f"{escape(label)}: {score:.4f}"
            for label, score in zip(classes, item["scores"], strict=True)
        )
        rows.append(
            f"<tr><td>{item['time']:.3f}</td><td>{item['start']:.3f}–{item['end']:.3f}</td><td>{escape(item['class_label'])}</td><td>{item['score']:.4f}</td><td>{scores}</td></tr>"
        )
    counts = " · ".join(
        f"{escape(item['label'])}: {item['count']}" for item in prediction["summary"]["counts"]
    )
    warnings = "；".join(escape(warning) for warning in prediction.get("warnings", []))
    return f"""<section><h2>模型输出 · {escape(model["name"])} / {escape(model["version"])}</h2>
    <p>{status} · {escape(prediction["created_at"])}<br>输入导联：{escape(", ".join(prediction["input_leads"]))} · 用时 {prediction["elapsed_ms"]:.1f} ms<br>运行 SHA256：{escape(model["sha256"])}</p>
    <p class="hint">{availability} 采样率 {preprocessing["source_sample_rate"]:g} → {preprocessing["sample_rate"]:g} Hz，窗口 {preprocessing["input_samples"]} 点；归一化 {escape(preprocessing["normalization"])}，输入滤波 {escape(preprocessing["input_filter"])}。这是导入模型的输出分数，不是已验证的医学置信度。</p>
    <p>运行 {prediction["summary"]["evaluated"]} 项 · 边界跳过 {prediction["summary"]["skipped"]} 项<br>{counts}</p>{"<p class='hint'>" + warnings + "</p>" if warnings else ""}
    <table><thead><tr><th>时间 / s</th><th>输入范围 / s</th><th>输出类别</th><th>模型分数</th><th>各类分数（配置顺序）</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
    <p class="hint">报告最多展示前 200 项；完整模型结果可从平台导出 JSON。人工复核与信号测量不被这些类别自动替换。</p></section>"""


def render_report(record: dict, prediction: dict | None = None) -> str:
    metrics = record["metrics"]
    fields = [
        ("平均心率", "heart_rate", " bpm"),
        ("检测心搏数", "beat_count", ""),
        ("平均 RR", "mean_rr", " ms"),
        ("SDRR", "sdnn", " ms"),
        ("相邻 RR 差均方根", "rmssd", " ms"),
        ("pRR50", "pnn50", " %"),
    ]
    cards = "".join(
        f'<div class="metric"><span>{label}</span><strong>{display(metrics.get(key), unit)}</strong></div>'
        for label, key, unit in fields
    )
    warnings = [finding for finding in record["findings"] if finding["level"] != "info"]
    findings = "".join(
        f"<li><b>{escape(item['title'])}</b><p>{escape(item['detail'])}</p></li>"
        for item in warnings
    )
    parser_warnings = "".join(
        f"<li>{escape(warning)}</li>" for warning in record["parser_warnings"]
    )
    reviews = record.get("beat_reviews", [])
    current = {
        review["index"]: review for review in reviews if review["lead"] == record["selected_lead"]
    }
    rows = "".join(
        f"<tr><td>{index + 1}</td><td>{item['time']:.3f}</td><td>{item['rr_ms']:.1f}</td><td>{display(item['heart_rate'])}</td><td>{'需复核' if item['flag'] else '—'}</td><td>{BEAT_LABELS.get(current.get(item.get('peak_index'), {}).get('status'), '未标记')}</td></tr>"
        for index, item in enumerate(record["rr"])
    )
    review_rows = "".join(
        f'<tr><td>{escape(item["lead"])}</td><td>{item["time"]:.3f}</td><td>{BEAT_LABELS.get(item["status"], "待确认")}</td><td class="note">{escape(item["note"]) or "—"}</td></tr>'
        for item in reviews
    )
    review_status = REVIEW_LABELS.get(record.get("review_status", "unreviewed"), "待复核")
    segment = record.get("segment")
    segment_html = ""
    if segment:
        segment_html = f'<p class="segment">独立选段 · 父记录 {escape(segment["parent_id"])} · 父记录内 {segment["start"]:.3f}–{segment["end"]:.3f} s<br>相对最初原始记录起点：{record.get("source_offset", 0):.3f} s；本报告时间轴从片段起点 0 s 开始。</p>'
    template = (record.get("tools") or {}).get("template") or {}
    template_html = template_svg(template)
    template_caption = f"参与统计 {template.get('used_count', 0)} / 完整候选 {template.get('available_count', 0)} 搏。中位数与 25–75% 范围；逐搏去局部基线，未归一化幅值，不代表正常心搏或诊断置信区间。"
    notch = f"{record['notch']} Hz 陷波" if record["notch"] else "不使用工频陷波"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>心电分析报告 · {escape(record["name"])}</title><style>
    *{{box-sizing:border-box}}body{{font:14px/1.7 "Microsoft YaHei",sans-serif;color:#243042;background:#f2f5f6;margin:0;padding:40px}}
    main{{max-width:1000px;margin:auto;background:#fff;padding:38px;border-radius:12px}}h1{{font-size:26px;margin:6px 0}}h2{{font-size:18px;margin-top:26px}}.brand{{color:#0d766e;font-weight:bold;letter-spacing:2px}}.muted,.hint{{color:#64748b}}.hint{{font-size:12px}}.notice{{padding:12px;background:#fff6e6;border-left:3px solid #d99d31}}.segment{{background:#edf8f5;padding:12px}}
    .metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{border:1px solid #e4e9ec;padding:14px;border-radius:8px}}.metric span{{display:block;color:#64748b;font-size:12px}}.metric strong{{font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{text-align:left;border-bottom:1px solid #e5e7eb;padding:7px;overflow-wrap:anywhere}}li p{{margin:2px 0 12px}}svg{{width:100%;height:auto}}.note{{white-space:pre-wrap}}a{{color:#0d766e}}@media print{{body{{padding:0;background:white}}main{{padding:15px;max-width:none}}.print-hint{{display:none}}tr{{break-inside:avoid}}h2{{break-after:avoid}}}}@media(max-width:600px){{body{{padding:12px}}main{{padding:20px}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}
    </style></head><body><main><div class="brand">PULSELAB / ECG RESEARCH</div><h1>心电信号分析报告</h1>
    <p class="muted">{escape(record["name"])} · {escape(record["created_at"])}</p><p class="hint print-hint">Ctrl+P 打印或另存为 PDF。</p>
    <p class="notice">{escape(record["disclaimer"])}</p>{segment_html}<h2>记录信息</h2>
    <p>记录 ID：{escape(record["id"])}<br>来源：{escape(record["source"])}<br>分析导联：{escape(record["selected_lead"])} / 全部导联：{escape(", ".join(record["lead_names"]))}<br>采样率：{record["sample_rate"]:g} Hz · 时长：{record["duration"]:.2f} s · 幅值：mV<br>滤波：0.5–40 Hz 带通，{notch}<br>人工状态：<b>{review_status}</b> · 当前导联标记 {len(current)} 个 · 原始采样：{"已保留" if record.get("has_original") else "旧记录不可用"}</p>
    <h2>测量结果</h2><div class="metrics">{cards}</div><p class="hint">统计对象是算法检出的 RR，未鉴别正常心搏；不是临床 NN-HRV。人工标记不改变算法指标。</p>
    <h2>波形 · 前 10 秒</h2>{waveform_svg(record)}<p class="hint">自适应幅值，非校准纸速与增益；检测标记需人工复核。</p>
    <h2>心搏模板 · {escape(record["selected_lead"])}</h2>{template_html}<p class="hint">{template_caption}</p>
    {prediction_section(prediction)}
    <h2>信号质量</h2><p>工程评分 {record["quality"]["score"]}/100 · {escape(record["quality"]["label"])}<br>低频功率 {record["quality"]["baseline_ratio"]:.1%} · 高频功率 {record["quality"]["high_frequency_ratio"]:.1%} · 平坦比例 {record["quality"]["flatline_ratio"]:.1%} · 极值重复 {record["quality"]["clipping_ratio"]:.1%}。</p>
    <h2>需复核事项</h2><ul>{findings or "<li>没有额外规则标记；这不排除任何疾病。</li>"}</ul>{"<h2>导入说明</h2><ul>" + parser_warnings + "</ul>" if parser_warnings else ""}
    <h2>人工备注</h2><p class="note">{escape(record["note"]) or "未填写"}</p>
    <h2>人工标记记录</h2><p class="hint">按导联保存的人工意见，不代表医学诊断；重新分析可能改变候选峰位置。复核状态由操作者手动设定。</p>
    <table><thead><tr><th>导联</th><th>记录内时间 / s</th><th>标记</th><th>备注</th></tr></thead><tbody>{review_rows or '<tr><td colspan="4">暂无人工标记</td></tr>'}</tbody></table>
    <h2>RR 间期明细</h2><table><thead><tr><th>序号</th><th>时间 / s</th><th>RR / ms</th><th>瞬时心率 / bpm</th><th>规则标记</th><th>人工标记</th></tr></thead><tbody>{rows or '<tr><td colspan="6">无法可靠计算 RR 序列</td></tr>'}</tbody></table>
    <h2>方法与边界</h2><p class="hint">SciPy 带通与陷波、QRS 导数能量及自适应阈值、Welch 频谱及逐搏模板。本工具不分析 P/QRS/QT 时限、ST 段或房颤，不包含临床验证的疾病分类模型。质量评分、形态差异和人工标记均不能排除或确诊疾病。</p>
    <p class="hint">PulseLab 2.1 · 本地存储。公开示例来自 <a href="https://physionet.org/content/mitdb/1.0.0/">MIT-BIH Arrhythmia Database</a>，Moody &amp; Mark (2001)，DOI 10.13026/C2F305，<a href="https://opendatacommons.org/licenses/by/1-0/">ODC Attribution 1.0</a>；合成示例仅为工程演示。</p>
    </main></body></html>"""
