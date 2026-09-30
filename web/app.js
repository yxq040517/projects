"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    records: [],
    demos: [],
    record: null,
    page: "workspace",
    lead: "",
    filtered: true,
    start: 0,
    window: 10,
    beatsPage: 0,
    busy: false,
    deleting: null,
    hover: null,
    view: "single",
    tool: "rr",
    api2: false,
    segmentPreview: null,
    segmentParent: null,
    reviewing: null,
    comparison: null,
    compareStart: 0,
    models: [],
    defaultModelId: null,
    recordModels: [],
    selectedModelId: "",
    modelRecordId: null,
    modelContextKey: "",
    modelRequest: 0,
    registryRequest: 0,
    predictionRequest: 0,
    prediction: null,
    predictionPage: 0,
    modelLoading: false,
    predictionLoading: false,
    editingModel: null,
    deletingModel: null,
    configRequest: 0,
    configLoading: false,
  };
  const number = (value, digits = 0) =>
    typeof value === "number" && Number.isFinite(value)
      ? value.toLocaleString("zh-CN", {
          maximumFractionDigits: digits,
          minimumFractionDigits: digits,
        })
      : "—";
  const text = (id, value) => {
    $(id).textContent = value == null || value === "" ? "—" : String(value);
  };
  const duration = (seconds) =>
    typeof seconds === "number" && Number.isFinite(seconds)
      ? seconds < 60
        ? `${number(seconds, 1)} 秒`
        : `${Math.floor(seconds / 60)} 分 ${number(seconds % 60, 0)} 秒`
      : "—";
  const date = (value) => {
    const d = new Date(value);
    return value && !Number.isNaN(d.getTime())
      ? d.toLocaleString("zh-CN", {
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
        })
      : "—";
  };
  const endpoint = (id) => `/api/records/${encodeURIComponent(id)}`;
  const el = (tag, className, value) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value != null) node.textContent = String(value);
    return node;
  };
  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }
  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(path, { cache: "no-store", ...options });
    } catch (_) {
      throw new Error(
        "无法连接本地服务。请确认工作站服务已启动，然后重新连接。",
      );
    }
    const type = response.headers.get("content-type") || "";
    let data = null;
    if (type.includes("json")) {
      try {
        data = await response.json();
      } catch (_) {
        throw new Error("本地服务返回了无法解析的响应，请重试。");
      }
    }
    if (!response.ok)
      throw new Error(
        typeof data?.detail === "string"
          ? data.detail
          : `请求未完成（HTTP ${response.status}），请稍后重试。`,
      );
    if (data === null)
      throw new Error("本地服务响应格式不正确，请检查服务是否正常。");
    return data;
  }
  const json = (data) => ({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  function notify(message, kind = "error") {
    text("notice-text", message);
    $("notice").className = `notice ${kind}`;
    $("notice").hidden = false;
  }
  function formError(id, message = "") {
    $(id).textContent = message;
    $(id).hidden = !message;
  }
  function busy(active, message = "正在处理波形…") {
    state.busy = active;
    $("busy-overlay").hidden = !active;
    text("busy-text", message);
    document.body.setAttribute("aria-busy", String(active));
    for (const dialog of document.querySelectorAll("dialog")) {
      dialog.setAttribute("aria-busy", String(active));
      let status = dialog.querySelector(".dialog-busy");
      if (!status) {
        status = el("div", "dialog-busy");
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
        dialog.prepend(status);
      }
      status.textContent = active ? message : "";
      status.hidden = !active;
      for (const node of dialog.querySelectorAll(
        "button, input, select, textarea",
      ))
        node.disabled = active;
    }
    if (!active) {
      updateOriginalControls();
      $("segment-save").disabled = !state.segmentPreview;
      $("beat-review-clear").disabled = !state.reviewing?.review;
    }
    updateModelControls();
    updateModelForm();
  }
  function openDialog(id) {
    if (state.busy) return;
    $(id).showModal();
  }
  function setSidebar(open) {
    $("sidebar").classList.toggle("open", open);
    $("sidebar-backdrop").hidden = !open;
    $("menu-toggle").setAttribute("aria-expanded", String(open));
    $("menu-toggle").setAttribute("aria-label", open ? "关闭导航" : "展开导航");
  }
  function changePage(page) {
    if (!["workspace", "history", "comparison", "models"].includes(page))
      page = "workspace";
    state.page = page;
    for (const name of ["workspace", "history", "comparison", "models"])
      $(`page-${name}`).hidden = name !== page;
    for (const link of document.querySelectorAll("[data-page]")) {
      const active = link.dataset.page === page;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    text(
      "page-label",
      {
        workspace: "分析工作台",
        history: "记录管理",
        comparison: "记录对比",
        models: "模型管理",
      }[page],
    );
    setSidebar(false);
    requestAnimationFrame(drawAll);
  }
  function navigate(page) {
    if (location.hash !== `#${page}`) location.hash = page;
    else changePage(page);
  }
  async function loadStats() {
    try {
      const stats = await api("/api/stats");
      text("stat-total", number(stats.total_records));
      text("stat-duration", duration(stats.total_duration));
      text("stat-review", number(stats.review_records));
    } catch (_) {
      text("stat-total", number(state.records.length));
      text(
        "stat-duration",
        duration(
          state.records.reduce(
            (total, r) => total + (Number(r.duration) || 0),
            0,
          ),
        ),
      );
      text("stat-review", "—");
    }
  }
  async function loadRecords() {
    const records = await api("/api/records");
    if (!Array.isArray(records)) throw new Error("记录列表格式不正确。");
    state.records = records
      .slice()
      .sort(
        (a, b) =>
          (Date.parse(b.created_at) || 0) - (Date.parse(a.created_at) || 0),
      );
    renderHistory();
    refreshCompareSelectors();
    text("record-count", state.records.length);
    await loadStats();
  }
  async function loadDemos() {
    const demos = await api("/api/demos");
    if (!Array.isArray(demos)) throw new Error("示例列表格式不正确。");
    state.demos = demos;
    renderDemos();
  }
  function renderEmpty(
    title = "还没有心电记录",
    description = "导入本地数值波形，或选择公开记录开始观察。合成示例仅用于功能演示。",
  ) {
    state.record = null;
    resetModelRecord();
    state.segmentPreview = null;
    state.segmentParent = null;
    state.reviewing = null;
    state.hover = null;
    $("segment-result").hidden = true;
    for (const id of ["segment-dialog", "beat-review-dialog"])
      if ($(id).open) $(id).close();
    updateOriginalControls();
    $("analysis-content").hidden = true;
    $("workspace-empty").hidden = false;
    text("empty-title", title);
    text("empty-description", description);
  }
  async function syncSavedRecord(record, message) {
    const summary = { ...record };
    delete summary.waveform;
    delete summary.peaks;
    delete summary.rr;
    const index = state.records.findIndex((item) => item.id === record.id);
    if (index >= 0) state.records[index] = summary;
    else state.records.unshift(summary);
    renderHistory();
    refreshCompareSelectors();
    text("record-count", state.records.length);
    try {
      await loadRecords();
      notify(message, "success");
    } catch (error) {
      notify(`${message} 但重新读取记录列表失败：${error.message}`);
    }
  }
  async function start() {
    if (state.busy) return;
    $("connection-retry").hidden = true;
    busy(true, "正在读取本地记录…");
    try {
      try {
        await loadDemos();
      } catch (error) {
        formError("demo-error", error.message);
      }
      await loadRecords();
      loadModels().catch((error) => formError("models-error", error.message));
      $("connection").classList.remove("offline");
      text("connection-text", "本地服务已连接");
      if (state.records.length)
        showRecord(await api(endpoint(state.records[0].id)));
      else {
        const firstPublic = state.demos.find((d) => d.kind === "public");
        if (firstPublic) {
          busy(true, "正在分析首条公开心电记录…");
          try {
            const record = await api(
              `/api/demos/${encodeURIComponent(firstPublic.id)}/analyze`,
              json({ notch: 50 }),
            );
            showRecord(record);
            await syncSavedRecord(record, "已载入首条公开心电记录。");
          } catch (error) {
            renderEmpty(
              "公开示例暂时无法载入",
              `${error.message} 您仍可导入本地文件或选择其他示例。`,
            );
            notify(error.message);
          }
        } else
          renderEmpty(
            "还没有心电记录",
            "当前没有可用的公开示例。可导入本地文件，或选择明确标注的合成示例。",
          );
      }
      api("/api/health")
        .then((health) => {
          state.api2 = /^2(?:\.|$)/.test(String(health.version || ""));
          updateOriginalControls();
          $("compare-run").disabled = state.records.length < 2 || !state.api2;
          renderBeats();
          text(
            "connection-text",
            health.local_only ? "本地处理 · 服务已连接" : "服务已连接",
          );
        })
        .catch(() => {});
    } catch (error) {
      $("connection").classList.add("offline");
      text("connection-text", "本地服务未连接");
      renderEmpty("无法读取本地记录", error.message);
      $("connection-retry").hidden = false;
      notify(error.message);
    } finally {
      busy(false);
    }
  }
  function showRecord(record) {
    state.record = record;
    state.segmentPreview = null;
    state.segmentParent = null;
    state.reviewing = null;
    state.view = "single";
    state.tool = "rr";
    $("segment-result").hidden = true;
    selectTool("rr");
    buildMultileadButtons();
    $("review-status").value = record.review_status || "unreviewed";
    updateOriginalControls();
    state.lead = record.selected_lead;
    state.filtered = true;
    state.start = 0;
    state.beatsPage = 0;
    state.hover = null;
    $("ecg-tooltip").hidden = true;
    $("workspace-empty").hidden = true;
    $("analysis-content").hidden = false;
    text("record-name", record.name);
    text(
      "record-meta",
      `${date(record.created_at)} · ${number(record.sample_rate)} Hz · ${duration(record.duration)} · ${(record.lead_names || []).length} 导联`,
    );
    text("analysis-lead", `分析导联 ${record.selected_lead}`);
    const metrics = record.metrics || {};
    const quality = record.quality || {};
    text("metric-heart-rate", number(metrics.heart_rate, 1));
    text("metric-beat-count", number(metrics.beat_count));
    text("metric-mean-rr", number(metrics.mean_rr, 1));
    text("metric-quality", number(quality.score));
    text(
      "heart-rate-range",
      `瞬时心率范围 ${number(metrics.min_heart_rate, 1)}–${number(metrics.max_heart_rate, 1)} bpm`,
    );
    text("beat-duration", `分析时长 ${duration(record.duration)}`);
    text("quality-label", quality.label);
    text("metric-sdnn", `${number(metrics.sdnn, 1)} ms`);
    text("metric-rmssd", `${number(metrics.rmssd, 1)} ms`);
    text("metric-pnn50", `${number(metrics.pnn50, 1)} %`);
    $("quality-bar").style.width =
      `${Math.max(0, Math.min(100, Number(quality.score) || 0))}%`;
    $("quality-bar").classList.toggle("poor", !quality.reliable);
    text(
      "quality-reliability",
      quality.reliable ? "质量检查通过 · 仍需复核" : "可靠性不足 · 谨慎解读",
    );
    $("lead-select").replaceChildren();
    for (const lead of record.lead_names || []) {
      const option = el("option", "", lead);
      option.value = lead;
      $("lead-select").append(option);
    }
    $("lead-select").value = state.lead;
    $("report-link").href = `${endpoint(record.id)}/report`;
    $("export-link").href = `${endpoint(record.id)}/export`;
    $("findings-list").replaceChildren();
    const routine = el("details", "routine-findings");
    routine.append(el("summary", "", "常规提示"));
    for (const finding of record.findings || []) {
      const warning = ["warning", "error"].includes(finding.level);
      const item = el("details", `finding ${warning ? "warning" : "info"}`);
      item.append(
        el("summary", "", finding.title),
        el("p", "", finding.detail),
      );
      (warning ? $("findings-list") : routine).append(item);
    }
    if (routine.children.length > 1) $("findings-list").append(routine);
    if (!(record.findings || []).length)
      $("findings-list").append(el("p", "muted", "暂无质量提示"));
    const meta = [
      ["来源", record.source],
      ["文件", record.filename],
      ["采样率", `${number(record.sample_rate)} Hz`],
      ["时长", duration(record.duration)],
      ["单位", record.units || "mV"],
      ["分析导联", record.selected_lead],
      ["导联", (record.lead_names || []).join(" / ")],
      [
        "原始样本",
        record.has_original ? "已保存完整分辨率" : "不可用（旧记录）",
      ],
      ["人工复核", reviewLabel(record.review_status)],
      ["来源时间偏移", `${number(record.source_offset || 0, 3)} s`],
      [
        "选段",
        record.segment
          ? `${number(record.segment.start, 3)}–${number(record.segment.end, 3)} s · 父记录 ${record.segment.parent_id}`
          : "完整记录",
      ],
      ["基线能量比", number(quality.baseline_ratio, 3)],
      ["高频能量比", number(quality.high_frequency_ratio, 3)],
      [
        "平线比例",
        typeof quality.flatline_ratio === "number"
          ? `${number(quality.flatline_ratio * 100, 1)} %`
          : "—",
      ],
      [
        "削顶比例",
        typeof quality.clipping_ratio === "number"
          ? `${number(quality.clipping_ratio * 100, 1)} %`
          : "—",
      ],
    ];
    $("metadata-list").replaceChildren();
    for (const [key, value] of meta) {
      const group = el("div");
      group.append(
        el("dt", "", key),
        el("dd", "", value || value === 0 ? value : "—"),
      );
      $("metadata-list").append(group);
    }
    text(
      "record-note",
      record.note || "暂无研究备注，可通过记录标题旁的编辑按钮添加。",
    );
    $("parser-warnings").replaceChildren();
    const warnings = record.parser_warnings || [];
    $("parser-warnings").hidden = !warnings.length;
    if (warnings.length) {
      $("parser-warnings").append(el("h3", "", "导入提示"));
      for (const warning of warnings)
        $("parser-warnings").append(el("p", "", warning));
    }
    text("rr-count", `${(record.rr || []).length} 个间期`);
    updateWaveControls();
    renderBeats();
    refreshModelRecord();
    requestAnimationFrame(drawAll);
  }
  function updateWaveControls() {
    if (!state.record) return;
    const selected =
      state.lead === state.record.selected_lead && state.view === "single";
    $("wave-filtered").disabled = !selected;
    $("wave-filtered").classList.toggle("selected", selected && state.filtered);
    $("wave-filtered").setAttribute(
      "aria-pressed",
      String(selected && state.filtered),
    );
    $("wave-raw").classList.toggle("selected", !selected || !state.filtered);
    $("wave-raw").setAttribute(
      "aria-pressed",
      String(!selected || !state.filtered),
    );
    $("peak-toggle").disabled = !selected;
    $("lead-notice").hidden = state.lead === state.record.selected_lead;
    text(
      "lead-notice",
      `仅查看 ${state.lead} 原始波形；指标仍来自 ${state.record.selected_lead}。点击“分析此导联”重新计算。`,
    );
    $("waveform-wrap").hidden = state.view === "multi";
    $("multilead-view").hidden = state.view !== "multi";
    for (const view of ["single", "multi"]) {
      $(`view-${view}`).classList.toggle("selected", state.view === view);
      $(`view-${view}`).setAttribute(
        "aria-pressed",
        String(state.view === view),
      );
    }
    for (const button of $("multilead-buttons").children) {
      button.classList.toggle("selected", button.dataset.lead === state.lead);
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.lead === state.lead),
      );
    }
    updateOriginalControls();
    updatePredictionOverlay();
    text(
      "waveform-subtitle",
      `查看 ${state.lead} · mV / 秒 · 振幅自动缩放，非固定纸速显示`,
    );
    text(
      "trace-label",
      state.view === "multi"
        ? "各导联原始信号"
        : selected && state.filtered
          ? "滤波信号"
          : "原始信号",
    );
    const max = Math.max(0, state.record.duration - state.window);
    state.start = Math.max(0, Math.min(max, state.start));
    $("time-slider").max = String(max);
    $("time-slider").value = String(state.start);
    $("time-slider").disabled = max === 0;
    $("window-prev").disabled = state.start <= 0;
    $("window-next").disabled = state.start >= max;
    text(
      "window-range",
      `${number(state.start, 1)}–${number(Math.min(state.record.duration, state.start + state.window), 1)} s`,
    );
  }
  function renderHistory() {
    const query = $("record-search").value.trim().toLocaleLowerCase();
    const status = $("history-status").value;
    const filtered = state.records.filter(
      (r) =>
        [
          r.name,
          r.source,
          r.filename,
          r.note,
          r.selected_lead,
          ...(r.lead_names || []),
        ]
          .join(" ")
          .toLocaleLowerCase()
          .includes(query) &&
        (!status || (r.review_status || "unreviewed") === status),
    );
    $("history-body").replaceChildren();
    text(
      "history-count",
      `${filtered.length} / ${state.records.length} 条记录`,
    );
    $("history-empty").hidden = !!filtered.length;
    for (const record of filtered) {
      const row = el("tr");
      row.dataset.recordId = record.id;
      const identity = el("td");
      const openName = el(
        "button",
        "record-name-button",
        record.name || "未命名记录",
      );
      openName.type = "button";
      openName.addEventListener("click", () => openRecord(record.id));
      identity.append(
        openName,
        el(
          "small",
          "table-source",
          record.source || record.filename || "本地记录",
        ),
      );
      row.append(identity);
      for (const value of [
        date(record.created_at),
        record.selected_lead,
        duration(record.duration),
        `${number(record.metrics?.heart_rate, 1)} bpm`,
      ])
        row.append(el("td", "", value));
      const quality = el("td");
      quality.append(
        el(
          "span",
          `quality-chip ${record.quality?.reliable ? "good" : "review"}`,
          record.quality?.label || "待复核",
        ),
      );
      row.append(quality);
      row.append(el("td", "", reviewLabel(record.review_status)));
      const actions = el("td", "table-actions");
      const open = el("button", "button secondary small", "打开");
      open.type = "button";
      open.dataset.action = "open";
      open.addEventListener("click", () => openRecord(record.id));
      const remove = el("button", "icon-button delete-button");
      remove.type = "button";
      remove.setAttribute("aria-label", `删除 ${record.name || "记录"}`);
      remove.title = "删除记录";
      remove.dataset.action = "delete";
      remove.append(icon("trash"));
      remove.addEventListener("click", () => askDelete(record));
      const compare = el("button", "button secondary small", "加入对比");
      compare.type = "button";
      compare.dataset.action = "compare";
      compare.addEventListener("click", () => {
        $("compare-left").value = record.id;
        if ($("compare-right").value === String(record.id))
          $("compare-right").value = String(
            state.records.find((r) => r.id !== record.id)?.id || "",
          );
        clearComparison();
        navigate("comparison");
      });
      actions.append(open, compare, remove);
      row.append(actions);
      $("history-body").append(row);
    }
  }
  async function openRecord(id) {
    if (state.busy) return;
    busy(true, "正在读取完整波形…");
    try {
      const record = await api(endpoint(id));
      showRecord(record);
      navigate("workspace");
    } catch (error) {
      notify(error.message);
    } finally {
      busy(false);
    }
  }
  function renderDemos() {
    $("demo-list").replaceChildren();
    if (!state.demos.length) {
      $("demo-list").append(
        el("p", "muted", "暂无可用示例。可刷新列表，或导入本地波形。"),
      );
      return;
    }
    for (const demo of state.demos) {
      const item = el("article", "demo-item");
      const content = el("div");
      const heading = el("div", "demo-item-heading");
      heading.append(
        el("h3", "", demo.name),
        el(
          "span",
          `pill ${demo.kind === "public" ? "" : "synthetic"}`,
          demo.kind === "public" ? "公开记录" : "合成示例",
        ),
      );
      content.append(
        heading,
        el("p", "", demo.description),
        el(
          "small",
          "demo-meta",
          `${number(demo.sample_rate)} Hz · ${duration(demo.duration)} · ${(demo.lead_names || []).join(" / ")}`,
        ),
        el("small", "demo-source", `来源：${demo.source || "未注明"}`),
      );
      const button = el("button", "button secondary small", "分析并载入");
      button.type = "button";
      button.dataset.demoId = demo.id;
      button.addEventListener("click", () => analyzeDemo(demo.id));
      item.append(content, button);
      $("demo-list").append(item);
    }
  }
  async function analyzeDemo(id) {
    if (state.busy) return;
    formError("demo-error");
    busy(true, "正在处理示例原始波形…");
    try {
      const record = await api(
        `/api/demos/${encodeURIComponent(id)}/analyze`,
        json({ notch: 50 }),
      );
      showRecord(record);
      $("demo-dialog").close();
      navigate("workspace");
      await syncSavedRecord(record, "示例已完成分析并保存到本机。");
    } catch (error) {
      formError("demo-error", error.message);
    } finally {
      busy(false);
    }
  }
  function renderBeats() {
    const rr = state.record?.rr || [];
    const pageSize = 8;
    const pages = Math.max(1, Math.ceil(rr.length / pageSize));
    state.beatsPage = Math.min(pages - 1, Math.max(0, state.beatsPage));
    $("beat-table-body").replaceChildren();
    rr.slice(
      state.beatsPage * pageSize,
      (state.beatsPage + 1) * pageSize,
    ).forEach((beat, index) => {
      const row = el("tr", beat.flag ? "flagged" : "");
      row.tabIndex = 0;
      row.setAttribute(
        "aria-label",
        `定位第 ${state.beatsPage * pageSize + index + 1} 个 RR 间期，时间 ${number(beat.time, 2)} 秒`,
      );
      for (const value of [
        state.beatsPage * pageSize + index + 1,
        number(beat.time, 2),
        number(beat.rr_ms, 1),
        number(beat.heart_rate, 1),
        beat.flag ? "建议复核" : "未标记",
      ])
        row.append(el("td", "", value));
      const review = (state.record.beat_reviews || []).find(
        (mark) =>
          mark.lead === state.record.selected_lead &&
          mark.index === beat.peak_index,
      );
      const cell = el("td");
      const mark = el(
        "button",
        `button secondary small beat-mark ${review?.status || ""}`,
        review ? beatLabel(review.status) : "标注",
      );
      mark.type = "button";
      mark.dataset.peakIndex = beat.peak_index;
      mark.disabled = !state.api2 || !Number.isInteger(beat.peak_index);
      mark.title = review?.note || `原始峰索引 ${beat.peak_index ?? "不可用"}`;
      mark.addEventListener("click", (event) => {
        event.stopPropagation();
        openBeatReview(beat, review);
      });
      cell.append(mark);
      row.append(cell);
      const locate = () => {
        state.lead = state.record.selected_lead;
        state.view = "single";
        $("lead-select").value = state.lead;
        state.start = Math.max(0, beat.time - state.window / 2);
        updateWaveControls();
        drawECG();
        $("waveform-title").scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      };
      row.addEventListener("click", locate);
      row.addEventListener("keydown", (event) => {
        if (
          event.target === row &&
          (event.key === "Enter" || event.key === " ")
        ) {
          event.preventDefault();
          locate();
        }
      });
      $("beat-table-body").append(row);
    });
    if (!rr.length) {
      const row = el("tr");
      const cell = el(
        "td",
        "table-empty",
        "没有足够或可靠的 R 峰，暂无可复核的 RR 间期。",
      );
      cell.colSpan = 6;
      row.append(cell);
      $("beat-table-body").append(row);
    }
    text(
      "beat-page-label",
      `${state.record?.selected_lead || ""} · ${rr.length} 个间期${rr.length && !Number.isInteger(rr[0].peak_index) ? " · 旧记录无峰索引" : ""}`,
    );
    text("beats-page", `${state.beatsPage + 1} / ${pages}`);
    $("beats-prev").disabled = state.beatsPage <= 0;
    $("beats-next").disabled = state.beatsPage >= pages - 1;
  }
  function askDelete(record) {
    if (state.busy) return;
    state.deleting = record;
    text(
      "delete-description",
      `将永久删除“${record.name || "未命名记录"}”及其本地保存的波形、分析结果和备注。`,
    );
    formError("delete-error");
    openDialog("delete-dialog");
  }
  function canvasSetup(id) {
    const canvas = $(id);
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.round(rect.width * ratio),
      height = Math.round(rect.height * ratio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.font = "11px system-ui, sans-serif";
    return { ctx, width: rect.width, height: rect.height };
  }
  function lowerBound(values, value) {
    let left = 0,
      right = values.length;
    while (left < right) {
      const mid = (left + right) >>> 1;
      if (values[mid] < value) left = mid + 1;
      else right = mid;
    }
    return left;
  }
  function waveData() {
    const record = state.record;
    if (!record) return null;
    const wave = record.waveform || {};
    const selected = state.lead === record.selected_lead;
    const values = selected
      ? state.filtered
        ? wave.filtered
        : wave.raw
      : wave.leads?.[state.lead];
    return { times: wave.time || [], values: values || [] };
  }
  function niceStep(range, count = 5) {
    const raw = range / count;
    const power = 10 ** Math.floor(Math.log10(raw || 1));
    const scale = raw / power;
    return (scale <= 1 ? 1 : scale <= 2 ? 2 : scale <= 5 ? 5 : 10) * power;
  }
  let ecgPlot = null;
  function drawECG() {
    if (state.view === "multi") {
      drawMultilead();
      return;
    }
    const setup = canvasSetup("ecg-canvas");
    const data = waveData();
    if (!setup || !data || !state.record) return;
    const { ctx, width, height } = setup;
    const { times, values } = data;
    const left = 49,
      right = width - 17,
      top = 26,
      bottom = height - 32;
    const end = Math.min(state.record.duration, state.start + state.window);
    const span = Math.max(0.01, end - state.start);
    const first = Math.max(0, lowerBound(times, state.start) - 1);
    const last = Math.min(times.length, lowerBound(times, end) + 1);
    let min = Infinity,
      max = -Infinity;
    for (let i = first; i < last; i++) {
      if (Number.isFinite(values[i])) {
        min = Math.min(min, values[i]);
        max = Math.max(max, values[i]);
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      ctx.fillStyle = "#647b81";
      ctx.fillText("当前导联没有可显示的波形数据", 24, height / 2);
      ecgPlot = null;
      return;
    }
    const midpoint = (min + max) / 2;
    const half = Math.max(0.5, (max - min) * 0.65);
    min = midpoint - half;
    max = midpoint + half;
    const x = (t) => left + ((t - state.start) / span) * (right - left);
    const y = (v) => bottom - ((v - min) / (max - min)) * (bottom - top);
    ecgPlot = {
      left,
      right,
      top,
      bottom,
      x,
      y,
      min,
      max,
      end,
      span,
      first,
      last,
      times,
      values,
    };
    ctx.fillStyle = "#fffcfb";
    ctx.fillRect(left, top, right - left, bottom - top);
    const gridTime = 0.04;
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = "#f5e4e1";
    ctx.beginPath();
    for (
      let t = Math.ceil(state.start / gridTime) * gridTime;
      t < end + gridTime / 2;
      t += gridTime
    ) {
      const px = x(t);
      if (px > right) break;
      ctx.moveTo(px, top);
      ctx.lineTo(px, bottom);
    }
    for (let v = Math.ceil(min / 0.1) * 0.1; v < max; v += 0.1) {
      const py = y(v);
      ctx.moveTo(left, py);
      ctx.lineTo(right, py);
    }
    ctx.stroke();
    ctx.strokeStyle = "#ebccca";
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    for (
      let t = Math.ceil(state.start / 0.2) * 0.2;
      t < end + 0.001;
      t += 0.2
    ) {
      ctx.moveTo(x(t), top);
      ctx.lineTo(x(t), bottom);
    }
    for (let v = Math.ceil(min / 0.5) * 0.5; v < max; v += 0.5) {
      ctx.moveTo(left, y(v));
      ctx.lineTo(right, y(v));
    }
    ctx.stroke();
    ctx.fillStyle = "#7a8a8c";
    ctx.textAlign = "center";
    const tickTime = niceStep(span, width < 500 ? 4 : 8);
    for (
      let t = Math.ceil(state.start / tickTime) * tickTime;
      t <= end + 0.001;
      t += tickTime
    )
      ctx.fillText(number(t, tickTime < 1 ? 1 : 0), x(t), height - 11);
    ctx.textAlign = "right";
    const tickY = niceStep(max - min, 4);
    for (let v = Math.ceil(min / tickY) * tickY; v < max; v += tickY)
      ctx.fillText(number(v, tickY < 1 ? 1 : 0), left - 8, y(v) + 4);
    ctx.textAlign = "left";
    ctx.fillText("mV", 12, 15);
    ctx.textAlign = "right";
    ctx.fillText("s", width - 5, height - 11);
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, right - left, bottom - top);
    ctx.clip();
    ctx.beginPath();
    let begun = false;
    for (let i = first; i < last; i++) {
      if (!Number.isFinite(values[i])) {
        begun = false;
        continue;
      }
      if (!begun) {
        ctx.moveTo(x(times[i]), y(values[i]));
        begun = true;
      } else ctx.lineTo(x(times[i]), y(values[i]));
    }
    ctx.strokeStyle = "#087e81";
    ctx.lineWidth = 1.6;
    ctx.lineJoin = "round";
    ctx.stroke();
    if ($("peak-toggle").checked && state.lead === state.record.selected_lead) {
      for (const peak of state.record.peaks || []) {
        if (peak.time < state.start || peak.time > end) continue;
        const index = Math.min(times.length - 1, lowerBound(times, peak.time));
        let value = values[index];
        if (index > 0 && times[index] > peak.time) {
          const delta = times[index] - times[index - 1];
          if (delta > 0)
            value =
              values[index - 1] +
              ((values[index] - values[index - 1]) *
                (peak.time - times[index - 1])) /
                delta;
        }
        if (!Number.isFinite(value)) continue;
        const px = x(peak.time),
          py = y(value);
        const mark = (state.record.beat_reviews || []).find(
          (review) => review.lead === state.lead && review.index === peak.index,
        );
        ctx.fillStyle = mark
          ? { confirmed: "#087f83", artifact: "#ac4e4e", uncertain: "#b37837" }[
              mark.status
            ] || "#c77a32"
          : "#c77a32";
        ctx.beginPath();
        ctx.arc(px, py, 3.3, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = "bold 9px system-ui";
        ctx.textAlign = "center";
        ctx.fillText(
          mark ? beatLabel(mark.status) : "R",
          px,
          Math.max(top + 10, py - 9),
        );
      }
    }
    drawPredictionOverlay(ctx, x, y, times, values, top, bottom, end);
    if (
      state.hover != null &&
      state.hover >= state.start &&
      state.hover <= end
    ) {
      const px = x(state.hover);
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "#718c92";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, top);
      ctx.lineTo(px, bottom);
      ctx.stroke();
    }
    ctx.restore();
  }
  function drawRR() {
    const setup = canvasSetup("rr-canvas");
    if (!setup || !state.record) return;
    const { ctx, width, height } = setup;
    const rr = (state.record.rr || []).filter(
      (item) => Number.isFinite(item.time) && Number.isFinite(item.rr_ms),
    );
    const left = 49,
      right = width - 20,
      top = 22,
      bottom = height - 29;
    if (!rr.length) {
      ctx.fillStyle = "#829195";
      ctx.textAlign = "center";
      ctx.fillText(
        "没有足够或可靠的 R 峰，无法绘制 RR 趋势",
        width / 2,
        height / 2,
      );
      return;
    }
    let min = Infinity,
      max = -Infinity;
    for (const item of rr) {
      min = Math.min(min, item.rr_ms);
      max = Math.max(max, item.rr_ms);
    }
    const padding = Math.max(60, (max - min) * 0.25);
    min = Math.max(0, min - padding);
    max += padding;
    const end = Math.max(1, state.record.duration);
    const x = (t) => left + (t / end) * (right - left),
      y = (value) => bottom - ((value - min) / (max - min)) * (bottom - top);
    ctx.font = "10px system-ui";
    ctx.strokeStyle = "#edf1f0";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#819091";
    ctx.textAlign = "right";
    const ys = niceStep(max - min, 4);
    for (let value = Math.ceil(min / ys) * ys; value <= max; value += ys) {
      const py = y(value);
      ctx.beginPath();
      ctx.moveTo(left, py);
      ctx.lineTo(right, py);
      ctx.stroke();
      ctx.fillText(number(value), left - 8, py + 3);
    }
    ctx.textAlign = "center";
    const xs = niceStep(end, width < 500 ? 4 : 7);
    for (let t = 0; t <= end; t += xs)
      ctx.fillText(number(t), x(t), height - 10);
    ctx.textAlign = "left";
    ctx.fillText("ms", 12, 13);
    ctx.textAlign = "right";
    ctx.fillText("s", width - 5, height - 10);
    ctx.beginPath();
    rr.forEach((item, index) => {
      if (!index) ctx.moveTo(x(item.time), y(item.rr_ms));
      else ctx.lineTo(x(item.time), y(item.rr_ms));
    });
    ctx.strokeStyle = "#318f94";
    ctx.lineWidth = 1.4;
    ctx.stroke();
    for (const item of rr) {
      ctx.fillStyle = item.flag ? "#ce8a41" : "#318f94";
      ctx.beginPath();
      ctx.arc(x(item.time), y(item.rr_ms), item.flag ? 3.5 : 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  function drawAll() {
    if (state.page === "workspace" && state.record) {
      drawECG();
      drawRR();
      drawTools();
    }
    if (state.page === "comparison") drawComparison();
    if ($("segment-dialog").open && state.segmentPreview) drawSegmentPreview();
  }
  function moveWindow(delta) {
    if (!state.record) return;
    state.start += delta;
    state.hover = null;
    $("ecg-tooltip").hidden = true;
    updateWaveControls();
    drawECG();
  }
  const reviewLabel = (status) =>
    ({ unreviewed: "未复核", in_progress: "复核中", reviewed: "已复核" })[
      status
    ] || "未复核";
  const beatLabel = (status) =>
    ({ confirmed: "确认", artifact: "伪差", uncertain: "存疑" })[status] ||
    "未标注";
  function updateOriginalControls() {
    const available = state.api2 && state.record?.has_original === true;
    for (const id of ["reanalyze-lead", "segment-open", "signal-export"]) {
      $(id).disabled = state.busy || !available;
      $(id).title = available
        ? "使用保存的完整原始样本"
        : state.api2
          ? "原始样本不可用，请重新导入原文件"
          : "需要工作站 2.0 服务";
    }
    $("review-status").disabled = state.busy || !state.record || !state.api2;
    $("original-notice").hidden = !state.record || available;
    text(
      "original-notice",
      state.api2
        ? "无完整原始样本：不能重新分析、选段或导出原始 CSV。请重新导入原文件。"
        : "当前服务不支持新增工具，请启动工作站 2.0。",
    );
  }
  function buildMultileadButtons() {
    $("multilead-buttons").replaceChildren();
    for (const lead of state.record?.lead_names || []) {
      const button = el(
        "button",
        "button secondary small",
        `${lead}${lead === state.record.selected_lead ? " · 分析" : ""}`,
      );
      button.type = "button";
      button.dataset.lead = lead;
      button.addEventListener("click", () => {
        state.lead = lead;
        $("lead-select").value = lead;
        state.hover = null;
        $("ecg-tooltip").hidden = true;
        updateWaveControls();
        drawECG();
      });
      $("multilead-buttons").append(button);
    }
    $("multilead-canvas").style.height =
      `${Math.max(1, state.record?.lead_names?.length || 0) * 112}px`;
  }
  function emptyChart(id, message) {
    const setup = canvasSetup(id);
    if (!setup) return;
    setup.ctx.fillStyle = "#677a70";
    setup.ctx.textAlign = "center";
    setup.ctx.fillText(message, setup.width / 2, setup.height / 2);
  }
  function limits(arrays, start = 0, end = Infinity) {
    let min = Infinity,
      max = -Infinity;
    for (const values of arrays)
      for (let i = start; i < Math.min(values.length, end); i++) {
        const value = values[i];
        if (Number.isFinite(value)) {
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      }
    return [min, max];
  }
  function chartFrame(setup, xMin, xMax, yMin, yMax, options = {}) {
    const { ctx, width, height } = setup;
    const left = 52,
      top = options.top ?? 22,
      bottom = options.bottom ?? height - 32;
    const right = options.square
      ? Math.min(width - 20, left + bottom - top)
      : width - 20;
    if (xMax <= xMin) xMax = xMin + 1;
    if (yMax <= yMin) {
      yMin -= 0.5;
      yMax += 0.5;
    }
    const x = (v) => left + ((v - xMin) / (xMax - xMin)) * (right - left);
    const y = (v) => bottom - ((v - yMin) / (yMax - yMin)) * (bottom - top);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#e5eae7";
    ctx.fillStyle = "#677a70";
    ctx.font = "10px system-ui";
    ctx.textAlign = "right";
    const ys = niceStep(yMax - yMin, 4);
    for (
      let value = Math.ceil(yMin / ys) * ys;
      value <= yMax + ys / 100;
      value += ys
    ) {
      ctx.beginPath();
      ctx.moveTo(left, y(value));
      ctx.lineTo(right, y(value));
      ctx.stroke();
      ctx.fillText(
        number(value, ys < 1 ? Math.min(3, Math.ceil(-Math.log10(ys))) : 0),
        left - 6,
        y(value) + 3,
      );
    }
    ctx.textAlign = "center";
    const xs = niceStep(xMax - xMin, width < 500 ? 4 : 7);
    for (
      let value = Math.ceil(xMin / xs) * xs;
      value <= xMax + xs / 100;
      value += xs
    ) {
      ctx.beginPath();
      ctx.moveTo(x(value), top);
      ctx.lineTo(x(value), bottom);
      ctx.stroke();
      ctx.fillText(number(value, xs < 1 ? 1 : 0), x(value), bottom + 18);
    }
    ctx.textAlign = "left";
    ctx.fillText(options.yLabel || "mV", 8, top - 8);
    ctx.textAlign = "right";
    ctx.fillText(options.xLabel || "s", right + 14, bottom + 18);
    return { ctx, left, right, top, bottom, x, y };
  }
  function trace(
    plot,
    xs,
    values,
    color,
    lineWidth = 1.4,
    first = 0,
    last = xs.length,
  ) {
    const { ctx, x, y } = plot;
    ctx.save();
    ctx.beginPath();
    ctx.rect(
      plot.left,
      plot.top,
      plot.right - plot.left,
      plot.bottom - plot.top,
    );
    ctx.clip();
    ctx.beginPath();
    let begun = false;
    for (let i = first; i < last; i++) {
      if (!Number.isFinite(xs[i]) || !Number.isFinite(values[i])) {
        begun = false;
        continue;
      }
      if (begun) ctx.lineTo(x(xs[i]), y(values[i]));
      else {
        ctx.moveTo(x(xs[i]), y(values[i]));
        begun = true;
      }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    ctx.restore();
  }
  function drawRecordWave(id, record, start, span, filtered) {
    const setup = canvasSetup(id);
    if (!setup || !record) return;
    const xs = record.waveform?.time || [],
      values =
        (filtered ? record.waveform?.filtered : record.waveform?.raw) || [];
    const end = start + span,
      first = Math.max(0, lowerBound(xs, start) - 1),
      last = Math.min(xs.length, lowerBound(xs, end) + 1);
    const [min, max] = limits([values], first, last);
    if (!Number.isFinite(min)) {
      emptyChart(id, "此时间窗口无波形数据");
      return;
    }
    const pad = Math.max(0.15, (max - min) * 0.15);
    trace(
      chartFrame(setup, start, end, min - pad, max + pad),
      xs,
      values,
      "#087f83",
      1.4,
      first,
      last,
    );
  }
  function drawMultilead() {
    const setup = canvasSetup("multilead-canvas");
    if (!setup || !state.record) return;
    const xs = state.record.waveform?.time || [],
      end = Math.min(state.record.duration, state.start + state.window);
    const first = Math.max(0, lowerBound(xs, state.start) - 1),
      last = Math.min(xs.length, lowerBound(xs, end) + 1);
    (state.record.lead_names || []).forEach((lead, index) => {
      const values =
        (lead === state.record.selected_lead
          ? state.record.waveform?.raw
          : state.record.waveform?.leads?.[lead]) || [];
      const top = index * 112 + 24,
        bottom = index * 112 + 85;
      const [min, max] = limits([values], first, last);
      setup.ctx.textAlign = "left";
      setup.ctx.fillStyle = lead === state.lead ? "#087f83" : "#203b42";
      setup.ctx.fillText(
        `${lead}${lead === state.record.selected_lead ? " · 分析导联" : ""} · 原始`,
        55,
        top - 9,
      );
      if (!Number.isFinite(min)) {
        setup.ctx.fillText("无显示样本", 55, top + 25);
        return;
      }
      const pad = Math.max(0.1, (max - min) * 0.15);
      trace(
        chartFrame(setup, state.start, end, min - pad, max + pad, {
          top,
          bottom,
        }),
        xs,
        values,
        lead === state.lead ? "#087f83" : "#6a9090",
        1.2,
        first,
        last,
      );
    });
  }
  function selectTool(tool) {
    state.tool = tool;
    for (const name of ["rr", "template", "spectrum"]) {
      const active = name === tool,
        tab = $(`tool-tab-${name}`);
      tab.classList.toggle("selected", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      $(`tool-panel-${name}`).hidden = !active;
    }
    requestAnimationFrame(drawAll);
  }
  function templateSummary(template) {
    return template?.median?.length
      ? `实际心搏 ${number(template.used_count)} / 可用 ${number(template.available_count)} · 显示 ${number(template.traces?.length || 0)} 条 · 去局部基线，未归一化`
      : template?.reason || "此记录无模板工具；有原始样本时可重新分析。";
  }
  function drawTemplate(id, template) {
    if (!template?.time_ms?.length || !template.median?.length) {
      emptyChart(id, template?.reason || "无模板数据");
      return;
    }
    const setup = canvasSetup(id);
    if (!setup) return;
    const xs = template.time_ms,
      [min, max] = limits([
        template.median,
        template.p25 || [],
        template.p75 || [],
        ...(template.traces || []),
      ]);
    if (!Number.isFinite(min)) {
      emptyChart(id, "无有效模板数据");
      return;
    }
    const pad = Math.max(0.1, (max - min) * 0.1),
      plot = chartFrame(setup, xs[0], xs[xs.length - 1], min - pad, max + pad, {
        xLabel: "ms",
      });
    const { ctx, x, y } = plot;
    if (
      template.p25?.length === xs.length &&
      template.p75?.length === xs.length
    ) {
      ctx.beginPath();
      xs.forEach((t, i) =>
        i
          ? ctx.lineTo(x(t), y(template.p25[i]))
          : ctx.moveTo(x(t), y(template.p25[i])),
      );
      for (let i = xs.length - 1; i >= 0; i--)
        ctx.lineTo(x(xs[i]), y(template.p75[i]));
      ctx.closePath();
      ctx.fillStyle = "#087f8324";
      ctx.fill();
    }
    for (const values of template.traces || [])
      trace(plot, xs, values, "#799d9d55", 1);
    trace(plot, xs, template.median, "#087f83", 2.5);
  }
  function drawHistogram(histogram) {
    const id = "histogram-canvas",
      xs = histogram?.edges_ms || [],
      counts = histogram?.counts || [];
    if (!counts.length || xs.length !== counts.length + 1) {
      emptyChart(id, "无可靠 RR 分布");
      return;
    }
    const setup = canvasSetup(id);
    if (!setup) return;
    const [, max] = limits([counts]),
      plot = chartFrame(setup, xs[0], xs[xs.length - 1], 0, Math.max(1, max), {
        yLabel: "个",
        xLabel: "ms",
      });
    plot.ctx.fillStyle = "#087f8380";
    counts.forEach((count, i) =>
      plot.ctx.fillRect(
        plot.x(xs[i]) + 1,
        plot.y(count),
        Math.max(1, plot.x(xs[i + 1]) - plot.x(xs[i]) - 2),
        plot.bottom - plot.y(count),
      ),
    );
  }
  function drawPoincare(points) {
    const id = "poincare-canvas",
      xs = points?.x_ms || [],
      ys = points?.y_ms || [];
    if (!xs.length || xs.length !== ys.length) {
      emptyChart(id, "无可靠相邻 RR 对");
      return;
    }
    const setup = canvasSetup(id);
    if (!setup) return;
    const [min, max] = limits([xs, ys]),
      pad = Math.max(30, (max - min) * 0.15);
    const plot = chartFrame(setup, min - pad, max + pad, min - pad, max + pad, {
      xLabel: "ms",
      yLabel: "ms",
      square: true,
    });
    plot.ctx.strokeStyle = "#9ab7b7";
    plot.ctx.setLineDash([3, 3]);
    plot.ctx.beginPath();
    plot.ctx.moveTo(plot.x(min - pad), plot.y(min - pad));
    plot.ctx.lineTo(plot.x(max + pad), plot.y(max + pad));
    plot.ctx.stroke();
    plot.ctx.setLineDash([]);
    plot.ctx.fillStyle = "#087f8380";
    xs.forEach((value, i) => {
      plot.ctx.beginPath();
      plot.ctx.arc(plot.x(value), plot.y(ys[i]), 2.5, 0, Math.PI * 2);
      plot.ctx.fill();
    });
  }
  function drawSpectrum(spectrum) {
    const id = "spectrum-canvas",
      xs = spectrum?.frequency_hz || [];
    if (!xs.length) {
      emptyChart(id, "此记录无频谱工具；可使用原始样本重新分析");
      return;
    }
    const setup = canvasSetup(id);
    if (!setup) return;
    const raw = (spectrum.raw_power || []).map((v) =>
        v > 0 ? Math.log10(v) : NaN,
      ),
      filtered = (spectrum.filtered_power || []).map((v) =>
        v > 0 ? Math.log10(v) : NaN,
      );
    const [min, max] = limits([raw, filtered]);
    if (!Number.isFinite(min)) {
      emptyChart(id, "功率为零或无有效谱数据");
      return;
    }
    const plot = chartFrame(
      setup,
      xs[0],
      xs[xs.length - 1],
      min - 0.3,
      max + 0.3,
      { xLabel: "Hz", yLabel: "log₁₀ PSD" },
    );
    trace(plot, xs, raw, "#a98556", 1.3);
    trace(plot, xs, filtered, "#087f83", 1.7);
  }
  function drawTools() {
    const tools = state.record?.tools;
    if (state.tool === "rr") {
      if (tools) {
        drawHistogram(tools.rr_histogram);
        drawPoincare(tools.poincare);
      } else {
        emptyChart("histogram-canvas", "旧记录无 RR 分布工具");
        emptyChart("poincare-canvas", "旧记录无 Poincaré 工具");
      }
    }
    if (state.tool === "template") {
      text("template-summary", templateSummary(tools?.template));
      drawTemplate("template-canvas", tools?.template);
    }
    if (state.tool === "spectrum") drawSpectrum(tools?.spectrum);
  }
  function applyRecordUpdate(record) {
    const previous = {
      id: state.record?.id,
      analysis: state.record?.selected_lead,
      lead: state.lead,
      start: state.start,
      filtered: state.filtered,
      page: state.beatsPage,
      tool: state.tool,
      view: state.view,
    };
    showRecord(record);
    if (
      previous.id === record.id &&
      previous.analysis === record.selected_lead
    ) {
      state.lead = record.lead_names?.includes(previous.lead)
        ? previous.lead
        : record.selected_lead;
      state.start = previous.start;
      state.filtered = previous.filtered;
      state.beatsPage = previous.page;
      state.view = previous.view;
      $("lead-select").value = state.lead;
      selectTool(previous.tool);
      updateWaveControls();
      renderBeats();
    }
  }
  async function mutateRecord(path, options, message) {
    if (state.busy || !state.record) return;
    busy(true, "正在保存并读取分析结果…");
    try {
      const record = await api(path, options);
      applyRecordUpdate(record);
      await syncSavedRecord(record, message);
    } catch (error) {
      notify(error.message);
      $("review-status").value = state.record.review_status || "unreviewed";
    } finally {
      busy(false);
    }
  }
  function openBeatReview(beat, review) {
    if (state.busy || !state.record || !Number.isInteger(beat.peak_index))
      return;
    state.reviewing = {
      recordId: state.record.id,
      lead: state.record.selected_lead,
      index: beat.peak_index,
      review,
    };
    text(
      "beat-review-context",
      `${state.record.selected_lead} · ${number(beat.time, 3)} s · 原始峰索引 ${beat.peak_index}`,
    );
    $("beat-review-status").value = review?.status || "uncertain";
    $("beat-review-note").value = review?.note || "";
    $("beat-review-clear").disabled = !review;
    formError("beat-review-error");
    openDialog("beat-review-dialog");
  }
  async function saveBeatReview(clear = false) {
    const target = state.reviewing;
    if (
      state.busy ||
      !target ||
      target.recordId !== state.record?.id ||
      target.lead !== state.record.selected_lead
    )
      return;
    busy(true, clear ? "正在清除人工标注…" : "正在保存人工标注…");
    formError("beat-review-error");
    try {
      const record = await api(
        `${endpoint(target.recordId)}/beats/${target.index}`,
        clear
          ? { method: "DELETE" }
          : {
              ...json({
                status: $("beat-review-status").value,
                note: $("beat-review-note").value,
              }),
              method: "PUT",
            },
      );
      $("beat-review-dialog").close();
      applyRecordUpdate(record);
      await syncSavedRecord(
        record,
        clear
          ? "人工标注已清除，算法指标未改变。"
          : "人工标注已保存，算法指标未改变。",
      );
    } catch (error) {
      formError("beat-review-error", error.message);
    } finally {
      busy(false);
    }
  }
  function segmentInput() {
    return {
      start: Number($("segment-start").value),
      end: Number($("segment-end").value),
      lead: $("segment-lead").value,
    };
  }
  function segmentKey(input) {
    return JSON.stringify([input.start, input.end, input.lead]);
  }
  function invalidateSegment() {
    state.segmentPreview = null;
    $("segment-result").hidden = true;
    $("segment-save").disabled = true;
  }
  function openSegment() {
    if (state.busy || !state.record?.has_original || !state.api2) return;
    state.segmentParent = state.record.id;
    invalidateSegment();
    $("segment-start").value = String(state.start);
    $("segment-end").value = String(
      Math.min(state.record.duration, state.start + Math.max(5, state.window)),
    );
    $("segment-start").max = String(state.record.duration - 5);
    $("segment-end").max = String(state.record.duration);
    $("segment-name").value = `${state.record.name || "心电记录"} · 选段`.slice(
      0,
      80,
    );
    $("segment-lead").replaceChildren();
    for (const lead of state.record.lead_names || []) {
      const option = el("option", "", lead);
      option.value = lead;
      $("segment-lead").append(option);
    }
    $("segment-lead").value = state.lead;
    text(
      "segment-source",
      `${state.record.name} · 完整原始样本 ${number(state.record.sample_rate)} Hz · ${duration(state.record.duration)}`,
    );
    formError("segment-error");
    openDialog("segment-dialog");
  }
  async function analyzeSegment(save) {
    if (
      state.busy ||
      !state.record ||
      state.segmentParent !== state.record.id ||
      !$("segment-form").reportValidity()
    )
      return;
    const input = segmentInput(),
      parent = state.segmentParent;
    if (
      !Number.isFinite(input.start) ||
      !Number.isFinite(input.end) ||
      input.start < 0 ||
      input.end > state.record.duration ||
      input.end - input.start < 5
    ) {
      formError("segment-error", "选段必须在当前记录内，且至少 5 秒。");
      return;
    }
    if (
      save &&
      (!state.segmentPreview || state.segmentPreview.key !== segmentKey(input))
    ) {
      invalidateSegment();
      formError("segment-error", "范围或导联已改变，请先重新预览。");
      return;
    }
    formError("segment-error");
    busy(true, save ? "正在保存独立原始选段…" : "正在分析完整分辨率选段…");
    try {
      const record = await api(
        `${endpoint(parent)}/segment`,
        json({ ...input, save, name: $("segment-name").value.trim() }),
      );
      if (save) {
        $("segment-dialog").close();
        showRecord(record);
        navigate("workspace");
        await syncSavedRecord(record, "选段已保存为独立记录，原记录未改变。");
      } else {
        state.segmentPreview = { record, key: segmentKey(input) };
        $("segment-result").hidden = false;
        const segment = record.segment || input;
        text(
          "segment-range",
          `父记录范围 ${number(segment.start, 4)}–${number(segment.end, 4)} s · 来源偏移 ${number(record.source_offset || 0, 4)} s · 预览横轴从 0 开始`,
        );
        $("segment-metrics").replaceChildren();
        for (const [label, value] of [
          ["心率", `${number(record.metrics?.heart_rate, 1)} bpm`],
          ["R 峰", number(record.metrics?.beat_count)],
          ["平均 RR", `${number(record.metrics?.mean_rr, 1)} ms`],
          ["质量", `${number(record.quality?.score)} /100`],
        ]) {
          const item = el("div");
          item.append(el("small", "", label), el("strong", "", value));
          $("segment-metrics").append(item);
        }
        text(
          "segment-template-summary",
          templateSummary(record.tools?.template),
        );
        requestAnimationFrame(drawSegmentPreview);
      }
    } catch (error) {
      formError("segment-error", error.message);
    } finally {
      busy(false);
    }
  }
  function drawSegmentPreview() {
    const record = state.segmentPreview?.record;
    if (!record) return;
    drawRecordWave("segment-canvas", record, 0, record.duration, true);
    drawTemplate("segment-template-canvas", record.tools?.template);
  }
  function clearComparison() {
    state.comparison = null;
    state.compareStart = 0;
    $("comparison-result").hidden = true;
    $("comparison-empty").hidden = false;
    text(
      "comparison-empty",
      state.records.length < 2
        ? "至少需要两条本地记录；可载入示例或导入数据。"
        : "选择两条不同记录后开始对比。",
    );
    formError("compare-error");
  }
  function refreshCompareSelectors() {
    const left = $("compare-left").value,
      right = $("compare-right").value;
    for (const id of ["compare-left", "compare-right"]) {
      $(id).replaceChildren();
      for (const record of state.records) {
        const option = el(
          "option",
          "",
          `${record.name || "未命名"} · ${record.selected_lead} · ${duration(record.duration)}`,
        );
        option.value = record.id;
        $(id).append(option);
      }
    }
    const has = (id) =>
      state.records.some((record) => String(record.id) === id);
    $("compare-left").value = has(left)
      ? left
      : String(state.records[0]?.id || "");
    $("compare-right").value =
      has(right) && right !== $("compare-left").value
        ? right
        : String(
            state.records.find(
              (record) => String(record.id) !== $("compare-left").value,
            )?.id || "",
          );
    $("compare-run").disabled = state.records.length < 2 || !state.api2;
    clearComparison();
  }
  async function runComparison() {
    if (state.busy) return;
    const left = $("compare-left").value,
      right = $("compare-right").value;
    if (!left || !right || left === right) {
      formError("compare-error", "请选择两条不同的本地记录。");
      return;
    }
    clearComparison();
    formError("compare-error");
    busy(true, "正在读取两条记录与实际指标…");
    try {
      const result = await api(
        `/api/compare?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`,
      );
      state.comparison = result;
      state.compareStart = 0;
      $("comparison-empty").hidden = true;
      $("comparison-result").hidden = false;
      $("comparison-lead-warning").hidden = result.same_lead;
      for (const side of ["left", "right"]) {
        const record = result[side];
        text(`compare-${side}-name`, record.name);
        text(
          `compare-${side}-context`,
          `${record.selected_lead} · ${duration(record.duration)} · 质量 ${number(record.quality?.score)} · ${reviewLabel(record.review_status)} · ${record.source || record.filename || "本地记录"}${record.segment ? ` · 选段 ${number(record.segment.start, 3)}–${number(record.segment.end, 3)} s` : ""}`,
        );
      }
      $("compare-metrics").replaceChildren();
      for (const item of result.differences || []) {
        const row = el("tr");
        row.dataset.metric = item.key;
        row.append(
          el("td", "", `${item.label}${item.unit ? ` (${item.unit})` : ""}`),
          el("td", "", number(item.left, 2)),
          el("td", "", number(item.right, 2)),
          el(
            "td",
            "",
            `${Number.isFinite(item.delta) && item.delta > 0 ? "+" : ""}${number(item.delta, 2)}`,
          ),
        );
        $("compare-metrics").append(row);
      }
      requestAnimationFrame(drawComparison);
    } catch (error) {
      formError("compare-error", error.message);
    } finally {
      busy(false);
    }
  }
  function drawComparison() {
    const result = state.comparison;
    if (!result) return;
    const span = Number($("compare-window").value),
      max = Math.max(
        0,
        Math.max(result.left.duration, result.right.duration) - span,
      );
    state.compareStart = Math.max(0, Math.min(max, state.compareStart));
    $("compare-slider").max = String(max);
    $("compare-slider").value = String(state.compareStart);
    $("compare-slider").disabled = max === 0;
    text(
      "compare-range",
      `${number(state.compareStart, 1)}–${number(state.compareStart + span, 1)} s · 独立幅值缩放`,
    );
    for (const side of ["left", "right"])
      drawRecordWave(
        `compare-${side}-canvas`,
        result[side],
        state.compareStart,
        span,
        $("compare-signal").value === "filtered",
      );
  }
  async function downloadOriginal() {
    if (state.busy || !state.record?.has_original || !state.api2) return;
    busy(true, "正在导出完整原始样本…");
    try {
      const response = await fetch(`${endpoint(state.record.id)}/signal.csv`, {
        cache: "no-store",
      });
      if (!response.ok) {
        const body = await response.json();
        throw new Error(body.detail || `导出失败（HTTP ${response.status}）`);
      }
      const url = URL.createObjectURL(await response.blob()),
        link = el("a");
      link.href = url;
      link.download = `${state.record.name || "signal"}.csv`;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      notify("已导出完整原始 CSV（所有导联，mV）。", "success");
    } catch (error) {
      notify(error.message || "无法下载原始样本。");
    } finally {
      busy(false);
    }
  }
  const modelEndpoint = (id) => `/api/models/${encodeURIComponent(id)}`;
  const taskLabel = (task) =>
    ({
      beat_classification: "心搏单标签",
      record_classification: "窗口单标签",
      record_multilabel: "窗口多标签",
    })[task] || task;
  const shapeLabel = (shape) =>
    Array.isArray(shape)
      ? `[${shape.map((dim) => (dim == null ? "B" : String(dim))).join(", ")}]`
      : "—";
  function modelAction(label, action, disabled = false) {
    const button = el("button", "button secondary small", label);
    button.type = "button";
    button.disabled = disabled;
    button.addEventListener("click", action);
    return button;
  }
  async function loadModels() {
    const token = ++state.registryRequest;
    formError("models-error");
    text("models-status", "正在读取模型…");
    try {
      const result = await api("/api/models");
      if (token !== state.registryRequest) return;
      state.models = result.models;
      state.defaultModelId = result.default_model_id;
      renderModels();
    } catch (error) {
      if (token !== state.registryRequest) return;
      text("models-status", "模型列表读取失败，请刷新重试。");
      throw error;
    }
  }
  function renderModels() {
    $("model-list").replaceChildren();
    $("models-empty").hidden = state.models.length !== 0;
    text(
      "models-status",
      state.models.length
        ? `${state.models.length} 个模型 · ${state.defaultModelId ? "已指定默认模型" : "未指定默认模型"}`
        : "未安装任何预测模型",
    );
    for (const model of state.models) {
      const manifest = model,
        signature = model.signature || {};
      const card = el("article", "card model-card");
      card.dataset.modelId = model.id;
      const header = el("div", "card-heading"),
        heading = el("div");
      heading.append(
        el("h2", "", model.name),
        el("p", "", `版本 ${model.version} · ${taskLabel(manifest.task)}`),
      );
      header.append(heading);
      if (model.is_default) header.append(el("span", "pill", "默认模型"));
      card.append(header);
      const facts = el("div", "model-facts");
      facts.append(
        el(
          "p",
          "",
          `类别（输出顺序）：${(manifest.classes || []).join(" / ")}`,
        ),
        el(
          "p",
          "",
          `${number(manifest.sample_rate)} Hz · ${number(manifest.input_samples)} 样本 · ${(manifest.lead_names || []).join(" / ")}`,
        ),
        el(
          "p",
          "",
          `输入 ${shapeLabel(signature.input_shape)} → 输出 ${shapeLabel(signature.output_shape)} · CPU`,
        ),
        el(
          "p",
          "",
          `${number(model.size_bytes / 1024 / 1024, 2)} MB · 导入 ${date(model.imported_at)}`,
        ),
      );
      if (manifest.description || model.description)
        facts.append(
          el(
            "p",
            "model-description",
            model.description || manifest.description,
          ),
        );
      card.append(facts);
      const actions = el("div", "model-actions");
      const details = modelAction("详情 / 编辑", () =>
        openModelDetails(model.id),
      );
      details.dataset.modelAction = "details";
      const setDefault = modelAction(
        "设为默认",
        () => setDefaultModel(model.id),
        model.is_default,
      );
      setDefault.dataset.modelAction = "default";
      const remove = modelAction("删除", () => {
        if (state.busy) return;
        state.deletingModel = model;
        text(
          "model-delete-context",
          `确认删除「${model.name}」版本 ${model.version}？`,
        );
        formError("model-delete-error");
        openDialog("model-delete-dialog");
      });
      remove.dataset.modelAction = "delete";
      const download = el("a", "button secondary small", "配置 JSON");
      download.href = `${modelEndpoint(model.id)}/manifest`;
      download.download = "";
      actions.append(details, setDefault, download, remove);
      card.append(actions);
      $("model-list").append(card);
    }
  }
  async function refreshAfterModelMutation() {
    await loadModels();
    if (state.record)
      await loadRecordModels(
        !state.recordModels.length && !state.selectedModelId,
      );
  }
  async function setDefaultModel(id) {
    if (state.busy) return;
    busy(true, "正在设置默认模型…");
    try {
      await api(`${modelEndpoint(id)}/default`, json({}));
      await refreshAfterModelMutation();
      notify("默认模型已保存；当前手动选择保持不变。", "success");
    } catch (error) {
      notify(error.message);
    } finally {
      busy(false);
    }
  }
  async function openModelDetails(id) {
    if (state.busy) return;
    busy(true, "正在读取模型配置…");
    try {
      const model = await api(modelEndpoint(id));
      state.editingModel = model;
      $("model-edit-name").value = model.name;
      $("model-edit-description").value = model.description || "";
      text(
        "model-details-context",
        `版本 ${model.version} · ${taskLabel(model.task)} · 导入 ${date(model.imported_at)}`,
      );
      text(
        "model-details-config",
        JSON.stringify(
          {
            manifest: Object.fromEntries(
              manifestFields
                .filter((key) => key in model)
                .map((key) => [key, model[key]]),
            ),
            signature: model.signature,
            sha256: model.sha256,
            filename: model.filename,
          },
          null,
          2,
        ),
      );
      $("model-manifest-download").href = `${modelEndpoint(id)}/manifest`;
      formError("model-edit-error");
      busy(false);
      openDialog("model-details-dialog");
    } catch (error) {
      notify(error.message);
    } finally {
      busy(false);
    }
  }
  function recordModelKey(record) {
    return JSON.stringify([
      record.id,
      record.selected_lead,
      record.notch,
      record.sample_rate,
      record.duration,
      record.has_original,
      record.quality?.reliable,
      (record.peaks || []).map((peak) => peak.index),
    ]);
  }
  function resetModelRecord() {
    ++state.modelRequest;
    ++state.predictionRequest;
    state.modelRecordId = null;
    state.modelContextKey = "";
    state.recordModels = [];
    state.selectedModelId = "";
    state.prediction = null;
    state.modelLoading = false;
    state.predictionLoading = false;
    renderModelSelector();
    renderPrediction();
  }
  function refreshModelRecord() {
    if (!state.record) return;
    const changedRecord = state.modelRecordId !== state.record.id;
    const key = recordModelKey(state.record);
    if (!changedRecord && key === state.modelContextKey) {
      updateModelControls();
      renderPrediction();
      return;
    }
    state.modelContextKey = key;
    state.modelRecordId = state.record.id;
    ++state.predictionRequest;
    state.prediction = null;
    state.predictionPage = 0;
    state.predictionLoading = false;
    if (changedRecord) {
      state.selectedModelId = "";
      state.recordModels = [];
      $("prediction-start").value = "0";
    }
    renderPrediction();
    loadRecordModels(changedRecord).catch((error) =>
      formError("prediction-error", error.message),
    );
  }
  async function loadRecordModels(selectDefault = false) {
    const recordId = state.record?.id;
    if (!recordId) return;
    const token = ++state.modelRequest;
    ++state.predictionRequest;
    state.predictionLoading = false;
    state.modelLoading = true;
    formError("prediction-error");
    updateModelControls();
    try {
      const result = await api(`${endpoint(recordId)}/models`);
      if (token !== state.modelRequest || state.record?.id !== recordId) return;
      state.recordModels = result.models;
      state.defaultModelId = result.default_model_id;
      if (selectDefault)
        state.selectedModelId = result.models.some(
          (model) => model.id === result.default_model_id,
        )
          ? result.default_model_id
          : "";
      else if (
        !result.models.some((model) => model.id === state.selectedModelId)
      )
        state.selectedModelId = "";
      state.modelLoading = false;
      renderModelSelector();
      await loadPrediction();
    } catch (error) {
      if (token !== state.modelRequest || state.record?.id !== recordId) return;
      state.modelLoading = false;
      state.recordModels = [];
      state.selectedModelId = "";
      state.prediction = null;
      renderModelSelector();
      renderPrediction();
      text("model-context", "模型兼容性读取失败，请在模型管理中刷新重试。");
      throw error;
    }
  }
  function selectedModel() {
    return state.recordModels.find(
      (model) => model.id === state.selectedModelId,
    );
  }
  function renderModelSelector() {
    $("model-select").replaceChildren();
    const placeholder = el(
      "option",
      "",
      state.recordModels.length
        ? "请选择模型（不自动选择替代模型）"
        : "未导入模型 · 前往模型管理",
    );
    placeholder.value = "";
    $("model-select").append(placeholder);
    for (const model of state.recordModels) {
      const option = el(
        "option",
        "",
        `${model.name} · ${model.version}${model.is_default ? " · 默认" : ""}${model.compatible ? "" : " · 不兼容"}`,
      );
      option.value = model.id;
      $("model-select").append(option);
    }
    $("model-select").value = state.selectedModelId;
    updateModelControls();
  }
  function updateModelControls() {
    const model = selectedModel(),
      manifest = model || {};
    const recordTask = model && manifest.task !== "beat_classification";
    $("model-select").disabled =
      state.busy ||
      state.modelLoading ||
      state.predictionLoading ||
      !state.recordModels.length;
    $("prediction-run").disabled =
      state.busy ||
      state.modelLoading ||
      state.predictionLoading ||
      !model?.compatible ||
      !state.record;
    $("prediction-rerun").disabled = $("prediction-run").disabled;
    $("prediction-start-field").hidden = !recordTask;
    $("prediction-start").disabled = state.busy || state.predictionLoading;
    if (recordTask)
      $("prediction-start").max = String(
        Math.max(
          0,
          state.record.duration - manifest.input_samples / manifest.sample_rate,
        ),
      );
    if (state.modelLoading)
      text("model-context", "正在读取当前记录的模型兼容性…");
    else if (!model)
      text(
        "model-context",
        state.recordModels.length
          ? "请选择模型。未指定默认模型时不会自动选择。"
          : "没有可用模型；在模型管理中导入 ONNX 和配置后再运行。",
      );
    else {
      const signature = model.signature || {};
      const window = recordTask
        ? `固定窗口 ${number(manifest.input_samples / manifest.sample_rate, 3)} s`
        : "完整记录心搏 · 边界窗口跳过";
      text(
        "model-context",
        `${taskLabel(manifest.task)} · ${window} · ${number(manifest.sample_rate)} Hz · 输入 ${shapeLabel(signature.input_shape)} · CPU${signature.runtime_version ? ` / ORT ${signature.runtime_version}` : ""}${model.compatible ? "" : ` · 不兼容：${model.reason}`}`,
      );
    }
    updatePredictionOverlay();
  }
  async function loadPrediction() {
    const recordId = state.record?.id,
      modelId = state.selectedModelId;
    if (!recordId) return;
    const token = ++state.predictionRequest;
    state.predictionLoading = true;
    state.prediction = null;
    state.predictionPage = 0;
    renderPrediction();
    updateModelControls();
    try {
      const result = await api(
        `${endpoint(recordId)}/predictions${modelId ? `?model_id=${encodeURIComponent(modelId)}` : ""}`,
      );
      if (
        token !== state.predictionRequest ||
        recordId !== state.record?.id ||
        modelId !== state.selectedModelId
      )
        return;
      state.prediction = result.predictions[0] || null;
      state.predictionLoading = false;
      renderPrediction();
      updateModelControls();
    } catch (error) {
      if (
        token !== state.predictionRequest ||
        recordId !== state.record?.id ||
        modelId !== state.selectedModelId
      )
        return;
      state.predictionLoading = false;
      renderPrediction();
      updateModelControls();
      formError("prediction-error", error.message);
    }
  }
  async function runPrediction() {
    const model = selectedModel(),
      recordId = state.record?.id;
    if (
      state.busy ||
      state.modelLoading ||
      state.predictionLoading ||
      !model?.compatible ||
      !recordId
    )
      return;
    const manifest = model,
      start =
        manifest.task === "beat_classification"
          ? 0
          : Number($("prediction-start").value);
    if (
      !Number.isFinite(start) ||
      start < 0 ||
      (manifest.task !== "beat_classification" &&
        (start + manifest.input_samples / manifest.sample_rate >
          state.record.duration + 1e-6 ||
          $("prediction-start").value.trim() === ""))
    ) {
      formError(
        "prediction-error",
        "请填写有效的非负起点，完整模型窗口必须位于记录内。",
      );
      return;
    }
    const modelId = model.id,
      token = ++state.predictionRequest;
    formError("prediction-error");
    busy(true, "正在使用 ONNX Runtime CPU 执行真实推理…");
    try {
      const prediction = await api(
        `${endpoint(recordId)}/predict`,
        json({ model_id: modelId, start }),
      );
      if (
        token !== state.predictionRequest ||
        state.record?.id !== recordId ||
        state.selectedModelId !== modelId
      )
        return;
      state.prediction = prediction;
      state.predictionPage = 0;
      renderPrediction();
      notify("模型推理完成，版本化结果已保存。", "success");
    } catch (error) {
      if (token === state.predictionRequest && state.record?.id === recordId)
        formError("prediction-error", error.message);
    } finally {
      busy(false);
    }
  }
  function predictionIsCurrent() {
    const result = state.prediction;
    return Boolean(
      result &&
        !state.modelLoading &&
        !state.predictionLoading &&
        !result.stale &&
        result.model_available &&
        result.record_id === state.record?.id &&
        result.model_id === state.selectedModelId &&
        result.record_context?.selected_lead === state.record.selected_lead,
    );
  }
  function updatePredictionOverlay() {
    const active =
      predictionIsCurrent() &&
      state.prediction.task === "beat_classification" &&
      state.view === "single" &&
      state.lead === state.record?.selected_lead;
    $("prediction-overlay").disabled = !active;
    $("prediction-overlay").title = active
      ? "紫色标签为模型类别，与 R 峰及人工标注独立"
      : "仅当前有效心搏结果、同一分析导联的单导联视图可叠加";
  }
  function drawPredictionOverlay(ctx, x, y, times, values, top, bottom, end) {
    if (
      !predictionIsCurrent() ||
      state.prediction.task !== "beat_classification" ||
      state.lead !== state.record.selected_lead ||
      !$("prediction-overlay").checked
    )
      return;
    ctx.save();
    ctx.font = "bold 10px system-ui";
    ctx.textAlign = "center";
    for (const prediction of state.prediction.predictions || []) {
      if (prediction.time < state.start || prediction.time > end) continue;
      const index = Math.min(
        times.length - 1,
        lowerBound(times, prediction.time),
      );
      if (!Number.isFinite(values[index])) continue;
      const px = x(prediction.time),
        py = Math.min(bottom - 8, Math.max(top + 28, y(values[index]) + 25));
      const label = String(prediction.class_label),
        width = ctx.measureText(label).width + 10;
      ctx.fillStyle = "#f1eaf9";
      ctx.fillRect(px - width / 2, py - 11, width, 16);
      ctx.fillStyle = "#76559b";
      ctx.fillText(label, px, py + 1);
    }
    ctx.restore();
  }
  function locatePrediction(prediction) {
    if (!state.record) return;
    const contextLead = state.prediction?.record_context?.selected_lead;
    if (state.record.lead_names.includes(contextLead)) state.lead = contextLead;
    state.view = "single";
    $("lead-select").value = state.lead;
    state.start = Math.max(0, prediction.time - state.window / 2);
    state.hover = prediction.time;
    updateWaveControls();
    drawECG();
    $("waveform-wrap").scrollIntoView({ behavior: "smooth", block: "center" });
  }
  function renderPrediction() {
    const result = state.prediction;
    $("prediction-content").hidden = !result;
    $("prediction-export").hidden = !result;
    $("prediction-rerun").hidden =
      !result?.stale ||
      !result.model_available ||
      result.model_id !== state.selectedModelId;
    if (state.record)
      $("report-link").href =
        `${endpoint(state.record.id)}/report${result && !result.stale ? `?prediction_id=${encodeURIComponent(result.id)}` : ""}`;
    $("prediction-table-body").replaceChildren();
    $("prediction-counts").replaceChildren();
    $("prediction-warnings").replaceChildren();
    if (!result) {
      text(
        "prediction-context",
        state.selectedModelId ? "当前选择模型的最新结果" : "尚未运行模型",
      );
      text(
        "prediction-status",
        state.predictionLoading
          ? "正在读取已保存结果…"
          : state.selectedModelId
            ? "该记录与模型尚无保存结果；点击“运行模型”。"
            : "导入并选择模型后运行；已有历史快照会在此保留。",
      );
      $("prediction-status").className = "prediction-status";
      updatePredictionOverlay();
      requestAnimationFrame(drawECG);
      return;
    }
    const manifest = result.model.manifest;
    const auditOnly = result.model_id !== state.selectedModelId;
    text(
      "prediction-context",
      `${result.model.name} · 版本 ${result.model.version} · ${(result.input_leads || []).join(" / ")} · ${number(result.elapsed_ms, 2)} ms · ${date(result.created_at)}`,
    );
    const messages = [];
    if (result.stale)
      messages.push(
        `已过期：${result.stale_reason || "记录分析已改变"}。请重新运行，不作为当前波形叠加。`,
      );
    if (!result.model_available)
      messages.push("模型已删除：仅显示保存的历史快照，不可重新运行或叠加。");
    if (auditOnly && result.model_available)
      messages.push("历史快照：尚未选择该模型，不作为当前波形叠加。");
    text(
      "prediction-status",
      messages.length
        ? messages.join(" ")
        : "已保存的真实推理结果 · 紫色波形标签为模型类别，独立于 R 峰与人工标注",
    );
    $("prediction-status").className =
      `prediction-status${messages.length ? " historical" : ""}`;
    $("prediction-export").href =
      `/api/predictions/${encodeURIComponent(result.id)}/export`;
    const total = result.summary.evaluated;
    for (const count of result.summary.counts || []) {
      const item = el("div", "class-count"),
        caption = el("div");
      caption.append(
        el("span", "", count.label),
        el("strong", "", number(count.count)),
      );
      const track = el("div", "class-count-track"),
        fill = el("span");
      fill.style.width = `${total ? Math.min(100, (count.count / total) * 100) : 0}%`;
      track.append(fill);
      item.append(caption, track);
      $("prediction-counts").append(item);
    }
    text(
      "prediction-summary",
      `${taskLabel(result.task)} · 实际评估 ${number(total)} · 跳过 ${number(result.summary.skipped)}${result.task === "record_multilabel" ? ` · 阈值 ${number(manifest.threshold, 4)} · 标签可同时成立` : ""}`,
    );
    const predictions = result.predictions || [],
      size = 20,
      pages = Math.max(1, Math.ceil(predictions.length / size));
    state.predictionPage = Math.max(
      0,
      Math.min(pages - 1, state.predictionPage),
    );
    for (const prediction of predictions.slice(
      state.predictionPage * size,
      (state.predictionPage + 1) * size,
    )) {
      const row = el("tr"),
        timeCell = el("td");
      row.className = "prediction-row";
      row.tabIndex = 0;
      row.setAttribute(
        "aria-label",
        `定位 ${number(prediction.time, 3)} 秒的 ${prediction.class_label} 结果`,
      );
      row.addEventListener("click", (event) => {
        if (!event.target.closest("button, details"))
          locatePrediction(prediction);
      });
      row.addEventListener("keydown", (event) => {
        if (event.target === row && ["Enter", " "].includes(event.key)) {
          event.preventDefault();
          locatePrediction(prediction);
        }
      });
      const timeLabel =
        result.task === "beat_classification"
          ? `${number(prediction.time, 3)} s`
          : `${number(prediction.start, 3)}–${number(prediction.end, 3)} s`;
      const locate = modelAction(timeLabel, () => locatePrediction(prediction));
      locate.className = "text-link prediction-locate";
      locate.title = "定位保存结果对应的原始记录时间";
      timeCell.append(locate);
      row.append(
        timeCell,
        el("td", "", prediction.class_label),
        el("td", "", number(prediction.score, 5)),
      );
      const scoresCell = el("td"),
        details = el("details", "score-details");
      details.append(el("summary", "", "展开全部分数"));
      const scores = el("ol", "ordered-scores");
      for (let index = 0; index < manifest.classes.length; index++) {
        const score = el("li");
        score.append(
          el("span", "", manifest.classes[index]),
          el("strong", "", number(prediction.scores[index], 6)),
        );
        if ((prediction.labels || []).includes(manifest.classes[index]))
          score.classList.add("selected-score");
        scores.append(score);
      }
      details.append(
        scores,
        el(
          "p",
          "form-hint",
          `实际窗口 ${number(prediction.start, 6)}–${number(prediction.end, 6)} s${prediction.index == null ? "" : ` · 原始峰索引 ${prediction.index}`}`,
        ),
      );
      scoresCell.append(details);
      row.append(scoresCell);
      $("prediction-table-body").append(row);
    }
    $("prediction-prev").disabled = state.predictionPage === 0;
    $("prediction-next").disabled = state.predictionPage >= pages - 1;
    text(
      "prediction-page",
      `${state.predictionPage + 1} / ${pages} · ${predictions.length} 条`,
    );
    text(
      "prediction-preprocessing",
      JSON.stringify(
        {
          preprocessing: result.preprocessing,
          task: result.task,
          input_leads: result.input_leads,
          sha256: result.model.sha256,
          record_context: result.record_context,
        },
        null,
        2,
      ),
    );
    for (const warning of result.warnings || [])
      $("prediction-warnings").append(el("p", "form-hint", warning));
    updatePredictionOverlay();
    requestAnimationFrame(drawECG);
  }
  const manifestFields = [
    "schema_version",
    "name",
    "version",
    "task",
    "classes",
    "sample_rate",
    "input_samples",
    "lead_names",
    "pre_samples",
    "normalization",
    "input_filter",
    "notch",
    "output_type",
    "threshold",
    "description",
  ];
  function validateManifest(config) {
    if (!config || typeof config !== "object" || Array.isArray(config))
      throw new Error("配置必须是一个 JSON 对象。");
    const unknown = Object.keys(config).filter(
      (key) => !manifestFields.includes(key),
    );
    if (unknown.length)
      throw new Error(`不支持的配置字段：${unknown.join("、")}`);
    if (config.schema_version !== 1)
      throw new Error("schema_version 必须为 1。");
    for (const [key, label, max] of [
      ["name", "名称", 80],
      ["version", "版本", 32],
    ])
      if (
        typeof config[key] !== "string" ||
        !config[key].trim() ||
        config[key].length > max
      )
        throw new Error(`${label}须为 1–${max} 字。`);
    if (
      config.description != null &&
      (typeof config.description !== "string" ||
        config.description.length > 1000)
    )
      throw new Error("描述最多 1000 字。");
    for (const [key, label, min, max] of [
      ["classes", "类别", 2, 64],
      ["lead_names", "导联", 1, 12],
    ]) {
      const values = config[key];
      if (
        !Array.isArray(values) ||
        values.length < min ||
        values.length > max ||
        values.some(
          (value) =>
            typeof value !== "string" || !value.trim() || value.length > 80,
        ) ||
        new Set(values).size !== values.length
      )
        throw new Error(
          `${label}须为 ${min}–${max} 个不重复的非空名称，每项最多 80 字。`,
        );
    }
    if (
      config.lead_names.includes("selected") &&
      config.lead_names.length !== 1
    )
      throw new Error("selected 只能作为唯一输入导联。");
    if (
      ![
        "beat_classification",
        "record_classification",
        "record_multilabel",
      ].includes(config.task)
    )
      throw new Error("请选择支持的分类任务。");
    if (
      typeof config.sample_rate !== "number" ||
      !Number.isFinite(config.sample_rate) ||
      config.sample_rate < 100 ||
      config.sample_rate > 1000
    )
      throw new Error("目标采样率须为 100–1000 Hz。");
    if (
      !Number.isInteger(config.input_samples) ||
      config.input_samples < 64 ||
      config.input_samples > 20000
    )
      throw new Error("窗口样本数须为 64–20000 的整数。");
    if (
      config.task === "beat_classification" &&
      (!Number.isInteger(config.pre_samples) ||
        config.pre_samples <= 0 ||
        config.pre_samples >= config.input_samples)
    )
      throw new Error("R 峰前样本数须为大于 0 且小于窗口样本数的整数。");
    if (!["none", "zscore", "minmax"].includes(config.normalization))
      throw new Error("归一化仅支持 none、zscore 或 minmax。");
    if (
      !["raw", "bandpass"].includes(config.input_filter) ||
      ![0, 50, 60].includes(config.notch)
    )
      throw new Error("请选择支持的滤波及陷波配置。");
    if (config.input_filter === "raw" && config.notch !== 0)
      throw new Error("原始输入 raw 的陷波必须为 0。");
    if (!["logits", "probabilities"].includes(config.output_type))
      throw new Error("输出类型仅支持 logits 或 probabilities。");
    if (
      config.threshold != null &&
      (typeof config.threshold !== "number" ||
        !Number.isFinite(config.threshold) ||
        config.threshold <= 0 ||
        config.threshold >= 1)
    )
      throw new Error("多标签阈值须严格在 0 与 1 之间。");
    if (config.task === "record_multilabel" && config.threshold == null)
      throw new Error("请填写多标签阈值。");
    return config;
  }
  function builderManifest() {
    const lines = (id) =>
      $(id)
        .value.split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
    return validateManifest({
      schema_version: 1,
      name: $("model-name").value.trim(),
      version: $("model-version").value.trim(),
      task: $("model-task").value,
      classes: lines("model-classes"),
      sample_rate: Number($("model-sample-rate").value),
      input_samples: Number($("model-input-samples").value),
      lead_names: lines("model-leads"),
      pre_samples:
        $("model-task").value === "beat_classification"
          ? Number($("model-pre-samples").value)
          : null,
      normalization: $("model-normalization").value,
      input_filter: $("model-input-filter").value,
      notch: Number($("model-notch").value),
      output_type: $("model-output-type").value,
      threshold: Number($("model-threshold").value),
      description: $("model-description").value,
    });
  }
  function applyManifest(config) {
    validateManifest(config);
    const fields = {
      name: "model-name",
      version: "model-version",
      task: "model-task",
      sample_rate: "model-sample-rate",
      input_samples: "model-input-samples",
      pre_samples: "model-pre-samples",
      normalization: "model-normalization",
      input_filter: "model-input-filter",
      notch: "model-notch",
      output_type: "model-output-type",
      threshold: "model-threshold",
      description: "model-description",
    };
    for (const [key, id] of Object.entries(fields))
      $(id).value = config[key] ?? (key === "threshold" ? 0.5 : "");
    $("model-classes").value = config.classes.join("\n");
    $("model-leads").value = config.lead_names.join("\n");
    $("model-config-json").value = JSON.stringify(config, null, 2);
    updateModelForm();
  }
  function updateModelForm() {
    const beat = $("model-task").value === "beat_classification",
      multi = $("model-task").value === "record_multilabel",
      raw = $("model-input-filter").value === "raw";
    $("model-pre-field").hidden = !beat;
    $("model-pre-samples").required = beat;
    $("model-pre-samples").disabled = state.busy || !beat;
    $("model-pre-samples").max = String(
      Number($("model-input-samples").value) - 1,
    );
    $("model-threshold-field").hidden = !multi;
    $("model-threshold").required = multi;
    $("model-threshold").disabled = state.busy || !multi;
    if (raw) $("model-notch").value = "0";
    $("model-notch").disabled = state.busy || raw;
    $("model-import-submit").disabled = state.busy || state.configLoading;
  }
  function downloadConfig(config) {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(config, null, 2)], { type: "application/json" }),
    );
    const link = el("a");
    link.href = url;
    link.download = "model-config.json";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function checkModelFile(file) {
    if (!file) throw new Error("请选择 ONNX 模型文件。");
    if (/\.(pt|pth)$/i.test(file.name))
      throw new Error(
        "不支持 .pt / .pth；请从训练框架导出 ONNX，再导入 .onnx 文件。",
      );
    if (!/\.onnx$/i.test(file.name))
      throw new Error("请选择 .onnx 模型文件；请从训练框架导出 ONNX。");
    if (file.size > 100 * 1024 * 1024)
      throw new Error("模型文件超过 100 MB 上限。");
    if (!file.size) throw new Error("模型文件为空。");
  }
  function registerModelEvents() {
    for (const id of ["model-import-open", "model-empty-import"])
      $(id).addEventListener("click", () => {
        formError("model-import-error");
        openDialog("model-import-dialog");
      });
    $("model-protocol-open").addEventListener("click", () =>
      openDialog("model-protocol-dialog"),
    );
    $("prediction-rerun").addEventListener("click", runPrediction);
    $("models-refresh").addEventListener("click", async () => {
      if (state.busy) return;
      busy(true, "正在刷新模型与记录兼容性…");
      try {
        await refreshAfterModelMutation();
      } catch (error) {
        formError("models-error", error.message);
      } finally {
        busy(false);
      }
    });
    $("model-select").addEventListener("change", () => {
      state.selectedModelId = $("model-select").value;
      $("prediction-start").value = "0";
      // Re-read compatibility; model changes never relabel an old prediction.
      loadRecordModels(false).catch((error) =>
        formError("prediction-error", error.message),
      );
    });
    $("prediction-run").addEventListener("click", runPrediction);
    $("prediction-overlay").addEventListener("change", drawECG);
    $("prediction-prev").addEventListener("click", () => {
      state.predictionPage--;
      renderPrediction();
    });
    $("prediction-next").addEventListener("click", () => {
      state.predictionPage++;
      renderPrediction();
    });
    for (const id of [
      "model-task",
      "model-input-filter",
      "model-input-samples",
    ])
      $(id).addEventListener("input", updateModelForm);
    $("model-file").addEventListener("change", () => {
      formError("model-import-error");
      try {
        checkModelFile($("model-file").files[0]);
      } catch (error) {
        formError("model-import-error", error.message);
      }
    });
    $("model-config-file").addEventListener("change", async () => {
      const file = $("model-config-file").files[0],
        token = ++state.configRequest;
      state.configLoading = Boolean(file);
      updateModelForm();
      if (!file) return;
      formError("model-import-error");
      try {
        if (file.size > 512 * 1024)
          throw new Error("配置文件超过 512 KB 上限。");
        const content = await file.text();
        if (token !== state.configRequest) return;
        applyManifest(JSON.parse(content));
      } catch (error) {
        if (token === state.configRequest)
          formError(
            "model-import-error",
            error instanceof SyntaxError
              ? "配置不是有效的 JSON。"
              : error.message,
          );
      } finally {
        if (token === state.configRequest) {
          state.configLoading = false;
          updateModelForm();
        }
      }
    });
    $("model-config-apply").addEventListener("click", () => {
      formError("model-import-error");
      try {
        applyManifest(JSON.parse($("model-config-json").value));
      } catch (error) {
        formError(
          "model-import-error",
          error instanceof SyntaxError
            ? "配置不是有效的 JSON。"
            : error.message,
        );
      }
    });
    for (const id of ["model-config-preview", "model-config-download"])
      $(id).addEventListener("click", () => {
        formError("model-import-error");
        try {
          const config = builderManifest();
          $("model-config-json").value = JSON.stringify(config, null, 2);
          if (id === "model-config-download") downloadConfig(config);
        } catch (error) {
          formError("model-import-error", error.message);
        }
      });
    $("model-import-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (state.busy || state.configLoading) return;
      formError("model-import-error");
      let body;
      try {
        const file = $("model-file").files[0];
        checkModelFile(file);
        const manifest = builderManifest();
        body = new FormData();
        body.append("file", file);
        body.append("config", JSON.stringify(manifest));
      } catch (error) {
        formError("model-import-error", error.message);
        return;
      }
      busy(true, "正在验证 ONNX 张量与 CPU 运行兼容性…");
      try {
        await api("/api/models/import", { method: "POST", body });
        $("model-import-dialog").close();
        $("model-import-form").reset();
        await refreshAfterModelMutation();
        notify("模型已导入。请在工作台选择模型后运行。", "success");
      } catch (error) {
        if ($("model-import-dialog").open)
          formError("model-import-error", error.message);
        else
          formError(
            "models-error",
            `导入已完成，但模型列表刷新失败：${error.message}`,
          );
      } finally {
        busy(false);
      }
    });
    $("model-edit-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (state.busy || !state.editingModel) return;
      const name = $("model-edit-name").value.trim();
      if (!name) {
        formError("model-edit-error", "请填写模型名称。");
        return;
      }
      formError("model-edit-error");
      busy(true, "正在保存模型名称与描述…");
      try {
        await api(modelEndpoint(state.editingModel.id), {
          ...json({ name, description: $("model-edit-description").value }),
          method: "PATCH",
        });
        $("model-details-dialog").close();
        await refreshAfterModelMutation();
        notify("模型信息已保存，历史结果保留原始快照。", "success");
      } catch (error) {
        if ($("model-details-dialog").open)
          formError("model-edit-error", error.message);
        else
          formError("models-error", `保存已完成，但刷新失败：${error.message}`);
      } finally {
        busy(false);
      }
    });
    $("model-delete-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (state.busy || !state.deletingModel) return;
      formError("model-delete-error");
      busy(true, "正在删除模型，保留历史预测…");
      try {
        await api(modelEndpoint(state.deletingModel.id), { method: "DELETE" });
        $("model-delete-dialog").close();
        state.deletingModel = null;
        await refreshAfterModelMutation();
        notify("模型已删除，历史预测快照仍保留。", "success");
      } catch (error) {
        if ($("model-delete-dialog").open)
          formError("model-delete-error", error.message);
        else
          formError("models-error", `删除已完成，但刷新失败：${error.message}`);
      } finally {
        busy(false);
      }
    });
    updateModelForm();
  }

  function registerUpgradeEvents() {
    for (const view of ["single", "multi"])
      $(`view-${view}`).addEventListener("click", () => {
        if (state.busy || !state.record) return;
        state.view = view;
        state.hover = null;
        $("ecg-tooltip").hidden = true;
        updateWaveControls();
        drawECG();
      });
    const tabs = [...document.querySelectorAll("[data-tool]")];
    for (const tab of tabs) {
      tab.addEventListener("click", () => selectTool(tab.dataset.tool));
      tab.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
          return;
        event.preventDefault();
        const index =
          event.key === "Home"
            ? 0
            : event.key === "End"
              ? tabs.length - 1
              : (tabs.indexOf(tab) +
                  (event.key === "ArrowRight" ? 1 : -1) +
                  tabs.length) %
                tabs.length;
        selectTool(tabs[index].dataset.tool);
        tabs[index].focus();
      });
    }
    $("reanalyze-lead").addEventListener("click", () => {
      if (state.record?.has_original && state.api2)
        mutateRecord(
          `${endpoint(state.record.id)}/reanalyze`,
          json({ lead: state.lead }),
          `已从原始样本分析导联 ${state.lead}。`,
        );
    });
    $("review-status").addEventListener("change", () => {
      if (state.record && state.api2)
        mutateRecord(
          `${endpoint(state.record.id)}/review`,
          { ...json({ status: $("review-status").value }), method: "PATCH" },
          "人工复核状态已保存。",
        );
    });
    $("history-status").addEventListener("change", renderHistory);
    $("signal-export").addEventListener("click", downloadOriginal);
    $("segment-open").addEventListener("click", openSegment);
    for (const id of ["segment-start", "segment-end", "segment-lead"])
      $(id).addEventListener("input", invalidateSegment);
    $("segment-form").addEventListener("submit", (event) => {
      event.preventDefault();
      analyzeSegment(false);
    });
    $("segment-save").addEventListener("click", () => analyzeSegment(true));
    $("beat-review-form").addEventListener("submit", (event) => {
      event.preventDefault();
      saveBeatReview(false);
    });
    $("beat-review-clear").addEventListener("click", () =>
      saveBeatReview(true),
    );
    $("compare-run").addEventListener("click", runComparison);
    for (const id of ["compare-left", "compare-right"])
      $(id).addEventListener("change", clearComparison);
    for (const id of ["compare-signal", "compare-window"])
      $(id).addEventListener("change", drawComparison);
    $("compare-slider").addEventListener("input", () => {
      state.compareStart = Number($("compare-slider").value);
      drawComparison();
    });
    $("comparison-demo").addEventListener("click", () =>
      openDialog("demo-dialog"),
    );
    refreshCompareSelectors();
  }
  $("menu-toggle").addEventListener("click", () =>
    setSidebar(!$("sidebar").classList.contains("open")),
  );
  $("sidebar-backdrop").addEventListener("click", () => setSidebar(false));
  window.addEventListener("hashchange", () =>
    changePage(location.hash.slice(1)),
  );
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setSidebar(false);
  });
  $("notice-close").addEventListener("click", () => {
    $("notice").hidden = true;
  });
  for (const dialog of document.querySelectorAll("dialog")) {
    for (const button of dialog.querySelectorAll("[data-close-dialog]"))
      button.addEventListener("click", () => dialog.close());
    dialog.addEventListener("cancel", (event) => {
      if (state.busy) event.preventDefault();
    });
  }
  for (const id of ["import-open", "empty-import"])
    $(id).addEventListener("click", () => {
      formError("import-error");
      openDialog("import-dialog");
    });
  for (const id of ["demo-open", "empty-demo"])
    $(id).addEventListener("click", () => openDialog("demo-dialog"));
  $("connection-retry").addEventListener("click", start);
  $("demo-refresh").addEventListener("click", async () => {
    if (state.busy) return;
    busy(true, "正在刷新示例列表…");
    formError("demo-error");
    try {
      await loadDemos();
    } catch (error) {
      formError("demo-error", error.message);
    } finally {
      busy(false);
    }
  });
  $("history-refresh").addEventListener("click", async () => {
    if (state.busy) return;
    busy(true, "正在刷新本地记录…");
    try {
      await loadRecords();
      notify("记录列表已刷新。", "success");
    } catch (error) {
      notify(error.message);
    } finally {
      busy(false);
    }
  });
  $("record-search").addEventListener("input", renderHistory);
  $("import-file").addEventListener("change", () =>
    text("file-label", $("import-file").files[0]?.name || "选择或拖入波形文件"),
  );
  const zone = $("upload-zone");
  for (const name of ["dragenter", "dragover"])
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  for (const name of ["dragleave", "drop"])
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  zone.addEventListener("drop", (event) => {
    const files = event.dataTransfer.files;
    if (files.length !== 1) {
      formError("import-error", "请一次导入一个波形文件。");
      return;
    }
    $("import-file").files = files;
    text("file-label", files[0].name);
  });
  $("import-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.busy) return;
    const file = $("import-file").files[0];
    if (!file || !/\.(csv|txt|json)$/i.test(file.name)) {
      formError("import-error", "请选择 .csv、.txt 或 .json 波形文件。");
      return;
    }
    if (!$("import-form").reportValidity()) return;
    formError("import-error");
    const body = new FormData($("import-form"));
    busy(true, "正在解析、滤波与分析原始波形…");
    try {
      const record = await api("/api/analyze", { method: "POST", body });
      showRecord(record);
      $("import-dialog").close();
      navigate("workspace");
      await syncSavedRecord(record, "导入完成，分析结果已保存在本机。");
    } catch (error) {
      formError("import-error", error.message);
    } finally {
      busy(false);
    }
  });
  $("record-edit").addEventListener("click", () => {
    if (!state.record) return;
    $("edit-name").value = state.record.name || "";
    $("edit-note").value = state.record.note || "";
    formError("edit-error");
    openDialog("edit-dialog");
  });
  $("edit-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.busy || !state.record) return;
    const name = $("edit-name").value.trim();
    if (!name) {
      formError("edit-error", "请填写记录名称。");
      return;
    }
    formError("edit-error");
    busy(true, "正在保存记录信息…");
    try {
      const record = await api(endpoint(state.record.id), {
        ...json({ name, note: $("edit-note").value }),
        method: "PATCH",
      });
      showRecord(record);
      $("edit-dialog").close();
      await syncSavedRecord(record, "记录名称与备注已保存。");
    } catch (error) {
      formError("edit-error", error.message);
    } finally {
      busy(false);
    }
  });
  $("delete-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.busy || !state.deleting) return;
    const id = state.deleting.id;
    busy(true, "正在删除本地记录…");
    formError("delete-error");
    try {
      await api(endpoint(id), { method: "DELETE" });
      $("delete-dialog").close();
      const wasCurrent = state.record?.id === id;
      state.records = state.records.filter((record) => record.id !== id);
      renderHistory();
      refreshCompareSelectors();
      text("record-count", state.records.length);
      if (wasCurrent) renderEmpty();
      state.deleting = null;
      await loadRecords();
      if (wasCurrent && state.records.length)
        showRecord(await api(endpoint(state.records[0].id)));
      notify("本地记录已删除。", "success");
    } catch (error) {
      if ($("delete-dialog").open) formError("delete-error", error.message);
      else notify(`删除已完成，但后续读取未完成：${error.message}`);
    } finally {
      busy(false);
    }
  });
  $("lead-select").addEventListener("change", () => {
    state.lead = $("lead-select").value;
    state.hover = null;
    $("ecg-tooltip").hidden = true;
    updateWaveControls();
    drawECG();
  });
  $("wave-filtered").addEventListener("click", () => {
    if (state.lead !== state.record?.selected_lead) return;
    state.filtered = true;
    updateWaveControls();
    drawECG();
  });
  $("wave-raw").addEventListener("click", () => {
    state.filtered = false;
    updateWaveControls();
    drawECG();
  });
  $("peak-toggle").addEventListener("change", drawECG);
  $("window-select").addEventListener("change", () => {
    state.window = Number($("window-select").value);
    state.hover = null;
    $("ecg-tooltip").hidden = true;
    updateWaveControls();
    drawECG();
  });
  $("time-slider").addEventListener("input", () => {
    state.start = Number($("time-slider").value);
    state.hover = null;
    $("ecg-tooltip").hidden = true;
    updateWaveControls();
    drawECG();
  });
  $("window-prev").addEventListener("click", () =>
    moveWindow(-state.window / 2),
  );
  $("window-next").addEventListener("click", () =>
    moveWindow(state.window / 2),
  );
  $("beats-prev").addEventListener("click", () => {
    state.beatsPage--;
    renderBeats();
  });
  $("beats-next").addEventListener("click", () => {
    state.beatsPage++;
    renderBeats();
  });
  $("ecg-canvas").addEventListener("pointermove", (event) => {
    if (!ecgPlot || !state.record) return;
    const rect = $("ecg-canvas").getBoundingClientRect();
    const px = event.clientX - rect.left,
      py = event.clientY - rect.top;
    const p = ecgPlot;
    if (px < p.left || px > p.right || py < p.top || py > p.bottom) {
      state.hover = null;
      $("ecg-tooltip").hidden = true;
      drawECG();
      return;
    }
    const time = state.start + ((px - p.left) / (p.right - p.left)) * p.span;
    const index = Math.min(p.times.length - 1, lowerBound(p.times, time));
    const nearest =
      index > 0 &&
      Math.abs(p.times[index - 1] - time) < Math.abs(p.times[index] - time)
        ? index - 1
        : index;
    state.hover = time;
    text(
      "ecg-tooltip",
      `${state.lead} · ${number(p.times[nearest], 3)} s · ${number(p.values[nearest], 3)} mV`,
    );
    const tip = $("ecg-tooltip");
    tip.hidden = false;
    tip.style.left = `${Math.max(8, Math.min(rect.width - tip.offsetWidth - 8, px + 12))}px`;
    tip.style.top = `${Math.max(4, py - 38)}px`;
    drawECG();
  });
  $("ecg-canvas").addEventListener("pointerleave", () => {
    state.hover = null;
    $("ecg-tooltip").hidden = true;
    drawECG();
  });
  let resizeFrame;
  const observer = new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(drawAll);
  });
  observer.observe($("waveform-wrap"));
  observer.observe($("rr-canvas").parentElement);
  for (const canvas of document.querySelectorAll("canvas"))
    observer.observe(canvas);
  registerUpgradeEvents();
  registerModelEvents();
  window.addEventListener("resize", () => {
    requestAnimationFrame(drawAll);
    if (window.innerWidth > 760) setSidebar(false);
  });
  changePage(location.hash.slice(1));
  renderHistory();
  renderDemos();
  start();
})();
