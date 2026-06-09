// Browser realtime client for Speech-to-Text.
//
// Captures mic audio via WebAudio, downsamples to the server's target sample
// rate (16 kHz), and streams float32 PCM frames over a WebSocket. Each
// connection is an isolated session on the server with its own speaker
// tracking and transcript file. The UI surfaces:
//   - server connection state (online/offline)
//   - current backend + model
//   - mic capture state with VU meter
//   - model loading overlay (when picking a new model)
//   - toast notifications for transient errors / events
//   - a persistent diagnostics log in the sidebar

(() => {
  // ── Element refs ─────────────────────────────────────────────────────
  const el = (id) => document.getElementById(id);
  const startBtn = el("startBtn");
  const stopBtn = el("stopBtn");
  const clearBtn = el("clearBtn");
  const clearLogBtn = el("clearLogBtn");
  const statusPill = el("statusPill");
  const statusText = el("statusText");
  const meterBar = el("meterBar");
  const transcriptEl = el("transcript");
  const footerEl = el("footer");
  const serverInfoEl = el("serverInfo");
  const backendSelect = el("backendSelect");
  const modelSelect = el("modelSelect");
  const modelHint = el("modelHint");
  const errorLogEl = el("errorLog");
  const connBadge = el("connBadge");
  const connLabel = el("connLabel");
  const overlayEl = el("loadingOverlay");
  const overlayTitle = el("overlayTitle");
  const overlaySub = el("overlaySub");
  const toastStack = el("toastStack");
  const recBanner = el("recBanner");
  const recElapsed = el("recElapsed");
  const recBackend = el("recBackend");
  const freqBars = el("freqBars");

  // Number of vertical bars in the visualizer (kept modest so the layout
  // doesn't get crushed on narrow screens)
  const BAR_COUNT = 24;
  // Build the bar elements once on page load
  for (let i = 0; i < BAR_COUNT; i++) {
    const b = document.createElement("span");
    b.className = "bar";
    freqBars.appendChild(b);
  }

  // ── State ────────────────────────────────────────────────────────────
  let audioCtx = null;
  let mediaStream = null;
  let processorNode = null;
  let sourceNode = null;
  let analyser = null;
  let analyserBuffer = null;
  let rafHandle = null;
  let recStartMs = 0;
  let elapsedHandle = null;
  let socket = null;
  let serverSampleRate = 16000;
  let segCount = 0;
  let availableModels = {};
  let serverDefault = null;
  let backendLabels = {
    "nemo": "NVIDIA NeMo",
    "qwen": "Qwen3-ASR",
    "gemini": "Gemini",
    "google-stt": "Google Cloud STT",
    "whisper": "Whisper (faster-whisper)",
  };
  let healthPollHandle = null;

  // speakerColors removed — diarization is no longer part of this project.

  // ── Toasts ───────────────────────────────────────────────────────────
  function toast(message, severity = "info", durationMs = 4000) {
    const node = document.createElement("div");
    node.className = "toast";
    node.setAttribute("data-severity", severity);
    node.innerHTML =
      `<span class="toast-msg"></span>` +
      `<button class="toast-close" aria-label="dismiss">×</button>`;
    node.querySelector(".toast-msg").textContent = message;
    const dismiss = () => {
      node.classList.add("toast-out");
      setTimeout(() => node.remove(), 220);
    };
    node.querySelector(".toast-close").addEventListener("click", dismiss);
    toastStack.appendChild(node);
    if (durationMs > 0) setTimeout(dismiss, durationMs);
  }

  // ── Diagnostics log ──────────────────────────────────────────────────
  function logEvent(message, severity = "info") {
    const empty = errorLogEl.querySelector(".log-empty");
    if (empty) empty.remove();

    const li = document.createElement("li");
    li.className = "log-item";
    li.setAttribute("data-severity", severity);
    li.innerHTML = `<span class="log-time"></span><span class="log-msg"></span>`;
    li.querySelector(".log-time").textContent = new Date().toLocaleTimeString();
    li.querySelector(".log-msg").textContent = message;
    errorLogEl.prepend(li);

    // Cap the list at 30 entries to bound DOM growth
    const items = errorLogEl.querySelectorAll(".log-item");
    if (items.length > 30) items[items.length - 1].remove();
  }

  function clearLog() {
    errorLogEl.innerHTML = `<li class="log-empty">No issues yet.</li>`;
  }

  // ── Status helpers ───────────────────────────────────────────────────
  function setStatus(state, text) {
    statusPill.setAttribute("data-state", state);
    statusText.textContent = text;
  }
  function setConn(state, label) {
    connBadge.setAttribute("data-state", state);
    connLabel.textContent = label;
  }

  // ── Loading overlay ──────────────────────────────────────────────────
  function showOverlay(title, sub) {
    overlayTitle.textContent = title;
    overlaySub.textContent = sub || "";
    overlayEl.hidden = false;
  }
  function hideOverlay() { overlayEl.hidden = true; }

  // ── Recording banner + frequency visualizer ──────────────────────────
  function showRecordingBanner(backendLabel) {
    recBanner.hidden = false;
    recBackend.textContent = backendLabel || "";
    recStartMs = performance.now();
    if (elapsedHandle) clearInterval(elapsedHandle);
    elapsedHandle = setInterval(() => {
      const s = Math.floor((performance.now() - recStartMs) / 1000);
      const m = Math.floor(s / 60);
      const sec = (s % 60).toString().padStart(2, "0");
      recElapsed.textContent = `${m}:${sec}`;
    }, 250);
    recElapsed.textContent = "0:00";
  }
  function hideRecordingBanner() {
    recBanner.hidden = true;
    if (elapsedHandle) { clearInterval(elapsedHandle); elapsedHandle = null; }
    freqBars.classList.remove("live");
    // Reset bars to their CSS-driven ambient state
    for (const b of freqBars.children) b.style.height = "";
  }

  // Animate the frequency bars from the AnalyserNode. Called via rAF.
  function pumpFrequencyBars() {
    if (!analyser || !analyserBuffer) return;
    analyser.getByteFrequencyData(analyserBuffer);

    // Bin the FFT bins into BAR_COUNT bars. Use a logarithmic-ish slice
    // so low frequencies don't dominate visually. We grab the first half
    // of the spectrum (the rest is mostly above-speech).
    const halfLen = Math.floor(analyserBuffer.length * 0.5);
    const bars = freqBars.children;
    for (let i = 0; i < bars.length; i++) {
      const start = Math.floor((i / bars.length) * halfLen);
      const end = Math.floor(((i + 1) / bars.length) * halfLen);
      let sum = 0; let count = 0;
      for (let j = start; j < end; j++) { sum += analyserBuffer[j]; count++; }
      const avg = count > 0 ? sum / count : 0;  // 0..255
      // Map 0..255 → 4..36 px. Apply a slight gamma so quieter sounds register.
      const norm = avg / 255.0;
      const h = 4 + Math.pow(norm, 0.7) * 32;
      bars[i].style.height = `${h}px`;
    }
    rafHandle = requestAnimationFrame(pumpFrequencyBars);
  }
  function startFrequencyBars() {
    freqBars.classList.add("live");
    rafHandle = requestAnimationFrame(pumpFrequencyBars);
  }
  function stopFrequencyBars() {
    if (rafHandle) { cancelAnimationFrame(rafHandle); rafHandle = null; }
  }

  // ── Transcript rendering ─────────────────────────────────────────────
  // Streaming render model:
  //   - server emits {type:"interim", id, text, ...} as words arrive
  //   - server emits {type:"segment", id, text, speaker, ..., is_final:true}
  //     when VAD detects an utterance end
  //   - the UI keeps a single "active row" for the current utterance, keyed
  //     by id. Each interim updates that row in place. The final replaces
  //     the row's content with the locked version (with speaker color) and
  //     starts a new active row for the next utterance.

  // Map: utterance id → DOM element (the row showing this utterance)
  const activeRows = new Map();

  function ensureRow(id) {
    if (activeRows.has(id)) return activeRows.get(id);
    if (segCount === 0) transcriptEl.innerHTML = "";
    segCount++;
    const div = document.createElement("div");
    div.className = "seg seg-interim";
    div.dataset.utteranceId = id;
    div.innerHTML =
      `<span class="ts"></span>` +
      `<span class="text"></span>`;
    transcriptEl.appendChild(div);
    activeRows.set(id, div);
    return div;
  }

  function applyInterim(m) {
    const div = ensureRow(m.id);
    div.classList.add("seg-interim");
    div.classList.remove("seg-final");
    div.querySelector(".ts").textContent =
      `[${m.start.toFixed(2)}s + ${m.duration.toFixed(2)}s]`;
    div.querySelector(".text").textContent = m.text;
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }

  function applyFinal(m) {
    const div = ensureRow(m.id);
    div.classList.remove("seg-interim");
    div.classList.add("seg-final");
    div.querySelector(".ts").textContent =
      `[${m.start.toFixed(2)}s – ${m.end.toFixed(2)}s]`;
    div.querySelector(".text").textContent = m.text;
    activeRows.delete(m.id);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }

  function clearTranscript() {
    transcriptEl.innerHTML =
      `<div class="empty-state">` +
      `  <div class="empty-icon">🎙</div>` +
      `  <div class="empty-title">Cleared</div>` +
      `  <div class="empty-sub">Start mic again to continue.</div>` +
      `</div>`;
    segCount = 0;
    activeRows.clear();
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  // ── Audio downsample ─────────────────────────────────────────────────
  function downsample(input, srcRate, dstRate) {
    if (dstRate === srcRate) return input;
    if (dstRate > srcRate) throw new Error("downsample only");
    const ratio = srcRate / dstRate;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const i0 = Math.floor(srcIdx);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const t = srcIdx - i0;
      out[i] = input[i0] * (1 - t) + input[i1] * t;
    }
    return out;
  }

  // ── Server discovery ─────────────────────────────────────────────────
  async function fetchServerInfo() {
    try {
      const r = await fetch("/health");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const info = await r.json();
      serverSampleRate = info.sample_rate || 16000;
      const backendNice = backendLabels[info.asr_backend] || info.asr_backend;
      serverInfoEl.textContent =
        `${backendNice} · ${info.asr_model} · ${info.sample_rate} Hz · chunk ${info.chunk_duration}s`;
      setConn("online", "online");
      return info;
    } catch (e) {
      serverInfoEl.textContent = "server unreachable";
      setConn("offline", "offline");
      throw e;
    }
  }

  async function fetchModels() {
    try {
      const r = await fetch("/models");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const m = await r.json();
      availableModels = m.available || {};
      serverDefault = m.default;
      populateBackendOptions();
    } catch (e) {
      logEvent(`Failed to fetch /models: ${e.message}`, "error");
      toast("Could not load model list — is the server up?", "error");
    }
  }

  function populateBackendOptions() {
    backendSelect.innerHTML = "";
    const backends = Object.keys(availableModels);
    if (backends.length === 0) {
      const opt = new Option("(none configured)", "");
      opt.disabled = true;
      backendSelect.add(opt);
      return;
    }
    for (const b of backends) {
      const label = backendLabels[b] || b;
      backendSelect.add(new Option(label, b));
    }
    if (serverDefault && backends.includes(serverDefault.backend)) {
      backendSelect.value = serverDefault.backend;
    }
    populateModelOptions();
  }

  function populateModelOptions() {
    const backend = backendSelect.value;
    const models = availableModels[backend] || {};
    modelSelect.innerHTML = "";
    const ids = Object.keys(models);
    if (ids.length === 0) {
      const opt = new Option("(no models listed)", "");
      opt.disabled = true;
      modelSelect.add(opt);
      modelHint.textContent = "No models configured for this backend.";
      return;
    }
    for (const id of ids) {
      modelSelect.add(new Option(id, id));
    }
    if (serverDefault && serverDefault.backend === backend &&
        ids.includes(serverDefault.model)) {
      modelSelect.value = serverDefault.model;
    }
    updateModelHint();
  }

  function updateModelHint() {
    const backend = backendSelect.value;
    const model = modelSelect.value;
    const desc = (availableModels[backend] || {})[model];
    modelHint.textContent = desc || "Pick a backend and model, then start the mic.";
  }

  function lockModelSelectors(locked) {
    backendSelect.disabled = locked;
    modelSelect.disabled = locked;
  }

  // ── Connection health polling ────────────────────────────────────────
  function startHealthPoll() {
    if (healthPollHandle) return;
    healthPollHandle = setInterval(async () => {
      try {
        const r = await fetch("/health");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const info = await r.json();
        if (info.status === "ready") setConn("online", "online");
        else setConn("loading", info.status || "loading");
      } catch {
        setConn("offline", "offline");
      }
    }, 8000);
  }

  // ── Capture / WebSocket lifecycle ────────────────────────────────────
  async function start() {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("idle", "requesting mic…");
    segCount = 0;
    transcriptEl.innerHTML =
      `<div class="empty-state">` +
      `  <div class="empty-icon">🎙</div>` +
      `  <div class="empty-title">Listening…</div>` +
      `  <div class="empty-sub">Speak normally; transcripts appear every few seconds.</div>` +
      `</div>`;

    try {
      await fetchServerInfo();
    } catch {
      setStatus("error", "server unreachable");
      toast("Server unreachable — check that the container is running.", "error", 6000);
      startBtn.disabled = false; stopBtn.disabled = true;
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("error", "mic API unavailable");
      const msg = "navigator.mediaDevices is undefined — open this page on http://localhost (not on a LAN IP) so the browser allows mic access.";
      toast(msg, "error", 8000);
      logEvent(msg, "error");
      startBtn.disabled = false; stopBtn.disabled = true;
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        video: false,
      });
    } catch (err) {
      setStatus("error", `mic denied: ${err.message}`);
      toast(`Mic permission denied: ${err.message}`, "error", 6000);
      logEvent(`getUserMedia failed: ${err.message}`, "error");
      startBtn.disabled = false; stopBtn.disabled = true;
      return;
    }

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") {
      try { await audioCtx.resume(); } catch (e) { console.warn("audioCtx.resume failed", e); }
    }
    const srcRate = audioCtx.sampleRate;

    const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${wsProto}//${location.host}/ws/transcribe`);
    socket.binaryType = "arraybuffer";

    let selectedBackend = backendSelect.value;
    let selectedModel = modelSelect.value;
    let modelReady = false;
    const setModelReady = (v) => { modelReady = v; };

    socket.onopen = () => {
      setStatus("idle", "connecting…");
      lockModelSelectors(true);
    };
    socket.onerror = () => {
      setStatus("error", "websocket error");
      logEvent("WebSocket error", "error");
    };
    socket.onclose = (e) => {
      setStatus("idle", "stopped");
      lockModelSelectors(false);
      hideOverlay();
      stopFrequencyBars();
      hideRecordingBanner();
      startBtn.disabled = false; stopBtn.disabled = true;
      if (e.code !== 1000) {
        logEvent(`WebSocket closed (code ${e.code})`, "warning");
      }
    };
    socket.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        handleServerMessage(m, selectedBackend, selectedModel, setModelReady);
      } catch { /* ignore non-JSON */ }
    };

    // Mic graph
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);

    // Analyser for the frequency visualizer in the recording banner.
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.7;
    analyserBuffer = new Uint8Array(analyser.frequencyBinCount);
    sourceNode.connect(analyser);

    // Show the live banner (banner shows even before the first segment so
    // users can immediately tell the mic is active).
    const backendNice = backendLabels[selectedBackend] || selectedBackend;
    showRecordingBanner(`${backendNice} · ${selectedModel}`);
    startFrequencyBars();

    processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
    let frameCount = 0;
    processorNode.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);

      let peak = 0;
      for (let i = 0; i < input.length; i++) {
        const a = Math.abs(input[i]); if (a > peak) peak = a;
      }
      meterBar.style.width = `${Math.min(100, peak * 200)}%`;

      if (socket && socket.readyState === WebSocket.OPEN && modelReady) {
        const ds = downsample(input, srcRate, serverSampleRate);
        socket.send(new Float32Array(ds).buffer);
        frameCount++;
        if (frameCount === 1) {
          logEvent(`Streaming started (peak ${peak.toFixed(3)})`, "info");
        }
      }
    };
    const muteGain = audioCtx.createGain();
    muteGain.gain.value = 0;
    sourceNode.connect(processorNode);
    processorNode.connect(muteGain);
    muteGain.connect(audioCtx.destination);
  }

  function handleServerMessage(m, selectedBackend, selectedModel, setModelReady) {
    if (m.type === "ready") {
      serverSampleRate = m.sample_rate;
      footerEl.textContent =
        `session ${m.session_id} · ${m.asr_backend} · ${m.asr_model}`;
      // If user picked something other than the server default, request swap
      if (selectedBackend &&
          (m.asr_backend !== selectedBackend || m.asr_model !== selectedModel)) {
        showOverlay(`Loading ${selectedBackend} / ${selectedModel}…`,
                    "First time loading this combination can take a minute.");
        socket.send(JSON.stringify({
          type: "select_model", backend: selectedBackend, model: selectedModel,
        }));
      } else {
        setModelReady(true);
        hideOverlay();
        setStatus("live", `streaming @ ${serverSampleRate} Hz`);
        logEvent(`Session ready: ${m.asr_backend}/${m.asr_model}`, "info");
      }
    }
    else if (m.type === "loading_model") {
      showOverlay(`Loading ${m.backend} / ${m.model}…`,
                  m.detail || "First time downloads can take a minute.");
      setStatus("loading", `loading ${m.backend}/${m.model.split('/').pop()}…`);
      setModelReady(false);
    }
    else if (m.type === "model_changed") {
      hideOverlay();
      setModelReady(true);
      setStatus("live", `streaming @ ${serverSampleRate} Hz`);
      const sessionPart = (footerEl.textContent.split(' · ')[0] || '').trim();
      footerEl.textContent = `${sessionPart} · ${m.backend} · ${m.model}`;
      toast(`Model active: ${m.backend} / ${m.model}`, "success");
      logEvent(`Model active: ${m.backend}/${m.model}`, "info");
    }
    else if (m.type === "interim") {
      applyInterim(m);
    }
    else if (m.type === "segment") {
      // Streaming finals carry is_final:true. (Legacy chunk-style messages
      // without an id still work — applyFinal handles either.)
      if (!m.id) m.id = `seg-${Date.now()}-${Math.random().toString(36).slice(2,6)}`;
      applyFinal(m);
    }
    else if (m.type === "closed") {
      const summary =
        `Session ended · ${m.total_segments} segment(s)` +
        (m.transcript_path ? ` · saved to ${m.transcript_path}` : "");
      footerEl.textContent = summary;
      logEvent(summary, "info");
    }
    else if (m.type === "error") {
      hideOverlay();
      setStatus("error", "server error");
      const sev = m.severity || "error";
      toast(m.message || "Server error", sev, 8000);
      logEvent(m.message || "Server error", sev);
    }
    else if (m.type === "info") {
      logEvent(m.message || "info", m.severity || "info");
    }
  }

  function stop() {
    stopBtn.disabled = true;
    stopFrequencyBars();
    hideRecordingBanner();
    if (socket && socket.readyState === WebSocket.OPEN) {
      try { socket.send(JSON.stringify({ type: "stop" })); } catch {}
    }
    setTimeout(() => {
      if (processorNode) { try { processorNode.disconnect(); } catch {} processorNode = null; }
      if (analyser) { try { analyser.disconnect(); } catch {} analyser = null; analyserBuffer = null; }
      if (sourceNode) { try { sourceNode.disconnect(); } catch {} sourceNode = null; }
      if (audioCtx) { try { audioCtx.close(); } catch {} audioCtx = null; }
      if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
      if (socket && socket.readyState <= 1) { try { socket.close(); } catch {} }
      socket = null;
      meterBar.style.width = "0%";
      startBtn.disabled = false;
    }, 300);
  }

  // ── Wire up ──────────────────────────────────────────────────────────
  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", stop);
  clearBtn.addEventListener("click", clearTranscript);
  clearLogBtn.addEventListener("click", clearLog);
  backendSelect.addEventListener("change", () => { populateModelOptions(); });
  modelSelect.addEventListener("change", updateModelHint);

  // Initial fetches + start health polling
  fetchServerInfo().catch(() => {});
  fetchModels().catch(() => {});
  startHealthPoll();
})();
