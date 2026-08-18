/**
 * YuzTanima — tarayici ici canli yuz tespiti demosu.
 *
 * Kamera goruntusu hicbir zaman cihazdan cikmaz: kareler yalnizca sayfa
 * icindeki ONNX Runtime oturumuna verilir, hicbir yere gonderilmez ve
 * kaydedilmez.
 *
 * Tasarim notu: video her zaman akici kalsin diye cizim dongusu ile
 * cikarim dongusu ayrilmistir. Cikarim ne kadar surerse sursun, video
 * requestAnimationFrame hizinda cizilmeye devam eder; kutular en son
 * tamamlanan tespitten gelir.
 */

import { decode, fitScale, rgbaToNchw } from "./scrfd.js";

const MODEL_URL = "./models/det_10g.onnx";
const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

const el = {
  view: document.getElementById("view"),
  stage: document.getElementById("stage"),
  idle: document.getElementById("overlay-idle"),
  error: document.getElementById("overlay-error"),
  errorTitle: document.getElementById("error-title"),
  errorText: document.getElementById("error-text"),
  startBtn: document.getElementById("start-btn"),
  retryBtn: document.getElementById("retry-btn"),
  modelNote: document.getElementById("model-note"),
  modelSize: document.getElementById("model-size"),
  statInfer: document.getElementById("stat-infer"),
  statRate: document.getElementById("stat-rate"),
  statFps: document.getElementById("stat-fps"),
  statFaces: document.getElementById("stat-faces"),
  statBackend: document.getElementById("stat-backend"),
  ctlSize: document.getElementById("ctl-size"),
  ctlThresh: document.getElementById("ctl-thresh"),
  ctlThreshVal: document.getElementById("ctl-thresh-val"),
  ctlKps: document.getElementById("ctl-kps"),
  ctlMirror: document.getElementById("ctl-mirror"),
};

const KPS_COLORS = ["#2dd4bf", "#2dd4bf", "#f59e0b", "#60a5fa", "#60a5fa"];

const state = {
  session: null,
  backend: null,
  inputName: null,
  outputNames: null,
  video: null,
  stream: null,
  running: false,
  detections: [],
  inferMs: 0,
  detPerSec: 0,
};

const work = document.createElement("canvas");
const workCtx = work.getContext("2d", { willReadFrequently: true });
const ctx = el.view.getContext("2d");

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

function configureOrt() {
  const ort = window.ort;
  if (!ort) throw new Error("ONNX Runtime yüklenemedi (CDN engellenmiş olabilir).");
  ort.env.wasm.wasmPaths = ORT_DIST;
  // Coklu is parcacigi yalnizca cross-origin isolation varsa mumkun;
  // GitHub Pages bu basliklari gondermedigi icin tek parcacik kullanilir.
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 4)
    : 1;
  ort.env.wasm.simd = true;
  ort.env.logLevel = "error";
  return ort;
}

async function fetchModel(onProgress) {
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`Model indirilemedi (HTTP ${res.status}).`);

  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) {
    const buf = await res.arrayBuffer();
    onProgress(1, buf.byteLength);
    return buf;
  }

  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total, total);
  }
  const merged = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    merged.set(c, off);
    off += c.length;
  }
  return merged.buffer;
}

async function createSession(ort, buffer) {
  const candidates = [];
  if (navigator.gpu) candidates.push("webgpu");
  candidates.push("wasm");

  let lastErr = null;
  for (const ep of candidates) {
    try {
      const session = await ort.InferenceSession.create(buffer, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      return { session, backend: ep };
    } catch (err) {
      lastErr = err;
      console.warn(`[demo] ${ep} arka ucu kullanilamadi:`, err);
    }
  }
  throw lastErr || new Error("Hiçbir arka uç başlatılamadı.");
}

async function initModel() {
  const ort = configureOrt();
  const buffer = await fetchModel((ratio, total) => {
    const mb = (total / 1048576).toFixed(1);
    el.modelSize.textContent = `${mb} MB`;
    el.modelNote.textContent =
      ratio >= 1 ? "Model indirildi, hazırlanıyor…" : `Model indiriliyor… %${Math.round(ratio * 100)}`;
  });

  const { session, backend } = await createSession(ort, buffer);
  state.session = session;
  state.backend = backend;
  state.inputName = session.inputNames[0];
  state.outputNames = session.outputNames;

  el.statBackend.textContent = backend === "webgpu" ? "WebGPU" : "WASM";
  el.modelNote.textContent = `Model hazır — ${backend === "webgpu" ? "WebGPU" : "WASM"} arka ucu.`;
  el.startBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// On isleme  (Python tarafiyla ayni: en-boy korunur, sol-uste hizalanir,
// bosluk siyah kalir, RGB, (v - 127.5) / 128, NCHW)
// ---------------------------------------------------------------------------

function preprocess(video, size) {
  if (work.width !== size || work.height !== size) {
    work.width = size;
    work.height = size;
  }
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = fitScale(vw, vh, size);

  workCtx.fillStyle = "#000";
  workCtx.fillRect(0, 0, size, size);
  workCtx.drawImage(video, 0, 0, vw, vh, 0, 0, vw * scale, vh * scale);

  const { data } = workCtx.getImageData(0, 0, size, size);
  return { tensor: rgbaToNchw(data, size), scale, size };
}

async function detectOnce(video, size, threshold) {
  const ort = window.ort;
  const { tensor, scale } = preprocess(video, size);
  const input = new ort.Tensor("float32", tensor, [1, 3, size, size]);

  const t0 = performance.now();
  const results = await state.session.run({ [state.inputName]: input });
  const elapsed = performance.now() - t0;

  const outputs = state.outputNames.map((n) => results[n].data);
  const dets = decode(outputs, size, { scoreThreshold: threshold, nmsThreshold: 0.4 });

  // Ag girdisi uzayindan orijinal video piksel uzayina don
  for (const d of dets) {
    for (let i = 0; i < 4; i++) d.bbox[i] /= scale;
    for (const k of d.kps) {
      k[0] /= scale;
      k[1] /= scale;
    }
  }
  return { dets, elapsed };
}

// ---------------------------------------------------------------------------
// Cikarim dongusu — cizimden bagimsiz, kendi hizinda calisir
// ---------------------------------------------------------------------------

async function inferenceLoop() {
  let detTimes = [];
  while (state.running) {
    const video = state.video;
    if (!video || video.readyState < 2 || !video.videoWidth) {
      await new Promise((r) => setTimeout(r, 60));
      continue;
    }
    try {
      const size = Number(el.ctlSize.value);
      const threshold = Number(el.ctlThresh.value);
      const { dets, elapsed } = await detectOnce(video, size, threshold);
      state.detections = dets;
      state.inferMs = state.inferMs ? state.inferMs * 0.7 + elapsed * 0.3 : elapsed;

      const now = performance.now();
      detTimes.push(now);
      detTimes = detTimes.filter((t) => now - t < 1000);
      state.detPerSec = detTimes.length;
    } catch (err) {
      console.error("[demo] cikarim hatasi", err);
      showError("Tespit sırasında hata", err.message || String(err));
      state.running = false;
      return;
    }
    // Tarayiciya nefes aldir (UI donmasin)
    await new Promise((r) => setTimeout(r, 0));
  }
}

// ---------------------------------------------------------------------------
// Cizim
// ---------------------------------------------------------------------------

function layout(video) {
  const dpr = window.devicePixelRatio || 1;
  const cw = el.stage.clientWidth;
  const ch = el.stage.clientHeight;
  if (el.view.width !== Math.round(cw * dpr) || el.view.height !== Math.round(ch * dpr)) {
    el.view.width = Math.round(cw * dpr);
    el.view.height = Math.round(ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const vw = video.videoWidth || 4;
  const vh = video.videoHeight || 3;
  const s = Math.min(cw / vw, ch / vh);
  const dw = vw * s;
  const dh = vh * s;
  return { cw, ch, vw, vh, s, dx: (cw - dw) / 2, dy: (ch - dh) / 2, dw, dh };
}

function drawFrame() {
  if (!state.running) return;
  requestAnimationFrame(drawFrame);

  const video = state.video;
  if (!video || video.readyState < 2) return;

  const L = layout(video);
  const mirror = el.ctlMirror.checked;
  const showKps = el.ctlKps.checked;

  ctx.clearRect(0, 0, L.cw, L.ch);

  // Video karesi (istenirse aynalanmis)
  ctx.save();
  if (mirror) {
    ctx.translate(L.cw, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, mirror ? L.cw - L.dx - L.dw : L.dx, L.dy, L.dw, L.dh);
  ctx.restore();

  // Video pikselinden ekran pikseline (aynalama dahil)
  const toX = (x) => (mirror ? L.dx + L.dw - x * L.s : L.dx + x * L.s);
  const toY = (y) => L.dy + y * L.s;

  ctx.lineWidth = 2;
  ctx.font = "600 13px 'Space Grotesk', system-ui, sans-serif";
  ctx.textBaseline = "alphabetic";

  for (const d of state.detections) {
    const x1 = toX(d.bbox[0]);
    const x2 = toX(d.bbox[2]);
    const left = Math.min(x1, x2);
    const right = Math.max(x1, x2);
    const top = toY(d.bbox[1]);
    const bottom = toY(d.bbox[3]);
    const w = right - left;
    const h = bottom - top;

    ctx.strokeStyle = "#2dd4bf";
    ctx.strokeRect(left, top, w, h);

    // Koseler
    const c = Math.min(w, h) * 0.16;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.moveTo(left, top + c); ctx.lineTo(left, top); ctx.lineTo(left + c, top);
    ctx.moveTo(right - c, top); ctx.lineTo(right, top); ctx.lineTo(right, top + c);
    ctx.moveTo(left, bottom - c); ctx.lineTo(left, bottom); ctx.lineTo(left + c, bottom);
    ctx.moveTo(right - c, bottom); ctx.lineTo(right, bottom); ctx.lineTo(right, bottom - c);
    ctx.stroke();
    ctx.lineWidth = 2;

    if (showKps) {
      for (let i = 0; i < d.kps.length; i++) {
        ctx.fillStyle = KPS_COLORS[i] || "#2dd4bf";
        ctx.beginPath();
        ctx.arc(toX(d.kps[i][0]), toY(d.kps[i][1]), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const label = `yüz  %${Math.round(d.score * 100)}`;
    const tw = ctx.measureText(label).width + 14;
    const ly = top - 22 < 0 ? bottom + 4 : top - 22;
    ctx.fillStyle = "#2dd4bf";
    ctx.fillRect(left, ly, tw, 20);
    ctx.fillStyle = "#04120f";
    ctx.fillText(label, left + 7, ly + 14);
  }
}

// ---------------------------------------------------------------------------
// Istatistikler
// ---------------------------------------------------------------------------

let frameCount = 0;
let lastStatTick = performance.now();

function statsLoop() {
  if (!state.running) return;
  const now = performance.now();
  const dt = now - lastStatTick;
  el.statFps.textContent = `${Math.round((frameCount * 1000) / dt)} fps`;
  el.statInfer.textContent = state.inferMs ? `${Math.round(state.inferMs)} ms` : "—";
  el.statRate.textContent = `${state.detPerSec.toFixed(1)}/sn`;
  el.statFaces.textContent = String(state.detections.length);
  frameCount = 0;
  lastStatTick = now;
  setTimeout(statsLoop, 1000);
}

function countFrames() {
  if (!state.running) return;
  frameCount++;
  requestAnimationFrame(countFrames);
}

// ---------------------------------------------------------------------------
// Kamera
// ---------------------------------------------------------------------------

function showError(title, text) {
  el.errorTitle.textContent = title;
  el.errorText.textContent = text;
  el.error.classList.remove("hidden");
}

function cameraErrorMessage(err) {
  switch (err && err.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Kamera izni verilmedi. Tarayıcının adres çubuğundaki kamera simgesinden izin verip tekrar deneyebilirsiniz.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "Bu cihazda kamera bulunamadı.";
    case "NotReadableError":
      return "Kamera başka bir uygulama tarafından kullanılıyor olabilir. Diğer uygulamaları kapatıp tekrar deneyin.";
    case "OverconstrainedError":
      return "Kamera istenen çözünürlüğü desteklemiyor.";
    default:
      return err && err.message ? err.message : "Kamera açılamadı.";
  }
}

async function startCamera() {
  el.startBtn.disabled = true;
  el.error.classList.add("hidden");

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("Kamera desteklenmiyor", "Tarayıcınız kamera erişimini desteklemiyor veya sayfa güvenli olmayan bir bağlantı üzerinden açılmış (HTTPS gerekir).");
    el.startBtn.disabled = false;
    return;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    showError("Kamera açılamadı", cameraErrorMessage(err));
    el.startBtn.disabled = false;
    return;
  }

  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;
  video.srcObject = state.stream;
  await video.play();
  state.video = video;

  el.idle.classList.add("hidden");
  state.running = true;
  requestAnimationFrame(drawFrame);
  requestAnimationFrame(countFrames);
  statsLoop();
  inferenceLoop();
}

// ---------------------------------------------------------------------------
// Baslatma
// ---------------------------------------------------------------------------

el.ctlThresh.addEventListener("input", () => {
  el.ctlThreshVal.textContent = Number(el.ctlThresh.value).toFixed(2);
});

el.startBtn.addEventListener("click", startCamera);
el.retryBtn.addEventListener("click", () => {
  el.error.classList.add("hidden");
  if (!state.running) startCamera();
});

window.addEventListener("beforeunload", () => {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
});

el.startBtn.disabled = true;
initModel().catch((err) => {
  console.error("[demo] model baslatilamadi", err);
  el.modelNote.textContent = "Model yüklenemedi.";
  showError("Model yüklenemedi", err.message || String(err));
});
