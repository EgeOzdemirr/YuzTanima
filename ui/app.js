const matchList = document.getElementById("match-list");
const galleryLabel = document.getElementById("gallery-id");
const captureLabel = document.getElementById("capture-id");
const liveVideo = document.getElementById("live-video");
const liveMjpeg = document.getElementById("live-mjpeg");
const statusTag = document.getElementById("status-tag");
const viewerCanvas = document.getElementById("viewer-canvas");
const cameraSelect = document.getElementById("camera-source");
const CAMERA_STREAM_BASE_URL = "http://localhost:9081";
const WEBRTC_SIGNAL_URL = `${CAMERA_STREAM_BASE_URL}/offer`;
const MJPEG_STREAM_URL = `${CAMERA_STREAM_BASE_URL}/stream.mjpg`;
const CLEAR_STATE_KEY = "awatar_clear_state_v2";

function logCamera(message, data) {
  if (data !== undefined) {
    console.log(`[camera] ${message}`, data);
  } else {
    console.log(`[camera] ${message}`);
  }
}

const MATCH_HOLD_MS = 7000; // tanınan kişi ekrandan çıktıktan sonra panelde bu kadar tutulur

let lastGlb = null;
let lastMatch = null;
let currentMatches = [];
let selectedPersonId = null;
let renderedMatchKey = null;
const personRegistry = new Map(); // personId -> { person, lastSeen }
let lastProcessedEventTs = 0;
let lastLook = { x: 0, y: 0 };
let viewer = null;
let peerConnection = null;
let lastEventKey = null;
let clearedAt = 0;
let clearedEventKey = null;

function loadClearState() {
  try {
    const raw = localStorage.getItem(CLEAR_STATE_KEY);
    if (!raw) return;
    const state = JSON.parse(raw);
    const ts = Number(state?.clearedAt || 0);
    clearedAt = Number.isFinite(ts) ? ts : 0;
    clearedEventKey = typeof state?.clearedEventKey === "string" ? state.clearedEventKey : null;
  } catch (err) {
    console.warn("clear state load failed", err);
  }
}

function saveClearState() {
  try {
    localStorage.setItem(
      CLEAR_STATE_KEY,
      JSON.stringify({
        clearedAt,
        clearedEventKey,
      })
    );
  } catch (err) {
    console.warn("clear state save failed", err);
  }
}

async function purgeBrowserCaches() {
  try {
    if ("caches" in window) {
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => caches.delete(k)));
    }
  } catch (err) {
    console.warn("cache storage cleanup failed", err);
  }
  try {
    sessionStorage.clear();
  } catch (err) {
    console.warn("sessionStorage cleanup failed", err);
  }
  try {
    const keep = localStorage.getItem(CLEAR_STATE_KEY);
    localStorage.clear();
    if (keep != null) {
      localStorage.setItem(CLEAR_STATE_KEY, keep);
    }
  } catch (err) {
    console.warn("localStorage cleanup failed", err);
  }
}

loadClearState();

function getEventKey(data) {
  return data?.captureId || data?.timestamp || data?.matchedPersonId || null;
}

function getEventTs(data) {
  const ts = Date.parse(data?.timestamp || "");
  return Number.isNaN(ts) ? 0 : ts;
}

function setStatus(status) {
  if (!statusTag) return;
  if (status === "SUSPICIOUS") {
    statusTag.classList.remove("hidden");
    statusTag.textContent = "ŞÜPHELİ";
  } else {
    statusTag.classList.add("hidden");
    statusTag.textContent = "";
  }
}

async function initViewer() {
  if (!viewerCanvas) return;
  try {
    const mod = await import("./viewer.js");
    viewer = new mod.GLBViewer(viewerCanvas);
    if (lastGlb && lastMatch) {
      viewer.load(lastGlb);
    }
    logCamera("GLB viewer initialized");
  } catch (err) {
    console.error("viewer init error", err);
  }
}

initViewer();

function galleryAssetPath(personId, filename) {
  if (!personId) return null;
  return `/public/gallery/persons/${encodeURIComponent(personId)}/${filename}`;
}

function attachPhotoWithFallback(img, personId, primaryUrl) {
  const candidates = [
    primaryUrl,
    galleryAssetPath(personId, "photo.jpg"),
    galleryAssetPath(personId, "photo.jpeg"),
    galleryAssetPath(personId, "photo.png"),
    galleryAssetPath(personId, "photo.webp"),
  ].filter(Boolean);
  let idx = 0;
  img.onerror = () => {
    idx += 1;
    if (idx >= candidates.length) {
      img.onerror = null;
      img.removeAttribute("src");
      return;
    }
    img.src = candidates[idx];
  };
  if (candidates.length) {
    img.src = `${candidates[0]}?t=${Date.now()}`;
  }
}

function personsFromEvent(data) {
  if (Array.isArray(data?.persons)) return data.persons;
  if (!data) return [];
  // Legacy single-person payload support.
  return [
    {
      trackId: null,
      matchedPersonId: data.matchedPersonId || null,
      similarity: data.similarity || 0,
      captureId: data.captureId || null,
      galleryPhotoPath: data.galleryPhotoPath || null,
      galleryFace3dPath: data.galleryFace3dPath || null,
      bbox: data.bbox || null,
    },
  ];
}

function formatSimilarity(similarity) {
  return `Benzerlik: %${Math.round((similarity || 0) * 100)}`;
}

function renderMatches() {
  if (!matchList) return;
  const key = `${currentMatches.map((p) => p.matchedPersonId).join("|")}@${selectedPersonId}`;
  if (key === renderedMatchKey) {
    // Same people on screen; only refresh the similarity numbers in place.
    matchList.querySelectorAll(".match-card").forEach((card) => {
      const person = currentMatches.find((p) => p.matchedPersonId === card.dataset.personId);
      const sim = card.querySelector(".match-sim");
      if (person && sim) sim.textContent = formatSimilarity(person.similarity);
    });
    return;
  }
  renderedMatchKey = key;
  matchList.innerHTML = "";
  if (!currentMatches.length) {
    const ph = document.createElement("div");
    ph.className = "placeholder match-placeholder";
    ph.textContent = "Eşleşme bekleniyor";
    matchList.appendChild(ph);
    return;
  }
  for (const p of currentMatches) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "match-card" + (p.matchedPersonId === selectedPersonId ? " selected" : "");
    card.dataset.personId = p.matchedPersonId;
    const img = document.createElement("img");
    img.alt = p.matchedPersonId;
    attachPhotoWithFallback(img, p.matchedPersonId, p.galleryPhotoPath);
    const info = document.createElement("div");
    info.className = "match-info";
    const name = document.createElement("div");
    name.className = "match-name";
    name.textContent = p.matchedPersonId;
    const sim = document.createElement("div");
    sim.className = "match-sim";
    sim.textContent = formatSimilarity(p.similarity);
    info.append(name, sim);
    card.append(img, info);
    card.addEventListener("click", () => {
      selectedPersonId = p.matchedPersonId;
      renderMatches();
      syncViewer();
    });
    matchList.appendChild(card);
  }
}

function syncViewer() {
  const sel = currentMatches.find((p) => p.matchedPersonId === selectedPersonId) || null;
  if (!sel) {
    if (lastMatch || lastGlb) {
      lastMatch = null;
      lastGlb = null;
      if (viewer) viewer.cancelLoad(true);
    }
    return;
  }
  const glb = sel.galleryFace3dPath || galleryAssetPath(sel.matchedPersonId, "face.glb");
  if (sel.matchedPersonId !== lastMatch) {
    lastMatch = sel.matchedPersonId;
    lastGlb = null;
    if (viewer) viewer.cancelLoad(true);
  }
  if (!glb) {
    lastGlb = null;
    if (viewer) viewer.cancelLoad(true);
  } else if (glb !== lastGlb) {
    lastGlb = glb;
    if (viewer) viewer.load(glb);
  } else if (viewer && viewer.loadedUrl !== glb && viewer.loadingUrl !== glb) {
    // Retry only if same matched person is still selected but previous load was canceled/failed.
    viewer.load(glb);
  }
}

function updateUI(evt) {
  const data = evt;
  const eventKey = getEventKey(data);
  const eventTs = getEventTs(data);
  const now = Date.now();

  // Only a fresh event (new timestamp) refreshes the registry; a stale file
  // (person left the frame, pipeline stopped writing) must not keep people alive.
  const isNewEvent = eventTs > lastProcessedEventTs;
  const clearedByOperator = clearedAt && eventTs && eventTs <= clearedAt;
  if (isNewEvent && !clearedByOperator) {
    lastProcessedEventTs = eventTs;
    for (const p of personsFromEvent(data)) {
      if (!p?.matchedPersonId) continue;
      personRegistry.set(p.matchedPersonId, { person: p, lastSeen: now });
    }
  }

  // Kişi ekrandan çıktıktan MATCH_HOLD_MS sonra panelden düşür.
  for (const [personId, entry] of personRegistry) {
    if (now - entry.lastSeen > MATCH_HOLD_MS) {
      personRegistry.delete(personId);
    }
  }

  const matched = [...personRegistry.values()]
    .map((entry) => entry.person)
    .sort((a, b) => (b.similarity || 0) - (a.similarity || 0));
  const hasGalleryMatch = matched.length > 0;

  setStatus(hasGalleryMatch ? "SUSPICIOUS" : "CLEAR");
  galleryLabel.textContent = hasGalleryMatch ? `${matched.length} kişi` : "-";
  captureLabel.textContent = data.captureId || matched[0]?.captureId || "-";

  currentMatches = matched;
  if (!hasGalleryMatch || !personRegistry.has(selectedPersonId)) {
    selectedPersonId = hasGalleryMatch ? matched[0].matchedPersonId : null;
  }
  renderMatches();
  syncViewer();

  const selected = currentMatches.find((p) => p.matchedPersonId === selectedPersonId) || null;
  const bbox = selected?.bbox || data.bbox;
  if (data.frameSize && data.frameSize.length === 2 && bbox && bbox.length === 4) {
    const [w, h] = data.frameSize;
    const [x1, y1, x2, y2] = bbox;
    if (w > 0 && h > 0) {
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const nx = (cx / w) * 2 - 1; // -1..1
      const ny = (cy / h) * 2 - 1;
      lastLook = { x: nx, y: ny };
      if (viewer) viewer.setLookOffset(nx, ny);
    }
  }
  lastEventKey = eventKey;
  if (clearedEventKey && eventKey && eventKey !== clearedEventKey) {
    clearedEventKey = null;
  }
}

async function poll() {
  try {
    const res = await fetch(`/data/events/latest_suspicious.json?ts=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const evt = await res.json();
    updateUI(evt);
  } catch (err) {
    console.error("poll error", err);
  } finally {
    setTimeout(poll, 1000);
  }
}

poll();

function stopWebRTC() {
  if (peerConnection) {
    logCamera("Closing WebRTC connection");
    peerConnection.close();
    peerConnection = null;
  }
}

function stopMjpegReceiver() {
  if (!liveMjpeg) return;
  liveMjpeg.onerror = null;
  liveMjpeg.onload = null;
  liveMjpeg.removeAttribute("src");
  liveMjpeg.hidden = true;
}

function startMjpegReceiver() {
  if (!liveMjpeg) {
    disableLiveCameraUI();
    return;
  }
  stopWebRTC();
  if (liveVideo) {
    liveVideo.pause();
    liveVideo.srcObject = null;
    liveVideo.hidden = true;
  }
  liveMjpeg.hidden = false;
  liveMjpeg.onload = () => logCamera("MJPEG stream connected");
  liveMjpeg.onerror = () => {
    logCamera("MJPEG stream failed");
    disableLiveCameraUI();
  };
  liveMjpeg.src = `${MJPEG_STREAM_URL}?ts=${Date.now()}`;
}

function waitForIceGathering(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const onStateChange = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", onStateChange);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", onStateChange);
  });
}

async function startWebRTCReceiver() {
  if (!window.RTCPeerConnection || !liveVideo) {
    disableLiveCameraUI();
    return;
  }
  try {
    stopMjpegReceiver();
    liveVideo.hidden = false;
    stopWebRTC();
    const pc = new RTCPeerConnection();
    peerConnection = pc;
    pc.oniceconnectionstatechange = () => logCamera("ICE state", pc.iceConnectionState);
    pc.onconnectionstatechange = () => logCamera("Peer state", pc.connectionState);
    pc.ontrack = (event) => {
      const [stream] = event.streams;
      if (stream) {
        liveVideo.srcObject = stream;
      } else {
        const fallback = new MediaStream([event.track]);
        liveVideo.srcObject = fallback;
      }
      liveVideo.onloadedmetadata = () => liveVideo.play();
      logCamera("Remote video track attached");
    };
    pc.addTransceiver("video", { direction: "recvonly" });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIceGathering(pc);

    const res = await fetch(WEBRTC_SIGNAL_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pc.localDescription),
    });
    if (!res.ok) throw new Error(`signal http ${res.status}`);
    const answer = await res.json();
    await pc.setRemoteDescription(answer);
    logCamera("WebRTC connected");
  } catch (err) {
    console.error("webrtc start error", err);
    logCamera("WebRTC start failed", { name: err?.name, message: err?.message });
    disableLiveCameraUI();
  }
}

async function clearFrozen() {
  clearedAt = Date.now();
  clearedEventKey = lastEventKey;
  await purgeBrowserCaches();
  saveClearState();
  lastMatch = null;
  lastGlb = null;
  currentMatches = [];
  personRegistry.clear();
  selectedPersonId = null;
  renderedMatchKey = null;
  renderMatches();
  setStatus("");
  galleryLabel.textContent = "-";
  captureLabel.textContent = "-";
  if (viewer) viewer.cancelLoad(true);
}

window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "c") {
    void clearFrozen();
  }
});

function setupCameraSelect() {
  if (!cameraSelect) return;
  cameraSelect.innerHTML = "";
  const opt = document.createElement("option");
  opt.value = "opencv";
  opt.textContent = "OpenCV Camera";
  cameraSelect.appendChild(opt);
  cameraSelect.disabled = true;
}

function parseUiConfig(yamlText) {
  const lines = yamlText.split(/\r?\n/);
  let inUi = false;
  const cfg = {
    liveCamera: true,
    liveStreamTransport: "mjpeg",
  };
  for (const line of lines) {
    const raw = line.replace(/\t/g, "  ");
    const trimmed = raw.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const indent = raw.match(/^\s*/)?.[0]?.length || 0;
    if (indent === 0) {
      inUi = trimmed === "ui:";
      continue;
    }
    if (inUi && trimmed.startsWith("live_camera:")) {
      const val = trimmed.split(":")[1]?.trim().toLowerCase();
      if (val) {
        cfg.liveCamera = !["false", "0", "no", "off"].includes(val);
      }
      continue;
    }
    if (inUi && trimmed.startsWith("live_stream_transport:")) {
      const val = trimmed.split(":")[1]?.trim().replace(/^["']|["']$/g, "").toLowerCase();
      if (val) {
        cfg.liveStreamTransport = val;
      }
    }
  }
  return cfg;
}

async function loadUiConfig() {
  try {
    const res = await fetch(`/services/camera_runtime/config.yaml?ts=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) return { liveCamera: true, liveStreamTransport: "mjpeg" };
    const text = await res.text();
    return parseUiConfig(text);
  } catch (err) {
    console.error("config load error", err);
    return { liveCamera: true, liveStreamTransport: "mjpeg" };
  }
}

function disableLiveCameraUI() {
  stopWebRTC();
  stopMjpegReceiver();
  if (liveVideo) {
    liveVideo.pause();
    liveVideo.srcObject = null;
    liveVideo.hidden = true;
  }
  const liveCard = liveVideo?.closest(".live-card") || liveMjpeg?.closest(".live-card");
  if (liveCard) {
    liveCard.style.display = "none";
  }
  document.body.classList.add("live-off");
  if (cameraSelect) {
    cameraSelect.disabled = true;
  }
}

async function boot() {
  setupCameraSelect();
  const cfg = await loadUiConfig();
  if (cfg.liveCamera === false) {
    disableLiveCameraUI();
    return;
  }
  if (cfg.liveStreamTransport === "webrtc") {
    startWebRTCReceiver();
  } else {
    startMjpegReceiver();
  }
}

boot();
