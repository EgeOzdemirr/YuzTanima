/**
 * SCRFD (InsightFace det_10g) cikti cozumlemesi.
 *
 * Bu modul, Python tarafindaki insightface implementasyonunun birebir
 * portudur ve `tools/demo/verify_decode.mjs` ile Python referansina karsi
 * dogrulanir. Hem tarayicida hem Node'da calisir (saf ESM, DOM bagimsiz).
 *
 * Model 9 cikti verir (stride 8/16/32 sirasiyla):
 *   [0..2] skor      (N, 1)
 *   [3..5] bbox      (N, 4)   -- merkeze uzaklik, stride ile carpilir
 *   [6..8] landmark  (N, 10)  -- 5 nokta, merkeze uzaklik
 */

export const STRIDES = [8, 16, 32];
export const NUM_ANCHORS = 2;

const anchorCache = new Map();

/**
 * Bir stride icin anchor merkezlerini uretir (x, y ikilileri).
 * Python karsiligi: np.stack(np.mgrid[:h,:w][::-1], -1) * stride
 */
export function anchorCenters(inputSize, stride) {
  const key = `${inputSize}:${stride}`;
  const cached = anchorCache.get(key);
  if (cached) return cached;

  const rows = Math.floor(inputSize / stride);
  const cols = Math.floor(inputSize / stride);
  const centers = new Float32Array(rows * cols * NUM_ANCHORS * 2);
  let p = 0;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      // Ayni konum NUM_ANCHORS kez tekrarlanir (np.stack([c]*n, axis=1))
      for (let a = 0; a < NUM_ANCHORS; a++) {
        centers[p++] = x * stride;
        centers[p++] = y * stride;
      }
    }
  }
  anchorCache.set(key, centers);
  return centers;
}

/** IoU tabanli NMS. insightface ile ayni: alan hesabinda +1 var. */
export function nms(dets, threshold) {
  const order = dets.map((d, i) => i).sort((a, b) => dets[b].score - dets[a].score);
  const areas = dets.map((d) => (d.bbox[2] - d.bbox[0] + 1) * (d.bbox[3] - d.bbox[1] + 1));
  const keep = [];

  const suppressed = new Uint8Array(dets.length);
  for (let oi = 0; oi < order.length; oi++) {
    const i = order[oi];
    if (suppressed[i]) continue;
    keep.push(i);
    for (let oj = oi + 1; oj < order.length; oj++) {
      const j = order[oj];
      if (suppressed[j]) continue;
      const xx1 = Math.max(dets[i].bbox[0], dets[j].bbox[0]);
      const yy1 = Math.max(dets[i].bbox[1], dets[j].bbox[1]);
      const xx2 = Math.min(dets[i].bbox[2], dets[j].bbox[2]);
      const yy2 = Math.min(dets[i].bbox[3], dets[j].bbox[3]);
      const w = Math.max(0, xx2 - xx1 + 1);
      const h = Math.max(0, yy2 - yy1 + 1);
      const inter = w * h;
      const ovr = inter / (areas[i] + areas[j] - inter);
      if (ovr > threshold) suppressed[j] = 1;
    }
  }
  return keep;
}

/**
 * Ham model ciktilarini yuz tespitlerine cevirir.
 *
 * @param {Float32Array[]} outputs 9 elemanli dizi (skor x3, bbox x3, kps x3)
 * @param {number} inputSize aga verilen kare boyutu (orn. 640)
 * @param {{scoreThreshold?: number, nmsThreshold?: number}} [opts]
 * @returns {{bbox: number[], score: number, kps: number[][]}[]}
 *          Koordinatlar ag girdisi uzayindadir; orijinal goruntuye donmek
 *          icin kullanilan olcege bolun.
 */
export function decode(outputs, inputSize, opts = {}) {
  const scoreThreshold = opts.scoreThreshold ?? 0.5;
  const nmsThreshold = opts.nmsThreshold ?? 0.4;
  const candidates = [];

  for (let level = 0; level < STRIDES.length; level++) {
    const stride = STRIDES[level];
    const scores = outputs[level];
    const bboxPreds = outputs[level + 3];
    const kpsPreds = outputs[level + 6];
    const centers = anchorCenters(inputSize, stride);
    const count = scores.length;

    for (let i = 0; i < count; i++) {
      const score = scores[i];
      if (score < scoreThreshold) continue;

      const cx = centers[i * 2];
      const cy = centers[i * 2 + 1];

      // distance2bbox: sol/ust cikarilir, sag/alt eklenir
      const b = i * 4;
      const bbox = [
        cx - bboxPreds[b] * stride,
        cy - bboxPreds[b + 1] * stride,
        cx + bboxPreds[b + 2] * stride,
        cy + bboxPreds[b + 3] * stride,
      ];

      // distance2kps: her nokta icin merkeze mesafe eklenir
      const k = i * 10;
      const kps = [];
      for (let j = 0; j < 5; j++) {
        kps.push([
          cx + kpsPreds[k + j * 2] * stride,
          cy + kpsPreds[k + j * 2 + 1] * stride,
        ]);
      }

      candidates.push({ bbox, score, kps });
    }
  }

  if (!candidates.length) return [];
  const keep = nms(candidates, nmsThreshold);
  return keep.map((i) => candidates[i]);
}

/**
 * En-boy oranini koruyan olcek katsayisi (sol-uste hizalanmis dolgu).
 * Python karsiligi: min(size/w, size/h)
 */
export function fitScale(width, height, inputSize) {
  return Math.min(inputSize / width, inputSize / height);
}

/**
 * Canvas RGBA piksel dizisini modelin bekledigi NCHW float tensorune cevirir.
 *
 * Python karsiligi:
 *   cv2.dnn.blobFromImage(bgr, 1/128, (size,size), (127.5,)*3, swapRB=True)
 * yani aga RGB kanallari, (v - 127.5) / 128 ile olceklenmis olarak gider.
 *
 * @param {Uint8ClampedArray|Uint8Array} rgba getImageData().data
 * @param {number} size kare kenar uzunlugu
 * @returns {Float32Array} uzunluk 3*size*size, kanal-once duzen
 */
export function rgbaToNchw(rgba, size) {
  const plane = size * size;
  const tensor = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    tensor[i] = (rgba[p] - 127.5) / 128;
    tensor[i + plane] = (rgba[p + 1] - 127.5) / 128;
    tensor[i + plane * 2] = (rgba[p + 2] - 127.5) / 128;
  }
  return tensor;
}
