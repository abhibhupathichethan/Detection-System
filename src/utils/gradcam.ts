export function jetColorMap(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 1)));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export interface GradCamField {
  gridWidth: number;
  gridHeight: number;
  values: Float32Array;
  heatCanvas: HTMLCanvasElement | null;
}

/**
 * Lightweight exterior border flood-fill on a small targetSize x targetSize RGBA buffer
 * so Grad-CAM heatmaps never spill purple/blue halos onto white or studio backgrounds.
 */
function buildForegroundLeafMask(
  rgba: Uint8ClampedArray,
  w: number,
  h: number
): Uint8Array {
  const N = w * h;
  const bg = new Uint8Array(N);

  const isGreenWall = (idx: number): boolean => {
    const p = idx * 4;
    const r = rgba[p];
    const g = rgba[p + 1];
    const b = rgba[p + 2];
    const exG = (2 * g - r - b) / 255;
    return exG > 0.048 && g > b + 4 && g > r - 6;
  };

  const isStudioBgPixel = (idx: number): boolean => {
    if (isGreenWall(idx)) return false;
    const p = idx * 4;
    const a = rgba[p + 3];
    if (a < 32) return true;
    const r = rgba[p];
    const g = rgba[p + 1];
    const b = rgba[p + 2];
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
    const luma = (r + g + b) / 3;
    const exG = (2 * g - r - b) / 255;

    if (b >= g - 3 && r <= g + 4 && luma > 125 && sat < 0.30) return true;
    if (luma > 175 && sat < 0.24 && exG < 0.04 && g - b < 14 && r - b < 16) return true;
    if (luma > 125 && exG < 0.025 && r - g < 42 && sat < 0.45) return true;
    return false;
  };

  const queue = new Int32Array(N);
  let head = 0;
  let tail = 0;
  let borderNonLeaf = 0;

  const trySeed = (x: number, y: number) => {
    const idx = y * w + x;
    if (bg[idx] || !isStudioBgPixel(idx)) return;
    bg[idx] = 1;
    queue[tail++] = idx;
    borderNonLeaf++;
  };

  for (let x = 0; x < w; x++) {
    trySeed(x, 0);
    trySeed(x, h - 1);
  }
  for (let y = 1; y < h - 1; y++) {
    trySeed(0, y);
    trySeed(w - 1, y);
  }

  // Only mask studio/bokeh backgrounds when exterior border actually has non-foliage pixels
  if (borderNonLeaf < Math.round((w + h) * 0.15)) {
    const allFg = new Uint8Array(N);
    allFg.fill(1);
    return allFg;
  }

  const dx = [1, -1, 0, 0];
  const dy = [0, 0, 1, -1];
  while (head < tail) {
    const cur = queue[head++];
    const cx = cur % w;
    const cy = (cur - cx) / w;
    const cp = cur * 4;
    const cr = rgba[cp];
    const cg = rgba[cp + 1];
    const cb = rgba[cp + 2];

    for (let dir = 0; dir < 4; dir++) {
      const nx = cx + dx[dir];
      const ny = cy + dy[dir];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const nIdx = ny * w + nx;
      if (bg[nIdx] || !isStudioBgPixel(nIdx)) continue;
      const np = nIdx * 4;
      const dist = Math.hypot(rgba[np] - cr, rgba[np + 1] - cg, rgba[np + 2] - cb);
      const luma = (rgba[np] + rgba[np + 1] + rgba[np + 2]) / 3;
      const maxC = Math.max(rgba[np], rgba[np + 1], rgba[np + 2]);
      const minC = Math.min(rgba[np], rgba[np + 1], rgba[np + 2]);
      const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
      const exG = (2 * rgba[np + 1] - rgba[np] - rgba[np + 2]) / 255;
      if (dist < 26 || (luma > 185 && sat < 0.22) || (luma > 125 && exG < 0.025 && dist < 22)) {
        bg[nIdx] = 1;
        queue[tail++] = nIdx;
      }
    }
  }

  const fg = new Uint8Array(N);
  let fgCount = 0;
  for (let i = 0; i < N; i++) {
    fg[i] = bg[i] ? 0 : 1;
    fgCount += fg[i];
  }
  if (fgCount < N * 0.08) {
    fg.fill(1);
  }
  return fg;
}

/**
 * Smoothly upsamples the ResNet50 Grad-CAM activation grid and pre-bakes a cached
 * leaf-masked COLORMAP_JET offscreen canvas so opacity slider drags run at 60fps in <1ms.
 */
export function computeGradCamField(
  img: HTMLImageElement,
  gradCamGrid: number[] | undefined,
  sourceGridSize = 28,
  isHealthy = false,
  targetSize = 168,
  threshold = 0.20
): GradCamField {
  const w = targetSize;
  const h = targetSize;
  const values = new Float32Array(w * h);

  if (!Array.isArray(gradCamGrid) || gradCamGrid.length === 0) {
    return { gridWidth: w, gridHeight: h, values, heatCanvas: null };
  }

  // Extract pixel-aligned foreground leaf mask directly from the rendered RGB image
  let fgMask: Uint8Array | null = null;
  try {
    const sampleCanvas = document.createElement('canvas');
    sampleCanvas.width = w;
    sampleCanvas.height = h;
    const sCtx = sampleCanvas.getContext('2d');
    if (sCtx) {
      sCtx.fillStyle = '#ffffff';
      sCtx.fillRect(0, 0, w, h);
      sCtx.drawImage(img, 0, 0, w, h);
      const imgData = sCtx.getImageData(0, 0, w, h);
      fgMask = buildForegroundLeafMask(imgData.data, w, h);
    }
  } catch {
    fgMask = null;
  }

  const gSize = Math.max(
    1,
    Math.round(sourceGridSize || Math.sqrt(gradCamGrid.length))
  );

  for (let y = 0; y < h; y++) {
    const fy = Math.max(
      0,
      Math.min(gSize - 1, ((y + 0.5) / h) * gSize - 0.5)
    );
    const y0 = Math.floor(fy);
    const y1 = Math.min(gSize - 1, y0 + 1);
    const ty = fy - y0;
    const wy = ty * ty * (3 - 2 * ty);

    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (fgMask && fgMask[idx] === 0) {
        values[idx] = 0;
        continue;
      }

      const fx = Math.max(
        0,
        Math.min(gSize - 1, ((x + 0.5) / w) * gSize - 0.5)
      );
      const x0 = Math.floor(fx);
      const x1 = Math.min(gSize - 1, x0 + 1);
      const tx = fx - x0;
      const wx = tx * tx * (3 - 2 * tx);

      const v00 = gradCamGrid[y0 * gSize + x0] ?? 0;
      const v10 = gradCamGrid[y0 * gSize + x1] ?? 0;
      const v01 = gradCamGrid[y1 * gSize + x0] ?? 0;
      const v11 = gradCamGrid[y1 * gSize + x1] ?? 0;

      const top = v00 * (1 - wx) + v10 * wx;
      const bottom = v01 * (1 - wx) + v11 * wx;
      const interpolated = top * (1 - wy) + bottom * wy;

      values[idx] = Math.max(
        0,
        Math.min(1, isHealthy ? Math.min(0.28, interpolated) : interpolated)
      );
    }
  }

  // Pre-render the COLORMAP_JET layer once per analysis
  const heatCanvas = document.createElement('canvas');
  heatCanvas.width = w;
  heatCanvas.height = h;
  const heatCtx = heatCanvas.getContext('2d');
  if (heatCtx) {
    const heatImg = heatCtx.createImageData(w, h);
    const { data } = heatImg;
    const effThresh = isHealthy ? 0.16 : threshold;

    for (let i = 0; i < values.length; i++) {
      const val = values[i];
      const p = i * 4;
      if (val < effThresh || (fgMask && fgMask[i] === 0)) {
        data[p] = 0;
        data[p + 1] = 0;
        data[p + 2] = 0;
        data[p + 3] = 0;
        continue;
      }

      const [r, g, b] = jetColorMap(val);
      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;

      const ramp = Math.min(1, (val - effThresh) / Math.max(0.08, 1 - effThresh));
      const baseOpacity = isHealthy
        ? 0.18 + 0.35 * ramp
        : 0.22 + 0.78 * Math.pow(ramp, 0.75);
      data[p + 3] = Math.round(baseOpacity * 255);
    }

    heatCtx.putImageData(heatImg, 0, 0);
  }

  return { gridWidth: w, gridHeight: h, values, heatCanvas };
}

export function clearGradCamCanvas(
  canvas: HTMLCanvasElement,
  img?: HTMLImageElement | null
): void {
  const natW = img?.naturalWidth || 512;
  const natH = img?.naturalHeight || 512;
  const maxDim = 512;
  const scale = maxDim / Math.max(natW, natH);
  const cw = Math.max(1, Math.round(natW * scale));
  const ch = Math.max(1, Math.round(natH * scale));
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, cw, ch);
  if (img) {
    ctx.drawImage(img, 0, 0, cw, ch);
  }
}

export function renderGradCamToCanvas(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  field: GradCamField,
  alpha: number
): void {
  const natW = img.naturalWidth || 512;
  const natH = img.naturalHeight || 512;
  const maxDim = 512;
  const scale = maxDim / Math.max(natW, natH);
  const cw = Math.max(1, Math.round(natW * scale));
  const ch = Math.max(1, Math.round(natH * scale));

  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw;
    canvas.height = ch;
  }

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(img, 0, 0, cw, ch);

  if (field.heatCanvas) {
    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(field.heatCanvas, 0, 0, cw, ch);
    ctx.restore();
  }
}

export function buildCompressedThumbnail(img: HTMLImageElement, size = 256): string {
  const natW = img.naturalWidth || size;
  const natH = img.naturalHeight || size;
  const scale = size / Math.max(natW, natH);
  const w = Math.max(1, Math.round(natW * scale));
  const h = Math.max(1, Math.round(natH * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  ctx.drawImage(img, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.8);
}

export function buildCompressedGradCamThumbnail(
  sourceCanvas: HTMLCanvasElement,
  size = 256
): string {
  const sw = sourceCanvas.width || size;
  const sh = sourceCanvas.height || size;
  const scale = size / Math.max(sw, sh);
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  ctx.drawImage(sourceCanvas, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.82);
}
