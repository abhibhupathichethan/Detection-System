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
 * Smoothly upsamples the ResNet50 Grad-CAM activation grid and pre-bakes a cached
 * COLORMAP_JET offscreen canvas so opacity slider drags run at 60fps in <1ms.
 */
export function computeGradCamField(
  _img: HTMLImageElement,
  gradCamGrid: number[] | undefined,
  sourceGridSize = 14,
  isHealthy = false,
  targetSize = 112,
  threshold = 0.14
): GradCamField {
  const w = targetSize;
  const h = targetSize;
  const values = new Float32Array(w * h);

  if (!Array.isArray(gradCamGrid) || gradCamGrid.length === 0) {
    return { gridWidth: w, gridHeight: h, values, heatCanvas: null };
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
    // Smoothstep interpolation eliminates square grid edges
    const wy = ty * ty * (3 - 2 * ty);

    for (let x = 0; x < w; x++) {
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

      values[y * w + x] = Math.max(
        0,
        Math.min(1, isHealthy ? Math.min(0.34, interpolated) : interpolated)
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

    for (let i = 0; i < values.length; i++) {
      const val = values[i];
      const p = i * 4;
      if (val < threshold) {
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

      const ramp = Math.min(1, (val - threshold) / Math.max(0.08, 1 - threshold));
      const baseOpacity = 0.25 + 0.75 * Math.pow(ramp, 0.7);
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
