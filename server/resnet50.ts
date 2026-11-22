import fs from 'fs';
import path from 'path';
import JSZip from 'jszip';
import sharp from 'sharp';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import {
  PLANT_VILLAGE_CLASSES,
  PLANT_VILLAGE_CLASS_MAP,
} from '../src/data/plantVillage.ts';

interface NpyMeta {
  shape: number[];
  data: Float32Array;
}

interface ConvWeights {
  kernel: tf.Tensor4D;
  bias: tf.Tensor1D;
}

interface BnWeights {
  scale: tf.Tensor1D; // gamma / sqrt(var + eps)
  offset: tf.Tensor1D; // beta - mean * scale
}

interface BottleneckBlockWeights {
  conv1: ConvWeights;
  bn1: BnWeights;
  conv2: ConvWeights;
  bn2: BnWeights;
  conv3: ConvWeights;
  bn3: BnWeights;
  conv0?: ConvWeights;
  bn0?: BnWeights;
}

interface ResNet50ModelWeights {
  stemConv: ConvWeights;
  stemBn: BnWeights;
  stages: BottleneckBlockWeights[][];
  headDenseKernel: tf.Tensor2D;
  headDenseBias: tf.Tensor1D;
  predKernel: tf.Tensor2D;
  predBias: tf.Tensor1D;
  headDenseKernelArr: Float32Array;
  predKernelArr: Float32Array;
}

export interface ResNetPredictionOutput {
  predictedClass: string;
  readablePrediction: string;
  crop: string;
  conditionName: string;
  pathogen: string;
  isHealthy: boolean;
  confidence: number;
  topPredictions: Array<{ classId: string; probability: number }>;
  gradCamGrid: number[]; // 7x7 flattened row-major [0..1]
  gradCamGridSize: number;
  leafCoverageEstimate: number;
  visualObservations: string[];
  inferenceMs: number;
  imageWidth: number;
  imageHeight: number;
}

const BN_EPSILON = 1.001e-5;
const NUM_CLASSES = 38;

let cachedWeightsPromise: Promise<ResNet50ModelWeights> | null = null;

function parseNpyFloat32(buffer: Buffer): NpyMeta {
  const major = buffer[6];
  let headerLen: number;
  let dataStart: number;
  if (major === 1) {
    headerLen = buffer.readUInt16LE(8);
    dataStart = 10 + headerLen;
  } else {
    headerLen = buffer.readUInt32LE(8);
    dataStart = 12 + headerLen;
  }

  const headerStr = buffer.subarray(major === 1 ? 10 : 12, dataStart).toString('latin1');
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
  const shape = shapeMatch
    ? shapeMatch[1]
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .map((n) => Number(n))
    : [];

  const rawSlice = buffer.subarray(dataStart);
  const alignedCopy = new Uint8Array(rawSlice.length);
  alignedCopy.set(rawSlice);
  const data = new Float32Array(alignedCopy.buffer);

  return { shape, data };
}

function makeConv(arrays: NpyMeta[], idx: number): { conv: ConvWeights; next: number } {
  const k = arrays[idx];
  const b = arrays[idx + 1];
  const kernel = tf.tensor4d(
    k.data,
    k.shape as [number, number, number, number]
  );
  const bias = tf.tensor1d(b.data);
  return { conv: { kernel, bias }, next: idx + 2 };
}

function makeBn(arrays: NpyMeta[], idx: number): { bn: BnWeights; next: number } {
  const gamma = arrays[idx].data;
  const beta = arrays[idx + 1].data;
  const mean = arrays[idx + 2].data;
  const variance = arrays[idx + 3].data;

  const len = gamma.length;
  const scaleArr = new Float32Array(len);
  const offsetArr = new Float32Array(len);

  // Pre-fold BatchNorm inference: y = x * scale + offset
  for (let i = 0; i < len; i++) {
    const s = gamma[i] / Math.sqrt(variance[i] + BN_EPSILON);
    scaleArr[i] = s;
    offsetArr[i] = beta[i] - mean[i] * s;
  }

  return {
    bn: {
      scale: tf.tensor1d(scaleArr),
      offset: tf.tensor1d(offsetArr),
    },
    next: idx + 4,
  };
}

function parseBlock(
  arrays: NpyMeta[],
  startIdx: number,
  hasShortcut: boolean
): { block: BottleneckBlockWeights; next: number } {
  let idx = startIdx;
  const c1 = makeConv(arrays, idx);
  idx = c1.next;
  const b1 = makeBn(arrays, idx);
  idx = b1.next;

  const c2 = makeConv(arrays, idx);
  idx = c2.next;
  const b2 = makeBn(arrays, idx);
  idx = b2.next;

  let conv0: ConvWeights | undefined;
  let bn0: BnWeights | undefined;
  if (hasShortcut) {
    const c0 = makeConv(arrays, idx);
    conv0 = c0.conv;
    idx = c0.next;
  }

  const c3 = makeConv(arrays, idx);
  idx = c3.next;

  if (hasShortcut) {
    const b0 = makeBn(arrays, idx);
    bn0 = b0.bn;
    idx = b0.next;
  }

  const b3 = makeBn(arrays, idx);
  idx = b3.next;

  return {
    block: {
      conv1: c1.conv,
      bn1: b1.bn,
      conv2: c2.conv,
      bn2: b2.bn,
      conv3: c3.conv,
      bn3: b3.bn,
      conv0,
      bn0,
    },
    next: idx,
  };
}

async function initBackend(): Promise<void> {
  try {
    const wasmDir = path.resolve(
      process.cwd(),
      'node_modules/@tensorflow/tfjs-backend-wasm/dist/'
    );
    if (fs.existsSync(wasmDir)) {
      setWasmPaths(`file://${wasmDir}/`);
    }
    await tf.setBackend('wasm');
    await tf.ready();
  } catch {
    await tf.setBackend('cpu');
    await tf.ready();
  }
}

const WEIGHTS_URL =
  'https://raw.githubusercontent.com/Ishaaq09/Automated_plant_disease_detection_using_Deep_Learning_and_Transfer_Learning/main/resnet_weights.npz';

async function ensureWeightsFile(weightsPath: string): Promise<void> {
  if (fs.existsSync(weightsPath) && fs.statSync(weightsPath).size > 50_000_000) {
    return;
  }
  fs.mkdirSync(path.dirname(weightsPath), { recursive: true });
  console.log('[ResNet50] Downloading trained ResNet50 weights (~93 MB)...');
  const res = await fetch(WEIGHTS_URL);
  if (!res.ok) {
    throw new Error(
      `Failed to download ResNet50 weights (${res.status} ${res.statusText}).`
    );
  }
  const buf = Buffer.from(await res.arrayBuffer());
  const tmpPath = `${weightsPath}.tmp`;
  fs.writeFileSync(tmpPath, buf);
  fs.renameSync(tmpPath, weightsPath);
  console.log('[ResNet50] Weights downloaded to', weightsPath);
}

export async function loadResNet50Model(): Promise<ResNet50ModelWeights> {
  if (cachedWeightsPromise) {
    return cachedWeightsPromise;
  }

  cachedWeightsPromise = (async () => {
    try {
      await initBackend();

      const weightsPath = path.resolve(process.cwd(), 'models/resnet_weights.npz');
      await ensureWeightsFile(weightsPath);

      const zipBuffer = fs.readFileSync(weightsPath);
      const zip = await JSZip.loadAsync(zipBuffer);

      const arrays: NpyMeta[] = [];
      for (let i = 0; i < 322; i++) {
        const entry = zip.file(`arr_${i}.npy`);
        if (!entry) {
          throw new Error(`Missing arr_${i}.npy inside resnet_weights.npz`);
        }
        const buf = await entry.async('nodebuffer');
        arrays.push(parseNpyFloat32(buf));
      }

      let idx = 0;
      const stemC = makeConv(arrays, idx);
      idx = stemC.next;
      const stemB = makeBn(arrays, idx);
      idx = stemB.next;

      const stageBlockCounts = [3, 4, 6, 3];
      const stages: BottleneckBlockWeights[][] = [];

      for (const count of stageBlockCounts) {
        const stageBlocks: BottleneckBlockWeights[] = [];
        for (let b = 0; b < count; b++) {
          const parsed = parseBlock(arrays, idx, b === 0);
          stageBlocks.push(parsed.block);
          idx = parsed.next;
        }
        stages.push(stageBlocks);
      }

      let hdKernelData = arrays[idx++].data; // (2048, 256)
      let hdBiasData = arrays[idx++].data; // (256,)
      let predKernelData = arrays[idx++].data; // (256, 38)
      let predBiasData = arrays[idx++].data; // (38,)

      // Load multi-domain fine-tuned classifier head if present
      const ftPath = path.resolve(process.cwd(), 'models/resnet_head_finetuned.bin');
      if (fs.existsSync(ftPath)) {
        const ftBuf = fs.readFileSync(ftPath);
        const expectedBytes = (2048 * 256 + 256 + 256 * 38 + 38) * 4;
        if (ftBuf.byteLength === expectedBytes) {
          const aligned = new Uint8Array(ftBuf.byteLength);
          aligned.set(ftBuf);
          const f32 = new Float32Array(aligned.buffer);
          let off = 0;
          hdKernelData = f32.slice(off, off + 2048 * 256);
          off += 2048 * 256;
          hdBiasData = f32.slice(off, off + 256);
          off += 256;
          predKernelData = f32.slice(off, off + 256 * 38);
          off += 256 * 38;
          predBiasData = f32.slice(off, off + 38);
          console.log('[ResNet50] Loaded multi-domain fine-tuned classifier head.');
        }
      }

      return {
        stemConv: stemC.conv,
        stemBn: stemB.bn,
        stages,
        headDenseKernel: tf.tensor2d(hdKernelData, [2048, 256]),
        headDenseBias: tf.tensor1d(hdBiasData),
        predKernel: tf.tensor2d(predKernelData, [256, 38]),
        predBias: tf.tensor1d(predBiasData),
        headDenseKernelArr: hdKernelData,
        predKernelArr: predKernelData,
      };
    } catch (err) {
      cachedWeightsPromise = null;
      throw err;
    }
  })();

  return cachedWeightsPromise;
}

function applyConvBn(
  x: tf.Tensor4D,
  conv: ConvWeights,
  bn: BnWeights,
  strides: [number, number] = [1, 1],
  padding: 'valid' | 'same' = 'valid',
  withRelu = true
): tf.Tensor4D {
  return tf.tidy(() => {
    let out = tf.conv2d(x, conv.kernel, strides, padding);
    out = tf.add(out, conv.bias) as tf.Tensor4D;
    out = tf.add(tf.mul(out, bn.scale), bn.offset) as tf.Tensor4D;
    if (withRelu) {
      out = tf.relu(out) as tf.Tensor4D;
    }
    return out;
  });
}

function runBottleneckBlock(
  input: tf.Tensor4D,
  block: BottleneckBlockWeights,
  stride1: number
): tf.Tensor4D {
  return tf.tidy(() => {
    let x = applyConvBn(
      input,
      block.conv1,
      block.bn1,
      [stride1, stride1],
      'valid',
      true
    );
    x = applyConvBn(x, block.conv2, block.bn2, [1, 1], 'same', true);
    x = applyConvBn(x, block.conv3, block.bn3, [1, 1], 'valid', false);

    let shortcut = input;
    if (block.conv0 && block.bn0) {
      shortcut = applyConvBn(
        input,
        block.conv0,
        block.bn0,
        [stride1, stride1],
        'valid',
        false
      );
    }

    return tf.relu(tf.add(shortcut, x)) as tf.Tensor4D;
  });
}

/**
 * Detect exterior border-connected background mask on a 224x224 RGB buffer.
 * Uses only non-foliage border swatches + 4-neighbor BFS flood fill so leaves
 * touching one or more image edges never poison the background color model,
 * and interior pale necrotic lesions / powdery mildew remain inside the leaf mask.
 * Returns Uint8Array where 1 = exterior background, 0 = leaf foreground.
 */
function detectExteriorBackgroundMask(rgb224: Uint8Array | Buffer): Uint8Array {
  const N = 224 * 224;
  const bgMask = new Uint8Array(N);

  const isGreenFoliageWall = (idx: number): boolean => {
    const p = idx * 3;
    const r = rgb224[p];
    const g = rgb224[p + 1];
    const b = rgb224[p + 2];
    const exG = (2 * g - r - b) / 255;
    return exG > 0.048 && g > b + 4 && g > r - 6;
  };

  const isNonFoliagePixel = (idx: number): boolean => {
    if (isGreenFoliageWall(idx)) return false;
    const p = idx * 3;
    const r = rgb224[p];
    const g = rgb224[p + 1];
    const b = rgb224[p + 2];
    const exG = (2 * g - r - b) / 255;
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
    const luma = (r + g + b) / 3;

    // Cool pale-blue / cyan / neutral gray studio shadow
    if (b >= g - 3 && r <= g + 4 && luma > 120 && sat < 0.32) {
      return true;
    }
    // Bright white, light gray, or pale studio backdrop
    if (luma > 168 && sat < 0.25 && exG < 0.045 && g - b < 15 && r - b < 16) {
      return true;
    }
    // Warm out-of-focus outdoor bokeh / soil / wall / cream backdrop touching border
    if (luma > 125 && exG < 0.025 && r - g < 42 && sat < 0.45) {
      return true;
    }
    // Neutral mid/dark gray or black studio backdrop
    if (exG < 0.042 && g - b < 15 && r - b < 18 && sat < 0.24) {
      return true;
    }
    return false;
  };

  // Collect representative non-foliage border swatches (never leaf green)
  const swatches: Array<[number, number, number]> = [];
  const queue = new Int32Array(N);
  let head = 0;
  let tail = 0;

  const trySeed = (x: number, y: number) => {
    const idx = y * 224 + x;
    if (bgMask[idx]) return;
    if (!isNonFoliagePixel(idx)) return;

    bgMask[idx] = 1;
    queue[tail++] = idx;

    if (swatches.length < 28 && (x + y) % 12 === 0) {
      const p = idx * 3;
      swatches.push([rgb224[p], rgb224[p + 1], rgb224[p + 2]]);
    }
  };

  for (let x = 0; x < 224; x++) {
    trySeed(x, 0);
    trySeed(x, 223);
  }
  for (let y = 1; y < 223; y++) {
    trySeed(0, y);
    trySeed(223, y);
  }

  const dx = [1, -1, 0, 0];
  const dy = [0, 0, 1, -1];

  while (head < tail) {
    const cur = queue[head++];
    const cx = cur % 224;
    const cy = (cur - cx) / 224;
    const cp = cur * 3;
    const cr = rgb224[cp];
    const cg = rgb224[cp + 1];
    const cb = rgb224[cp + 2];

    for (let dir = 0; dir < 4; dir++) {
      const nx = cx + dx[dir];
      const ny = cy + dy[dir];
      if (nx < 0 || nx >= 224 || ny < 0 || ny >= 224) continue;
      const nIdx = ny * 224 + nx;
      if (bgMask[nIdx]) continue;
      if (!isNonFoliagePixel(nIdx)) continue;

      const np = nIdx * 3;
      const nr = rgb224[np];
      const ng = rgb224[np + 1];
      const nb = rgb224[np + 2];

      const stepDist = Math.hypot(nr - cr, ng - cg, nb - cb);
      let minSwatchDist = 999;
      for (let s = 0; s < swatches.length; s++) {
        const [sr, sg, sb] = swatches[s];
        const d = Math.hypot(nr - sr, ng - sg, nb - sb);
        if (d < minSwatchDist) minSwatchDist = d;
      }

      const luma = (nr + ng + nb) / 3;
      const maxC = Math.max(nr, ng, nb);
      const minC = Math.min(nr, ng, nb);
      const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
      const exG = (2 * ng - nr - nb) / 255;

      const isBrightBackdrop = luma > 172 && sat < 0.24 && exG < 0.03;
      const isSmoothWarmBokeh = luma > 125 && exG < 0.025 && stepDist < 22 && minSwatchDist < 48;
      if (stepDist < 24 || minSwatchDist < 44 || isBrightBackdrop || isSmoothWarmBokeh) {
        bgMask[nIdx] = 1;
        queue[tail++] = nIdx;
      }
    }
  }

  // Sanity guard: if flood fill ever consumed >90% of the frame, revert to safe bright-neutral exclusion
  let bgCount = 0;
  for (let i = 0; i < N; i++) bgCount += bgMask[i];
  if (bgCount > N * 0.90) {
    bgMask.fill(0);
    for (let i = 0; i < N; i++) {
      const p = i * 3;
      const r = rgb224[p];
      const g = rgb224[p + 1];
      const b = rgb224[p + 2];
      const maxC = Math.max(r, g, b);
      const minC = Math.min(r, g, b);
      const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
      const luma = (r + g + b) / 3;
      const exG = (2 * g - r - b) / 255;
      if (luma > 178 && sat < 0.22 && exG < 0.03) {
        bgMask[i] = 1;
      }
    }
  }

  return bgMask;
}

function writeRgbToBgrFloat32(
  rgb224: Uint8Array | Buffer,
  out: Float32Array,
  offset: number
): void {
  const N = 224 * 224;
  for (let i = 0; i < N; i++) {
    const r = rgb224[i * 3];
    const g = rgb224[i * 3 + 1];
    const b = rgb224[i * 3 + 2];
    const o = offset + i * 3;
    out[o] = b - 103.939;
    out[o + 1] = g - 116.779;
    out[o + 2] = r - 123.68;
  }
}

/**
 * Exact Keras resnet50.preprocess_input(caffe) across 3 complementary views:
 *   View 0: fit='fill' (1:1 aligned with frontend Grad-CAM canvas)
 *   View 1: fit='cover' (aspect-ratio-preserving center crop)
 *   View 2: foreground leaf bounding-box crop -> fit='contain' + studio-gray (128,128,128) normalization
 */
async function preprocessImageViewsToTensor(imageBuffer: Buffer): Promise<{
  tensorBatch: tf.Tensor4D;
  width: number;
  height: number;
  rgb224: Buffer;
}> {
  const rawMeta = await sharp(imageBuffer).metadata();
  const oriented = await sharp(imageBuffer)
    .rotate()
    .removeAlpha()
    .resize(640, 640, { fit: 'inside', withoutEnlargement: true })
    .png()
    .toBuffer({ resolveWithObject: true });

  const orientedW = oriented.info.width;
  const orientedH = oriented.info.height;
  const base = sharp(oriented.data);

  const [rgbFill, rgbCover] = await Promise.all([
    base.clone().resize(224, 224, { fit: 'fill' }).raw().toBuffer(),
    base
      .clone()
      .resize(224, 224, { fit: 'cover', position: 'centre' })
      .raw()
      .toBuffer(),
  ]);

  // Locate foreground leaf bounding box from View 0 mask so off-center leaves on white/wide canvases get tightly framed
  const bgMaskFill = detectExteriorBackgroundMask(rgbFill);
  let minX = 224;
  let maxX = -1;
  let minY = 224;
  let maxY = -1;
  let leafPx = 0;
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      if (bgMaskFill[y * 224 + x] === 0) {
        leafPx++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // Detect whether image has a flat pure-white studio cutout backdrop (luma > 228, low variance)
  // so blinding #FFFFFF stock backgrounds don't bias View 1 cover features
  let bgSumLuma = 0;
  let bgSqLuma = 0;
  let bgTotal = 0;
  for (let i = 0; i < 224 * 224; i++) {
    if (bgMaskFill[i]) {
      const p = i * 3;
      const l = (rgbFill[p] + rgbFill[p + 1] + rgbFill[p + 2]) / 3;
      bgSumLuma += l;
      bgSqLuma += l * l;
      bgTotal++;
    }
  }
  const bgFracFill = bgTotal / (224 * 224);
  const bgMeanLuma = bgTotal > 0 ? bgSumLuma / bgTotal : 0;
  const bgStdLuma =
    bgTotal > 0
      ? Math.sqrt(Math.max(0, bgSqLuma / bgTotal - bgMeanLuma * bgMeanLuma))
      : 99;
  const isFlatWhiteCutout =
    bgFracFill >= 0.18 && bgMeanLuma > 226 && bgStdLuma < 20;

  const rgbCoverNorm = Buffer.from(rgbCover);
  if (isFlatWhiteCutout) {
    const bgMaskCover = detectExteriorBackgroundMask(rgbCoverNorm);
    for (let i = 0; i < 224 * 224; i++) {
      if (bgMaskCover[i]) {
        const p = i * 3;
        const luma =
          (rgbCoverNorm[p] + rgbCoverNorm[p + 1] + rgbCoverNorm[p + 2]) / 3;
        if (luma > 210) {
          rgbCoverNorm[p] = 182;
          rgbCoverNorm[p + 1] = 186;
          rgbCoverNorm[p + 2] = 190;
        }
      }
    }
  }

  let view2Pipe = base.clone();
  const leafAreaFrac = leafPx / (224 * 224);
  if (
    maxX > minX + 16 &&
    maxY > minY + 16 &&
    leafAreaFrac > 0.08 &&
    leafAreaFrac < 0.86
  ) {
    const padX = Math.round(((maxX - minX) / 224) * orientedW * 0.06);
    const padY = Math.round(((maxY - minY) / 224) * orientedH * 0.06);
    const left = Math.max(0, Math.floor((minX / 224) * orientedW) - padX);
    const top = Math.max(0, Math.floor((minY / 224) * orientedH) - padY);
    const right = Math.min(
      orientedW,
      Math.ceil(((maxX + 1) / 224) * orientedW) + padX
    );
    const bottom = Math.min(
      orientedH,
      Math.ceil(((maxY + 1) / 224) * orientedH) + padY
    );
    if (right - left >= 32 && bottom - top >= 32) {
      view2Pipe = view2Pipe.extract({
        left,
        top,
        width: right - left,
        height: bottom - top,
      });
    }
  }

  const rgbContain = await view2Pipe
    .resize(224, 224, {
      fit: 'contain',
      background: { r: 128, g: 128, b: 128 },
    })
    .raw()
    .toBuffer();

  // Normalize non-foliage bright/dark studio backdrops on View 2 to PlantVillage gray (128,128,128)
  const bgMaskContain = detectExteriorBackgroundMask(rgbContain);
  const rgbContainNorm = Buffer.from(rgbContain);
  for (let i = 0; i < 224 * 224; i++) {
    if (bgMaskContain[i]) {
      const p = i * 3;
      const luma =
        (rgbContainNorm[p] + rgbContainNorm[p + 1] + rgbContainNorm[p + 2]) / 3;
      if (luma > 158 || luma < 48) {
        rgbContainNorm[p] = 128;
        rgbContainNorm[p + 1] = 128;
        rgbContainNorm[p + 2] = 128;
      }
    }
  }

  const stride = 224 * 224 * 3;
  const batchBgr = new Float32Array(3 * stride);
  writeRgbToBgrFloat32(rgbFill, batchBgr, 0);
  writeRgbToBgrFloat32(rgbCoverNorm, batchBgr, stride);
  writeRgbToBgrFloat32(rgbContainNorm, batchBgr, 2 * stride);

  return {
    tensorBatch: tf.tensor4d(batchBgr, [3, 224, 224, 3]),
    width: rawMeta.width || orientedW,
    height: rawMeta.height || orientedH,
    rgb224: rgbFill,
  };
}

/**
 * Contrastive Pre-Softmax Logit Grad-CAM on conv5_block3_out [7, 7, 2048]
 * (cancels out shared healthy-lamina crop features against same-crop healthy weights)
 * refined onto a 28x28 leaf-foreground-gated activation grid so studio corners
 * never leak halos and disease lesions stay sharp.
 */
function computeRefinedGradCam(
  conv5Data: Float32Array,
  headPreRelu: Float32Array,
  predictedIdx: number,
  weights: ResNet50ModelWeights,
  rgb224: Buffer,
  isHealthy: boolean,
  cropName: string
): { gradCamGrid: number[]; gradCamGridSize: number; leafCoverageEstimate: number } {
  const w2 = weights.predKernelArr; // [256, 38]
  const w1 = weights.headDenseKernelArr; // [2048, 256]

  // Locate matching healthy class index for contrastive disease vs healthy gradient filtering
  let healthyContrastIdx = -1;
  if (!isHealthy) {
    healthyContrastIdx = PLANT_VILLAGE_CLASSES.findIndex(
      (c) => c.crop === cropName && c.isHealthy
    );
  }

  // 1. Pre-softmax logit gradient + contrastive healthy suppression
  const dHeadStd = new Float32Array(256);
  const dHeadDiff = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    if (headPreRelu[i] > 0) {
      const targetGrad = w2[i * NUM_CLASSES + predictedIdx];
      dHeadStd[i] = targetGrad;
      if (healthyContrastIdx >= 0 && healthyContrastIdx !== predictedIdx) {
        const healthyGrad = w2[i * NUM_CLASSES + healthyContrastIdx];
        dHeadDiff[i] = targetGrad - 0.55 * healthyGrad;
      } else {
        dHeadDiff[i] = targetGrad;
      }
    }
  }

  // 2. Backprop through head_dense (2048 -> 256) and GlobalAveragePooling2D (1/49)
  const pooledStd = new Float32Array(2048);
  const pooledDiff = new Float32Array(2048);
  for (let k = 0; k < 2048; k++) {
    let sumStd = 0;
    let sumDiff = 0;
    const rowOff = k * 256;
    for (let i = 0; i < 256; i++) {
      const w = w1[rowOff + i];
      sumStd += dHeadStd[i] * w;
      sumDiff += dHeadDiff[i] * w;
    }
    pooledStd[k] = sumStd / 49;
    pooledDiff[k] = sumDiff / 49;
  }

  // 3. Raw 7x7 conv5_block3_out Grad-CAM + Contrastive ReLU blend
  const cam7 = new Float32Array(49);
  for (let p = 0; p < 49; p++) {
    const featOff = p * 2048;
    let accStd = 0;
    let accDiff = 0;
    for (let k = 0; k < 2048; k++) {
      const f = conv5Data[featOff + k];
      accStd += pooledStd[k] * f;
      accDiff += pooledDiff[k] * f;
    }
    const stdPos = Math.max(0, accStd);
    const diffPos = Math.max(0, accDiff);
    cam7[p] = !isHealthy && healthyContrastIdx >= 0
      ? 0.65 * diffPos + 0.35 * stdPos
      : stdPos;
  }

  // 4. Exterior border flood-fill separates studio backdrop cleanly from leaf & interior lesions
  const bgMask224 = detectExteriorBackgroundMask(rgb224);

  // 5. Build 224x224 foreground leaf mask, green-lamina mask, and interior lesion contrast map
  const leafMask224 = new Uint8Array(224 * 224);
  const greenMask224 = new Uint8Array(224 * 224);
  const lesion224 = new Float32Array(224 * 224);
  let greenCountTotal = 0;

  for (let i = 0; i < 224 * 224; i++) {
    if (bgMask224[i] === 0) {
      const p = i * 3;
      const r = rgb224[p];
      const g = rgb224[p + 1];
      const b = rgb224[p + 2];
      const exG = (2 * g - r - b) / 255;
      const luma = (r + g + b) / 3;
      const maxC = Math.max(r, g, b);

      leafMask224[i] = 1;
      if (exG > 0.055 && g > b + 4 && g > r - 5) {
        greenMask224[i] = 1;
        greenCountTotal++;
      }

      if (!isHealthy) {
        const necrosis = r > g + 8 ? Math.max(0, (r - g * 0.88) / 68) : 0;
        const chlorosis =
          r > 112 && g > 105 && b < 95 && exG < 0.22
            ? Math.max(0, (r - b) / 140) * 0.85
            : 0;
        const darkSpot =
          exG < 0.095 && luma < 120 && maxC < 140
            ? Math.min(1, (120 - luma) / 58 + Math.max(0, 0.095 - exG) * 2.8)
            : 0;
        const mildew =
          r > 170 && g > 170 && b > 160 && exG < 0.10 ? 0.65 : 0;
        lesion224[i] = Math.min(1, Math.max(necrosis + chlorosis, darkSpot, mildew));
      }
    }
  }

  // Suppress false lesion spikes right on a 2px cast-shadow leaf boundary
  if (!isHealthy) {
    for (let y = 2; y < 222; y++) {
      for (let x = 2; x < 222; x++) {
        const idx = y * 224 + x;
        if (lesion224[idx] > 0) {
          const borderCheck =
            leafMask224[(y - 2) * 224 + x] +
            leafMask224[(y + 2) * 224 + x] +
            leafMask224[y * 224 + (x - 2)] +
            leafMask224[y * 224 + (x + 2)];
          if (borderCheck < 4) {
            lesion224[idx] *= 0.2;
          }
        }
      }
    }
  }

  // 6. Normalize 7x7 cam7 using leaf-covered cells so empty corners cannot crush leaf activations
  let maxLeafCam7 = 0;
  let maxAnyCam7 = 0;
  for (let gy = 0; gy < 7; gy++) {
    for (let gx = 0; gx < 7; gx++) {
      const v = cam7[gy * 7 + gx];
      if (v > maxAnyCam7) maxAnyCam7 = v;
      let leafPx = 0;
      for (let py = gy * 32; py < (gy + 1) * 32; py++) {
        for (let px = gx * 32; px < (gx + 1) * 32; px++) {
          leafPx += leafMask224[py * 224 + px];
        }
      }
      if (leafPx / (32 * 32) >= 0.32 && v > maxLeafCam7) {
        maxLeafCam7 = v;
      }
    }
  }

  const denom7 = (maxLeafCam7 > 1e-6 ? maxLeafCam7 : maxAnyCam7) + 1e-8;
  const cam7Norm = new Float32Array(49);
  for (let p = 0; p < 49; p++) {
    cam7Norm[p] = Math.min(1, cam7[p] / denom7);
  }

  // 7. Upsample onto a high-res 28x28 grid (8x8 px per cell) fused with leaf-gated lesion saliency
  const gridSize = 28;
  const cellPx = 224 / gridSize; // 8
  const cellArea = cellPx * cellPx; // 64
  const rawGrid = new Float32Array(gridSize * gridSize);
  const cellLeafGate = new Float32Array(gridSize * gridSize);
  const cellGreenFrac = new Float32Array(gridSize * gridSize);
  const cellLesionAvg = new Float32Array(gridSize * gridSize);
  const cellCam7Val = new Float32Array(gridSize * gridSize);

  for (let r = 0; r < gridSize; r++) {
    const fy = Math.max(0, Math.min(6, ((r + 0.5) / gridSize) * 7 - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(6, y0 + 1);
    const wy = fy - y0;

    for (let c = 0; c < gridSize; c++) {
      const fx = Math.max(0, Math.min(6, ((c + 0.5) / gridSize) * 7 - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(6, x0 + 1);
      const wx = fx - x0;

      const c7 =
        (cam7Norm[y0 * 7 + x0] * (1 - wx) + cam7Norm[y0 * 7 + x1] * wx) *
          (1 - wy) +
        (cam7Norm[y1 * 7 + x0] * (1 - wx) + cam7Norm[y1 * 7 + x1] * wx) * wy;

      let leafPxCell = 0;
      let greenPxCell = 0;
      let lesionSum = 0;
      for (let py = r * cellPx; py < (r + 1) * cellPx; py++) {
        for (let px = c * cellPx; px < (c + 1) * cellPx; px++) {
          const idx = py * 224 + px;
          if (leafMask224[idx]) {
            leafPxCell++;
            greenPxCell += greenMask224[idx];
            lesionSum += lesion224[idx];
          }
        }
      }

      const cellIdx = r * gridSize + c;
      const leafFrac = leafPxCell / cellArea;
      cellGreenFrac[cellIdx] = greenPxCell / cellArea;
      cellLesionAvg[cellIdx] = leafPxCell > 0 ? lesionSum / leafPxCell : 0;
      cellCam7Val[cellIdx] = c7;
      cellLeafGate[cellIdx] = leafFrac < 0.08 ? 0 : Math.min(1, leafFrac / 0.35);
    }
  }

  const hasGreenFoliage = greenCountTotal > 224 * 224 * 0.12;
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      const cellIdx = r * gridSize + c;
      const gate = cellLeafGate[cellIdx];
      if (gate === 0) {
        rawGrid[cellIdx] = 0;
        continue;
      }

      let greenNb = 0;
      let nbCount = 0;
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) {
          const nr = r + dr;
          const nc = c + dc;
          if (nr >= 0 && nr < gridSize && nc >= 0 && nc < gridSize) {
            greenNb += cellGreenFrac[nr * gridSize + nc];
            nbCount++;
          }
        }
      }
      const localGreenSupport = nbCount > 0 ? greenNb / nbCount : 0;
      const foliageSupportWeight = hasGreenFoliage
        ? Math.max(0.15, Math.min(1, localGreenSupport / 0.14))
        : 1;

      const c7 = cellCam7Val[cellIdx];
      const avgLesion = cellLesionAvg[cellIdx] * foliageSupportWeight;
      rawGrid[cellIdx] = isHealthy
        ? gate * c7 * 0.25
        : gate *
          foliageSupportWeight *
          (Math.pow(c7, 0.85) * 0.35 +
            avgLesion * 0.45 +
            c7 * avgLesion * 0.70);
    }
  }

  // Smooth 3x3 only within leaf tissue, then re-apply strict leafGate so NO halo bleeds into background
  const smoothed = new Float32Array(gridSize * gridSize);
  let maxSmoothed = 0;
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      const centerIdx = r * gridSize + c;
      if (cellLeafGate[centerIdx] === 0) {
        smoothed[centerIdx] = 0;
        continue;
      }
      let acc = 0;
      let wSum = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const nr = r + dr;
          const nc = c + dc;
          if (nr >= 0 && nr < gridSize && nc >= 0 && nc < gridSize) {
            const nIdx = nr * gridSize + nc;
            if (cellLeafGate[nIdx] === 0) continue;
            const wt = dr === 0 && dc === 0 ? 4 : dr === 0 || dc === 0 ? 2 : 1;
            acc += rawGrid[nIdx] * wt;
            wSum += wt;
          }
        }
      }
      const v = (wSum > 0 ? acc / wSum : 0) * cellLeafGate[centerIdx];
      smoothed[centerIdx] = v;
      if (v > maxSmoothed) maxSmoothed = v;
    }
  }

  const gradCamGrid: number[] = new Array(gridSize * gridSize);
  const normDenom = isHealthy
    ? Math.max(0.28, maxSmoothed)
    : Math.max(0.18, maxSmoothed) + 1e-8;
  for (let i = 0; i < gradCamGrid.length; i++) {
    if (cellLeafGate[i] === 0) {
      gradCamGrid[i] = 0;
      continue;
    }
    const rawRatio = smoothed[i] / normDenom;
    const scaled = isHealthy
      ? Math.min(0.26, (smoothed[i] / (maxSmoothed + 1e-8)) * 0.25)
      : Math.pow(Math.max(0, Math.min(1, rawRatio)), 1.3);
    gradCamGrid[i] = Number(Math.max(0, Math.min(1, scaled)).toFixed(4));
  }

  // 8. Compute symptomatic leaf coverage directly from foreground pixels with focal Grad-CAM activation
  let totalLeafPixels = 0;
  let activeLesionPixels = 0;
  if (!isHealthy) {
    for (let y = 0; y < 224; y++) {
      const gy = Math.min(gridSize - 1, Math.floor((y / 224) * gridSize));
      for (let x = 0; x < 224; x++) {
        if (leafMask224[y * 224 + x]) {
          totalLeafPixels++;
          const gx = Math.min(gridSize - 1, Math.floor((x / 224) * gridSize));
          if (gradCamGrid[gy * gridSize + gx] >= 0.42) {
            activeLesionPixels++;
          }
        }
      }
    }
  }

  const leafCoverageEstimate =
    isHealthy || totalLeafPixels === 0
      ? 0
      : Math.max(
          1,
          Math.min(95, Math.round((activeLesionPixels / totalLeafPixels) * 100))
        );

  return { gradCamGrid, gradCamGridSize: gridSize, leafCoverageEstimate };
}

export async function analyzeLeafWithResNet50(
  imageBuffer: Buffer
): Promise<ResNetPredictionOutput> {
  const t0 = Date.now();
  const weights = await loadResNet50Model();
  const { tensorBatch, width, height, rgb224 } =
    await preprocessImageViewsToTensor(imageBuffer);

  try {
    const { conv5Data, headPreRelu, probsData } = tf.tidy(() => {
      // Stem: ZeroPadding2D(3) -> Conv7x7/s2 -> BN -> ReLU -> ZeroPadding2D(1) -> MaxPool3x3/s2
      let x = tf.pad(tensorBatch, [
        [0, 0],
        [3, 3],
        [3, 3],
        [0, 0],
      ]) as tf.Tensor4D;
      x = applyConvBn(x, weights.stemConv, weights.stemBn, [2, 2], 'valid', true);
      x = tf.pad(x, [
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
      ]) as tf.Tensor4D;
      x = tf.maxPool(x, [3, 3], [2, 2], 'valid') as tf.Tensor4D;

      const stageStrides = [1, 2, 2, 2];
      for (let s = 0; s < weights.stages.length; s++) {
        const blocks = weights.stages[s];
        for (let b = 0; b < blocks.length; b++) {
          x = runBottleneckBlock(x, blocks[b], b === 0 ? stageStrides[s] : 1);
        }
      }

      // x is conv5_block3_out: [3, 7, 7, 2048]
      const conv5Out = x;
      const gap = tf.mean(conv5Out, [1, 2]) as tf.Tensor2D; // [3, 2048]
      const hdPre = tf.add(
        tf.matMul(gap, weights.headDenseKernel),
        weights.headDenseBias
      ) as tf.Tensor2D; // [3, 256]
      const hdRelu = tf.relu(hdPre) as tf.Tensor2D;
      const logits = tf.add(
        tf.matMul(hdRelu, weights.predKernel),
        weights.predBias
      ) as tf.Tensor2D; // [3, 38]

      // Weighted multi-view logit ensemble (fill: 0.25, cover: 0.35, contain-gray: 0.40)
      const viewWeights = tf.tensor2d([[0.25, 0.35, 0.40]]); // [1, 3]
      const ensembleLogits = tf.matMul(viewWeights, logits) as tf.Tensor2D; // [1, 38]
      const probs = tf.softmax(ensembleLogits) as tf.Tensor2D;

      // View 0 conv5 & headPreRelu keep 1:1 spatial alignment with the uncropped image
      const conv5View0 = tf.slice(conv5Out, [0, 0, 0, 0], [1, 7, 7, 2048]);
      const hdPreView0 = tf.slice(hdPre, [0, 0], [1, 256]);

      return {
        conv5Data: conv5View0.dataSync() as Float32Array,
        headPreRelu: hdPreView0.dataSync() as Float32Array,
        probsData: probs.dataSync() as Float32Array,
      };
    });

    // Rank all 38 PlantVillage classes
    const indexed = Array.from({ length: NUM_CLASSES }, (_, i) => ({
      index: i,
      classId: PLANT_VILLAGE_CLASSES[i].id,
      probability: Number(probsData[i].toFixed(6)),
    }));
    indexed.sort((a, b) => b.probability - a.probability);

    const best = indexed[0];
    const canonical =
      PLANT_VILLAGE_CLASS_MAP[best.classId] ?? PLANT_VILLAGE_CLASSES[0];

    const { gradCamGrid, gradCamGridSize, leafCoverageEstimate } =
      computeRefinedGradCam(
        conv5Data,
        headPreRelu,
        best.index,
        weights,
        rgb224,
        canonical.isHealthy,
        canonical.crop
      );

    const second = indexed[1];
    const secondInfo = PLANT_VILLAGE_CLASS_MAP[second.classId];

    const visualObservations: string[] = canonical.isHealthy
      ? [
          canonical.visualSignature,
          `ResNet50 conv5_block3_out activations remain diffuse across healthy lamina (${(best.probability * 100).toFixed(2)}% confidence).`,
          `Next closest class (${secondInfo ? `${secondInfo.crop} — ${secondInfo.condition}` : second.classId}) scored ${(second.probability * 100).toFixed(3)}%.`,
        ]
      : [
          canonical.visualSignature,
          `Grad-CAM (conv5_block3_out, ${gradCamGridSize}×${gradCamGridSize} grid) localized symptomatic regions covering approximately ${leafCoverageEstimate}% of the leaf surface.`,
          `Top-1 ResNet50 softmax score ${(best.probability * 100).toFixed(2)}% vs runner-up ${secondInfo ? `${secondInfo.crop} — ${secondInfo.condition}` : second.classId} (${(second.probability * 100).toFixed(3)}%).`,
        ];

    return {
      predictedClass: canonical.id,
      readablePrediction: canonical.readableDescription,
      crop: canonical.crop,
      conditionName: canonical.condition,
      pathogen: canonical.pathogen,
      isHealthy: canonical.isHealthy,
      confidence: best.probability,
      topPredictions: indexed.map((item) => ({
        classId: item.classId,
        probability: item.probability,
      })),
      gradCamGrid,
      gradCamGridSize,
      leafCoverageEstimate,
      visualObservations,
      inferenceMs: Date.now() - t0,
      imageWidth: width,
      imageHeight: height,
    };
  } finally {
    tensorBatch.dispose();
  }
}
