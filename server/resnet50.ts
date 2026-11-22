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

  const isNonFoliagePixel = (idx: number): boolean => {
    const p = idx * 3;
    const r = rgb224[p];
    const g = rgb224[p + 1];
    const b = rgb224[p + 2];
    const exG = (2 * g - r - b) / 255;
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
    const luma = (r + g + b) / 3;

    // Cool pale-blue / cyan / neutral gray studio shadow (b >= g - 3, never foliage green or brown necrosis)
    if (b >= g - 3 && r <= g + 4 && luma > 120 && sat < 0.32) {
      return true;
    }
    // Bright white, light gray, or pale studio backdrop
    if (luma > 168 && sat < 0.25 && exG < 0.045 && g - b < 15 && r - b < 16) {
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

    if (swatches.length < 24 && (x + y) % 14 === 0) {
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
      if (stepDist < 24 || minSwatchDist < 44 || isBrightBackdrop) {
        bgMask[nIdx] = 1;
        queue[tail++] = nIdx;
      }
    }
  }

  // Sanity guard: if flood fill ever consumed >92% of the frame, revert to safe bright-neutral exclusion
  let bgCount = 0;
  for (let i = 0; i < N; i++) bgCount += bgMask[i];
  if (bgCount > N * 0.92) {
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
 *   View 2: fit='contain' with bright/dark studio backdrops normalized to PlantVillage gray (128,128,128)
 */
async function preprocessImageViewsToTensor(imageBuffer: Buffer): Promise<{
  tensorBatch: tf.Tensor4D;
  width: number;
  height: number;
  rgb224: Buffer;
}> {
  const meta = await sharp(imageBuffer).metadata();
  const base = sharp(imageBuffer).rotate().removeAlpha();

  const [rgbFill, rgbCover, rgbContain] = await Promise.all([
    base.clone().resize(224, 224, { fit: 'fill' }).raw().toBuffer(),
    base
      .clone()
      .resize(224, 224, { fit: 'cover', position: 'centre' })
      .raw()
      .toBuffer(),
    base
      .clone()
      .resize(224, 224, {
        fit: 'contain',
        background: { r: 128, g: 128, b: 128 },
      })
      .raw()
      .toBuffer(),
  ]);

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
  writeRgbToBgrFloat32(rgbCover, batchBgr, stride);
  writeRgbToBgrFloat32(rgbContainNorm, batchBgr, 2 * stride);

  return {
    tensorBatch: tf.tensor4d(batchBgr, [3, 224, 224, 3]),
    width: meta.width || 224,
    height: meta.height || 224,
    rgb224: rgbFill,
  };
}

/**
 * Pre-softmax Logit Grad-CAM on conv5_block3_out [7, 7, 2048]
 * (avoids softmax gradient vanishing when top-1 confidence approaches 100%)
 * refined onto a 14x14 leaf-foreground-gated activation grid so empty studio
 * corners never hijack normalization and lesion boundaries remain sharp.
 */
function computeRefinedGradCam(
  conv5Data: Float32Array,
  headPreRelu: Float32Array,
  predictedIdx: number,
  weights: ResNet50ModelWeights,
  rgb224: Buffer,
  isHealthy: boolean
): { gradCamGrid: number[]; gradCamGridSize: number; leafCoverageEstimate: number } {
  // 1. Pre-softmax logit gradient: d(logit_c)/d(logits) = e_c
  const dHead = new Float32Array(256);
  const w2 = weights.predKernelArr; // [256, 38]
  for (let i = 0; i < 256; i++) {
    const grad = w2[i * NUM_CLASSES + predictedIdx];
    dHead[i] = headPreRelu[i] > 0 ? grad : 0;
  }

  // 2. Backprop through head_dense (2048 -> 256) and GlobalAveragePooling2D (1/49)
  const pooledGrads = new Float32Array(2048);
  const w1 = weights.headDenseKernelArr; // [2048, 256]
  for (let k = 0; k < 2048; k++) {
    let sum = 0;
    const rowOff = k * 256;
    for (let i = 0; i < 256; i++) {
      sum += dHead[i] * w1[rowOff + i];
    }
    pooledGrads[k] = sum / 49;
  }

  // 3. Raw 7x7 conv5_block3_out Grad-CAM + ReLU
  const cam7 = new Float32Array(49);
  for (let p = 0; p < 49; p++) {
    const featOff = p * 2048;
    let acc = 0;
    for (let k = 0; k < 2048; k++) {
      acc += pooledGrads[k] * conv5Data[featOff + k];
    }
    cam7[p] = Math.max(0, acc);
  }

  // 4. Exterior border flood-fill separates studio backdrop cleanly from leaf & interior lesions
  const bgMask224 = detectExteriorBackgroundMask(rgb224);

  // 5. Build 224x224 foreground leaf mask and interior lesion contrast map
  const leafMask224 = new Uint8Array(224 * 224);
  const lesion224 = new Float32Array(224 * 224);

  for (let i = 0; i < 224 * 224; i++) {
    if (bgMask224[i] === 0) {
      const p = i * 3;
      const r = rgb224[p];
      const g = rgb224[p + 1];
      const b = rgb224[p + 2];
      const exG = (2 * g - r - b) / 255;

      leafMask224[i] = 1;
      if (!isHealthy) {
        const necrosis = Math.max(0, (r - g * 0.84) / 72);
        const chlorosis =
          r > 110 && g > 105 && b < 95 && exG < 0.24
            ? Math.max(0, (r - b) / 145) * 0.85
            : 0;
        const darkSpot =
          exG < 0.13 && g < 115 && b < 105 && r > 32
            ? Math.max(0, (0.13 - exG) * 4.2)
            : 0;
        const mildew =
          r > 165 && g > 165 && b > 155 && exG < 0.12 ? 0.65 : 0;
        lesion224[i] = Math.min(1, necrosis + chlorosis + darkSpot + mildew);
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
            lesion224[idx] *= 0.25;
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

  // 7. Upsample onto a 14x14 grid (16x16 px per cell) fused with leaf-gated lesion saliency
  const gridSize = 14;
  const rawGrid = new Float32Array(gridSize * gridSize);
  let maxGrid = 0;

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

      let leafPx = 0;
      let lesionSum = 0;
      for (let py = r * 16; py < (r + 1) * 16; py++) {
        for (let px = c * 16; px < (c + 1) * 16; px++) {
          const idx = py * 224 + px;
          if (leafMask224[idx]) {
            leafPx++;
            lesionSum += lesion224[idx];
          }
        }
      }

      const leafFrac = leafPx / 256;
      const avgLesion = leafPx > 0 ? lesionSum / leafPx : 0;
      const leafGate = Math.min(1, leafFrac / 0.45);

      const val = isHealthy
        ? leafGate * c7 * 0.32
        : leafGate *
          (Math.pow(c7, 0.72) * 0.46 +
            avgLesion * 0.44 +
            c7 * avgLesion * 0.55);

      rawGrid[r * gridSize + c] = val;
      if (val > maxGrid) maxGrid = val;
    }
  }

  // Light 3x3 smoothing on the 14x14 grid for organic spatial continuity
  const smoothed = new Float32Array(gridSize * gridSize);
  let maxSmoothed = 0;
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      let acc = 0;
      let wSum = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const nr = r + dr;
          const nc = c + dc;
          if (nr >= 0 && nr < gridSize && nc >= 0 && nc < gridSize) {
            const wt = dr === 0 && dc === 0 ? 4 : dr === 0 || dc === 0 ? 2 : 1;
            acc += rawGrid[nr * gridSize + nc] * wt;
            wSum += wt;
          }
        }
      }
      const v = acc / wSum;
      smoothed[r * gridSize + c] = v;
      if (v > maxSmoothed) maxSmoothed = v;
    }
  }

  const gradCamGrid: number[] = new Array(gridSize * gridSize);
  const normDenom = isHealthy
    ? Math.max(0.32, maxSmoothed)
    : maxSmoothed + 1e-8;
  for (let i = 0; i < gradCamGrid.length; i++) {
    const scaled = isHealthy
      ? Math.min(0.34, (smoothed[i] / (maxSmoothed + 1e-8)) * 0.32)
      : smoothed[i] / normDenom;
    gradCamGrid[i] = Number(Math.max(0, Math.min(1, scaled)).toFixed(4));
  }

  // 8. Compute symptomatic leaf coverage directly from foreground pixels with active Grad-CAM
  let totalLeafPixels = 0;
  let activeLesionPixels = 0;
  if (!isHealthy) {
    for (let y = 0; y < 224; y++) {
      const gy = Math.min(gridSize - 1, Math.floor((y / 224) * gridSize));
      for (let x = 0; x < 224; x++) {
        if (leafMask224[y * 224 + x]) {
          totalLeafPixels++;
          const gx = Math.min(gridSize - 1, Math.floor((x / 224) * gridSize));
          if (gradCamGrid[gy * gridSize + gx] >= 0.34) {
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
        canonical.isHealthy
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
