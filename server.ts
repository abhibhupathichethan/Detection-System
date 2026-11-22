import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import JSZip from 'jszip';
import {
  analyzeLeafWithResNet50,
  loadResNet50Model,
} from './server/resnet50.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const app = express();

app.use(express.json({ limit: '15mb' }));

let weightsLoaded = false;
let weightsError: string | null = null;

loadResNet50Model()
  .then(() => {
    weightsLoaded = true;
    console.log('[ResNet50] PlantVillage 38-class weights loaded and ready.');
  })
  .catch((err: unknown) => {
    weightsError =
      err instanceof Error ? err.message : 'Failed to load ResNet50 weights.';
    console.error('[ResNet50] Weight warmup error:', weightsError);
  });

app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'PhytoScan ResNet50 Detection & Grad-CAM',
    model: 'ResNet50-PlantVillage-38',
    weightsLoaded,
    ...(weightsError ? { weightsError } : {}),
  });
});

function addDirectoryToZip(zip: JSZip, dirPath: string, zipPrefix: string): void {
  if (!fs.existsSync(dirPath)) return;
  for (const entry of fs.readdirSync(dirPath, { withFileTypes: true })) {
    if (
      entry.name === 'node_modules' ||
      entry.name === 'dist' ||
      entry.name === '.git' ||
      entry.name === '.DS_Store'
    ) {
      continue;
    }
    const fullPath = path.join(dirPath, entry.name);
    const relZipPath = zipPrefix ? `${zipPrefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      addDirectoryToZip(zip, fullPath, relZipPath);
    } else if (entry.isFile()) {
      zip.file(relZipPath, fs.readFileSync(fullPath));
    }
  }
}

app.get('/api/download-source', async (_req, res) => {
  try {
    const rootDir = process.cwd();
    const zip = new JSZip();

    const rootFiles = [
      'package.json',
      'tsconfig.json',
      'vite.config.ts',
      'index.html',
      'metadata.json',
      '.env.example',
      '.gitignore',
      'README_MAC_M2.md',
      'server.ts',
    ];

    for (const fileName of rootFiles) {
      const fullPath = path.join(rootDir, fileName);
      if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
        zip.file(fileName, fs.readFileSync(fullPath));
      }
    }

    addDirectoryToZip(zip, path.join(rootDir, 'src'), 'src');
    addDirectoryToZip(zip, path.join(rootDir, 'server'), 'server');

    const fineTunedHeadPath = path.join(
      rootDir,
      'models',
      'resnet_head_finetuned.bin'
    );
    if (fs.existsSync(fineTunedHeadPath)) {
      zip.file(
        'models/resnet_head_finetuned.bin',
        fs.readFileSync(fineTunedHeadPath)
      );
    }

    const zipBuffer = await zip.generateAsync({
      type: 'nodebuffer',
      compression: 'DEFLATE',
      compressionOptions: { level: 6 },
    });

    res.setHeader('Content-Type', 'application/zip');
    res.setHeader(
      'Content-Disposition',
      'attachment; filename="phytoscan-resnet50.zip"'
    );
    res.setHeader('Content-Length', String(zipBuffer.length));
    return res.send(zipBuffer);
  } catch (err: unknown) {
    const message =
      err instanceof Error ? err.message : 'Failed to build source archive.';
    console.error('[download-source] error:', message);
    return res.status(500).json({ error: message });
  }
});

app.post('/api/analyze-leaf', async (req, res) => {
  try {
    const { imageBase64, imageWidth, imageHeight } = req.body ?? {};

    if (!imageBase64 || typeof imageBase64 !== 'string') {
      return res
        .status(400)
        .json({ error: 'Missing base64-encoded leaf image.' });
    }

    const cleanBase64 = imageBase64.replace(/^data:image\/[a-z0-9.+-]+;base64,/i, '');
    const imageBuffer = Buffer.from(cleanBase64, 'base64');

    if (imageBuffer.length === 0) {
      return res
        .status(400)
        .json({ error: 'Uploaded leaf image payload is empty or invalid.' });
    }

    const result = await analyzeLeafWithResNet50(imageBuffer);
    weightsLoaded = true;

    return res.json({
      predictedClass: result.predictedClass,
      readablePrediction: result.readablePrediction,
      crop: result.crop,
      conditionName: result.conditionName,
      pathogen: result.pathogen,
      isHealthy: result.isHealthy,
      confidence: result.confidence,
      topPredictions: result.topPredictions,
      gradCamGrid: result.gradCamGrid,
      gradCamGridSize: result.gradCamGridSize,
      leafCoverageEstimate: result.leafCoverageEstimate,
      visualObservations: result.visualObservations,
      inferenceMs: result.inferenceMs,
      analyzedAt: new Date().toISOString(),
      imageWidth: Number(imageWidth) || result.imageWidth,
      imageHeight: Number(imageHeight) || result.imageHeight,
    });
  } catch (err: unknown) {
    const message =
      err instanceof Error
        ? err.message
        : 'Unexpected server error during ResNet50 leaf analysis.';
    console.error('[analyze-leaf] error:', message);
    return res.status(500).json({ error: message });
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`PhytoScan ResNet50 server running at http://localhost:${PORT}`);
  });
}

startServer();
