# PhytoScan — Running Locally on Apple Silicon (Mac M2)

This guide explains how to extract the PhytoScan source archive and run the **38-Class ResNet50 Plant Leaf Disease Detection & Grad-CAM** application locally on your **MacBook / Mac mini / iMac with Apple Silicon (M2)**.

---

## 1. Prerequisites (Mac M2)

1. **Node.js (v20 LTS or v22 LTS — native ARM64)**
   - Recommended: Install via [Homebrew](https://brew.sh) or the official macOS ARM64 installer from [nodejs.org](https://nodejs.org/).
   - Check in **Terminal**:
     ```bash
     uname -m        # Should print: arm64
     node -v         # Should be v20.x or v22.x (or v18.18+)
     npm -v
     ```
   - If Node is not installed yet (via Homebrew):
     ```bash
     brew install node@22
     ```

2. **Optional (only if you use Safari/Chrome)**: Any modern macOS browser (Chrome, Safari, Firefox, Edge, Arc).

---

## 2. Extract the Downloaded ZIP Archive

1. Locate the downloaded file `phytoscan-resnet50.zip` in your **Downloads** folder (or double-click it in Finder to extract it).
2. Or extract it from **Terminal**:
   ```bash
   cd ~/Downloads
   unzip phytoscan-resnet50.zip -d phytoscan-resnet50
   cd phytoscan-resnet50
   ```

---

## 3. Install Dependencies

Inside the extracted project folder (`phytoscan-resnet50`), run:

```bash
npm install
```

> **Apple Silicon (`arm64`) note for `sharp`**:
> `sharp` automatically downloads prebuilt native `darwin-arm64` binaries during `npm install`. If you previously ran Node under Rosetta (`x86_64`), make sure your Terminal is running natively (`uname -m` -> `arm64`) before running `npm install`.

---

## 4. ResNet50 Model Weights

- The fine-tuned 38-class classification head (`models/resnet_head_finetuned.bin`) is included directly inside this archive.
- On your **very first startup**, the backend (`server/resnet50.ts`) automatically checks for `models/resnet_weights.npz` (~93 MB) and downloads it into `models/resnet_weights.npz` if it is not already present (requires an internet connection once on first boot; afterwards inference runs 100% offline locally).

---

## 5. Run the Application

Start the full-stack Express + Vite + ResNet50 server:

```bash
npm run dev
```

Wait until Terminal logs:
```
PhytoScan ResNet50 server running at http://localhost:3000
[ResNet50] Loaded multi-domain fine-tuned classifier head.
[ResNet50] PlantVillage 38-class weights loaded and ready.
```

Then open your browser and go to:

**http://localhost:3000**

---

## 6. Production Build (Optional)

To compile the frontend and run in production mode:

```bash
npm run build
NODE_ENV=production npm run dev
```

Or check TypeScript types with:

```bash
npm run lint
```

---

## 7. Troubleshooting on Mac M2

- **Port 3000 already in use**:
  Stop whatever process is using port 3000:
  ```bash
  lsof -ti:3000 | xargs kill -9
  ```
- **Native `sharp` error after switching Node versions**:
  Rebuild `sharp` cleanly for `arm64`:
  ```bash
  rm -rf node_modules
  npm install
  ```
- **First-run weight download fails offline**:
  Ensure internet access for the initial startup so `models/resnet_weights.npz` finishes downloading (~93 MB), or download it manually into `models/resnet_weights.npz` from:
  `https://raw.githubusercontent.com/Ishaaq09/Automated_plant_disease_detection_using_Deep_Learning_and_Transfer_Learning/main/resnet_weights.npz`
