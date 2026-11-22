import React from 'react';
import { Download, Printer } from 'lucide-react';
import { PLANT_VILLAGE_CLASS_MAP } from '../data/plantVillage';
import type { DiagnosisRecord } from '../types/diagnostics';

interface ReportViewProps {
  records: DiagnosisRecord[];
  selectedRecord: DiagnosisRecord | null;
  onSelectRecordId: (id: string) => void;
  onGoToAnalyze: () => void;
}

function describeGridZone(row: number, col: number, gridSize: number): string {
  const vBand =
    row < gridSize / 3
      ? 'Upper'
      : row < (2 * gridSize) / 3
      ? 'Mid'
      : 'Lower';
  const hBand =
    col < gridSize / 3
      ? 'Left'
      : col < (2 * gridSize) / 3
      ? 'Center'
      : 'Right';
  return vBand === 'Mid' && hBand === 'Center'
    ? 'Central Lamina'
    : `${vBand}-${hBand}`;
}

interface GradCamPeakCell {
  row: number;
  col: number;
  intensity: number;
  zone: string;
}

function summarizeGradCamGrid(
  grid: number[] | undefined,
  gridSize = 7
): { peakCells: GradCamPeakCell[]; activeCount: number; peakIntensity: number } {
  if (!Array.isArray(grid) || grid.length === 0) {
    return { peakCells: [], activeCount: 0, peakIntensity: 0 };
  }
  const size = Math.max(1, Math.round(gridSize || Math.sqrt(grid.length)));
  const cells: GradCamPeakCell[] = [];
  let activeCount = 0;
  let peakIntensity = 0;

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const val = Number(grid[r * size + c] ?? 0);
      if (val > peakIntensity) peakIntensity = val;
      if (val >= 0.35) activeCount++;
      if (val >= 0.25) {
        cells.push({
          row: r,
          col: c,
          intensity: val,
          zone: describeGridZone(r, c, size),
        });
      }
    }
  }

  cells.sort((a, b) => b.intensity - a.intensity);
  return {
    peakCells: cells.slice(0, 6),
    activeCount,
    peakIntensity,
  };
}

export const ReportView: React.FC<ReportViewProps> = ({
  records,
  selectedRecord,
  onSelectRecordId,
  onGoToAnalyze,
}) => {
  const buildMarkdownReport = (): string => {
    if (!selectedRecord) {
      return '# Leaf Diagnosis Report\n\nNo analyzed leaf specimen selected.\n';
    }

    const { analysis, specimenLabel } = selectedRecord;
    const timestamp = new Date(analysis.analyzedAt).toLocaleString();
    const topMatches = analysis.topPredictions.slice(0, 10);

    const { peakCells, activeCount, peakIntensity } = summarizeGradCamGrid(
      analysis.gradCamGrid,
      analysis.gradCamGridSize
    );

    const gridRows =
      !analysis.isHealthy && peakCells.length > 0
        ? peakCells
            .map(
              (cell, i) =>
                `| ${i + 1} | ${cell.zone} | Row ${cell.row + 1}, Col ${cell.col + 1} | ${(cell.intensity * 100).toFixed(1)}% |`
            )
            .join('\n')
        : '| — | Diffuse baseline activation (Healthy leaf) | — | — |';

    const reportGridSize =
      analysis.gradCamGridSize ||
      Math.max(1, Math.round(Math.sqrt(analysis.gradCamGrid?.length || 196)));

    return `# Leaf Pathology Diagnostic Report: ${analysis.crop} — ${analysis.conditionName}

- **Specimen**: ${specimenLabel}
- **Analyzed At**: ${timestamp}
- **Health Status**: ${analysis.isHealthy ? 'Healthy' : 'Diseased'}
- **Predicted Classification**: ${analysis.readablePrediction}
- **PlantVillage Class ID**: \`${analysis.predictedClass}\`
- **Causal Agent / Pathogen**: ${analysis.pathogen}
- **ResNet50 Confidence**: ${(analysis.confidence * 100).toFixed(2)}%
- **Estimated Affected Leaf Surface**: ${analysis.leafCoverageEstimate}%
- **Grad-CAM Active Cells (≥35% intensity)**: ${activeCount} / ${reportGridSize ** 2} (Peak: ${(peakIntensity * 100).toFixed(1)}%)

---

## Visual Pathology Observations

${analysis.visualObservations.map((o) => `- ${o}`).join('\n')}

---

## Grad-CAM Spatial Activation Summary (conv5_block3_out)

| # | Leaf Region | ${reportGridSize}×${reportGridSize} Feature Cell | Activation Intensity |
| :---: | :--- | :---: | :---: |
${gridRows}

---

## Top Matching PlantVillage Classes (38-Class Model)

| Rank | Crop | Condition | Probability |
| :---: | :--- | :--- | :---: |
${topMatches
  .map((p, idx) => {
    const info = PLANT_VILLAGE_CLASS_MAP[p.classId];
    return `| ${idx + 1} | ${info ? info.crop : '—'} | ${info ? info.condition : p.classId} | ${(p.probability * 100).toFixed(3)}% |`;
  })
  .join('\n')}

---

## Model & Dataset Reference

- **Implementation**: Python (TensorFlow / Keras ResNet50, OpenCV Grad-CAM on \`conv5_block3_out\`)
- **Dataset**: PlantVillage Dataset (Source: Kaggle · 38 crop & disease classes)
`;
  };

  const handleDownloadMarkdown = () => {
    const content = buildMarkdownReport();
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const safeName = (
      selectedRecord?.analysis.predictedClass || 'leaf_diagnosis'
    ).replace(/[^a-z0-9_-]/gi, '_');
    link.href = url;
    link.download = `Report_${safeName}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (!selectedRecord) {
    return (
      <div className="max-w-[1440px] mx-auto px-6 py-8">
        <div className="pb-6 border-b border-slate-200 mb-6">
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
            Diagnostic Report
          </h1>
        </div>
        <section className="bg-white border border-slate-200 rounded-lg p-12 text-center max-w-2xl mx-auto">
          <p className="text-base font-medium text-slate-900">
            No analyzed leaf selected for report generation
          </p>
          <p className="text-sm text-slate-500 mt-1.5">
            Upload and analyze a leaf image first so the report is generated specifically for that specimen.
          </p>
          <button
            type="button"
            onClick={onGoToAnalyze}
            className="mt-5 px-4 py-2 text-xs font-medium text-white bg-teal-700 rounded-md hover:bg-teal-800 transition-colors"
          >
            Analyze a Leaf
          </button>
        </section>
      </div>
    );
  }

  const { analysis, specimenLabel, thumbnailDataUrl, gradCamDataUrl } =
    selectedRecord;
  const analyzedDate = new Date(analysis.analyzedAt).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
  const topCandidates = analysis.topPredictions.slice(0, 8);
  const { peakCells, activeCount, peakIntensity } = summarizeGradCamGrid(
    analysis.gradCamGrid,
    analysis.gradCamGridSize
  );
  const gridSize = analysis.gradCamGridSize || 7;

  return (
    <div className="max-w-[1440px] mx-auto px-6 py-8">
      {/* Top Action Bar (Hidden on Print) */}
      <div className="no-print flex flex-col lg:flex-row lg:items-end justify-between gap-4 pb-6 border-b border-slate-200 mb-6">
        <div>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
            Diagnostic Report
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {records.length > 0 && (
            <select
              value={selectedRecord.id}
              onChange={(e) => onSelectRecordId(e.target.value)}
              aria-label="Select analyzed leaf for report"
              className="px-3 py-2 text-xs bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-600 max-w-[260px]"
            >
              {records.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.specimenLabel} — {r.analysis.conditionName} (
                  {(r.analysis.confidence * 100).toFixed(1)}%)
                </option>
              ))}
            </select>
          )}
          <button
            type="button"
            onClick={handleDownloadMarkdown}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5" />
            Download (.md)
          </button>
          <button
            type="button"
            onClick={() => window.print()}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-white bg-teal-700 rounded-md hover:bg-teal-800 transition-colors whitespace-nowrap"
          >
            <Printer className="w-3.5 h-3.5" />
            Print / Save PDF
          </button>
        </div>
      </div>

      {/* Focused Specimen Diagnostic Report Document */}
      <article className="print-report-container bg-white border border-slate-200 rounded-lg px-8 sm:px-12 py-10 max-w-4xl mx-auto">
        {/* Specimen Header */}
        <header className="pb-6 border-b border-slate-200 mb-8">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 mb-2">
            <span>Specimen: {specimenLabel}</span>
            <span className="font-mono tabular-nums">{analyzedDate}</span>
          </div>
          <h2
            className="text-2xl sm:text-3xl font-bold text-slate-950 leading-tight"
            style={{ fontFamily: 'var(--font-serif)' }}
          >
            {analysis.readablePrediction}
          </h2>
          <p className="text-xs font-mono text-slate-500 mt-2">
            Class ID: {analysis.predictedClass} · Causal Agent: {analysis.pathogen}
          </p>

          {/* Key Quantitative Findings for This Leaf */}
          <dl className="mt-6 pt-5 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div>
              <dt className="text-slate-500">Crop Type</dt>
              <dd className="font-semibold text-slate-900 mt-0.5">
                {analysis.crop}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Condition Status</dt>
              <dd
                className={`font-semibold mt-0.5 ${
                  analysis.isHealthy ? 'text-emerald-700' : 'text-rose-700'
                }`}
              >
                {analysis.isHealthy ? '● Healthy' : '▲ Diseased'}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Confidence</dt>
              <dd className="font-mono tabular-nums font-semibold text-teal-800 mt-0.5">
                {(analysis.confidence * 100).toFixed(2)}%
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Affected Leaf Area</dt>
              <dd className="font-mono tabular-nums font-semibold text-slate-900 mt-0.5">
                {analysis.leafCoverageEstimate}%
              </dd>
            </div>
          </dl>
        </header>

        <div className="space-y-8 text-sm text-slate-800">
          {/* 1. Original vs Grad-CAM Visual Evidence */}
          <section className="print-break-avoid">
            <h3 className="text-base font-semibold text-slate-950 mb-3">
              1. Visual Specimen &amp; Grad-CAM Localization
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 max-w-2xl">
              <div>
                <div className="aspect-square rounded border border-slate-200 overflow-hidden bg-slate-50 flex items-center justify-center">
                  <img
                    src={thumbnailDataUrl}
                    alt={`Original ${specimenLabel}`}
                    referrerPolicy="no-referrer"
                    className="w-full h-full object-contain"
                  />
                </div>
                <p className="text-xs font-mono text-slate-500 mt-1.5">
                  Original RGB Input ({specimenLabel})
                </p>
              </div>
              <div>
                <div className="aspect-square rounded border border-slate-200 overflow-hidden bg-slate-50 flex items-center justify-center">
                  {gradCamDataUrl ? (
                    <img
                      src={gradCamDataUrl}
                      alt={`Grad-CAM Overlay for ${specimenLabel}`}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-xs text-slate-400">
                      Grad-CAM overlay unavailable
                    </div>
                  )}
                </div>
                <p className="text-xs font-mono text-slate-500 mt-1.5">
                  Grad-CAM Activation Overlay (conv5_block3_out)
                </p>
              </div>
            </div>
          </section>

          {/* 2. Specific Visual Pathology Findings */}
          <section className="print-break-avoid">
            <h3 className="text-base font-semibold text-slate-950 mb-3">
              2. Observed Pathology Findings ({analysis.crop} — {analysis.conditionName})
            </h3>
            <ul className="space-y-2 text-slate-700 leading-relaxed">
              {analysis.visualObservations.map((obs, idx) => (
                <li key={idx} className="flex items-start gap-2.5">
                  <span className="font-mono text-xs text-slate-400 mt-0.5 tabular-nums">
                    0{idx + 1}.
                  </span>
                  <span>{obs}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* 3. ResNet50 Grad-CAM Spatial Activation Map for This Specimen */}
          <section className="print-break-avoid">
            <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
              <h3 className="text-base font-semibold text-slate-950">
                3. Grad-CAM Spatial Activation (conv5_block3_out · {gridSize}×{gridSize} Grid)
              </h3>
              <span className="text-xs font-mono tabular-nums text-slate-500">
                {activeCount}/{gridSize * gridSize} cells ≥35% · Peak {(peakIntensity * 100).toFixed(1)}%
              </span>
            </div>
            {!analysis.isHealthy && peakCells.length > 0 ? (
              <div className="overflow-x-auto border border-slate-200 rounded-md">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-medium">
                      <th className="py-2 px-3 w-12 font-mono">#</th>
                      <th className="py-2 px-3">Leaf Region</th>
                      <th className="py-2 px-3 text-right font-mono">
                        Feature Cell ({gridSize}×{gridSize})
                      </th>
                      <th className="py-2 px-3 text-right font-mono">
                        Activation Intensity
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {peakCells.map((cell, idx) => (
                      <tr key={`${cell.row}-${cell.col}`}>
                        <td className="py-2 px-3 font-mono tabular-nums text-slate-400">
                          {String(idx + 1).padStart(2, '0')}
                        </td>
                        <td className="py-2 px-3 text-slate-900 font-medium">
                          {cell.zone}
                        </td>
                        <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-600">
                          Row {cell.row + 1}, Col {cell.col + 1}
                        </td>
                        <td className="py-2 px-3 text-right font-mono tabular-nums text-teal-800">
                          {(cell.intensity * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-md p-3.5">
                No focal disease hotspots localized. ResNet50 conv5_block3_out activations remain diffuse across healthy lamina.
              </p>
            )}
          </section>

          {/* 4. Top Candidate Classes for This Specimen */}
          <section className="print-break-avoid">
            <h3 className="text-base font-semibold text-slate-950 mb-3">
              4. Top Candidate Classifications (from 38-Class Model)
            </h3>
            <div className="overflow-x-auto border border-slate-200 rounded-md">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-medium">
                    <th className="py-2 px-3 w-12 font-mono">Rank</th>
                    <th className="py-2 px-3">Crop</th>
                    <th className="py-2 px-3">Condition</th>
                    <th className="py-2 px-3 text-right font-mono">
                      Probability
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {topCandidates.map((pred, idx) => {
                    const info = PLANT_VILLAGE_CLASS_MAP[pred.classId];
                    return (
                      <tr
                        key={pred.classId}
                        className={idx === 0 ? 'bg-teal-50/40 font-medium' : ''}
                      >
                        <td className="py-2 px-3 font-mono tabular-nums text-slate-500">
                          {String(idx + 1).padStart(2, '0')}
                        </td>
                        <td className="py-2 px-3 text-slate-900">
                          {info ? info.crop : '—'}
                        </td>
                        <td className="py-2 px-3 text-slate-800">
                          {info ? info.condition : pred.classId}
                        </td>
                        <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-900">
                          {(pred.probability * 100).toFixed(3)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* Compact Technical Basis Footer inside Report */}
          <section className="pt-5 border-t border-slate-200 text-xs text-slate-500 flex flex-wrap items-center justify-between gap-2">
            <span>
              Implementation: Python (TensorFlow / Keras ResNet50 &amp; OpenCV Grad-CAM)
            </span>
            <span className="font-mono">
              Dataset: PlantVillage (Source: Kaggle · 38 Classes)
            </span>
          </section>
        </div>
      </article>
    </div>
  );
};
