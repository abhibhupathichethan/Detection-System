import { useCallback, useEffect, useRef, useState } from 'react';
import { Download } from 'lucide-react';
import { AnalyzeView } from './components/AnalyzeView';
import { HistoryView } from './components/HistoryView';
import { ReportView } from './components/ReportView';
import { SAMPLE_SPECIMENS } from './data/sampleSpecimens';
import type { DiagnosisRecord, LeafAnalysisResult } from './types/diagnostics';

type ActiveTab = 'analyze' | 'history' | 'report';

const HISTORY_STORAGE_KEY = 'phytoscan_resnet50_history_v1';

async function imageUrlToBase64(
  src: string
): Promise<{ base64: string; mimeType: string; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    if (/^https?:\/\//i.test(src)) {
      img.crossOrigin = 'anonymous';
    }
    img.onload = () => {
      const maxDim = 768;
      const scale = Math.min(
        1,
        maxDim / Math.max(img.naturalWidth, img.naturalHeight)
      );
      const w = Math.max(1, Math.round(img.naturalWidth * scale));
      const h = Math.max(1, Math.round(img.naturalHeight * scale));
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Could not initialize image canvas context.'));
        return;
      }
      ctx.fillStyle = '#808080';
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
      const base64 = dataUrl.split(',')[1] ?? '';
      resolve({
        base64,
        mimeType: 'image/jpeg',
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    };
    img.onerror = () => reject(new Error('Failed to load selected leaf image.'));
    img.src = src;
  });
}

function loadSavedHistory(): DiagnosisRecord[] {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('analyze');
  const [history, setHistory] = useState<DiagnosisRecord[]>(() =>
    loadSavedHistory()
  );
  const [activeRecordId, setActiveRecordId] = useState<string | null>(null);

  const [activeImageSrc, setActiveImageSrc] = useState<string | null>(
    SAMPLE_SPECIMENS[0].imageUrl
  );
  const [activeSpecimenLabel, setActiveSpecimenLabel] = useState<string>(
    SAMPLE_SPECIMENS[0].label
  );
  const [analysis, setAnalysis] = useState<LeafAnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const requestCounterRef = useRef(0);
  const activeLabelRef = useRef(activeSpecimenLabel);
  activeLabelRef.current = activeSpecimenLabel;

  useEffect(() => {
    try {
      localStorage.setItem(
        HISTORY_STORAGE_KEY,
        JSON.stringify(history.slice(0, 25))
      );
    } catch {
      // Ignore quota errors
    }
  }, [history]);

  const runDiagnosisForImage = useCallback(async (src: string, label: string) => {
    const reqId = ++requestCounterRef.current;
    setIsAnalyzing(true);
    setError(null);

    try {
      const { base64, mimeType, width, height } = await imageUrlToBase64(src);
      const response = await fetch('/api/analyze-leaf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageBase64: base64,
          mimeType,
          imageWidth: width,
          imageHeight: height,
        }),
      });

      const payload = await response.json();
      if (reqId !== requestCounterRef.current) return;

      if (!response.ok) {
        throw new Error(
          payload?.error || `Server responded with ${response.status}`
        );
      }

      const result = payload as LeafAnalysisResult;
      const newRecordId = `${Date.now()}-${Math.random()
        .toString(36)
        .slice(2, 7)}`;

      setAnalysis(result);
      setActiveRecordId(newRecordId);
      setHistory((prev) => [
        {
          id: newRecordId,
          specimenLabel: label,
          thumbnailDataUrl: src.startsWith('data:') ? src : '',
          gradCamDataUrl: null,
          analysis: result,
        },
        ...prev.slice(0, 24),
      ]);
    } catch (err: unknown) {
      if (reqId !== requestCounterRef.current) return;
      const msg =
        err instanceof Error ? err.message : 'Unable to complete leaf diagnosis.';
      setError(msg);
      setAnalysis(null);
    } finally {
      if (reqId === requestCounterRef.current) {
        setIsAnalyzing(false);
      }
    }
  }, []);

  const initialHistoryRef = useRef<DiagnosisRecord[]>(history);

  useEffect(() => {
    const saved = initialHistoryRef.current;
    if (saved.length > 0) {
      const latest = saved[0];
      setActiveRecordId(latest.id);
      if (latest.thumbnailDataUrl) {
        setActiveImageSrc(latest.thumbnailDataUrl);
      }
      setActiveSpecimenLabel(latest.specimenLabel);
      setAnalysis(latest.analysis);
    } else {
      runDiagnosisForImage(
        SAMPLE_SPECIMENS[0].imageUrl,
        SAMPLE_SPECIMENS[0].label
      );
    }
  }, [runDiagnosisForImage]);

  const handleSelectImage = useCallback(
    (imageSrc: string, label: string, autoAnalyze = true) => {
      setActiveImageSrc(imageSrc);
      setActiveSpecimenLabel(label);
      setAnalysis(null);
      setActiveRecordId(null);
      if (autoAnalyze) {
        runDiagnosisForImage(imageSrc, label);
      }
    },
    [runDiagnosisForImage]
  );

  const handleRunCurrentAnalysis = useCallback(() => {
    if (activeImageSrc) {
      runDiagnosisForImage(activeImageSrc, activeLabelRef.current);
    }
  }, [activeImageSrc, runDiagnosisForImage]);

  const handleSnapshotsReady = useCallback(
    (thumbUrl: string | null, gradCamUrl: string | null) => {
      if (!activeRecordId) return;
      setHistory((prev) =>
        prev.map((rec) =>
          rec.id === activeRecordId
            ? {
                ...rec,
                thumbnailDataUrl: thumbUrl || rec.thumbnailDataUrl,
                gradCamDataUrl: gradCamUrl || rec.gradCamDataUrl,
              }
            : rec
        )
      );
    },
    [activeRecordId]
  );

  const handleInspectRecord = useCallback((record: DiagnosisRecord) => {
    setActiveRecordId(record.id);
    setActiveImageSrc(record.thumbnailDataUrl);
    setActiveSpecimenLabel(record.specimenLabel);
    setAnalysis(record.analysis);
    setError(null);
    setActiveTab('analyze');
  }, []);

  const handleIncludeInReport = useCallback((record: DiagnosisRecord) => {
    setActiveRecordId(record.id);
    setActiveImageSrc(record.thumbnailDataUrl);
    setActiveSpecimenLabel(record.specimenLabel);
    setAnalysis(record.analysis);
    setActiveTab('report');
  }, []);

  const handleDeleteRecord = useCallback(
    (id: string) => {
      setHistory((prev) => prev.filter((r) => r.id !== id));
      if (activeRecordId === id) {
        setActiveRecordId(null);
      }
    },
    [activeRecordId]
  );

  const handleClearAll = useCallback(() => {
    setHistory([]);
    setActiveRecordId(null);
  }, []);

  const selectedReportRecord =
    history.find((r) => r.id === activeRecordId) ?? history[0] ?? null;

  const navItems: Array<{ id: ActiveTab; label: string }> = [
    { id: 'analyze', label: 'Analyze' },
    { id: 'history', label: 'Previous Diagnoses' },
    { id: 'report', label: 'Report' },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#F8FAFC] text-slate-900">
      {/* Strict 3-Zone Top Navigation Bar */}
      <header className="no-print sticky top-0 z-30 bg-white border-b border-slate-200">
        <div className="max-w-[1440px] mx-auto px-6 h-16 flex items-center justify-between gap-8">
          <button
            type="button"
            onClick={() => setActiveTab('analyze')}
            className="text-lg font-bold tracking-tight text-slate-950 whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 rounded"
          >
            PhytoScan
          </button>

          <nav
            aria-label="Primary sections"
            className="flex items-center gap-6 overflow-x-auto"
          >
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  className={`py-1 text-sm font-medium transition-colors whitespace-nowrap shrink-0 border-b-2 ${
                    isActive
                      ? 'text-teal-800 border-teal-700'
                      : 'text-slate-600 border-transparent hover:text-slate-950 hover:border-slate-300'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          <div className="hidden sm:flex items-center gap-2.5 shrink-0">
            <a
              href="/api/download-source"
              download="phytoscan-resnet50.zip"
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
            >
              <Download className="w-3.5 h-3.5" />
              Download Code (.zip)
            </a>
            <button
              type="button"
              onClick={() => setActiveTab('report')}
              className="px-4 py-2 text-xs font-medium text-white bg-teal-700 rounded-md hover:bg-teal-800 transition-colors whitespace-nowrap shrink-0"
            >
              Generate Report
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 pb-10">
        {activeTab === 'analyze' && (
          <AnalyzeView
            activeImageSrc={activeImageSrc}
            activeSpecimenLabel={activeSpecimenLabel}
            analysis={analysis}
            isAnalyzing={isAnalyzing}
            error={error}
            onSelectImage={handleSelectImage}
            onRunAnalysis={handleRunCurrentAnalysis}
            onSnapshotsReady={handleSnapshotsReady}
            onOpenReport={() => setActiveTab('report')}
          />
        )}

        {activeTab === 'history' && (
          <HistoryView
            records={history}
            activeRecordId={activeRecordId}
            onInspectRecord={handleInspectRecord}
            onIncludeInReport={handleIncludeInReport}
            onDeleteRecord={handleDeleteRecord}
            onClearAll={handleClearAll}
            onGoToAnalyze={() => setActiveTab('analyze')}
          />
        )}

        {activeTab === 'report' && (
          <ReportView
            records={history}
            selectedRecord={selectedReportRecord}
            onSelectRecordId={(id) => setActiveRecordId(id)}
            onGoToAnalyze={() => setActiveTab('analyze')}
          />
        )}
      </main>
    </div>
  );
}
