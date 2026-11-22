import React from 'react';
import { Eye, FileText, Trash2 } from 'lucide-react';
import type { DiagnosisRecord } from '../types/diagnostics';

interface HistoryViewProps {
  records: DiagnosisRecord[];
  activeRecordId: string | null;
  onInspectRecord: (record: DiagnosisRecord) => void;
  onIncludeInReport: (record: DiagnosisRecord) => void;
  onDeleteRecord: (id: string) => void;
  onClearAll: () => void;
  onGoToAnalyze: () => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  records,
  activeRecordId,
  onInspectRecord,
  onIncludeInReport,
  onDeleteRecord,
  onClearAll,
  onGoToAnalyze,
}) => {
  return (
    <div className="max-w-[1440px] mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-6 border-b border-slate-200 mb-6">
        <div>
          <p className="text-xs font-mono text-slate-500 mb-1.5">
            Saved Session Archive · Local Diagnosis Log
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
            Previous Diagnoses
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs font-mono tabular-nums text-slate-500">
            {records.length} {records.length === 1 ? 'record' : 'records'}
          </span>
          {records.length > 0 && (
            <button
              type="button"
              onClick={onClearAll}
              className="px-3.5 py-2 text-xs font-medium text-rose-700 bg-white border border-slate-300 rounded-md hover:bg-rose-50 transition-colors whitespace-nowrap"
            >
              Clear History
            </button>
          )}
        </div>
      </div>

      {records.length === 0 ? (
        <section className="bg-white border border-slate-200 rounded-lg p-12 text-center">
          <p className="text-base font-medium text-slate-900">
            No previous diagnoses recorded yet
          </p>
          <p className="text-sm text-slate-500 mt-1.5 max-w-md mx-auto">
            Upload or select a leaf image in the Analyze workspace. Every completed classification and Grad-CAM result is automatically logged here.
          </p>
          <button
            type="button"
            onClick={onGoToAnalyze}
            className="mt-5 px-4 py-2 text-xs font-medium text-white bg-teal-700 rounded-md hover:bg-teal-800 transition-colors whitespace-nowrap"
          >
            Analyze a Leaf
          </button>
        </section>
      ) : (
        <section className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-medium">
                  <th className="py-3 px-4">Specimen</th>
                  <th className="py-3 px-4">Grad-CAM</th>
                  <th className="py-3 px-4">Predicted Disease / Class</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4 text-right font-mono">Confidence</th>
                  <th className="py-3 px-4 text-right font-mono">Affected</th>
                  <th className="py-3 px-4">Analyzed At</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {records.map((rec) => {
                  const isCurrent = rec.id === activeRecordId;
                  const dateLabel = new Date(rec.analysis.analyzedAt).toLocaleString(
                    undefined,
                    {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    }
                  );

                  return (
                    <tr
                      key={rec.id}
                      className={`transition-colors ${
                        isCurrent ? 'bg-teal-50/40' : 'hover:bg-slate-50'
                      }`}
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <div className="w-12 h-12 rounded border border-slate-200 bg-slate-100 overflow-hidden shrink-0">
                            <img
                              src={rec.thumbnailDataUrl}
                              alt={rec.specimenLabel}
                              referrerPolicy="no-referrer"
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <div className="min-w-0">
                            <p className="font-medium text-slate-900 truncate max-w-[160px]">
                              {rec.specimenLabel}
                            </p>
                            <p className="text-[11px] font-mono text-slate-500">
                              {rec.analysis.crop}
                            </p>
                          </div>
                        </div>
                      </td>

                      <td className="py-3 px-4">
                        <div className="w-12 h-12 rounded border border-slate-200 bg-slate-100 overflow-hidden">
                          {rec.gradCamDataUrl ? (
                            <img
                              src={rec.gradCamDataUrl}
                              alt={`Grad-CAM ${rec.specimenLabel}`}
                              referrerPolicy="no-referrer"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-400">
                              —
                            </div>
                          )}
                        </div>
                      </td>

                      <td className="py-3 px-4">
                        <p className="font-medium text-slate-900">
                          {rec.analysis.readablePrediction}
                        </p>
                        <p className="text-[11px] font-mono text-slate-500 mt-0.5">
                          {rec.analysis.predictedClass}
                        </p>
                      </td>

                      <td className="py-3 px-4 whitespace-nowrap">
                        <span
                          className={`font-medium ${
                            rec.analysis.isHealthy
                              ? 'text-emerald-700'
                              : 'text-rose-700'
                          }`}
                        >
                          {rec.analysis.isHealthy ? '● Healthy' : '▲ Diseased'}
                        </span>
                      </td>

                      <td className="py-3 px-4 text-right font-mono tabular-nums font-medium text-slate-900">
                        {(rec.analysis.confidence * 100).toFixed(2)}%
                      </td>

                      <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-700">
                        {rec.analysis.leafCoverageEstimate}%
                      </td>

                      <td className="py-3 px-4 font-mono tabular-nums text-slate-500 whitespace-nowrap">
                        {dateLabel}
                      </td>

                      <td className="py-3 px-4 text-right whitespace-nowrap">
                        <div className="inline-flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => onInspectRecord(rec)}
                            className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded hover:bg-slate-50 transition-colors"
                          >
                            <Eye className="w-3.5 h-3.5" />
                            View
                          </button>
                          <button
                            type="button"
                            onClick={() => onIncludeInReport(rec)}
                            className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-teal-800 bg-white border border-slate-300 rounded hover:bg-teal-50 transition-colors"
                          >
                            <FileText className="w-3.5 h-3.5" />
                            Report
                          </button>
                          <button
                            type="button"
                            onClick={() => onDeleteRecord(rec.id)}
                            aria-label={`Delete ${rec.specimenLabel}`}
                            className="inline-flex items-center justify-center p-1.5 text-slate-500 bg-white border border-slate-300 rounded hover:text-rose-700 hover:bg-rose-50 transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
};
