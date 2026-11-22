export interface TopPrediction {
  classId: string;
  probability: number;
}

export interface LeafAnalysisResult {
  predictedClass: string;
  readablePrediction: string;
  crop: string;
  conditionName: string;
  pathogen: string;
  isHealthy: boolean;
  confidence: number;
  topPredictions: TopPrediction[];
  gradCamGrid: number[];
  gradCamGridSize: number;
  visualObservations: string[];
  leafCoverageEstimate: number;
  inferenceMs?: number;
  analyzedAt: string;
  imageWidth: number;
  imageHeight: number;
}

export interface DiagnosisRecord {
  id: string;
  specimenLabel: string;
  thumbnailDataUrl: string;
  gradCamDataUrl: string | null;
  analysis: LeafAnalysisResult;
}

export interface SampleSpecimen {
  id: string;
  label: string;
  crop: string;
  expectedClass: string;
  imageUrl: string;
  specimenCode: string;
}
