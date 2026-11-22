import type { SampleSpecimen } from '../types/diagnostics';

export const SAMPLE_SPECIMENS: SampleSpecimen[] = [
  {
    id: 'tomato-early-blight',
    label: 'Tomato · Early Blight',
    crop: 'Tomato',
    expectedClass: 'Tomato___Early_blight',
    imageUrl: '/src/assets/images/pv_tomato_early_blight.jpg',
    specimenCode: 'PV-TOM-029',
  },
  {
    id: 'apple-scab',
    label: 'Apple · Apple Scab',
    crop: 'Apple',
    expectedClass: 'Apple___Apple_scab',
    imageUrl: '/src/assets/images/pv_apple_scab.jpg',
    specimenCode: 'PV-APP-001',
  },
  {
    id: 'tomato-healthy',
    label: 'Tomato · Healthy Control',
    crop: 'Tomato',
    expectedClass: 'Tomato___healthy',
    imageUrl: '/src/assets/images/pv_tomato_healthy.jpg',
    specimenCode: 'PV-TOM-038',
  },
];
