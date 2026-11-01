import { api, qs } from './client'
import type {
  Alert, Analysis, AnalysisDetail, Analytics, Area, DetectionImage, DetectionRun, ExtractedFrame, Extraction, ExtractionPlan,
  FrameResults,
  AnnotatedVideo, Health, HistoryRecord, Layout, MasterCsvStatus, Media, MonthGrouping, MonthReport, ReportsOverview,
  Overview, SamplingStrategy, ScreenshotEvidence, Settings, SettingSchema, UniformCompliance,
} from './types'

/** Filters shared by the Results dashboard and its CSV exports. */
export type ResultsFilter = { source_video?: string; camera_id?: string; store_id?: string; date?: string }

export const Api = {
  health: () => api<Health>('/api/health'),
  overview: () => api<Overview>('/api/overview'),
  layouts: () => api<Layout[]>('/api/layouts'),

  upload: (file: File, signal?: AbortSignal) => {
    const f = new FormData()
    f.append('file', file)
    return api<Media>('/api/media', { form: f, signal })
  },
  media: (id: string) => api<Media>(`/api/media/${id}`),

  analyse: (media_id: string, area: Area) => api<AnalysisDetail>('/api/analyses', { body: { media_id, area } }),
  analysis: (id: string) => api<AnalysisDetail>(`/api/analyses/${id}`),
  analyses: (p: { limit?: number; offset?: number; kind?: string; status?: string; q?: string } = {}) =>
    api<{ analyses: Analysis[]; total: number }>(`/api/analyses${qs(p)}`),
  cancel: (id: string) => api<{ ok: boolean }>(`/api/analyses/${id}/cancel`, { method: 'POST' }),
  deleteAnalysis: (id: string) => api<{ ok: boolean }>(`/api/analyses/${id}`, { method: 'DELETE' }),

  alerts: (p: { state?: string; kind?: string; limit?: number; offset?: number } = {}) =>
    api<{ alerts: Alert[]; total: number; new: number }>(`/api/alerts${qs(p)}`),
  setAlertState: (id: number, state: Alert['state']) => api<{ ok: boolean }>(`/api/alerts/${id}/state`, { body: { state } }),

  analytics: (start?: string) => api<Analytics>(`/api/analytics${qs({ start })}`),
  search: (q: string) => api<{ analyses: Analysis[]; alerts: Alert[] }>(`/api/search${qs({ q })}`),

  settings: () => api<{ settings: Settings; schema: SettingSchema[]; roi_preset: Area | null }>('/api/settings'),
  saveSettings: (s: Partial<Settings>) => api<{ settings: Settings }>('/api/settings', { method: 'PUT', body: s }),
  resetSettings: () => api<{ settings: Settings }>('/api/settings/reset', { method: 'POST' }),
  saveRoiPreset: (a: Area) => api<{ ok: boolean }>('/api/roi-preset', { method: 'PUT', body: a }),
  clearRoiPreset: () => api<{ ok: boolean }>('/api/roi-preset', { method: 'DELETE' }),

  extractionPlan: (mediaId: string, p: { sampling_strategy: SamplingStrategy; interval_seconds?: number; sample_count?: number;
    random_seed?: number; include_initial: boolean }) =>
    api<ExtractionPlan>(`/api/media/${mediaId}/extraction-plan${qs(p)}`),
  startExtraction: (b: { media_id: string; sampling_strategy?: SamplingStrategy; interval_seconds?: number; sample_count?: number;
    random_seed?: number; include_initial: boolean; new_run?: boolean; run_detection?: boolean; threshold?: number;
    store_id?: string; camera_id?: string; recording_start_time?: string }) =>
    api<Extraction>('/api/extractions', { body: b }),
  extraction: (id: string) => api<Extraction>(`/api/extractions/${id}`),
  extractions: (media_id?: string) => api<{ extractions: Extraction[] }>(`/api/extractions${qs({ media_id })}`),
  cancelExtraction: (id: string) => api<{ ok: boolean }>(`/api/extractions/${id}/cancel`, { method: 'POST' }),
  extractedFrames: (id: string, limit = 200) =>
    api<{ total: number; frames: ExtractedFrame[] }>(`/api/extractions/${id}/frames${qs({ limit })}`),

  startDetection: (extractionId: string, b: { threshold?: number; new_run?: boolean } = {}) =>
    api<DetectionRun>(`/api/extractions/${extractionId}/detect`, { body: b }),
  detectionRuns: (extractionId: string) => api<{ runs: DetectionRun[] }>(`/api/extractions/${extractionId}/detections`),
  cancelDetection: (id: string) => api<{ ok: boolean }>(`/api/detection-runs/${id}/cancel`, { method: 'POST' }),
  detectionResults: (id: string) => api<{ fields: string[]; images: DetectionImage[] }>(`/api/detection-runs/${id}/results`),
  /** One CSV row's evidence; `path` = the row's evidence_url (job + image + run). */
  screenshotEvidence: (path: string) => api<ScreenshotEvidence>(path),
  uniformCompliance: (extractionId: string) => api<UniformCompliance>(`/api/extractions/${extractionId}/uniform-compliance`),
  /** Cumulative uniform_compliance_master.csv (every finished video). */
  masterCsv: () => api<MasterCsvStatus>('/api/uniform-compliance/master'),
  syncMasterCsv: () => api<{ jobs_appended: number; rows_appended: number; status: MasterCsvStatus }>('/api/uniform-compliance/master/sync', { method: 'POST' }),
  /** Business reports (derived from persisted records): overview + latest, months, all rows. */
  reportsOverview: () => api<ReportsOverview>('/api/reports/overview'),
  monthReport: (month: string, by: MonthGrouping) => api<MonthReport>(`/api/reports/months/${month}${qs({ by })}`),
  monthCsvPath: (month: string, by: MonthGrouping) => `/api/reports/months/${month}.csv${qs({ by })}`,
  reportRecords: (p: { month?: string; by?: MonthGrouping; store?: string; video?: string; result?: string; q?: string; limit?: number } = {}) =>
    api<{ total: number; fields: string[]; records: HistoryRecord[] }>(`/api/reports/records${qs(p)}`),
  annotatedVideo: (extractionId: string) => api<AnnotatedVideo>(`/api/extractions/${extractionId}/annotated-video`),
  startAnnotatedVideo: (extractionId: string) => api<AnnotatedVideo>(`/api/extractions/${extractionId}/annotated-video`, { method: 'POST' }),

  frameResults: (p: ResultsFilter & { outcome?: string; processing_status?: string; limit?: number } = {}) =>
    api<FrameResults>(`/api/frame-results${qs(p)}`),
  /** Combined CSV exports (download links, not fetches). */
  manifestCsvPath: (p: ResultsFilter = {}) => `/api/frame-results/image_manifest.csv${qs(p)}`,
  resultsCsvPath: (p: ResultsFilter & { outcome?: string; processing_status?: string } = {}) =>
    `/api/frame-results/detection_results.csv${qs(p)}`,
  failuresCsvPath: (p: ResultsFilter = {}) => `/api/frame-results/processing_failures.csv${qs({ ...p, date: undefined })}`,
  hourlyCsvPath: (p: ResultsFilter = {}) => `/api/frame-results/hourly_summary.csv${qs(p)}`,
  /** Uniform compliance reports: one row per screenshot / one row per person found. */
  screenshotSummaryCsvPath: (p: ResultsFilter = {}) => `/api/frame-results/screenshot_summary.csv${qs(p)}`,
  complianceDetailsCsvPath: (p: ResultsFilter = {}) => `/api/frame-results/uniform_compliance_details.csv${qs(p)}`,
}
