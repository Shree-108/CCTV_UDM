// Types mirror the FastAPI responses in backend/api/routes.py.

export type ComplianceStatus = 'compliant' | 'no_uniform' | 'uncertain' | 'no_person'
export type PersonLabel = 'uniform' | 'non_uniform' | 'uncertain'
export type MediaKind = 'image' | 'video'
export type JobState = 'queued' | 'running' | 'done' | 'failed'
export type Point = [number, number]
export type Rect = [number, number, number, number]

export interface Media {
  id: string
  created_at: string
  kind: MediaKind
  filename: string
  size_bytes: number
  width: number
  height: number
  fps: number | null
  frames: number | null
  duration_sec: number | null
  suggested_layout: string
  preview_url: string
  file_url: string
}

export interface Layout { id: string; label: string; rows: number; cols: number; cells: Rect[] }

export interface Area {
  rect: Rect
  roi: Point[] | null
  label: string
  layout: string
  cell: number | null
}

export interface LiveProgress {
  frame: number
  frames_total: number
  processed: number
  progress: number | null
  people: number
  uniformed: number
  non_uniform: number
  uncertain: number
  people_now: number
  current_status: ComplianceStatus
  monitoring?: boolean
  processing_fps: number
  media_time_sec: number
  alerts: number
  segments?: [number, number][]
}

export interface PersonResult {
  track_id: number
  label: PersonLabel
  confidence: number
  det_confidence: number
  first_seen_sec: number | null
  last_seen_sec: number | null
  frames: number
  uniform_votes: number
  non_uniform_votes: number
  uncertain_votes: number
  note: string
  box: Rect | null
}

export interface Alert {
  id: number
  created_at: string
  analysis_id: string
  kind: 'no_uniform' | 'low_confidence'
  severity: 'crit' | 'warn'
  title: string
  message: string
  media_time_sec: number | null
  people_inside: number | null
  uniformed: number | null
  confidence: number | null
  track_id: number | null
  filename: string
  state: 'new' | 'acknowledged' | 'resolved'
  evidence_url: string | null
}

export interface Analysis {
  id: string
  created_at: string
  finished_at: string | null
  media_id: string
  kind: MediaKind
  filename: string
  state: JobState
  progress: number
  error: string | null
  status: ComplianceStatus | null
  status_title: string | null
  people: number | null
  uniformed: number | null
  non_uniform: number | null
  uncertain: number | null
  avg_uniform_conf: number | null
  max_uniform_conf: number | null
  duration_sec: number | null
  processing_sec: number | null
  frames_total: number | null
  frames_processed: number | null
  processing_fps: number | null
  longest_presence_sec: number | null
  alerts: number
  uniform_obs: number | null
  non_uniform_obs: number | null
  uncertain_obs: number | null
  area: Area | null
  settings: Settings | null
  classifier: string | null
  model_version: string | null
  /** 'capture' for scheduled/manual camera captures; null or 'upload' for uploads */
  source: 'upload' | 'capture' | null
  store_id: string | null
  camera_id: string | null
  capture_id: string | null
  /** UTC time the frame was taken (captures only) */
  captured_at: string | null
  device: string | null
  live: LiveProgress | null
  files: Partial<Record<'annotated' | 'keyframe' | 'report_json' | 'report_csv' | 'report_txt', string>>
  preview_url: string
}

export interface AnalysisDetail extends Analysis {
  persons: PersonResult[]
  alert_list: Alert[]
  live_frame_url: string | null
}

export interface Settings {
  person_confidence: number
  uniform_confidence: number
  analysis_fps: number
  inference_size: number
  temporal_confirmation_frames: number
  gpu_acceleration: 'auto' | 'on' | 'off'
  output_video_quality: 'low' | 'medium' | 'high'
  absence_alert_seconds: number
  non_uniform_confidence: number
  frame_skip: number
  vote_window: number
  presence_gap_tolerance: number
  min_track_observations: number
  min_person_height_px: number
  roi_reference_point: 'bottom_center' | 'center' | 'upper_center'
  untrained_model_mode: 'baseline' | 'disabled'
  low_confidence_alerts: boolean
  capture_enabled: boolean
  capture_interval_minutes: number
  capture_active_start: string
  capture_active_end: string
  default_store_id: string
  video_step_seconds: number
}

export interface SettingSchema {
  key: keyof Settings
  default: number | string | boolean
  type: 'float' | 'int' | 'str' | 'bool'
  min: number | null
  max: number | null
  advanced: boolean
  description: string
  choices: string[] | null
}

export interface Health {
  status: 'ok' | 'degraded'
  model_error: string | null
  person_detector: { state: 'ready' | 'missing'; model: string; path: string }
  uniform_classifier: {
    state: 'ready' | 'not_trained' | 'mismatch'
    label: string
    backend: string
    version: string
    model: ModelIdentity
    classes: PersonLabel[]
    overall_rule: 'any_uniform' | 'all_uniform'
    uniform_confidence: number
    non_uniform_confidence: number
    config: string
  }
  tracking: { state: 'ready'; tracker: string }
  gpu: { available: boolean; setting: string; device: string }
  evaluation: EvaluationReport | null
  queue: number
  live_cctv: 'not_implemented'
}

/** Which uniform weights file this installation runs (config/uniform_config.yaml). */
export interface ModelIdentity {
  name: string
  version: string
  status: string | null
  dataset_version: string | null
  path: string
  expected_path: string
  present: boolean
  sha256: string | null
  expected_sha256: string | null
  verified: boolean | null
}

export interface EvaluationReport {
  model: string
  UNIFORM_CONFIDENCE: number
  sets: Record<string, {
    n: number
    accuracy: number
    per_class: Record<PersonLabel, { precision: number; recall: number; f1: number; support: number }>
    confusion_matrix: { labels: PersonLabel[]; rows_true_cols_pred: number[][] }
    uniform_false_positives: number
    uniform_false_positive_rate: number | null
  }>
}

export interface Overview {
  latest: AnalysisDetail | null
  recent: Analysis[]
  analyses_total: number
  alerts_new: number
  alerts_total: number
  recent_alerts: Alert[]
  running: Analysis[]
}

export interface AnalyticsAgg {
  analyses: number
  with_people: number
  compliant: number
  no_uniform: number
  uncertain_status: number
  no_person: number
  people: number
  uniformed: number
  non_uniform: number
  uncertain: number
  alerts: number
  uniform_obs: number
  non_uniform_obs: number
  uncertain_obs: number
  videos: number
  images: number
  processing_sec: number
  footage_sec: number
  avg_uniform_confidence: number | null
  compliance_rate: number | null
}

export interface Analytics extends AnalyticsAgg {
  by_day: { day: string; analyses: number; compliant: number; with_people: number }[]
  today: AnalyticsAgg
}
