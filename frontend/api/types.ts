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
  sha256?: string | null
  /** only in the upload response: earlier deliveries of the identical file */
  previously_received?: PreviouslyReceived[]
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
  extraction_default_interval_seconds: number
  extraction_default_strategy: SamplingStrategy
  extraction_default_sample_count: number
  extraction_image_format: 'png' | 'jpg'
  extraction_max_frames: number
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
  storage: StorageStatus
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

/** Exactly which frames an extraction job would produce (GET /api/media/{id}/extraction-plan). */
/** random_segments = "Random across video": one random frame from each of N equal parts */
export type SamplingStrategy = 'interval' | 'random' | 'random_segments'

export interface ExtractionPlan {
  total: number
  /** samples that are not the initial frame (scheduled or random) */
  scheduled: number
  initial: number
  strategy: SamplingStrategy
  interval_seconds: number | null
  sample_count: number | null
  random_seed: number | null
  coverage_note: string
  /** random_segments: [start, end] seconds of each part */
  segments?: [number, number][]
  duration_seconds: number
  last_frame_seconds: number
  explanation: string
  first_timestamps: number[]
  max_frames: number
  allowed: boolean
  reason?: string
}

export interface ExtractionSample {
  image_id: string
  sample_type: 'initial' | 'scheduled' | 'random'
  target_seconds: number
  target_hhmmss: string
  target_frame: number
  status: 'saved' | 'failed' | 'pending' | 'not_extracted'
  frame_number: number | null
  video_timestamp_hhmmss: string | null
}

export interface ExtractionFailure {
  image_id: string
  sample_type: 'initial' | 'scheduled' | 'random'
  target_seconds: number
  target_hhmmss: string
  target_frame: number
  error_category?: string
  error_message: string
}

export interface Extraction {
  id: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  media_id: string
  source_video: string
  state: 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted'
  sampling_strategy: SamplingStrategy
  /** interval strategy only */
  interval_seconds: number | null
  /** random strategy only */
  sample_count: number | null
  random_seed: number | null
  /** null = not supplied (never guessed) */
  store_id: string | null
  camera_id: string | null
  recording_start_time: string | null
  source: 'upload' | 'cctv_adapter' | 's3'
  include_initial: boolean
  image_format: 'png' | 'jpg'
  planned: number
  extracted: number
  failed: number
  progress: number
  error: string | null
  output_dir: string
  plan: ExtractionPlan
  /** the selected timestamps with their status (jobs of up to 50 samples; null for older / larger jobs) */
  samples: ExtractionSample[] | null
  failures: ExtractionFailure[]
  manifest_url: string
  failures_url: string
  reused?: boolean
}

export type FrameOutcome = 'UNIFORM_DETECTED' | 'NO_UNIFORM_DETECTED' | 'REVIEW_REQUIRED' | 'PROCESSING_FAILED'

/** A run of the uniform model over an extraction's frames. */
export interface DetectionRun {
  id: string
  extraction_id: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  state: 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted'
  threshold: number
  model_version: string | null
  total: number
  processed: number
  succeeded: number
  failed: number
  uniform_images: number
  no_uniform_images: number
  review_images: number
  detections: number
  progress: number
  error: string | null
  output_dir: string | null
  results_csv_url: string
  results_url: string
  screenshot_summary_url: string
  compliance_details_url: string
  reused?: boolean
}

/** One image of detection_results.csv (rows grouped by image_id). */
export interface DetectionImage {
  image_id: string
  extraction_job_id: string
  detection_run_id: string
  source_video: string
  store_id: string
  camera_id: string
  video_timestamp_seconds: number
  video_timestamp_hhmmss: string
  /** real clock time - only when the recording start time is known, otherwise '' */
  recording_timestamp: string
  person_count: number | null
  prediction: FrameOutcome
  processing_status: string
  review_status: string
  review_reason: string
  model_version: string
  detection_count: number
  original_url: string
  annotated_url: string | null
  detections: { confidence: number; bbox: [number, number, number, number] }[]
}

export interface StorageStatus {
  mode: 'local'
  path: string
  frames_path: string
  s3: { configured: boolean; bucket: string | null; uploads_enabled: boolean; note: string }
}

/** One processed frame on the Results dashboard (rows of detection_results.csv grouped by image). */
export interface FrameResult {
  extraction_id: string
  run_id: string
  image_id: string
  source_video: string
  store_id: string
  camera_id: string
  /** video-relative position (HH:MM:SS.mmm) */
  timestamp: string
  timestamp_seconds: number
  /** real clock time - only when the recording start time is known, otherwise '' */
  recording_timestamp: string
  sampling_strategy: SamplingStrategy
  prediction: FrameOutcome
  detection_count: number
  person_count: number | null
  people: { type: 'uniform_shirt' | 'not_uniform' | 'uncertain'; confidence: number | null; bbox: [number, number, number, number] | null }[]
  confidences: number[]
  original_image_path: string
  annotated_image_path: string
  processing_status: 'ok' | 'failed' | 'annotation_failed'
  review_status: string
  review_reason: string
  model_version: string
  threshold: number
  original_url: string
  annotated_url: string | null
  /** human-readable summary generated from this screenshot's actual detections */
  remark: string
}

/** The six columns of uniform_compliance_job_<job>.csv, in order (Sample CCTV CSV format). */
export type ComplianceCsvRow = {
  'Store Name': string; 'Date': string; 'Time Stamp': string; 'img name': string; 'confidence score': string; 'remark (Example)': string
}

/** Preview of a job's business-facing uniform compliance CSV: one record per sampled screenshot. */
export interface UniformCompliance {
  job_id: string
  screenshots_sampled: number
  extracted: number
  extraction_state: string
  run_id: string | null
  run_state: string | null
  /** newest detection run of any state; detection_pending = requested, run not created yet */
  detection_state: string | null
  detection_error: string | null
  detection_pending: boolean
  file_name: string
  saved: boolean
  csv_url: string
  fields: (keyof ComplianceCsvRow)[]
  records: ({
    row: ComplianceCsvRow
    image_id: string
    processing_status: string
    prediction: FrameOutcome | ''
    extraction_status: 'saved' | 'failed' | 'not_extracted'
    recording_timestamp: string
    job_id: string
    run_id: string | null
  } & EvidenceLinks)[]
}

/** Where ONE CSV row's evidence is: addressed by job id + image id + run id, never by the file name alone. */
export interface EvidenceLinks {
  original_url: string | null
  /** small JPEG made in memory for hover previews */
  thumb_url: string | null
  annotated_url: string | null
  evidence_url: string
}

/** GET /api/extractions/{job}/evidence/{image_id}?run_id= — stored records only, the model is not re-run. */
export interface ScreenshotEvidence {
  job_id: string; run_id: string | null; run_state: string | null; image_id: string
  source_video: string; store_id: string; camera_id: string
  video_timestamp_hhmmss: string; video_timestamp_seconds: number; frame_number: number | null
  recording_timestamp: string | null; extraction_status: string; extraction_error: string
  prediction: FrameOutcome | ''; processing_status: string; people_detected: number | null
  people: { person: number; type: string; confidence: number | null; person_confidence: number | null; reason: string }[]
  confidence_threshold: number | null; model_version: string; review_reason: string
  row: ComplianceCsvRow
  original: { filename: string | null; url: string | null; thumb_url: string | null; available: boolean }
  annotated: { filename: string | null; url: string | null; available: boolean }
}

export interface ResultsVideo {
  extraction_id: string
  source_video: string
  media_id: string
  store_id: string | null
  camera_id: string | null
  recording_start_time: string | null
  source: 'upload' | 'cctv_adapter' | 's3'
  created_at: string
  state: Extraction['state']
  sampling_strategy: SamplingStrategy
  interval_seconds: number | null
  sample_count: number | null
  random_seed: number | null
  duration_seconds: number | null
  planned: number
  extracted: number
  extraction_failed: number
  extraction_error: string | null
  progress: number
  run_id: string | null
  run_state: DetectionRun['state'] | null
  processed: number
  threshold: number | null
  model_version: string | null
  last_run_state: DetectionRun['state'] | null
  last_run_error: string | null
  last_run_progress: number | null
}

export interface HourlyRow {
  source_video: string
  extraction_job_id: string
  detection_run_id: string
  store_id: string
  camera_id: string
  hour_basis: 'recording_time' | 'video_relative'
  hour_start: string
  hour_end: string
  expected_samples: number
  successful_samples: number
  failed_samples: number
  analysed_images: number
  processing_failed_images: number
  uniform_images: number
  no_uniform_images: number
  review_required_images: number
  target_shirt_detections: number
  person_detections: number
  summary_status: string
}

export interface ProcessingFailure {
  stage: 'extraction_job' | 'extraction' | 'detection_run' | 'inference' | 'annotation'
  extraction_job_id: string
  detection_run_id: string
  image_id: string
  source_video: string
  store_id: string
  camera_id: string
  video_timestamp_seconds: string | number
  video_timestamp_hhmmss: string
  target_frame: string | number
  original_image_path: string
  error_category: string
  error_message: string
  retry_status: 'not_retried' | 'retried_ok' | 'retried_failed'
}

export interface CctvStatus {
  state: 'not_configured'
  adapter: null
  note: string
  requirements: string[]
}

export interface FrameResults {
  summary: {
    videos: number; extraction_jobs: number; frames_extracted: number; extraction_failures: number
    frames_processed: number; frames_analysed: number
    uniform_images: number; no_uniform_images: number; review_images: number; failed_images: number
    detections: number; people_detected: number; frames_not_processed: number; incomplete_jobs: number
    processing_failures: number
  }
  videos: ResultsVideo[]
  source_videos: string[]
  cameras: string[]
  stores: string[]
  recording_dates: string[]
  images: FrameResult[]
  images_total: number
  hourly: { available: false; reason: string } | { available: true; basis: string; rows: HourlyRow[] }
  failures: ProcessingFailure[]
  failures_total: number
  storage: StorageStatus
  cctv: CctvStatus
  counting_note: string
}

/** One row of image_manifest.csv (+ the image URL). */
export interface ExtractedFrame {
  image_id: string
  extraction_job_id: string
  source_video: string
  store_id: string
  camera_id: string
  video_timestamp_seconds: string
  video_timestamp_hhmmss: string
  frame_number: string
  sampling_strategy: SamplingStrategy
  sampling_interval_seconds: string
  image_filename: string
  original_image_path: string
  image_width: string
  image_height: string
  extraction_status: string
  source_video_id: string
  sample_type: 'initial' | 'scheduled' | 'random'
  random_seed: string
  recording_start_time: string
  recording_timestamp: string
  url: string
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

/** GET /api/uniform-compliance/master — what uniform_compliance_master.csv contains (local disk only). */
export interface MasterCsvStatus {
  file_name: string
  path: string
  exists: boolean
  size_bytes: number
  rows: number
  videos: number
  fields: (keyof ComplianceCsvRow)[]
  updated_at: string | null
  /** finished jobs whose rows are not in the file yet (e.g. after a failed write) */
  pending_jobs: { job_id: string; source_video: string }[]
  last_error: { job_id: string; run_id: string | null; message: string; at: string } | null
  storage: string
  csv_url: string
}

/** GET /api/extractions/{id}/annotated-video — the full-video analysis pass of the job's source video. */
export interface AnnotatedVideo {
  job_id: string
  media_id: string
  analysis_id: string | null
  state: 'not_started' | 'unavailable' | 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted'
  progress: number
  /** true only when the annotated file exists and decodes */
  available: boolean
  video_url: string | null
  download_url: string | null
  playable_in_browser?: boolean
  error: string | null
  note: string
}

/** One analysis (extraction job + its reporting run) as the business reports see it. */
export interface AnalysisSummary {
  job_id: string
  run_id: string | null
  media_id: string
  source_video: string
  source: string
  store_id: string | null
  camera_id: string | null
  recording_start_time: string | null
  recording_date: string | null
  received_at: string
  size_bytes: number | null
  duration_seconds: number | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  ended_at: string | null
  status: 'completed' | 'failed' | 'in_progress' | 'not_analysed'
  error: string | null
  sampling_strategy: string
  random_seed: number | null
  screenshots_planned: number
  screenshots_extracted: number
  screenshots_analysed: number
  uniform: number
  no_uniform: number
  review: number
  processing_failed: number
  in_master_csv: boolean
}

export type MonthGrouping = 'completion' | 'recording'

/** GET /api/reports/overview */
export interface ReportsOverview {
  summary: {
    videos_received: number; analyses: number; analyses_completed: number; analyses_failed: number
    analyses_in_progress: number; analyses_not_analysed: number; screenshots_extracted: number
    screenshots_analysed: number; business_records: number
  }
  latest: (AnalysisSummary & { fields: (keyof ComplianceCsvRow)[]; records: UniformCompliance['records']; csv_url: string }) | null
  newest_other: AnalysisSummary | null
  months: Record<MonthGrouping, string[]>
  grouping_rules: Record<MonthGrouping, string>
}

/** GET /api/reports/months/{YYYY-MM} */
export interface MonthReport {
  month: string; by: MonthGrouping; rule: string
  videos_received: number; analyses: number; analyses_completed: number; analyses_failed: number; analyses_in_progress: number
  screenshots_extracted: number; screenshots_analysed: number
  uniform: number; no_uniform: number; review: number; processing_failed: number
  business_records: number; fields: (keyof ComplianceCsvRow)[]; analyses_list: AnalysisSummary[]; csv_url: string
}

/** GET /api/reports/records — one business row of a completed analysis (= a master CSV row). */
export interface HistoryRecord extends EvidenceLinks {
  row: ComplianceCsvRow
  job_id: string; run_id: string; source_video: string; image_id: string
  prediction: FrameOutcome; processing_status: string; month: string | null; completed_at: string | null; in_master_csv: boolean
}

/** Earlier deliveries of the same file (same SHA-256), returned by POST /api/media. */
export interface PreviouslyReceived {
  media_id: string; filename: string; received_at: string
  jobs: { job_id: string; state: string; created_at: string }[]
}
