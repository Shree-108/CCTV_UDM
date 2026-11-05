'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState, type ReactNode } from 'react'
import { AlertTriangle, Cctv, Clapperboard, Clock, Database, Download, FileSpreadsheet, Film, HardDrive, Images, RotateCcw, Table2, TriangleAlert } from 'lucide-react'
import { downloadUrl, url } from '../../api/client'
import { Api, type ResultsFilter } from '../../api/endpoints'
import type {
  AnalysisSummary, DetectionRun, Extraction, FrameOutcome, FrameResult, FrameResults, MonthGrouping, ProcessingFailure, ReportsOverview,
  ResultsVideo, UniformCompliance,
} from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx, dateTime, duration } from '../../lib/format'
import { Badge, Button, Field, PageHeader, Panel, Select } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { Configure, SampleList, UploadStep, samplingLabel } from '../extraction/ExtractionPage'
import { ImageNameCell, OUTCOME } from './ScreenshotEvidence'

const PROC_TONE: Record<string, 'ok' | 'warn' | 'crit'> = { ok: 'ok', annotation_failed: 'warn', failed: 'crit' }
const HOUR_TONE: Record<string, 'ok' | 'warn' | 'crit' | 'info' | 'neutral'> = {
  complete: 'ok', complete_with_failures: 'warn', extraction_incomplete: 'warn', analysis_incomplete: 'warn', not_analysed: 'neutral', in_progress: 'info',
}
const active = (s?: string | null) => s === 'queued' || s === 'running'
const unknown = <span className="text-fg-3">unknown</span>

/** Video upload → extraction → existing uniform model → frame-level results, from real backend files. */
export default function ResultsDashboard() {
  const router = useRouter()
  const params = useSearchParams()
  // uploads moved to Video Analysis: an old /results?media=<id> link continues there
  useEffect(() => { const m = params.get('media'); if (m) router.replace(`/analysis/video?quick=${m}`) }, [params, router])
  const [video, setVideo] = useState('')
  const [camera, setCamera] = useState('')
  const [store, setStore] = useState('')
  const [date, setDate] = useState('')
  const [outcome, setOutcome] = useState('')
  const [proc, setProc] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const filter: ResultsFilter = { source_video: video || undefined, camera_id: camera || undefined, store_id: store || undefined, date: date || undefined }
  const health = useQuery('health', Api.health, { poll: 20000 })
  const res = useQuery<FrameResults>(`frame-results:${video}:${camera}:${store}:${date}:${outcome}:${proc}`,
    () => Api.frameResults({ ...filter, outcome: outcome || undefined, processing_status: proc || undefined, limit: 1000 }),
    { poll: 5000 })
  const d = res.data
  const sel = d?.images.find((i) => `${i.run_id}/${i.image_id}` === selected) ?? null
  const u = health.data?.uniform_classifier
  const modelProblem = health.data && (health.data.status !== 'ok' || u?.state !== 'ready')
  const anyFilter = video || camera || store || date || outcome || proc
  const clear = () => { setVideo(''); setCamera(''); setStore(''); setDate(''); setOutcome(''); setProc(''); setSelected(null) }

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Results Dashboard"
        description="Results of every analysed video: the latest analysis, monthly reports, all historical CSV records and downloads. Every number is counted from the saved records. To analyse a new video, use Video Analysis."
        badge={d && <StatusBadges d={d} />} />

      {modelProblem && (
        <div className="flex items-start gap-2 rounded-md border border-crit/40 bg-crit-bg px-3 py-2 text-[12px] text-crit" role="alert">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
          <div><b>Uniform model not available for detection.</b> {health.data?.model_error ?? u?.label}.{' '}
            Frame extraction still works; detection runs will be refused until the trained model loads.{' '}
            <Link href="/settings#model-status" className="underline">Model status</Link></div>
        </div>
      )}
      {health.error && <ErrorState compact error={health.error} title="Backend not reachable" />}

      {params.get('job') && <>
        <SectionTitle>Selected analysis</SectionTitle>
        <ProcessPanel jobId={params.get('job')} mediaId={null} go={(q) => router.replace(q.startsWith('job=') ? `/results?${q}` : '/analysis/video')} />
      </>}
      <ReportSections />

      {res.error && !d ? <ErrorState error={res.error} onRetry={res.refetch} title="Results could not be loaded" /> : !d ? (
        <div className="py-10 text-center text-[13px] text-fg-3">Loading results…</div>
      ) : (
        <>
          <SectionTitle>F · Video history and outputs</SectionTitle>
          <VideosPanel videos={d.videos} />
          <SectionTitle>Technical details <span className="font-normal normal-case tracking-normal text-fg-3">— per-image and per-person data, filters and the technical CSV files</span></SectionTitle>
          <div className="flex flex-wrap items-end gap-3 rounded-md border border-line bg-panel px-3.5 py-2.5">
            <Field label="Source video" htmlFor="fv">
              <Select id="fv" value={video} onChange={(e) => { setVideo(e.target.value); setSelected(null) }} className="min-w-56">
                <option value="">All videos</option>
                {d.source_videos.map((v) => <option key={v} value={v}>{v}</option>)}
              </Select>
            </Field>
            <Field label="Camera" htmlFor="fc">
              <Select id="fc" value={camera} onChange={(e) => { setCamera(e.target.value); setSelected(null) }} disabled={!d.cameras.length}>
                <option value="">{d.cameras.length ? 'All cameras' : 'none recorded'}</option>
                {d.cameras.map((c) => <option key={c} value={c}>{c}</option>)}
              </Select>
            </Field>
            {d.stores.length > 0 && (
              <Field label="Store" htmlFor="fs">
                <Select id="fs" value={store} onChange={(e) => { setStore(e.target.value); setSelected(null) }}>
                  <option value="">All stores</option>
                  {d.stores.map((c) => <option key={c} value={c}>{c}</option>)}
                </Select>
              </Field>
            )}
            <Field label="Recording date" htmlFor="fd">
              <Select id="fd" value={date} onChange={(e) => { setDate(e.target.value); setSelected(null) }} disabled={!d.recording_dates.length}>
                <option value="">{d.recording_dates.length ? 'All dates' : 'no recording times known'}</option>
                {d.recording_dates.map((c) => <option key={c} value={c}>{c}</option>)}
              </Select>
            </Field>
            <Field label="Result" htmlFor="fo">
              <Select id="fo" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
                <option value="">All results</option>
                {(Object.keys(OUTCOME) as FrameOutcome[]).map((o) => <option key={o} value={o}>{o}</option>)}
              </Select>
            </Field>
            <Field label="Processing status" htmlFor="fp">
              <Select id="fp" value={proc} onChange={(e) => setProc(e.target.value)}>
                <option value="">All statuses</option>
                {['ok', 'failed', 'annotation_failed'].map((s) => <option key={s} value={s}>{s}</option>)}
              </Select>
            </Field>
            {anyFilter && <Button size="sm" variant="ghost" onClick={clear}>Clear filters</Button>}
          </div>

          <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
            <MetricCard label="Videos" value={d.summary.videos}
              hint={`${d.summary.extraction_jobs} job${d.summary.extraction_jobs === 1 ? '' : 's'}${d.summary.incomplete_jobs ? ` · ${d.summary.incomplete_jobs} incomplete` : ''}`} />
            <MetricCard label="Images extracted" value={d.summary.frames_extracted}
              hint={d.summary.extraction_failures ? `${d.summary.extraction_failures} could not be extracted` : 'screenshots saved'} />
            <MetricCard label="Images analysed" value={d.summary.frames_analysed}
              hint={d.summary.frames_not_processed ? `${d.summary.frames_not_processed} not analysed yet` : 'all extracted images'} />
            <MetricCard label="Uniform detected" value={d.summary.uniform_images} tone="ok" hint="images with ≥1 target shirt" />
            <MetricCard label="No uniform detected" value={d.summary.no_uniform_images} hint="images · not a confirmed violation" />
            <MetricCard label="Review required" value={d.summary.review_images} tone={d.summary.review_images ? 'warn' : undefined} hint="images a person should check" />
            <MetricCard label="Processing failed" value={d.summary.failed_images} tone={d.summary.failed_images ? 'crit' : undefined} hint="images with no prediction" />
            <MetricCard label="Target-shirt detections" value={d.summary.detections}
              hint={`detection records, not images or people · ${d.summary.people_detected} people found`} />
          </div>
          <p className="text-[11px] text-fg-3">
            <b className="text-fg-2">Image counts are screenshots, not people.</b> A person seen in 10 screenshots counts 10 times, and people are
            not linked across screenshots. Only the sampled screenshots are checked, so someone who appears between samples is not seen.
            Confidence is the model&apos;s score, not a guaranteed probability. The model checks the shirt pattern only — not logos, exact
            colours or the full dress code.
          </p>

          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[12px] text-fg-3"><Download className="mr-1 inline size-3.5" aria-hidden />CSV exports{anyFilter ? ' (filtered)' : ''}:</span>
            <a href={downloadUrl(Api.screenshotSummaryCsvPath(filter))}><Button size="sm" variant="primary">screenshot_summary</Button></a>
            <a href={downloadUrl(Api.complianceDetailsCsvPath(filter))}><Button size="sm" variant="primary">uniform_compliance_details</Button></a>
            <a href={downloadUrl(Api.manifestCsvPath(filter))}><Button size="sm">image_manifest</Button></a>
            <a href={downloadUrl(Api.resultsCsvPath({ ...filter, outcome: outcome || undefined, processing_status: proc || undefined }))}>
              <Button size="sm">detection_results</Button></a>
            <a href={downloadUrl(Api.failuresCsvPath(filter))}><Button size="sm">processing_failures</Button></a>
            <a href={downloadUrl(Api.hourlyCsvPath(filter))}><Button size="sm">hourly_summary</Button></a>
          </div>

          <Panel title="Screenshot results" icon={Table2} noPad
            subtitle={`${d.images_total} screenshot${d.images_total === 1 ? '' : 's'}${outcome || proc ? ' (filtered)' : ''} · click a row to see the evidence`}>
            <div className="grid xl:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)]">
              <div className="min-w-0 border-line xl:border-r">
                <DataTable rows={d.images} rowKey={(i) => `${i.run_id}/${i.image_id}`} dense pageSize={15}
                  selectedKey={selected} onRowClick={(i) => setSelected(`${i.run_id}/${i.image_id}`)}
                  empty={<EmptyState compact title={d.summary.frames_processed ? 'No screenshots match these filters' : 'No analysed screenshots yet'}
                    body={d.summary.frames_processed ? undefined : 'Upload a video above and start processing.'} />}
                  columns={[
                    { key: 'video', header: 'Video', cell: (i) => <span className="block max-w-36 truncate text-fg-2" title={i.source_video}>{i.source_video}</span>, sortValue: (i) => i.source_video },
                    { key: 'cam', header: 'Camera', hideable: true, cell: (i) => (i.camera_id ? <span className="font-mono text-[11px]">{i.camera_id}</span> : unknown), sortValue: (i) => i.camera_id },
                    { key: 'ts', header: 'Video time', cell: (i) => <span className="font-mono text-fg">{i.timestamp}</span>, sortValue: (i) => i.timestamp_seconds },
                    { key: 'rec', header: 'Recorded at', hideable: true, cell: (i) => (i.recording_timestamp ? <span className="font-mono text-[11px]">{i.recording_timestamp.replace('T', ' ').slice(0, 19)}</span> : unknown), sortValue: (i) => i.recording_timestamp },
                    { key: 'res', header: 'Result', cell: (i) => <Badge tone={OUTCOME[i.prediction].tone} dot>{OUTCOME[i.prediction].label}</Badge>, sortValue: (i) => i.prediction },
                    { key: 'n', header: 'Target shirts', align: 'right', cell: (i) => <span className="tabular">{i.prediction === 'PROCESSING_FAILED' ? '—' : i.detection_count}</span>, sortValue: (i) => i.detection_count },
                    { key: 'p', header: 'People', align: 'right', hideable: true, cell: (i) => <span className="tabular">{i.person_count ?? '—'}</span>, sortValue: (i) => i.person_count ?? -1 },
                    { key: 'conf', header: 'Confidence', cell: (i) => <span className="tabular font-mono text-[11px] text-fg-2">{i.confidences.length ? i.confidences.map((c) => c.toFixed(2)).join(', ') : '—'}</span>,
                      sortValue: (i) => (i.confidences.length ? Math.max(...i.confidences) : -1) },
                    { key: 'st', header: 'Processing', cell: (i) => <Badge tone={PROC_TONE[i.processing_status] ?? 'warn'}>{i.processing_status}</Badge>, sortValue: (i) => i.processing_status },
                    { key: 'rm', header: 'Remark', hideable: true, cell: (i) => <span className="block max-w-72 truncate text-[11px] text-fg-2" title={i.remark}>{i.remark}</span> },
                  ]} />
              </div>
              <FramePreview frame={sel} />
            </div>
          </Panel>

          <HourlyPanel hourly={d.hourly} />
          <FailuresPanel failures={d.failures} total={d.failures_total} />
          <p className="text-[11px] text-fg-3"><Cctv className="mr-1 inline size-3.5" aria-hidden />Live CCTV: {d.cctv.note}</p>
        </>
      )}
    </div>
  )
}

function StatusBadges({ d }: { d: FrameResults }) {
  const s = d.storage
  return (
    <span className="flex flex-wrap gap-1.5" title={`${s.s3.note} Files: ${s.frames_path}/`}>
      <Badge tone="neutral" dot><HardDrive className="mr-0.5 inline size-3" aria-hidden />Local storage</Badge>
      <Badge tone={s.s3.configured ? 'warn' : 'neutral'}>
        <Database className="mr-0.5 inline size-3" aria-hidden />{s.s3.configured ? 'S3: cart reports only · evidence stays local' : 'S3 not configured'}
      </Badge>
      <Badge tone="neutral" title={d.cctv.note}><Cctv className="mr-0.5 inline size-3" aria-hidden />CCTV API not connected</Badge>
    </span>
  )
}

// ---------------------------------------------------------------- upload → sampling → processing
/** THE quick sampled analysis (upload → validate → 4 random frames → existing model → CSVs), shared by the
 *  Results Dashboard and Video Analysis: both pages render this, so both call the same backend service
 *  (POST /api/media, POST /api/extractions with run_detection). `go` receives "", "media=<id>" or "job=<id>". */
export function ProcessPanel({ jobId, mediaId, go, startLabel = 'START PROCESSING', extra }: {
  jobId: string | null; mediaId: string | null; go: (q: string) => void; startLabel?: string; extra?: (jobId: string) => ReactNode
}) {
  if (jobId) return <><JobProgress id={jobId} onNew={() => go('')} extra={extra?.(jobId)} /><ComplianceCsvPanel id={jobId} /><AnnotatedVideoPanel id={jobId} /></>
  if (mediaId) {
    return <>
      <StageBar stages={[{ label: 'Received', status: 'done' }, { label: 'Validating', status: 'done' }, ...PENDING_STAGES]} />
      <DuplicateNotice mediaId={mediaId} />
      <Configure mediaId={mediaId} startLabel={startLabel} onReset={() => go('')}
        onStarted={(id) => { invalidate('frame-results'); invalidate('master-csv'); go(`job=${id}`) }} />
    </>
  }
  return <>
    <StageBar stages={[{ label: 'Received', status: 'active' }, { label: 'Validating', status: 'pending' }, ...PENDING_STAGES]} />
    <UploadStep showRecent={false} onUploaded={(id) => go(`media=${id}`)} />
  </>
}

type StageStatus = 'done' | 'active' | 'pending' | 'failed' | 'skipped'
const PENDING_STAGES: { label: string; status: StageStatus }[] = ['Sampling frames', 'Analyzing', 'Saving results', 'Completed']
  .map((label) => ({ label, status: 'pending' }))

/** Stages of one job, derived only from the stored job / run states (nothing is estimated). */
function jobStages(j: Extraction, run: DetectionRun | undefined): { label: string; status: StageStatus }[] {
  const bad = (s?: string) => s === 'failed' || s === 'cancelled' || s === 'interrupted'
  const sampling: StageStatus = active(j.state) ? 'active' : bad(j.state) ? 'failed' : 'done'
  const detecting: StageStatus = !run ? (active(j.state) ? 'pending' : bad(j.state) ? 'pending' : 'skipped')
    : run.state === 'done' ? 'done' : bad(run.state) ? 'failed' : run.state === 'running' && (run.progress ?? 0) >= 1 ? 'done' : run.state === 'running' ? 'active' : 'pending'
  const saving: StageStatus = !run ? 'pending' : run.state === 'done' ? 'done' : bad(run.state) ? 'pending'
    : run.state === 'running' && (run.progress ?? 0) >= 1 ? 'active' : 'pending'
  const failed = bad(j.state) || bad(run?.state)
  const end: { label: string; status: StageStatus } = failed ? { label: 'Failed', status: 'failed' }
    : run?.state === 'done' ? { label: 'Completed', status: 'done' }
      : !run && j.state === 'done' ? { label: 'Completed (no detection requested)', status: 'done' } : { label: 'Completed', status: 'pending' }
  return [{ label: 'Received', status: 'done' }, { label: 'Validating', status: 'done' }, { label: 'Sampling frames', status: sampling },
    { label: 'Analyzing', status: detecting }, { label: 'Saving results', status: saving }, end]
}

/** The same file (SHA-256) was delivered before: say so, link the earlier results, let the user decide. */
function DuplicateNotice({ mediaId }: { mediaId: string }) {
  const q = useQuery(`media:${mediaId}`, () => Api.media(mediaId))
  const prev = q.data?.previously_received ?? []
  if (!prev.length) return null
  const jobs = prev.flatMap((p) => p.jobs.map((j) => ({ ...j, received_at: p.received_at })))
  return (
    <div className="rounded-md border border-warn/40 bg-warn-bg px-3 py-2 text-[12px] text-warn" role="status">
      <b>This exact video was received before</b> ({prev.length} earlier deliver{prev.length === 1 ? 'y' : 'ies'}, first on {dateTime(prev[0].received_at)}).{' '}
      {jobs.length ? <>Earlier analyses: {jobs.slice(0, 4).map((j, i) => <span key={j.job_id}>{i > 0 && ', '}<Link className="underline" href={`/results?job=${j.job_id}`}>{j.job_id}</Link> ({j.state})</span>)}.
        {' '}Starting again creates a new, separate analysis with new random frames.</> : 'It was not analysed then.'}
    </div>
  )
}

const STAGE_TONE: Record<StageStatus, string> = {
  done: 'border-ok/40 bg-ok-bg text-ok', active: 'border-info/50 bg-info-bg text-info', pending: 'border-line text-fg-3',
  failed: 'border-crit/40 bg-crit-bg text-crit', skipped: 'border-line text-fg-3 line-through',
}

export function StageBar({ stages }: { stages: { label: string; status: StageStatus }[] }) {
  return (
    <ol className="flex flex-wrap items-center gap-1.5" aria-label="Analysis progress">
      {stages.map((s, i) => (
        <li key={s.label} className="flex items-center gap-1.5" aria-current={s.status === 'active' ? 'step' : undefined}>
          {i > 0 && <span className="text-fg-3" aria-hidden>→</span>}
          <span className={cx('rounded-md border px-2 py-0.5 text-[11px] font-semibold', STAGE_TONE[s.status])}>
            {s.status === 'done' ? '✓ ' : s.status === 'failed' ? '✕ ' : s.status === 'active' ? '● ' : ''}{s.label}
          </span>
        </li>
      ))}
    </ol>
  )
}

function JobProgress({ id, onNew, extra }: { id: string; onNew: () => void; extra?: ReactNode }) {
  const ext = useQuery(`extraction:${id}`, () => Api.extraction(id), { poll: 1000 })
  const runs = useQuery(`detruns:${id}`, () => Api.detectionRuns(id), { poll: 1000 })
  const j = ext.data?.id === id ? ext.data : undefined
  const run = runs.data?.runs[0]
  const finished = j && !active(j.state) && (!run || !active(run.state))
  useEffect(() => { if (finished) { invalidate('frame-results'); invalidate('master-csv') } }, [finished, run?.state])
  if (ext.error) return <Panel title="Processing"><ErrorState error={ext.error} title="This job could not be loaded" action={<Button size="sm" onClick={onNew}>Process a video</Button>} /></Panel>
  if (!j) return <Panel title="Processing"><div className="py-6 text-center text-[12px] text-fg-3">Loading…</div></Panel>
  const ePct = active(j.state) ? Math.round((j.progress ?? 0) * 100) : j.state === 'done' ? 100 : Math.round(((j.extracted + j.failed) / Math.max(1, j.planned)) * 100)
  const dPct = run ? Math.round((run.progress ?? 0) * 100) : 0
  return (
    <Panel title="Processing" icon={Film} subtitle={`${j.source_video} · ${samplingLabel(j)} · job ${j.id}`}
      actions={<>
        {extra}
        <Link href={`/extract?job=${id}`}><Button size="sm" variant="ghost">Job details</Button></Link>
        <Button size="sm" variant="ghost" icon={RotateCcw} onClick={onNew}>Process another video</Button>
      </>}>
      <div className="mb-3"><StageBar stages={jobStages(j, run)} /></div>
      <div className="grid gap-3 md:grid-cols-2">
        <Stage title="1 · Screenshot extraction" state={j.state} pct={ePct}
          lines={[`Video duration ${duration(j.plan.duration_seconds)}`, `${j.extracted} of ${j.planned} screenshots saved${j.failed ? ` · ${j.failed} failed` : ''}`,
            `Store ${j.store_id ?? 'unknown'} · camera ${j.camera_id ?? 'unknown'}`]}
          error={j.error} />
        <Stage title="2 · Uniform detection" state={run?.state ?? (active(j.state) ? 'waiting' : 'not started')} pct={dPct}
          lines={run ? [`${run.processed} of ${run.total} screenshots processed · ${run.failed} failed`,
            `${run.uniform_images} uniform detected · ${run.no_uniform_images} no uniform · ${run.review_images} review required`,
            run.model_version ? `Model ${run.model_version.split(' · ')[0]} · threshold ${run.threshold.toFixed(2)} · run ${run.id}` : '']
            : [active(j.state) ? 'Starts when extraction finishes (if selected).' : 'Detection was not requested for this job. Open "Job details" to run it.']}
          error={run?.error ?? null} />
      </div>
      {j.samples && (
        <div className="mt-3">
          <p className="mb-1.5 text-[12px] font-semibold text-fg">Selected timestamps <span className="font-normal text-fg-3">(position in the video{j.random_seed != null ? ` · seed ${j.random_seed}` : ''})</span></p>
          <SampleList samples={j.samples} />
        </div>
      )}
    </Panel>
  )
}

/** The business-facing uniform_compliance_job_<job>.csv: six columns, one row per sampled screenshot. */
function ComplianceCsvPanel({ id }: { id: string }) {
  // Keep polling while extraction OR detection is requested / queued / running. (Stopping when extraction
  // finished left the panel at "not analysed" while the detection run was still to start.)
  const [busyPoll, setBusyPoll] = useState(true)
  const q = useQuery(`uniform-compliance:${id}`, () => Api.uniformCompliance(id), { poll: busyPoll ? 1500 : 15000 })
  const c = q.data?.job_id === id ? q.data : undefined
  const busy = active(c?.extraction_state) || active(c?.run_state) || active(c?.detection_state) || !!c?.detection_pending
  useEffect(() => { if (c) setBusyPoll(busy) }, [c, busy])
  return (
    <Panel title="Uniform Compliance CSV" icon={FileSpreadsheet} noPad={!!c?.records.length}
      subtitle={c ? `job ${c.job_id} · ${c.screenshots_sampled} screenshot${c.screenshots_sampled === 1 ? '' : 's'} sampled · ${c.file_name}` : undefined}
      actions={c && !busy ? <a href={downloadUrl(c.csv_url)}><Button size="sm" variant="primary" icon={Download}>Download this video&apos;s CSV</Button></a>
        : <Button size="sm" variant="primary" icon={Download} disabled>Download this video&apos;s CSV</Button>}>
      {q.error ? <ErrorState compact error={q.error} title="The compliance report could not be loaded" onRetry={q.refetch} />
        : !c ? <div className="py-6 text-center text-[12px] text-fg-3">Loading…</div>
          : !c.records.length ? <p className="text-[12px] text-fg-3">{active(c.extraction_state) ? 'Screenshots are being extracted…' : 'This job has no sampled screenshots.'}</p> : (
            <>
              {(busy || !c.run_id) && (
                <p className={cx('px-3.5 pt-2.5 text-[11px]', !busy && c.detection_state && c.detection_state !== 'done' ? 'text-crit' : 'text-fg-3')}>
                  {busy ? 'Processing — the rows fill in as the uniform model analyses the screenshots.'
                    : c.detection_state && c.detection_state !== 'done' ? `The detection run ended "${c.detection_state}" before analysing these screenshots${c.detection_error ? `: ${c.detection_error}` : ''}. No prediction was made, so the scores are blank.`
                      : 'Detection was not requested for this job, so no prediction was made and the scores are blank. Open "Job details" to run it.'}</p>
              )}
              <EvidenceStrip records={c.records} />
              <BusinessTable fields={c.fields} records={c.records} />
            </>
          )}
    </Panel>
  )
}

function Stage({ title, state, pct, lines, error }: { title: string; state: string; pct: number; lines: string[]; error: string | null }) {
  const tone = state === 'done' ? 'ok' : state === 'failed' ? 'crit' : active(state) ? 'info' : 'warn'
  return (
    <div className="rounded-md border border-line bg-panel-2 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-[12px] font-semibold text-fg">{title}</span>
        <Badge tone={tone} dot>{state}</Badge>
      </div>
      <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-panel" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className={cx('h-full transition-[width] duration-500', state === 'failed' ? 'bg-crit' : 'bg-info')} style={{ width: `${pct}%` }} />
      </div>
      {lines.filter(Boolean).map((l) => <p key={l} className="text-[12px] text-fg-2">{l}</p>)}
      {error && <p className="mt-1.5 rounded border border-crit/30 bg-crit-bg px-2 py-1 text-[12px] text-crit" role="alert">{error}</p>}
    </div>
  )
}

// ---------------------------------------------------------------- evidence
function FramePreview({ frame }: { frame: FrameResult | null }) {
  if (!frame) {
    return <div className="flex min-h-48 items-center justify-center p-6 text-center text-[12px] text-fg-3"><span><Images className="mx-auto mb-1 size-5" aria-hidden />Select a screenshot to see the original, the annotated copy and its details.</span></div>
  }
  const o = OUTCOME[frame.prediction]
  const shown = frame.people.length
  return (
    <div className="flex min-w-0 flex-col gap-2 p-3">
      <div className="flex flex-wrap items-center gap-2 text-[12px]">
        <Badge tone={o.tone} dot>{o.label}</Badge>
        <span className="font-mono text-fg">{frame.timestamp}</span>
        <span className="truncate text-fg-3" title={frame.source_video}>{frame.source_video}</span>
      </div>
      <p className="text-[11px] text-fg-2">{o.help}.</p>
      <div className="grid gap-2 sm:grid-cols-2">
        <Figure title="Original (unchanged)" src={url(frame.original_url)} />
        {frame.annotated_url ? <Figure title="Annotated copy" src={url(frame.annotated_url)} />
          : <div className="flex aspect-video items-center justify-center rounded-md border border-dashed border-line text-[11px] text-fg-3">No annotated image ({frame.processing_status})</div>}
      </div>
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-[11px]">
        <dt className="text-fg-3">Video time</dt><dd className="font-mono text-fg-2">{frame.timestamp} <span className="font-sans text-fg-3">(position in the video)</span></dd>
        <dt className="text-fg-3">Recorded at</dt><dd className="text-fg-2">{frame.recording_timestamp ? <span className="font-mono">{frame.recording_timestamp}</span> : 'unknown — the recording start time was not supplied'}</dd>
        <dt className="text-fg-3">Store / camera</dt><dd className="text-fg-2">{frame.store_id || 'unknown'} / {frame.camera_id || 'unknown'}</dd>
        <dt className="text-fg-3">Target shirts</dt><dd className="text-fg-2">{frame.prediction === 'PROCESSING_FAILED' ? 'no prediction' : frame.detection_count}{frame.confidences.length ? ` · confidence ${frame.confidences.map((c) => c.toFixed(2)).join(', ')}` : ''}</dd>
        <dt className="text-fg-3">People found</dt><dd className="text-fg-2">{frame.person_count ?? '—'}{shown ? ` · ${frame.people.map((p) => `${p.type}${p.confidence != null ? ` ${p.confidence.toFixed(2)}` : ''}`).join(', ')}` : ''}</dd>
        <dt className="text-fg-3">Threshold</dt><dd className="text-fg-2">{frame.threshold.toFixed(2)}</dd>
        <dt className="text-fg-3">Remark</dt><dd className="text-fg-2">{frame.remark}</dd>
        <dt className="text-fg-3">Processing</dt><dd className="text-fg-2">{frame.processing_status}{frame.review_reason ? ` · ${frame.review_reason}` : ''}</dd>
        <dt className="text-fg-3">Review</dt><dd className="text-fg-2">{frame.review_status}</dd>
        <dt className="text-fg-3">Model</dt><dd className="truncate font-mono text-fg-2" title={frame.model_version}>{frame.model_version}</dd>
        <dt className="text-fg-3">Image ID</dt><dd className="truncate font-mono text-fg-2">{frame.image_id}</dd>
        <dt className="text-fg-3">Job / run</dt><dd className="truncate font-mono text-fg-2">{frame.extraction_id} / {frame.run_id}</dd>
        <dt className="text-fg-3">Files</dt><dd className="truncate font-mono text-fg-3" title={`${frame.original_image_path}\n${frame.annotated_image_path}`}>{frame.original_image_path}</dd>
      </dl>
    </div>
  )
}

function Figure({ title, src }: { title: string; src: string }) {
  return (
    <figure className="min-w-0">
      <a href={src} target="_blank" rel="noreferrer">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={title} className="block max-h-[320px] w-full rounded-md border border-line bg-video object-contain" />
      </a>
      <figcaption className="mt-0.5 text-[11px] text-fg-3">{title}</figcaption>
    </figure>
  )
}

// ---------------------------------------------------------------- jobs / hourly / failures
function VideosPanel({ videos }: { videos: ResultsVideo[] }) {
  const router = useRouter()
  if (!videos.length) return null
  return (
    <Panel title="Analysis history" icon={Film} noPad subtitle="Every processed video and job · click a row to open its screenshots, CSV rows and annotated video">
      <DataTable rows={videos} rowKey={(v) => v.extraction_id} dense pageSize={10} onRowClick={(v) => router.push(`/results?job=${v.extraction_id}`)}
        columns={[
          { key: 'v', header: 'Source video', cell: (v) => <span className="block max-w-56 truncate text-fg" title={`${v.source_video} · job ${v.extraction_id}`}>{v.source_video}</span>, sortValue: (v) => v.source_video },
          { key: 'rd', header: 'Recording date', cell: (v) => (v.recording_start_time ? <span className="font-mono text-[11px]">{v.recording_start_time.slice(0, 10)}</span> : unknown), sortValue: (v) => v.recording_start_time ?? '' },
          { key: 'sc', header: 'Store / camera', cell: (v) => <span className="text-[11px] text-fg-2">{v.store_id ?? 'unknown'} / {v.camera_id ?? 'unknown'}</span>, sortValue: (v) => v.camera_id ?? '' },
          { key: 'i', header: 'Sampling', cell: (v) => <span className="text-[11px]">{samplingLabel(v)}</span> },
          { key: 'd', header: 'Duration', cell: (v) => duration(v.duration_seconds), sortValue: (v) => v.duration_seconds ?? 0 },
          { key: 'e', header: 'Extracted', align: 'right', cell: (v) => <span className="tabular">{v.extracted}/{v.planned}</span> },
          { key: 'p', header: 'Analysed', align: 'right', cell: (v) => <span className="tabular">{v.processed}</span> },
          { key: 's', header: 'Status', cell: (v) => <span className="text-[11px] text-fg-2">extraction {v.state}{v.last_run_state ? ` · detection ${v.last_run_state}` : ' · no detection run'}</span> },
          { key: 'err', header: 'Problem', cell: (v) => <span className="block max-w-72 truncate text-[11px] text-crit" title={v.last_run_error ?? v.extraction_error ?? ''}>{v.last_run_error ?? v.extraction_error ?? ''}</span> },
          { key: 'c', header: 'Created', cell: (v) => <span className="text-[11px] text-fg-3">{dateTime(v.created_at)}</span>, sortValue: (v) => v.created_at },
        ]} />
    </Panel>
  )
}

function HourlyPanel({ hourly }: { hourly: FrameResults['hourly'] }) {
  return (
    <Panel title="Hourly summary" icon={Clock} noPad={hourly.available}>
      {!hourly.available ? <p className="text-[12px] text-fg-3">{hourly.reason}</p> : (
        <>
          <p className="px-3.5 pt-2.5 text-[11px] text-fg-3">Grouped by {hourly.basis}. Only hours that contain footage are listed. Counts are screenshots, not people.</p>
          <DataTable rows={hourly.rows} rowKey={(r) => `${r.extraction_job_id}/${r.hour_start}`} dense pageSize={12}
            columns={[
              { key: 'v', header: 'Source video', cell: (r) => <span className="block max-w-48 truncate" title={r.extraction_job_id}>{r.source_video}</span> },
              { key: 'c', header: 'Camera', cell: (r) => r.camera_id || unknown },
              { key: 'h', header: 'Hour', cell: (r) => <span className="font-mono text-[11px]">{r.hour_start.replace('T', ' ')}</span> },
              { key: 'x', header: 'Expected', align: 'right', cell: (r) => r.expected_samples },
              { key: 's', header: 'Saved', align: 'right', cell: (r) => r.successful_samples },
              { key: 'f', header: 'Failed', align: 'right', cell: (r) => r.failed_samples + r.processing_failed_images },
              { key: 'a', header: 'Analysed', align: 'right', cell: (r) => r.analysed_images },
              { key: 'u', header: 'Uniform', align: 'right', cell: (r) => r.uniform_images },
              { key: 'n', header: 'No uniform', align: 'right', cell: (r) => r.no_uniform_images },
              { key: 'r', header: 'Review', align: 'right', cell: (r) => r.review_required_images },
              { key: 'd', header: 'Shirt detections', align: 'right', cell: (r) => r.target_shirt_detections },
              { key: 'st', header: 'Status', cell: (r) => <Badge tone={HOUR_TONE[r.summary_status] ?? 'neutral'}>{r.summary_status.replace(/_/g, ' ')}</Badge> },
            ]} />
        </>
      )}
    </Panel>
  )
}

function FailuresPanel({ failures, total }: { failures: ProcessingFailure[]; total: number }) {
  return (
    <Panel title={`Processing errors (${total})`} icon={TriangleAlert} noPad={total > 0}>
      {!total ? <p className="text-[12px] text-fg-3">No failed extractions, inferences or runs for these filters.</p> : (
        <>
          <DataTable rows={failures} rowKey={(f) => `${f.stage}/${f.extraction_job_id}/${f.detection_run_id}/${f.image_id}`} dense pageSize={10}
            columns={[
              { key: 'st', header: 'Stage', cell: (f) => <Badge tone="crit">{f.stage.replace(/_/g, ' ')}</Badge>, sortValue: (f) => f.stage },
              { key: 'v', header: 'Video', cell: (f) => <span className="block max-w-44 truncate" title={f.source_video}>{f.source_video}</span> },
              { key: 't', header: 'Video time', cell: (f) => <span className="font-mono text-[11px]">{f.video_timestamp_hhmmss || '—'}</span> },
              { key: 'c', header: 'Category', cell: (f) => <span className="font-mono text-[11px]">{f.error_category}</span> },
              { key: 'm', header: 'Message', cell: (f) => <span className="block max-w-96 truncate text-[11px] text-fg-2" title={f.error_message}>{f.error_message}</span> },
              { key: 'r', header: 'Retry', cell: (f) => <span className="text-[11px]">{f.retry_status.replace(/_/g, ' ')}</span> },
            ]} />
          {total > failures.length && <p className="px-3.5 pb-3 text-[11px] text-fg-3">Showing {failures.length} of {total}; the processing_failures CSV lists all of them.</p>}
        </>
      )}
    </Panel>
  )
}

// ---------------------------------------------------------------- cumulative CSV / evidence / annotated video
/** E · uniform_compliance_master.csv: the six business columns for every finished video, appended job by
 *  job, with all its rows searchable below. */
function MasterCsvPanel({ months }: { months: string[] }) {
  const q = useQuery('master-csv', Api.masterCsv, { poll: 5000 })
  const [syncing, setSyncing] = useState(false)
  const [syncError, setSyncError] = useState<string | null>(null)
  const m = q.data
  const sync = async () => {
    setSyncing(true); setSyncError(null)
    try { await Api.syncMasterCsv(); invalidate('master-csv') } catch (e) { setSyncError(e instanceof Error ? e.message : String(e)) } finally { setSyncing(false) }
  }
  return (
    <Panel title="All historical CSV data (master CSV)" icon={FileSpreadsheet}
      subtitle={m ? `${m.file_name} · all videos · ${m.rows} row${m.rows === 1 ? '' : 's'} from ${m.videos} video${m.videos === 1 ? '' : 's'}${m.updated_at ? ` · last appended ${dateTime(m.updated_at)}` : ''}` : undefined}
      actions={m?.exists ? <a href={downloadUrl(m.csv_url)}><Button size="sm" variant="primary" icon={Download}>Download Master CSV</Button></a>
        : <Button size="sm" variant="primary" icon={Download} disabled title="No video has finished processing yet">Download Master CSV</Button>}>
      {q.error ? <ErrorState compact error={q.error} title="The master CSV status could not be loaded" onRetry={q.refetch} />
        : !m ? <div className="py-2 text-center text-[12px] text-fg-3">Loading…</div> : (
          <div className="flex flex-col gap-1.5 text-[12px] text-fg-2">
            <p>Columns: <span className="font-mono text-[11px]">{m.fields.join(', ')}</span>. One row per sampled screenshot; each finished video is
              appended once and earlier rows are never changed. Stored on {m.storage} at <span className="font-mono text-[11px]">{m.path}</span>.</p>
            {m.last_error && (
              <p className="rounded-md border border-crit/40 bg-crit-bg px-2.5 py-1.5 text-crit" role="alert">
                <b>The last append failed</b> (job {m.last_error.job_id}): {m.last_error.message}</p>
            )}
            {m.pending_jobs.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 rounded-md border border-warn/40 bg-warn-bg px-2.5 py-1.5 text-warn">
                <span>{m.pending_jobs.length} finished video{m.pending_jobs.length === 1 ? ' is' : 's are'} not in the master CSV yet
                  ({m.pending_jobs.slice(0, 3).map((p) => p.source_video).join(', ')}{m.pending_jobs.length > 3 ? ', …' : ''}).</span>
                <Button size="sm" onClick={sync} disabled={syncing}>{syncing ? 'Appending…' : 'Append missing rows'}</Button>
              </div>
            )}
            {syncError && <p className="text-crit" role="alert">{syncError}</p>}
            <HistoryTable months={months} />
          </div>
        )}
    </Panel>
  )
}

/** The job's sampled screenshots (annotated copy when it exists) with their timestamp and remark. */
function EvidenceStrip({ records }: { records: UniformCompliance['records'] }) {
  const shots = records.filter((r) => r.original_url)
  if (!shots.length) return null
  return (
    <div className="grid grid-cols-2 gap-2 border-b border-line px-3.5 py-2.5 lg:grid-cols-4">
      {shots.map((r) => {
        const src = url(r.annotated_url ?? r.original_url!)
        return (
          <figure key={r.image_id} className="min-w-0">
            <a href={src} target="_blank" rel="noreferrer" title="Open full size">
              <img src={src} alt={`Screenshot at ${r.row['Time Stamp']}`} loading="lazy"
                className="block aspect-video w-full rounded-md border border-line bg-video object-contain" />
            </a>
            <figcaption className="mt-1 text-[11px] leading-snug">
              <span className="font-mono text-fg">{r.row['Time Stamp']}</span>
              <span className="text-fg-3"> · {r.annotated_url ? 'annotated' : 'original'}{r.row['confidence score'] ? ` · ${r.row['confidence score']}` : ''}</span>
              <span className="block text-fg-2">{r.row['remark (Example)']}</span>
            </figcaption>
          </figure>
        )
      })}
    </div>
  )
}

/** Full-video annotated output of the job's source video — offered only when the file exists and decodes. */
function AnnotatedVideoPanel({ id }: { id: string }) {
  const [starting, setStarting] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const q = useQuery(`annotated-video:${id}`, () => Api.annotatedVideo(id), { poll: 3000 })
  const v = q.data?.job_id === id ? q.data : undefined
  const start = async () => {
    setStarting(true); setErr(null)
    try { await Api.startAnnotatedVideo(id); invalidate(`annotated-video:${id}`) } catch (e) { setErr(e instanceof Error ? e.message : String(e)) } finally { setStarting(false) }
  }
  return (
    <Panel title="Annotated video" icon={Clapperboard} subtitle={v?.note}
      actions={v?.available && v.download_url ? <a href={url(v.download_url)}><Button size="sm" variant="primary" icon={Download}>Download annotated video</Button></a> : undefined}>
      {q.error ? <ErrorState compact error={q.error} title="The annotated video status could not be loaded" onRetry={q.refetch} />
        : !v ? <div className="py-2 text-center text-[12px] text-fg-3">Loading…</div>
          : v.available && v.video_url ? (
            <div className="flex flex-col gap-1.5">
              {v.playable_in_browser ? <video src={url(v.video_url)} controls preload="metadata" className="max-h-[420px] w-full rounded-md border border-line bg-video" />
                : <p className="text-[12px] text-warn">This file could not be converted to H.264, so the browser may not play it. Download it and open it in a video player.</p>}
              <p className="text-[11px] text-fg-3">Full-video analysis {v.analysis_id}</p>
            </div>
          ) : active(v.state) ? (
            <p className="text-[12px] text-fg-2">Analysing the whole video… {Math.round(v.progress * 100)}% (analysis {v.analysis_id}). The download appears when the file is saved and verified.</p>
          ) : (
            <div className="flex flex-wrap items-center gap-2 text-[12px] text-fg-2">
              <span>{v.state === 'unavailable' ? v.error
                : v.state === 'not_started' ? 'No annotated video yet. The screenshots above do not cover the whole video; this runs a separate pass over the full recording and takes a while.'
                  : `The last attempt ended "${v.state}"${v.error ? `: ${v.error}` : ''}.`}</span>
              {v.state !== 'unavailable' && <Button size="sm" onClick={start} disabled={starting}>{starting ? 'Starting…' : v.state === 'not_started' ? 'Generate annotated video' : 'Try again'}</Button>}
              {err && <span className="text-crit" role="alert">{err}</span>}
            </div>
          )}
    </Panel>
  )
}

// ---------------------------------------------------------------- business reports (A–E)
function SectionTitle({ children }: { children: ReactNode }) {
  return <h2 className="mt-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-fg-2">{children}</h2>
}

const STATUS_TONE: Record<AnalysisSummary['status'], 'ok' | 'crit' | 'info' | 'neutral'> = {
  completed: 'ok', failed: 'crit', in_progress: 'info', not_analysed: 'neutral',
}
const SAMPLE_NOTE = 'Quick sample-based report: only the sampled screenshots were checked, not every frame of the video, so a violation between samples is not seen.'

/** A–E: summary, latest analysis, its CSV, monthly reports, all historical rows — all from /api/reports. */
function ReportSections() {
  const ov = useQuery('reports-overview', Api.reportsOverview, { poll: 5000 })
  const o = ov.data
  if (ov.error && !o) return <ErrorState error={ov.error} onRetry={ov.refetch} title="The reports could not be loaded" />
  if (!o) return <div className="py-6 text-center text-[12px] text-fg-3">Loading reports…</div>
  const s = o.summary
  return (
    <>
      <SectionTitle>A · Summary</SectionTitle>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
        <MetricCard label="Videos received" value={s.videos_received} hint="uploaded videos stored" />
        <MetricCard label="Analyses completed" value={s.analyses_completed} tone="ok"
          hint={`${s.analyses} analyses${s.analyses_in_progress ? ` · ${s.analyses_in_progress} running` : ''}${s.analyses_not_analysed ? ` · ${s.analyses_not_analysed} not analysed` : ''}`} />
        <MetricCard label="Screenshots extracted" value={s.screenshots_extracted} hint="sampled frames saved" />
        <MetricCard label="Screenshots analysed" value={s.screenshots_analysed} hint={`${s.business_records} CSV records`} />
        <MetricCard label="Failed analyses" value={s.analyses_failed} tone={s.analyses_failed ? 'crit' : undefined} hint="extraction or detection failed" />
      </div>

      <SectionTitle>B · Latest video analysis</SectionTitle>
      <LatestAnalysisPanel o={o} />

      <SectionTitle>C · Latest analysis CSV</SectionTitle>
      <Panel title="Latest analysis CSV" icon={FileSpreadsheet} noPad={!!o.latest?.records.length}
        subtitle={o.latest ? `job ${o.latest.job_id} only · ${o.latest.records.length} row${o.latest.records.length === 1 ? '' : 's'} · six business columns` : undefined}
        actions={o.latest ? <a href={downloadUrl(o.latest.csv_url)}><Button size="sm" variant="primary" icon={Download}>Download Latest Analysis CSV</Button></a>
          : <Button size="sm" variant="primary" icon={Download} disabled>Download Latest Analysis CSV</Button>}>
        {o.latest ? <BusinessTable fields={o.latest.fields} records={o.latest.records} />
          : <p className="text-[12px] text-fg-3">No analysis has completed yet.</p>}
      </Panel>

      <SectionTitle>D · Monthly reports</SectionTitle>
      <MonthlyPanel months={o.months} rules={o.grouping_rules} />

      <SectionTitle>E · All historical CSV data</SectionTitle>
      <MasterCsvPanel months={o.months.completion} />
    </>
  )
}

function LatestAnalysisPanel({ o }: { o: ReportsOverview }) {
  const a = o.latest
  const n = o.newest_other
  return (
    <Panel title="Latest completed analysis" icon={Images} noPad={!!a?.records.length}
      subtitle={a ? `${a.source_video} · job ${a.job_id}` : undefined}
      actions={a ? <Link href={`/results?job=${a.job_id}`}><Button size="sm" variant="ghost">Open job · annotated video</Button></Link> : undefined}>
      {n && (
        <p className={cx('mx-3.5 mt-2.5 rounded-md border px-2.5 py-1.5 text-[12px]', n.status === 'failed' ? 'border-crit/40 bg-crit-bg text-crit' : 'border-info/40 bg-info-bg text-fg-2')} role={n.status === 'failed' ? 'alert' : 'status'}>
          The newest job ({n.source_video}, job {n.job_id}) is <b>{n.status.replace('_', ' ')}</b>{n.error ? `: ${n.error}` : ''}.{' '}
          {a ? 'The result below is the latest COMPLETED analysis, not that job.' : ''}
        </p>
      )}
      {!a ? <p className="text-[12px] text-fg-3">{n ? '' : 'No analysis yet. '}Analyse a video in <Link className="underline" href="/analysis/video">Video Analysis</Link>.</p> : (
        <>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 border-b border-line px-3.5 py-2.5 text-[12px] md:grid-cols-4">
            <div><dt className="text-fg-3">Store</dt><dd className="text-fg">{a.store_id ?? unknown}</dd></div>
            <div><dt className="text-fg-3">Recording date</dt><dd className="font-mono text-fg">{a.recording_start_time ? a.recording_start_time.replace('T', ' ').slice(0, 16) : unknown}</dd></div>
            <div><dt className="text-fg-3">Completed</dt><dd className="text-fg">{dateTime(a.completed_at)}</dd></div>
            <div><dt className="text-fg-3">Status</dt><dd><Badge tone={STATUS_TONE[a.status]} dot>{a.status}</Badge></dd></div>
            <div><dt className="text-fg-3">Screenshots</dt><dd className="text-fg">{a.screenshots_analysed} analysed of {a.screenshots_extracted} extracted ({a.screenshots_planned} planned)</dd></div>
            <div><dt className="text-fg-3">Results</dt><dd className="text-fg">{a.uniform} uniform · {a.no_uniform} no uniform · {a.review} review{a.processing_failed ? ` · ${a.processing_failed} failed` : ''}</dd></div>
            <div><dt className="text-fg-3">Video</dt><dd className="text-fg">{duration(a.duration_seconds)} · received {dateTime(a.received_at)}</dd></div>
            <div><dt className="text-fg-3">Seed</dt><dd className="font-mono text-fg">{a.random_seed ?? '—'}</dd></div>
          </dl>
          <EvidenceStrip records={a.records} />
          <p className="px-3.5 py-2 text-[11px] text-fg-3">{SAMPLE_NOTE}</p>
        </>
      )}
    </Panel>
  )
}

/** The six business columns, with links to the evidence (hover / click the img name for the row's own screenshot). */
function BusinessTable({ fields, records }: { fields: UniformCompliance['fields']; records: UniformCompliance['records'] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12px]">
        <thead><tr className="border-b border-line text-left text-[11px] text-fg-3">
          {fields.map((f) => <th key={f} scope="col" className="px-3 py-2 font-semibold whitespace-nowrap">{f}</th>)}
          <th scope="col" className="px-3 py-2 font-semibold">Evidence</th>
        </tr></thead>
        <tbody>
          {records.map((r) => (
            <tr key={r.image_id} className="border-b border-line/60 align-top last:border-0">
              <td className="px-3 py-1.5 whitespace-nowrap">{r.row['Store Name'] || unknown}</td>
              <td className="px-3 py-1.5 font-mono whitespace-nowrap" title={r.row.Date ? undefined : 'The recording start time was not supplied'}>{r.row.Date || unknown}</td>
              <td className="px-3 py-1.5 font-mono">{r.row['Time Stamp']}</td>
              <td className="px-3 py-1.5 font-mono text-[11px]"><ImageNameCell name={r.row['img name']} links={r} /></td>
              <td className="px-3 py-1.5 font-mono text-[11px]">{r.row['confidence score'] || '—'}</td>
              <td className="min-w-64 px-3 py-1.5 text-fg-2">{r.row['remark (Example)']}</td>
              <td className="px-3 py-1.5 whitespace-nowrap">
                {r.original_url && <a className="underline" href={url(r.original_url)} target="_blank" rel="noreferrer">original</a>}
                {r.annotated_url && <> · <a className="underline" href={url(r.annotated_url)} target="_blank" rel="noreferrer">annotated</a></>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MonthlyPanel({ months, rules }: { months: ReportsOverview['months']; rules: ReportsOverview['grouping_rules'] }) {
  const [by, setBy] = useState<MonthGrouping>('completion')
  const [picked, setPicked] = useState('')
  const list = months[by]
  const month = list.includes(picked) ? picked : list[0] ?? ''
  const q = useQuery(`month-report:${by}:${month}`, () => (month ? Api.monthReport(month, by) : Promise.resolve(null)), { poll: 10000 })
  const r = q.data && q.data.month === month && q.data.by === by ? q.data : null
  const label = (m: string) => new Date(`${m}-01T00:00:00`).toLocaleDateString(undefined, { month: 'long', year: 'numeric' })
  return (
    <Panel title="Monthly reports" icon={Clock}
      subtitle={month ? `${label(month)} · grouped by ${by === 'completion' ? 'analysis completion date' : 'recording date'}` : undefined}
      actions={r ? <a href={downloadUrl(r.csv_url)}><Button size="sm" variant="primary" icon={Download}>Download Monthly CSV</Button></a>
        : <Button size="sm" variant="primary" icon={Download} disabled>Download Monthly CSV</Button>}>
      <div className="flex flex-col gap-2.5">
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Group by" htmlFor="mg">
            <Select id="mg" value={by} onChange={(e) => setBy(e.target.value as MonthGrouping)}>
              <option value="completion">Analysis completion date (default)</option>
              <option value="recording">Recording date (only when supplied)</option>
            </Select>
          </Field>
          <Field label="Month" htmlFor="mm">
            <Select id="mm" value={month} onChange={(e) => setPicked(e.target.value)} disabled={!list.length}>
              {list.length ? list.map((m) => <option key={m} value={m}>{label(m)}</option>) : <option value="">no months with records</option>}
            </Select>
          </Field>
          <p className="max-w-xl text-[11px] text-fg-3">{rules[by]}</p>
        </div>
        {!list.length ? <EmptyState compact title="No records for this grouping"
          body={by === 'recording' ? 'No analysed video has a recording date. Enter the recording start time when starting an analysis.' : 'No video has been received yet.'} />
          : q.error ? <ErrorState compact error={q.error} title="This month could not be loaded" onRetry={q.refetch} />
            : !r ? <div className="py-4 text-center text-[12px] text-fg-3">Loading…</div> : (
              <>
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
                  <MetricCard label="Videos received" value={r.videos_received} hint={by === 'completion' ? 'by receipt date' : 'with this recording month'} />
                  <MetricCard label="Analyses completed" value={r.analyses_completed} tone="ok" />
                  <MetricCard label="Failed analyses" value={r.analyses_failed} tone={r.analyses_failed ? 'crit' : undefined} />
                  <MetricCard label="Screenshots extracted" value={r.screenshots_extracted} />
                  <MetricCard label="Screenshots analysed" value={r.screenshots_analysed} />
                  <MetricCard label="Uniform detected" value={r.uniform} tone="ok" hint="screenshots" />
                  <MetricCard label="No uniform / review" value={`${r.no_uniform} / ${r.review}`} tone={r.review ? 'warn' : undefined} hint="screenshots" />
                  <MetricCard label="CSV records" value={r.business_records} hint="rows in the monthly CSV" />
                </div>
                <DataTable rows={r.analyses_list} rowKey={(a) => a.job_id} dense pageSize={8}
                  columns={[
                    { key: 'v', header: 'Video', cell: (a) => <Link className="block max-w-56 truncate text-fg hover:underline" href={`/results?job=${a.job_id}`} title={`job ${a.job_id}`}>{a.source_video}</Link>, sortValue: (a) => a.source_video },
                    { key: 's', header: 'Store', cell: (a) => a.store_id ?? unknown },
                    { key: 'r', header: 'Recording date', cell: (a) => a.recording_date ?? unknown },
                    { key: 'st', header: 'Status', cell: (a) => <Badge tone={STATUS_TONE[a.status]} dot>{a.status}</Badge>, sortValue: (a) => a.status },
                    { key: 'n', header: 'Screenshots', align: 'right', cell: (a) => <span className="tabular">{a.screenshots_analysed}/{a.screenshots_extracted}</span> },
                    { key: 'e', header: 'Ended', cell: (a) => <span className="text-[11px] text-fg-3">{dateTime(a.ended_at)}</span>, sortValue: (a) => a.ended_at ?? '' },
                    { key: 'x', header: 'Problem', cell: (a) => <span className="block max-w-64 truncate text-[11px] text-crit" title={a.error ?? ''}>{a.error ?? ''}</span> },
                  ]} />
              </>
            )}
      </div>
    </Panel>
  )
}

/** Every master-CSV row (completed analyses), filtered in the browser. */
function HistoryTable({ months }: { months: string[] }) {
  const q = useQuery('report-records', () => Api.reportRecords({ limit: 5000 }), { poll: 10000 })
  const [month, setMonth] = useState('')
  const [store, setStore] = useState('')
  const [video, setVideo] = useState('')
  const [result, setResult] = useState('')
  const [text, setText] = useState('')
  const all = q.data?.records ?? []
  const stores = [...new Set(all.map((r) => r.row['Store Name']))].sort()
  const videos = [...new Set(all.map((r) => r.source_video))].sort()
  const t = text.trim().toLowerCase()
  const rows = all.filter((r) => (!month || r.month === month) && (!store || r.row['Store Name'] === store) && (!video || r.source_video === video)
    && (!result || r.prediction === result) && (!t || `${Object.values(r.row).join(' ')} ${r.source_video} ${r.job_id}`.toLowerCase().includes(t)))
  if (q.error && !q.data) return <ErrorState compact error={q.error} title="The historical records could not be loaded" onRetry={q.refetch} />
  return (
    <div className="mt-1 flex flex-col gap-2">
      <div className="flex flex-wrap items-end gap-3">
        <Field label="Month (completion)" htmlFor="hm"><Select id="hm" value={month} onChange={(e) => setMonth(e.target.value)}>
          <option value="">All months</option>{months.map((m) => <option key={m} value={m}>{m}</option>)}</Select></Field>
        <Field label="Store" htmlFor="hs"><Select id="hs" value={store} onChange={(e) => setStore(e.target.value)}>
          <option value="">All stores</option>{stores.map((s) => <option key={s} value={s}>{s || '(unknown)'}</option>)}</Select></Field>
        <Field label="Video" htmlFor="hv"><Select id="hv" value={video} onChange={(e) => setVideo(e.target.value)} className="min-w-48">
          <option value="">All videos</option>{videos.map((v) => <option key={v} value={v}>{v}</option>)}</Select></Field>
        <Field label="Result" htmlFor="hr"><Select id="hr" value={result} onChange={(e) => setResult(e.target.value)}>
          <option value="">All results</option>{(Object.keys(OUTCOME) as FrameOutcome[]).map((o) => <option key={o} value={o}>{o}</option>)}</Select></Field>
        <Field label="Search" htmlFor="hq"><input id="hq" value={text} onChange={(e) => setText(e.target.value)} placeholder="image, remark, job…"
          className="h-8 rounded-md border border-line bg-panel px-2 text-[12px] text-fg placeholder:text-fg-3" /></Field>
        <span className="pb-1.5 text-[11px] text-fg-3">{rows.length} of {q.data?.total ?? 0} rows</span>
      </div>
      <DataTable rows={rows} rowKey={(r) => `${r.job_id}/${r.image_id}`} dense pageSize={15}
        empty={<EmptyState compact title={all.length ? 'No rows match these filters' : 'No completed analyses yet'} />}
        columns={[
          { key: 'sn', header: 'Store Name', cell: (r) => r.row['Store Name'] || unknown, sortValue: (r) => r.row['Store Name'] },
          { key: 'd', header: 'Date', cell: (r) => <span className="font-mono">{r.row.Date || unknown}</span>, sortValue: (r) => r.row.Date },
          { key: 'ts', header: 'Time Stamp', cell: (r) => <span className="font-mono">{r.row['Time Stamp']}</span> },
          { key: 'img', header: 'img name', cell: (r) => <ImageNameCell name={r.row['img name']} links={r} />, sortValue: (r) => r.row['img name'] },
          { key: 'c', header: 'confidence score', cell: (r) => <span className="font-mono text-[11px]">{r.row['confidence score'] || '—'}</span> },
          { key: 'rm', header: 'remark (Example)', cell: (r) => <span className="block max-w-80 truncate text-[11px] text-fg-2" title={r.row['remark (Example)']}>{r.row['remark (Example)']}</span> },
          { key: 'v', header: 'Video', hideable: true, cell: (r) => <Link className="block max-w-40 truncate text-[11px] text-fg-2 hover:underline" href={`/results?job=${r.job_id}`} title={`job ${r.job_id}`}>{r.source_video}</Link>, sortValue: (r) => r.source_video },
        ]} />
    </div>
  )
}
