'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import { Ban, Download, Film, Info, ListChecks, Play, RotateCcw, Upload } from 'lucide-react'
import { ApiError, downloadUrl, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Extraction, ExtractionPlan, ExtractionSample, SamplingStrategy } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx, dateTime, duration } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Segmented, Toggle } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { DropZone, MediaInfo } from '../analysis/common'
import { DetectionPanel } from './DetectionPanel'

const ACCEPT = 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mp4,.mov,.avi,.mkv,.m4v,.webm'
const STATE_TONE: Record<Extraction['state'], 'ok' | 'warn' | 'crit' | 'info' | 'neutral'> = {
  queued: 'info', running: 'info', done: 'ok', failed: 'crit', cancelled: 'warn', interrupted: 'warn',
}
const intervalLabel = (s: number) => (s >= 60 && s % 60 === 0 ? `${s / 60} min` : `${s} s`)
/** "every 30 s" / "5 random frames (seed 42)" - the sampling strategy as recorded on the job. */
export const samplingLabel = (j: Pick<Extraction, 'sampling_strategy' | 'interval_seconds' | 'sample_count' | 'random_seed'>) =>
  j.sampling_strategy === 'random_segments' ? `${j.sample_count ?? '?'} screenshots across the video (seed ${j.random_seed ?? '?'})`
    : j.sampling_strategy === 'random' ? `${j.sample_count ?? '?'} random frames (seed ${j.random_seed ?? '?'})`
    : `every ${intervalLabel(j.interval_seconds ?? 0)}`
/** Store / camera / recording start, with "unknown" for anything that was not supplied. */
export const sourceLabel = (j: Pick<Extraction, 'store_id' | 'camera_id' | 'recording_start_time'>) =>
  `store ${j.store_id ?? 'unknown'} · camera ${j.camera_id ?? 'unknown'} · recording start ${j.recording_start_time ?? 'unknown'}`

/** Video upload → interval → real frames saved unchanged + image_manifest.csv (no model is run here). */
export default function ExtractionPage() {
  const params = useSearchParams()
  const jobId = params.get('job')
  const mediaId = params.get('media')
  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Frame Extraction"
        description="Upload a CCTV video and save real frames at a fixed interval, with a CSV manifest of every saved image. Frames are saved exactly as decoded — nothing is added, removed or changed." />
      {jobId ? <JobView id={jobId} /> : mediaId ? <Configure mediaId={mediaId} /> : <UploadStep />}
    </div>
  )
}

// ---------------------------------------------------------------- 1. upload
/** Reused by the Results dashboard: `onUploaded` keeps it on the same page. */
export function UploadStep({ onUploaded, showRecent = true }: { onUploaded?: (mediaId: string) => void; showRecent?: boolean } = {}) {
  const router = useRouter()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const recent = useQuery('extractions', () => Api.extractions())
  const onFile = async (f: File) => {
    setBusy(true); setError(null)
    try {
      const m = await Api.upload(f)
      if (m.kind !== 'video') throw new Error('That is an image. Frame extraction needs a video file.')
      if (onUploaded) onUploaded(m.id)
      else router.push(`/extract?media=${m.id}`)
    } catch (e) {
      setError(e as Error)
    } finally {
      setBusy(false)
    }
  }
  return (
    <>
      <Panel title="Upload CCTV video" icon={Upload}>
        <DropZone accept={ACCEPT} onFile={onFile} busy={busy} label="DROP CCTV VIDEO" formats={['MP4', 'AVI', 'MOV', 'MKV', 'M4V', 'WEBM']}
          hint="or click to browse · the file is checked for a readable duration, resolution and frame rate" />
        {error && <div className="mt-3"><ErrorState compact error={error} title="Upload failed" onRetry={() => setError(null)} /></div>}
      </Panel>
      {showRecent && <RecentJobs jobs={recent.data?.extractions ?? []} />}
    </>
  )
}

function RecentJobs({ jobs }: { jobs: Extraction[] }) {
  const router = useRouter()
  if (!jobs.length) return null
  return (
    <Panel title="Recent extractions" icon={ListChecks} noPad>
      <ul className="divide-y divide-line">
        {jobs.slice(0, 10).map((j) => (
          <li key={j.id}>
            <button onClick={() => router.push(`/extract?job=${j.id}`)} className="flex w-full items-center gap-3 px-3.5 py-2 text-left text-[12px] hover:bg-hover">
              <Badge tone={STATE_TONE[j.state]} dot>{j.state}</Badge>
              <span className="min-w-0 flex-1 truncate text-fg">{j.source_video}</span>
              <span className="text-fg-3">{samplingLabel(j)}</span>
              {j.camera_id && <span className="hidden font-mono text-fg-3 md:inline">{j.camera_id}</span>}
              <span className="tabular font-mono text-fg-2">{j.extracted}/{j.planned}</span>
              <span className="hidden text-fg-3 sm:inline">{dateTime(j.created_at)}</span>
            </button>
          </li>
        ))}
      </ul>
    </Panel>
  )
}

// ---------------------------------------------------------------- 2. choose interval
/** Reused by the Results dashboard: `onStarted` / `onReset` keep it on the same page. */
export function Configure({ mediaId, onStarted, onReset, startLabel = 'EXTRACT FRAMES' }: {
  mediaId: string; onStarted?: (jobId: string) => void; onReset?: () => void; startLabel?: string
}) {
  const router = useRouter()
  const { toast } = useApp()
  const media = useQuery(`media:${mediaId}`, () => Api.media(mediaId))
  const settings = useQuery('settings', Api.settings)
  const def = settings.data?.settings.extraction_default_interval_seconds ?? 30
  const [value, setValue] = useState<string | null>(null)
  const [unit, setUnit] = useState<'s' | 'min'>('s')
  const [initialV, setInitial] = useState<boolean | null>(null)
  const [detectAfter, setDetectAfter] = useState(true)
  const [starting, setStarting] = useState(false)
  const [strategyV, setStrategy] = useState<SamplingStrategy | null>(null)
  const [countV, setCount] = useState<string | null>(null)
  // defaults come from Settings: random across the video, 4 screenshots, no extra initial frame
  const strategy: SamplingStrategy = strategyV ?? settings.data?.settings.extraction_default_strategy ?? 'random_segments'
  const count = countV ?? String(settings.data?.settings.extraction_default_sample_count ?? 4)
  const initial = initialV ?? strategy === 'interval'
  const [seed, setSeed] = useState('')
  const [storeId, setStoreId] = useState('')
  const [cameraId, setCameraId] = useState('')
  const [recStart, setRecStart] = useState('')          // datetime-local value, '' = unknown
  const shown = value ?? String(unit === 'min' ? def / 60 : def)
  const seconds = Number(shown) * (unit === 'min' ? 60 : 1)
  const nCount = Number(count)
  const nSeed = seed.trim() === '' ? undefined : Number(seed)
  const valid = strategy === 'interval'
    ? Number.isFinite(seconds) && seconds >= 1 && seconds <= 86400
    : Number.isInteger(nCount) && nCount >= 1 && (nSeed === undefined || (Number.isInteger(nSeed) && nSeed >= 0))
  const sampling = strategy === 'interval'
    ? { sampling_strategy: strategy, interval_seconds: seconds }
    : { sampling_strategy: strategy, sample_count: nCount, random_seed: nSeed }
  const plan = useQuery<ExtractionPlan>(media.data?.kind === 'video' && valid ? `xplan:${mediaId}:${JSON.stringify(sampling)}:${initial}` : null,
    () => Api.extractionPlan(mediaId, { ...sampling, include_initial: initial }))

  const preset = (v: number, u: 's' | 'min') => { setValue(String(v)); setUnit(u) }
  const start = async (newRun = false) => {
    setStarting(true)
    try {
      const j = await Api.startExtraction({ media_id: mediaId, ...sampling, include_initial: initial, new_run: newRun,
        run_detection: detectAfter, store_id: storeId.trim() || undefined, camera_id: cameraId.trim() || undefined,
        recording_start_time: recStart ? (recStart.length === 16 ? `${recStart}:00` : recStart) : undefined })
      invalidate('extractions')
      if (j.reused) toast({ tone: 'info', title: 'Already extracted', body: 'An identical extraction of this video exists — showing it. Use "Extract again" for a new run.' })
      if (onStarted) onStarted(j.id)
      else router.push(`/extract?job=${j.id}`)
    } catch (e) {
      const d = e instanceof ApiError ? (e.detail as { job_id?: string; message?: string } | undefined) : undefined
      if (d?.job_id) {
        toast({ tone: 'warn', title: 'Extraction already running', body: d.message ?? '' })
        if (onStarted) onStarted(d.job_id)
        else router.push(`/extract?job=${d.job_id}`)
      } else toast({ tone: 'crit', title: 'Could not start extraction', body: (e as Error).message })
    } finally {
      setStarting(false)
    }
  }

  const reset = () => (onReset ? onReset() : router.push('/extract'))
  if (media.error) return <Panel title="Video"><ErrorState error={media.error} title="Uploaded video not found" action={<Button size="sm" onClick={reset}>Upload again</Button>} /></Panel>
  if (!media.data) return <Panel title="Video"><div className="py-10 text-center text-[13px] text-fg-3">Loading…</div></Panel>
  const m = media.data
  return (
    <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Panel title="Uploaded video" icon={Film} actions={<Button size="sm" variant="ghost" icon={RotateCcw} onClick={reset}>Other video</Button>}>
        <div className="mb-3"><MediaInfo media={m} /></div>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={url(m.preview_url)} alt="First frame preview" className="block max-h-[360px] w-full rounded-md border border-line bg-video object-contain" />
        {m.kind !== 'video' && <p className="mt-2 text-[12px] text-crit">This upload is an image. Frame extraction needs a video.</p>}
      </Panel>
      <Panel title="Sampling" icon={ListChecks}>
        <div className="flex flex-col gap-3">
          <Segmented label="Strategy" value={strategy} onChange={setStrategy}
            options={[{ value: 'random_segments', label: 'Random across video' }, { value: 'interval', label: 'Fixed interval' },
              { value: 'random', label: 'Random frames' }]} />
          {strategy === 'interval' ? (
            <div className="flex flex-wrap items-end gap-2">
              <Field label="Every" htmlFor="xint">
                <Input id="xint" type="number" min={unit === 'min' ? 0.0167 : 1} step="any" value={shown} onChange={(e) => setValue(e.target.value)} className="w-28" />
              </Field>
              <Segmented label="Unit" value={unit} onChange={(u) => { setUnit(u); setValue(String(u === 'min' ? +(seconds / 60).toFixed(4) : seconds)) }}
                options={[{ value: 's', label: 'seconds' }, { value: 'min', label: 'minutes' }]} />
              <Button size="sm" variant="ghost" onClick={() => preset(30, 's')}>30 s · testing</Button>
              <Button size="sm" variant="ghost" onClick={() => preset(15, 'min')}>15 min · monitoring</Button>
            </div>
          ) : (
            <div className="flex flex-wrap items-end gap-2">
              <Field label={strategy === 'random_segments' ? 'Number of screenshots' : 'Number of random frames'} htmlFor="xcount">
                <Input id="xcount" type="number" min={1} step={1} value={count} onChange={(e) => setCount(e.target.value)} className="w-28" />
              </Field>
              <Field label="Seed (optional)" htmlFor="xseed">
                <Input id="xseed" type="number" min={0} step={1} placeholder="generated" value={seed} onChange={(e) => setSeed(e.target.value)} className="w-32" />
              </Field>
              <p className="text-[11px] text-fg-3">{strategy === 'random_segments'
                ? 'The video is split into equal parts and one random screenshot is taken from each. '
                : 'Frames are drawn at random from the whole video. '}The seed is recorded on the job; the same seed and settings give the same timestamps.</p>
            </div>
          )}
          <div className="grid gap-2 sm:grid-cols-3">
            <Field label="Store ID (optional)" htmlFor="xstore">
              <Input id="xstore" value={storeId} maxLength={64} placeholder="unknown" onChange={(e) => setStoreId(e.target.value)} />
            </Field>
            <Field label="Camera ID (optional)" htmlFor="xcam">
              <Input id="xcam" value={cameraId} maxLength={64} placeholder="unknown" onChange={(e) => setCameraId(e.target.value)} />
            </Field>
            <Field label="Recording start (optional)" htmlFor="xrec">
              <Input id="xrec" type="datetime-local" step={1} value={recStart} onChange={(e) => setRecStart(e.target.value)} />
            </Field>
          </div>
          <p className="-mt-1 text-[11px] text-fg-3">Leave these empty when unknown — nothing is guessed. Without a recording start time,
            results show only the position in the video (e.g. 00:05:00), never a clock time.</p>
          <Toggle checked={detectAfter} onChange={setDetectAfter} label="Run uniform detection on the frames afterwards"
            description="Uses the existing trained model; annotated copies and detection_results.csv are saved separately." />
          <Toggle checked={initial} onChange={setInitial} label="Also save the initial frame (00:00:00)"
            description="Recorded separately as sample type 'initial' — it is not counted as a scheduled sample." />
          <div className={cx('rounded-md border px-3 py-2 text-[12px]', plan.data && !plan.data.allowed ? 'border-warn/40 bg-warn-bg text-warn' : 'border-line bg-panel-2 text-fg-2')}>
            {!valid ? (strategy === 'interval' ? 'Enter an interval between 1 second and 24 hours.' : 'Enter a whole number of random frames (and a whole-number seed, or leave it empty).')
              : plan.error ? (plan.error as Error).message
                : !plan.data ? 'Calculating…' : (
                  <>
                    <div className="mb-1 flex items-center gap-1.5 font-semibold text-fg"><Info className="size-3.5" aria-hidden />
                      {plan.data.total} image{plan.data.total === 1 ? '' : 's'} will be extracted
                      <span className="font-normal text-fg-3">({plan.data.initial} initial + {plan.data.scheduled} {plan.data.strategy === 'interval' ? 'scheduled' : 'random'})</span></div>
                    <p>{plan.data.explanation}</p>
                    <p className="mt-1 text-fg-3">{plan.data.coverage_note}</p>
                    {plan.data.reason && <p className="mt-1 font-medium">{plan.data.reason}</p>}
                  </>
                )}
          </div>
          <Button variant="primary" icon={Play} className="h-10 text-[14px]" loading={starting}
            disabled={!plan.data?.allowed || m.kind !== 'video'} onClick={() => start(false)}>
            {startLabel}
          </Button>
          <p className="text-[11px] text-fg-3">Frames are saved as {(settings.data?.settings.extraction_image_format ?? 'png').toUpperCase()} in
            <span className="font-mono"> storage/extractions/&lt;job&gt;/frames/</span> with an image_manifest.csv (successful frames) and failures.csv.</p>
        </div>
      </Panel>
    </div>
  )
}

const SAMPLE_TONE: Record<ExtractionSample['status'], 'ok' | 'crit' | 'info' | 'warn'> = {
  saved: 'ok', failed: 'crit', pending: 'info', not_extracted: 'warn',
}

/** The timestamps chosen for a job, each with its extraction status. Reused by the Results dashboard. */
export function SampleList({ samples }: { samples: ExtractionSample[] }) {
  return (
    <ol className="flex flex-wrap gap-1.5 text-[12px]">
      {samples.map((s) => (
        <li key={s.image_id} className="flex items-center gap-1.5 rounded-md border border-line bg-panel-2 px-2 py-1"
          title={`${s.image_id} · frame ${s.target_frame}`}>
          <span className="text-fg-3">{s.sample_type === 'initial' ? 'initial' : `#${s.image_id.slice(-4).replace(/^0+/, '')}`}</span>
          <span className="font-mono text-fg">{s.video_timestamp_hhmmss ?? s.target_hhmmss}</span>
          <Badge tone={SAMPLE_TONE[s.status]}>{s.status.replace('_', ' ')}</Badge>
        </li>
      ))}
    </ol>
  )
}

// ---------------------------------------------------------------- 3. progress + results
function JobView({ id }: { id: string }) {
  const router = useRouter()
  const { toast } = useApp()
  const [settled, setSettled] = useState(false)
  const q = useQuery<Extraction>(`extraction:${id}`, () => Api.extraction(id), { poll: settled ? undefined : 1000 })
  const j = q.data?.id === id ? q.data : undefined
  const active = j?.state === 'queued' || j?.state === 'running'
  useEffect(() => {          // stop polling once the job has finished, and refresh the recent-jobs list
    if (j && !active && !settled) { setSettled(true); invalidate('extractions') }
  }, [j, active, settled])
  const frames = useQuery(j && !active && j.extracted > 0 ? `xframes:${id}:${j.extracted}` : null, () => Api.extractedFrames(id))

  if (!j) return q.error ? <ErrorState error={q.error} onRetry={q.refetch} /> : <div className="py-16 text-center text-[13px] text-fg-3">Loading…</div>
  const done = j.extracted + j.failed
  const pctDone = active ? Math.round((j.progress ?? 0) * 100) : j.state === 'done' ? 100 : Math.round((done / Math.max(1, j.planned)) * 100)
  return (
    <div className="flex flex-col gap-3">
      <Panel title="Extraction" icon={Film} subtitle={`${j.source_video} · ${samplingLabel(j)} · ${j.include_initial ? 'with' : 'without'} initial frame · ${j.id}`}
        actions={<>
          <Badge tone={STATE_TONE[j.state]} dot>{j.state}</Badge>
          {active && <Button size="sm" variant="danger" icon={Ban} onClick={async () => { await Api.cancelExtraction(id); toast({ tone: 'info', title: 'Stopping extraction…' }) }}>Stop</Button>}
          {!active && <Button size="sm" variant="ghost" icon={RotateCcw} onClick={() => router.push(`/extract?media=${j.media_id}`)}>Extract again</Button>}
        </>}>
        <div className="flex flex-col gap-3">
          <div className="h-1.5 overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-valuenow={pctDone} aria-valuemin={0} aria-valuemax={100}>
            <div className={cx('h-full transition-[width] duration-500', j.state === 'failed' ? 'bg-crit' : 'bg-info')} style={{ width: `${pctDone}%` }} />
          </div>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <MetricCard label="Planned" value={j.planned} hint={`${j.plan.initial} initial + ${j.plan.scheduled} ${j.sampling_strategy === 'interval' ? 'scheduled' : 'random'}`} />
            <MetricCard label="Extracted" value={j.extracted} tone="ok" hint="saved and listed in image_manifest.csv" />
            <MetricCard label="Failed" value={j.failed} tone={j.failed ? 'crit' : undefined} hint={j.failed ? 'see the list below' : undefined} />
            <MetricCard label="Progress" value={`${pctDone}%`} hint={j.finished_at ? `finished ${dateTime(j.finished_at)}` : active ? 'working…' : undefined} />
          </div>
          <p className="text-[12px] text-fg-2">{j.plan.explanation}</p>
          <p className="text-[12px] text-fg-3">{sourceLabel(j)}{j.source === 'cctv_adapter' ? ' · received through the CCTV adapter' : ''}</p>
          {j.error && <p className="rounded-md border border-warn/30 bg-warn-bg px-2.5 py-1.5 text-[12px] text-warn">{j.error}</p>}
          {j.state === 'cancelled' && <p className="text-[12px] text-warn">Stopped early: {j.extracted} of {j.planned} frames were saved; the manifest lists exactly those.</p>}
          <div className="flex flex-wrap gap-1.5">
            <a href={downloadUrl(j.manifest_url)}><Button size="sm" variant="primary" icon={Download} disabled={active}>image_manifest.csv</Button></a>
            <a href={downloadUrl(j.failures_url)}><Button size="sm" icon={Download} disabled={active}>failures.csv</Button></a>
            <span className="self-center font-mono text-[11px] text-fg-3">{j.output_dir}/</span>
          </div>
        </div>
      </Panel>

      {j.samples && <Panel title="Selected timestamps" icon={ListChecks}><SampleList samples={j.samples} /></Panel>}

      <DetectionPanel extraction={j} />

      {j.failures.length > 0 && (
        <Panel title={`Failed timestamps (${j.failed})`} noPad>
          <ul className="divide-y divide-line text-[12px]">
            {j.failures.map((f) => (
              <li key={f.image_id} className="flex gap-3 px-3.5 py-1.5">
                <span className="font-mono text-fg">{f.target_hhmmss}</span>
                <span className="font-mono text-fg-3">frame {f.target_frame}</span>
                <span className="text-fg-2">{f.error_message}</span>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      {!active && (
        <Panel title={`Extracted frames${frames.data ? ` (${frames.data.total})` : ''}`} noPad>
          {!j.extracted ? <EmptyState compact title="No frames were saved" /> : !frames.data ? <div className="p-6 text-center text-[12px] text-fg-3">Loading…</div> : (
            <div className="grid grid-cols-2 gap-2 p-3 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-6">
              {frames.data.frames.map((f) => (
                <a key={f.image_id} href={url(f.url)} target="_blank" rel="noreferrer" className="group overflow-hidden rounded-md border border-line bg-panel-2 hover:border-line-strong">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={url(f.url)} alt={`Frame ${f.frame_number} at ${f.video_timestamp_hhmmss}`} loading="lazy" className="aspect-video w-full bg-video object-contain" />
                  <div className="flex items-center justify-between gap-1 px-2 py-1 text-[11px]">
                    <span className="font-mono text-fg" title={f.recording_timestamp ? `recorded ${f.recording_timestamp}` : 'position in the video'}>{f.video_timestamp_hhmmss}</span>
                    <span className="font-mono text-fg-3">#{f.frame_number}</span>
                    {f.sample_type !== 'scheduled' && <Badge tone="info">{f.sample_type}</Badge>}
                  </div>
                </a>
              ))}
            </div>
          )}
          {frames.data && frames.data.total > frames.data.frames.length && (
            <p className="px-3.5 pb-3 text-[11px] text-fg-3">Showing the first {frames.data.frames.length} of {frames.data.total}; the manifest lists all of them.</p>
          )}
        </Panel>
      )}
      <p className="text-[11px] text-fg-3">Video length {duration(j.plan.duration_seconds)} · last frame at {j.plan.last_frame_seconds.toFixed(2)} s</p>
    </div>
  )
}
