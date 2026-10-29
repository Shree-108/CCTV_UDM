'use client'

import { useEffect, useState } from 'react'
import { Ban, Download, Play, RotateCcw, ScanSearch } from 'lucide-react'
import { ApiError, downloadUrl, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { DetectionRun, Extraction, FrameOutcome } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { cx, dateTime } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, Panel } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { EmptyState } from '../../components/ui/states'

const OUTCOME: Record<FrameOutcome, { label: string; tone: 'ok' | 'neutral' | 'warn' | 'crit' }> = {
  UNIFORM_DETECTED: { label: 'Uniform detected', tone: 'ok' },
  NO_UNIFORM_DETECTED: { label: 'No uniform detected', tone: 'neutral' },
  REVIEW_REQUIRED: { label: 'Review required', tone: 'warn' },
  PROCESSING_FAILED: { label: 'Processing failed', tone: 'crit' },
}
const RUN_TONE: Record<DetectionRun['state'], 'ok' | 'warn' | 'crit' | 'info'> = {
  queued: 'info', running: 'info', done: 'ok', failed: 'crit', cancelled: 'warn', interrupted: 'warn',
}

/** Runs the existing uniform model (person detector + shirt classifier) on an extraction's frames. */
export function DetectionPanel({ extraction }: { extraction: Extraction }) {
  const { toast } = useApp()
  const settings = useQuery('settings', Api.settings)
  const [busy, setBusy] = useState(false)
  const [thr, setThr] = useState<string | null>(null)
  const [settled, setSettled] = useState(false)
  const extActive = extraction.state === 'queued' || extraction.state === 'running'
  const runs = useQuery(`detruns:${extraction.id}`, () => Api.detectionRuns(extraction.id), { poll: settled ? undefined : 1000 })
  const run = runs.data?.runs[0]
  const active = run?.state === 'queued' || run?.state === 'running'
  useEffect(() => { setSettled(!extActive && !active && runs.data !== undefined) }, [extActive, active, runs.data])
  const results = useQuery(run && !active && run.processed > 0 ? `detres:${run.id}:${run.processed}` : null, () => Api.detectionResults(run!.id))
  const shownThr = thr ?? String(run?.threshold ?? settings.data?.settings.uniform_confidence ?? 0.7)

  const start = async (newRun: boolean) => {
    setBusy(true)
    try {
      const r = await Api.startDetection(extraction.id, { threshold: Number(shownThr), new_run: newRun })
      if (r.reused) toast({ tone: 'info', title: 'Same run already exists', body: 'Showing it. Change the threshold or use "Run again".' })
      setSettled(false)
      runs.refetch()
    } catch (e) {
      const d = e instanceof ApiError ? e.detail : undefined
      toast({ tone: 'crit', title: 'Uniform detection not started', body: typeof d === 'string' ? d : (e as Error).message })
    } finally {
      setBusy(false)
    }
  }

  const controls = (
    <div className="flex flex-wrap items-end gap-2">
      <Field label="Threshold (P uniform)" htmlFor="dthr">
        <Input id="dthr" type="number" min={0.5} max={0.99} step={0.01} value={shownThr} onChange={(e) => setThr(e.target.value)} className="w-28" />
      </Field>
      <Button variant="primary" icon={run ? RotateCcw : Play} loading={busy} disabled={extActive || active || extraction.extracted === 0}
        onClick={() => start(!!run)}>{run ? 'Run again' : 'Run uniform detection'}</Button>
    </div>
  )

  return (
    <Panel title="Uniform detection" icon={ScanSearch}
      subtitle="Existing model: person detector + shirt classifier · the original frames are never modified"
      actions={run && <><Badge tone={RUN_TONE[run.state]} dot>{run.state}</Badge>
        {active && <Button size="sm" variant="danger" icon={Ban} onClick={() => Api.cancelDetection(run.id)}>Stop</Button>}</>}>
      <div className="flex flex-col gap-3">
        {extActive ? <p className="text-[12px] text-fg-3">Waiting for the extraction to finish…</p> : controls}
        {run && (
          <>
            <div className="h-1.5 overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-valuenow={Math.round(run.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
              <div className={cx('h-full transition-[width] duration-500', run.state === 'failed' ? 'bg-crit' : 'bg-info')} style={{ width: `${Math.round(run.progress * 100)}%` }} />
            </div>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              <MetricCard label="Images processed" value={`${run.processed} / ${run.total}`} />
              <MetricCard label="Successful predictions" value={run.succeeded} tone="ok" />
              <MetricCard label="Failed inferences" value={run.failed} tone={run.failed ? 'crit' : undefined} hint={run.failed ? 'marked PROCESSING_FAILED — no prediction' : undefined} />
              <MetricCard label="Target-shirt detections" value={run.detections} hint="per image, not unique people" />
            </div>
            <div className="flex flex-wrap gap-1.5 text-[12px]">
              <Badge tone="ok" dot>Uniform detected {run.uniform_images}</Badge>
              <Badge tone="neutral" dot>No uniform detected {run.no_uniform_images}</Badge>
              <Badge tone="warn" dot>Review required {run.review_images}</Badge>
              {run.failed > 0 && <Badge tone="crit" dot>Processing failed {run.failed}</Badge>}
            </div>
            {run.error && <p className="rounded-md border border-crit/30 bg-crit-bg px-2.5 py-1.5 text-[12px] text-crit">{run.error}</p>}
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-[11px]">
              <dt className="text-fg-3">Model</dt><dd className="truncate font-mono text-fg-2">{run.model_version ?? '—'}</dd>
              <dt className="text-fg-3">Threshold</dt><dd className="text-fg-2">{run.threshold.toFixed(2)}</dd>
              <dt className="text-fg-3">Run</dt><dd className="truncate font-mono text-fg-2">{run.id}{run.finished_at ? ` · finished ${dateTime(run.finished_at)}` : ''}</dd>
            </dl>
            <div className="flex flex-wrap gap-1.5">
              <a href={downloadUrl(run.results_csv_url)}><Button size="sm" variant="primary" icon={Download} disabled={active || !run.processed}>detection_results.csv</Button></a>
              <a href={downloadUrl(run.screenshot_summary_url)}><Button size="sm" icon={Download} disabled={active || !run.processed}>screenshot_summary.csv</Button></a>
              <a href={downloadUrl(run.compliance_details_url)}><Button size="sm" icon={Download} disabled={active || !run.processed}>uniform_compliance_details.csv</Button></a>
            </div>
            <p className="text-[11px] text-fg-3">
              Confidence is the classifier&apos;s score for &quot;uniform&quot;, not a calibrated probability of compliance.
              &quot;No uniform detected&quot; does not mean anyone is non-compliant. Detections are counted per image and are
              not linked across frames, so they are not a count of unique people.
            </p>
          </>
        )}
        {run && !active && (
          !results.data ? (run.processed ? <p className="text-[12px] text-fg-3">Loading results…</p> : null) : !results.data.images.length ? <EmptyState compact title="No results" /> : (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
              {results.data.images.map((im) => {
                const o = OUTCOME[im.prediction]
                const best = im.detections.length ? Math.max(...im.detections.map((d) => d.confidence)) : null
                return (
                  <div key={im.image_id} className="overflow-hidden rounded-md border border-line bg-panel-2">
                    <a href={url(im.annotated_url ?? im.original_url)} target="_blank" rel="noreferrer">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={url(im.annotated_url ?? im.original_url)} alt={`${o.label} at ${im.video_timestamp_hhmmss}`} loading="lazy" className="aspect-video w-full bg-video object-contain" />
                    </a>
                    <div className="flex flex-col gap-0.5 px-2 py-1.5 text-[11px]">
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-mono text-fg" title={im.recording_timestamp ? `recorded ${im.recording_timestamp}` : 'position in the video'}>{im.video_timestamp_hhmmss}</span>
                        <Badge tone={o.tone}>{o.label}</Badge>
                      </div>
                      <span className="text-fg-3">{im.prediction === 'PROCESSING_FAILED' ? 'no prediction' : `${im.detection_count} target shirt${im.detection_count === 1 ? '' : 's'}`}
                        {im.person_count != null ? ` · ${im.person_count} ${im.person_count === 1 ? 'person' : 'people'}` : ''}{best != null ? ` · best ${best.toFixed(2)}` : ''}
                        {im.processing_status !== 'ok' ? ` · ${im.processing_status}` : ''}</span>
                      {im.review_reason && <span className="truncate text-warn" title={im.review_reason}>{im.review_reason}</span>}
                      <a href={url(im.original_url)} target="_blank" rel="noreferrer" className="text-info hover:underline">original</a>
                    </div>
                  </div>
                )
              })}
            </div>
          )
        )}
      </div>
    </Panel>
  )
}
