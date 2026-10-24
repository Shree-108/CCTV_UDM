'use client'

import Link from 'next/link'
import { useEffect, useRef, useState } from 'react'
import { Ban, Crop, Download, FileText, ImageIcon, ScanSearch, ShieldAlert, Timer, Users } from 'lucide-react'
import { downloadUrl, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { AnalysisDetail, ComplianceStatus } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx, dateTime, duration, num, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Panel } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { LABEL_META, LabelBadge, STATUS_META, StatusHero, statusMessage } from '../../components/ui/status'

export const clock = (t: number | null | undefined) => t == null ? '—' : `${Math.floor(t / 60).toString().padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`

/** Polls an analysis while it is queued/running; stops once done/failed. */
export function useAnalysis(id: string | null) {
  const [finished, setFinished] = useState(false)
  const q = useQuery<AnalysisDetail>(id ? `analysis:${id}` : null, () => Api.analysis(id!), { poll: id && !finished ? 1000 : undefined })
  const a = q.data?.id === id ? q.data : undefined
  useEffect(() => { setFinished(false) }, [id])
  useEffect(() => {
    const f = a?.state === 'done' || a?.state === 'failed'
    setFinished(f)
    if (f) { invalidate('overview'); invalidate('alerts'); invalidate('analyses'); invalidate('analytics') }
  }, [a?.state])
  return { ...q, data: a }
}

/** Dynamic badges at the top of the analysis panel (never hard-coded). */
export function StatusBadges({ uniformed, nonUniform, uncertain, status }: { uniformed: number; nonUniform: number; uncertain: number; status?: ComplianceStatus | null }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {status && <Badge tone={STATUS_META[status].tone} dot>{STATUS_META[status].title}</Badge>}
      <Badge tone="ok" dot>Uniform {uniformed}</Badge>
      <Badge tone="crit" dot>Non-uniform {nonUniform}</Badge>
      <Badge tone="warn" dot>Uncertain {uncertain}</Badge>
    </div>
  )
}

function LiveFrame({ id, running }: { id: string; running: boolean }) {
  const [tick, setTick] = useState(0)
  const [ok, setOk] = useState(false)
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => setTick((n) => n + 1), 900)
    return () => clearInterval(t)
  }, [running])
  return (
    <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded-md border border-line bg-video">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={url(`/api/analyses/${id}/live.jpg?t=${tick}`)} alt="Frame currently being analysed" onLoad={() => setOk(true)} onError={() => setOk(false)}
        className={cx('absolute inset-0 size-full object-contain', !ok && 'opacity-0')} />
      {!ok && <span className="text-[12px] text-fg-3">Waiting for the first analysed frame…</span>}
      <span className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/70 px-2 py-1 text-[11px] font-semibold tracking-wide text-info">
        <span className="live-dot size-1.5 rounded-full bg-info" aria-hidden /> ANALYZING VIDEO…
      </span>
    </div>
  )
}

export function AnalysisProgress({ a }: { a: AnalysisDetail }) {
  const { toast } = useApp()
  const lv = a.live
  const p = lv?.progress ?? a.progress ?? 0
  return (
    <div className="flex flex-col gap-3" aria-live="polite">
      <LiveFrame id={a.id} running={a.state === 'running'} />
      <div className="flex flex-wrap items-center justify-between gap-2 text-[13px]">
        <span className="flex items-center gap-2 text-fg">
          <Badge tone="info" dot className="animate-pulse">{a.state === 'queued' ? 'Queued' : 'Analyzing video…'}</Badge>
          {a.state === 'queued' ? 'Waiting for the analysis worker…' : 'Detecting people, tracking them and checking their uniform'}
        </span>
        <span className="flex items-center gap-2">
          <span className="tabular font-mono text-[15px] font-semibold text-info">{(p * 100).toFixed(1)}%</span>
          <Button size="sm" variant="danger" icon={Ban} onClick={async () => { await Api.cancel(a.id); toast({ tone: 'info', title: 'Stopping analysis…' }) }}>Stop</Button>
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-valuenow={Math.round(p * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full bg-info transition-[width] duration-500" style={{ width: `${p * 100}%` }} />
      </div>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-7">
        <MetricCard label="Frames" value={lv ? `${num(lv.frame)} / ${num(lv.frames_total)}` : '—'} hint={lv ? `${num(lv.processed)} analysed` : undefined} />
        <MetricCard label="People" value={lv?.people ?? '—'} hint={lv ? `${lv.people_now} in view now` : undefined} />
        <MetricCard label="Uniformed" value={lv?.uniformed ?? '—'} tone="ok" />
        <MetricCard label="Non-uniform" value={lv?.non_uniform ?? '—'} tone="crit" />
        <MetricCard label="Uncertain" value={lv?.uncertain ?? '—'} tone="warn" />
        <MetricCard label="Current status" value={!lv ? '—' : lv.monitoring ? 'Monitoring' : STATUS_META[lv.current_status].short}
          tone={!lv ? undefined : lv.monitoring ? 'warn' : STATUS_META[lv.current_status].tone} hint={lv?.monitoring ? 'confirming uniform…' : undefined} />
        <MetricCard label="Processing FPS" value={lv ? num(lv.processing_fps, 1) : '—'} hint={lv ? `at ${clock(lv.media_time_sec)}` : undefined} />
      </div>
    </div>
  )
}

/** First-paint fallback; useFitHeight then measures the real space left in the workspace. */
const MEDIA_MAX_H = 'max-h-[clamp(240px,calc(100dvh-300px),760px)]'

/**
 * Largest height that keeps the element inside the visible workspace (the scrolling <main>),
 * however much the headers above it wrap. Re-measured when the window or workspace resizes.
 */
function useFitHeight<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [h, setH] = useState<number>()
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const scroller = el.closest('main') ?? document.documentElement
    const fit = () => {
      const top = el.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop
      setH(Math.round(Math.max(240, Math.min(900, scroller.clientHeight - top - 20))))
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(scroller)                                       // window / workspace resized
    for (const c of Array.from(scroller.children)) ro.observe(c) // headers above it wrapped or loaded
    window.addEventListener('resize', fit)
    return () => { ro.disconnect(); window.removeEventListener('resize', fit) }
  }, [])
  return [ref, h] as const
}

function AnnotatedMedia({ a }: { a: AnalysisDetail }) {
  const ref = useRef<HTMLVideoElement>(null)
  const [err, setErr] = useState(false)
  const [box, maxH] = useFitHeight<HTMLDivElement>()
  if (a.kind === 'image' || !a.files.annotated) {
    const src = a.files.annotated ?? a.files.keyframe
    return src ? (
      <div ref={box} className="flex min-w-0 items-center justify-center self-start overflow-hidden rounded-md border border-line bg-video">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={url(src)} alt="Annotated analysis result" style={{ maxHeight: maxH }}
          className={cx('block h-auto w-auto max-w-full object-contain', MEDIA_MAX_H)} />
      </div>
    ) : <EmptyState compact title="No annotated output" />
  }
  if (err) {
    return (
      <div className="flex aspect-video flex-col items-center justify-center gap-2 rounded-md border border-line bg-video text-center">
        <p className="text-[13px] text-fg">This browser could not play the annotated video.</p>
        <a href={downloadUrl(a.files.annotated)}><Button size="sm" icon={Download}>Download annotated video</Button></a>
      </div>
    )
  }
  return (
    <div ref={box} className="min-w-0 self-start">
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <video ref={ref} data-analysis={a.id} src={url(a.files.annotated)} controls autoPlay muted playsInline preload="auto" onError={() => setErr(true)}
        style={{ maxHeight: maxH }} className={cx('aspect-video w-full rounded-md border border-line bg-video object-contain', MEDIA_MAX_H)} />
    </div>
  )
}

export function seekTo(id: string, t: number) {
  const v = document.querySelector<HTMLVideoElement>(`video[data-analysis="${id}"]`)
  if (v) { v.currentTime = Math.max(0, t - 0.3); void v.play().catch(() => {}); v.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) }
}

/** Image result: one line per detected person (clothing only, never identity) + overall store status. */
function PersonVerdicts({ a, status }: { a: AnalysisDetail; status: ComplianceStatus }) {
  if (!a.persons.length) return null
  return (
    <div className="rounded-md border border-line">
      <ul className="max-h-48 divide-y divide-line overflow-y-auto">
        {a.persons.map((p) => (
          <li key={p.track_id} className="flex items-center gap-2 px-2.5 py-1.5 text-[12px]" title={p.note || undefined}>
            <span className="font-mono text-fg">Person {p.track_id}</span>
            <span className="text-fg-3" aria-hidden>→</span>
            <LabelBadge label={p.label} />
            <span className="ml-auto tabular font-mono" style={{ color: LABEL_META[p.label].color }}>{p.confidence ? pct(p.confidence, 0) : '—'}</span>
          </li>
        ))}
      </ul>
      <div className="flex items-center gap-2 border-t border-line bg-panel-2/60 px-2.5 py-1.5 text-[12px]">
        <span className="font-semibold text-fg">Overall</span><span className="text-fg-3" aria-hidden>→</span>
        <Badge tone={STATUS_META[status].tone} dot>{STATUS_META[status].title}</Badge>
      </div>
    </div>
  )
}

function ModelLine({ a }: { a: AnalysisDetail }) {
  return (
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-[11px]">
      <dt className="text-fg-3">Analysed</dt><dd className="truncate text-fg-2">{dateTime(a.finished_at ?? a.created_at)}</dd>
      <dt className="text-fg-3">Model</dt><dd className="truncate font-mono text-fg-2" title={a.model_version ?? a.classifier ?? ''}>{a.model_version ?? a.classifier ?? '—'}</dd>
    </dl>
  )
}

/** Complete result of an analysis: annotated media, status, metrics, people, alerts, downloads. */
export function AnalysisResult({ a, headerExtra }: { a: AnalysisDetail; headerExtra?: React.ReactNode }) {
  const status = a.status ?? 'no_person'
  const video = a.kind === 'video'
  return (
    <div className="flex flex-col gap-3">
      <Panel title="Uniform analysis" icon={ScanSearch} subtitle={`${a.filename} · ${a.area?.label ?? ''}${a.area?.roi ? ' + store ROI' : ''} · ${dateTime(a.finished_at ?? a.created_at)}`}
        actions={<><Badge tone="ok" dot>Analysis complete</Badge>
          {!video && <Link href={`/analysis/image?media=${a.media_id}&setup=1`}><Button size="sm" variant="ghost" icon={Crop}>Change camera / ROI</Button></Link>}
          {headerExtra}</>}>
        <div className="mb-3"><StatusBadges uniformed={a.uniformed ?? 0} nonUniform={a.non_uniform ?? 0} uncertain={a.uncertain ?? 0} /></div>
        <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_340px]">
          <AnnotatedMedia a={a} />
          <div className="flex flex-col gap-3">
            <StatusHero status={status} message={statusMessage(status, a.uniformed ?? 0, a.people ?? 0)} />
            {!video && <PersonVerdicts a={a} status={status} />}
            <div className="grid grid-cols-2 gap-2">
              <MetricCard label="People detected" value={a.people ?? 0} hint={video ? 'unique tracked people' : 'inside the store area'} />
              <MetricCard label="Uniformed" value={a.uniformed ?? 0} tone="ok" />
              <MetricCard label="Non-uniform" value={a.non_uniform ?? 0} tone={a.non_uniform ? 'crit' : undefined} />
              <MetricCard label="Uncertain" value={a.uncertain ?? 0} tone={a.uncertain ? 'warn' : undefined} />
              <MetricCard label="Uniform confidence" value={a.avg_uniform_conf != null ? pct(a.avg_uniform_conf, 0) : '—'} hint={a.max_uniform_conf != null ? `max ${pct(a.max_uniform_conf, 0)}` : 'no uniform found'} />
              <MetricCard label="Alerts" value={a.alerts} tone={a.alerts ? 'crit' : undefined} />
            </div>
            <ModelLine a={a} />
            <Downloads a={a} />
            {a.classifier !== 'trained' && (
              <p className="rounded-md border border-warn/30 bg-warn-bg px-2.5 py-2 text-[11px] text-warn">
                <b>Trained uniform model missing.</b> {a.classifier === 'stripe-baseline'
                  ? 'This result uses the stripe-pattern baseline and is indicative only.' : 'Uniform classification was disabled; all people are UNCERTAIN.'}{' '}
                <Link href="/settings#model-status" className="underline">Model status →</Link>
              </p>
            )}
            {a.error && <p className="text-[11px] text-warn">{a.error}</p>}
          </div>
        </div>
      </Panel>

      {video && (
        <Panel title="Video analytics" icon={Timer}>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
            <MetricCard label="Video duration" value={clock(a.duration_sec)} />
            <MetricCard label="Processing time" value={duration(a.processing_sec)} hint={a.device ? `on ${a.device.toUpperCase()}` : undefined} />
            <MetricCard label="Frames processed" value={num(a.frames_processed)} hint={`of ${num(a.frames_total)}`} />
            <MetricCard label="Processing FPS" value={num(a.processing_fps, 1)} hint="analysed frames / s" />
            <MetricCard label="Longest uniform presence" value={clock(a.longest_presence_sec)} tone={a.longest_presence_sec ? 'ok' : undefined} hint="confirmed, occlusion-tolerant" />
            <MetricCard label="Unique people" value={a.people ?? 0} />
            <MetricCard label="Uniformed people" value={a.uniformed ?? 0} tone="ok" />
            <MetricCard label="Non-uniform people" value={a.non_uniform ?? 0} />
            <MetricCard label="Uncertain" value={a.uncertain ?? 0} />
            <MetricCard label="Uniform alerts" value={a.alert_list.filter((x) => x.kind === 'no_uniform').length} tone={a.alert_list.some((x) => x.kind === 'no_uniform') ? 'crit' : undefined} />
          </div>
          <PresenceTimeline a={a} />
        </Panel>
      )}

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Panel title="People" icon={Users} noPad subtitle={video ? 'One row per tracked person · final label from temporal voting' : 'One row per detected person inside the store area'}>
          <DataTable rows={a.persons} rowKey={(p) => p.track_id} dense
            onRowClick={video ? (p) => p.first_seen_sec != null && seekTo(a.id, p.first_seen_sec) : undefined}
            empty={<EmptyState compact title="No person inside the selected store area" />}
            columns={[
              { key: 'id', header: 'Person', cell: (p) => <span className="font-mono text-fg">Person #{String(p.track_id).padStart(2, '0')}</span>, sortValue: (p) => p.track_id },
              { key: 'label', header: 'Result', cell: (p) => <LabelBadge label={p.label} />, sortValue: (p) => p.label },
              { key: 'conf', header: 'Confidence', align: 'right', cell: (p) => <span className="tabular" style={{ color: LABEL_META[p.label].color }}>{p.confidence ? pct(p.confidence, 0) : '—'}</span>, sortValue: (p) => p.confidence },
              { key: 'det', header: 'Person conf.', align: 'right', cell: (p) => <span className="tabular text-fg-2">{pct(p.det_confidence, 0)}</span>, sortValue: (p) => p.det_confidence },
              ...(video ? [
                { key: 'seen', header: 'Seen', cell: (p: typeof a.persons[number]) => <span className="font-mono text-[11px] text-info">{clock(p.first_seen_sec)}–{clock(p.last_seen_sec)}</span> },
                { key: 'votes', header: 'Votes U / N / ?', cell: (p: typeof a.persons[number]) => <span className="tabular font-mono text-[11px] text-fg-2">{p.uniform_votes} / {p.non_uniform_votes} / {p.uncertain_votes}</span> },
              ] : [
                { key: 'note', header: 'Note', cell: (p: typeof a.persons[number]) => <span className="text-[11px] text-fg-3">{p.note || '—'}</span> },
              ]),
            ]} />
        </Panel>
        <Panel title="Alerts & evidence" icon={ShieldAlert} noPad>
          {a.alert_list.length === 0 ? <EmptyState compact title="No alerts" body="No uniform violation was confirmed in this analysis." /> : (
            <ul className="divide-y divide-line">
              {a.alert_list.map((al) => (
                <li key={al.id} className="flex gap-3 p-3">
                  {al.evidence_url && (
                    <a href={url(al.evidence_url)} target="_blank" rel="noreferrer" className="shrink-0">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={url(al.evidence_url)} alt="Evidence frame" className="h-14 w-24 rounded border border-line bg-video object-cover" />
                    </a>
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5"><Badge tone={al.severity}>{al.title}</Badge>{al.media_time_sec != null && (
                      <button className="font-mono text-[11px] text-info hover:underline" onClick={() => seekTo(a.id, al.media_time_sec!)}>@ {clock(al.media_time_sec)}</button>)}</div>
                    <p className="mt-1 text-[12px] text-fg-2">{al.message}</p>
                    <Link href={`/alerts?id=${al.id}`} className="text-[11px] text-info hover:underline">View evidence →</Link>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  )
}

function PresenceTimeline({ a }: { a: AnalysisDetail }) {
  const segs = a.live?.segments ?? []
  const dur = a.duration_sec || 1
  return (
    <div className="mt-3">
      <div className="mb-1 flex items-center justify-between text-[11px] text-fg-3">
        <span>Confirmed uniform presence over the video (click to seek)</span><span className="font-mono">{clock(0)} – {clock(a.duration_sec)}</span>
      </div>
      <div className="relative h-7 overflow-hidden rounded-md border border-line bg-panel-2">
        {segs.map(([s, e], i) => (
          <button key={i} onClick={() => seekTo(a.id, s)} title={`Uniform present ${clock(s)}–${clock(e)}`}
            className="absolute inset-y-1 rounded-sm bg-ok/70 hover:bg-ok" style={{ left: `${(s / dur) * 100}%`, width: `max(3px, ${((e - s) / dur) * 100}%)` }} />
        ))}
        {a.alert_list.filter((x) => x.media_time_sec != null).map((al) => (
          <button key={al.id} onClick={() => seekTo(a.id, al.media_time_sec!)} title={`${al.title} @ ${clock(al.media_time_sec)}`}
            className={cx('absolute inset-y-0 w-1 -translate-x-1/2', al.severity === 'crit' ? 'bg-crit' : 'bg-warn')} style={{ left: `${(al.media_time_sec! / dur) * 100}%` }} />
        ))}
        {!segs.length && <span className="absolute inset-0 flex items-center justify-center text-[11px] text-fg-3">No confirmed uniform presence</span>}
      </div>
    </div>
  )
}

function Downloads({ a }: { a: AnalysisDetail }) {
  const f = a.files
  return (
    <div className="flex flex-wrap gap-1.5">
      {a.kind === 'video' && f.annotated && <a href={downloadUrl(f.annotated)}><Button size="sm" variant="primary" icon={Download}>Download annotated video</Button></a>}
      {a.kind === 'image' && f.annotated && <a href={downloadUrl(f.annotated)}><Button size="sm" variant="primary" icon={ImageIcon}>Download annotated image</Button></a>}
      {a.kind === 'video' && f.keyframe && <a href={downloadUrl(f.keyframe)}><Button size="sm" icon={ImageIcon}>Annotated frame</Button></a>}
      {f.report_txt && <a href={downloadUrl(f.report_txt)}><Button size="sm" icon={FileText}>Download report</Button></a>}
      {f.report_json && <a href={downloadUrl(f.report_json)}><Button size="sm" variant="ghost">JSON</Button></a>}
      {f.report_csv && <a href={downloadUrl(f.report_csv)}><Button size="sm" variant="ghost">CSV</Button></a>}
    </div>
  )
}

/** Any-state view: progress while running, error if failed, result when done. */
export function AnalysisView({ id, headerExtra }: { id: string; headerExtra?: React.ReactNode }) {
  const q = useAnalysis(id)
  const a = q.data
  if (!a) return q.error ? <ErrorState error={q.error} onRetry={q.refetch} /> : <div className="py-16 text-center text-[13px] text-fg-3">Loading analysis…</div>
  if (a.state === 'failed') return <Panel title="Uniform analysis"><ErrorState error={new Error(a.error ?? 'Analysis failed')} title="Analysis failed" /></Panel>
  if (a.state !== 'done') return <Panel title="Uniform analysis" icon={ScanSearch} subtitle={a.filename}><AnalysisProgress a={a} /></Panel>
  return <AnalysisResult a={a} headerExtra={headerExtra} />
}
