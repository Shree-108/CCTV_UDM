'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { AlertTriangle, CheckCircle2, CircleHelp, Cpu, FileVideo, ImageIcon, ScanFace, ScanSearch, ShieldAlert, Upload, UserRound, Users } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Analysis, Health } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { ago, cx, dateTime } from '../../lib/format'
import { Badge, Button, Dot, PageHeader, Panel } from '../../components/ui/primitives'
import { KpiCard } from '../../components/ui/kpi'
import { EmptyState, QueryView } from '../../components/ui/states'
import { STATUS_META, StatusBadgeC, StatusHero, statusMessage } from '../../components/ui/status'
import { StatusBadges, clock } from '../analysis/AnalysisView'

export default function OverviewPage() {
  const router = useRouter()
  const ov = useQuery('overview', Api.overview, { poll: 5000 })
  const health = useQuery('health', Api.health, { poll: 20000 })
  const latest = ov.data?.latest
  const ready = health.data && health.data.status === 'ok'

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Uniform Compliance" description="Latest result from your uploaded CCTV images and videos. Every number on this page comes from a real analysis."
        badge={health.data ? <Badge tone={ready ? 'ok' : 'crit'} dot className="ml-1">{ready ? 'SYSTEM READY' : 'MODEL ERROR'}</Badge> : undefined}
        actions={<>
          <Link href="/analysis/image"><Button icon={ImageIcon}>Analyse image</Button></Link>
          <Link href="/analysis/video"><Button variant="primary" icon={FileVideo}>Analyse video</Button></Link>
        </>} />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <KpiCard label="People detected" icon={Users} tone="info" loading={ov.loading} value={latest ? latest.people : ov.data ? '—' : null} sub={latest ? 'latest analysis' : 'no analysis yet'} />
        <KpiCard label="Uniformed" icon={CheckCircle2} tone="ok" loading={ov.loading} value={latest ? latest.uniformed : ov.data ? '—' : null} sub={latest?.avg_uniform_conf != null ? `avg confidence ${Math.round(latest.avg_uniform_conf * 100)}%` : ''} />
        <KpiCard label="Non-uniform" icon={UserRound} tone="crit" loading={ov.loading} value={latest ? latest.non_uniform : ov.data ? '—' : null} />
        <KpiCard label="Uncertain" icon={CircleHelp} tone="warn" loading={ov.loading} value={latest ? latest.uncertain : ov.data ? '—' : null} />
        <KpiCard label="Current status" icon={ScanSearch} tone={latest?.status ? STATUS_META[latest.status].tone : 'neutral'} loading={ov.loading}
          value={latest?.status ? <span style={{ color: STATUS_META[latest.status].color }} className="text-[17px]">{STATUS_META[latest.status].title}</span> : ov.data ? '—' : null}
          sub={latest ? latest.filename : ''} />
        <KpiCard label="Alerts" icon={ShieldAlert} tone={ov.data?.alerts_new ? 'crit' : 'neutral'} loading={ov.loading} value={ov.data ? ov.data.alerts_new : null}
          sub={ov.data ? `new · ${ov.data.alerts_total} total` : ''} onClick={() => router.push('/alerts?state=new')} />
      </div>

      {ov.data && ov.data.running.length > 0 && (
        <Panel title="Analysis in progress" icon={ScanSearch}>
          <ul className="flex flex-col gap-1.5">
            {ov.data.running.map((r) => (
              <li key={r.id}><Link href={`/analysis/${r.kind}?analysis=${r.id}`} className="flex items-center gap-3 rounded-md border border-line bg-panel-2 px-3 py-2 hover:border-info">
                <Badge tone="info" dot className="animate-pulse">{r.state === 'queued' ? 'Queued' : 'Analyzing video…'}</Badge>
                <span className="truncate text-[13px] text-fg">{r.filename}</span>
                <span className="ml-auto tabular font-mono text-[12px] text-info">{Math.round((r.progress ?? 0) * 100)}%</span>
              </Link></li>
            ))}
          </ul>
        </Panel>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
        <Panel title="Latest analysis" icon={ScanFace}
          subtitle={latest ? `${latest.filename} · ${latest.area?.label ?? ''} · ${dateTime(latest.finished_at)}` : undefined}
          actions={latest && <Link href={`/history?id=${latest.id}`} className="text-[12px] text-info hover:underline">Open analysis →</Link>}>
          <QueryView q={ov} isEmpty={(d) => !d.latest}
            empty={<EmptyState icon={Upload} title="No analysis yet" body="Upload a CCTV image or video to check whether the people inside the store are wearing the approved uniform."
              action={<Link href="/upload"><Button variant="primary" icon={Upload}>Upload CCTV media</Button></Link>} />}>
            {(d) => {
              const a = d.latest!
              const src = a.kind === 'video' ? a.files.keyframe : a.files.annotated
              return (
                <div className="flex flex-col gap-3">
                  <StatusBadges status={a.status} uniformed={a.uniformed ?? 0} nonUniform={a.non_uniform ?? 0} uncertain={a.uncertain ?? 0} />
                  <Link href={`/history?id=${a.id}`} className="group relative block overflow-hidden rounded-md border border-line bg-video">
                    {src ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={url(src)} alt="Latest annotated result" className="w-full object-contain transition-opacity group-hover:opacity-90" />
                    ) : <div className="aspect-video" />}
                    {a.kind === 'video' && <span className="absolute left-2 top-2 rounded bg-black/70 px-2 py-0.5 text-[11px] text-white">Key frame · video {clock(a.duration_sec)}</span>}
                  </Link>
                  {a.status && <StatusHero status={a.status} message={statusMessage(a.status, a.uniformed ?? 0, a.people ?? 0)} uniformed={a.uniformed} people={a.people} compact />}
                </div>
              )
            }}
          </QueryView>
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="Recent analysis" noPad actions={<Link href="/history" className="text-[12px] text-info hover:underline">History →</Link>}>
            <QueryView q={ov} isEmpty={(d) => d.recent.length === 0} empty={<EmptyState compact title="Nothing analysed yet" />}>
              {(d) => <RecentList rows={d.recent} />}
            </QueryView>
          </Panel>
          <ModelStatus h={health.data} />
          <Panel title="Recent alerts" noPad actions={<Link href="/alerts" className="text-[12px] text-info hover:underline">All alerts →</Link>}>
            <QueryView q={ov} isEmpty={(d) => d.recent_alerts.length === 0}
              empty={<EmptyState icon={CheckCircle2} compact title="No alerts" body="No uniform violation has been confirmed." />}>
              {(d) => (
                <ul className="divide-y divide-line">
                  {d.recent_alerts.map((al) => (
                    <li key={al.id}><Link href={`/alerts?id=${al.id}`} className="flex items-center gap-3 px-3.5 py-2 hover:bg-hover/60">
                      <AlertTriangle className={cx('size-4 shrink-0', al.severity === 'crit' ? 'text-crit' : 'text-warn')} aria-hidden />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-[13px] font-medium text-fg">{al.title}</div>
                        <div className="truncate text-[11px] text-fg-3">{al.filename} · {ago(al.created_at)}</div>
                      </div>
                      {al.state === 'new' && <Badge tone="crit">new</Badge>}
                    </Link></li>
                  ))}
                </ul>
              )}
            </QueryView>
          </Panel>
        </div>
      </div>
    </div>
  )
}

export function RecentList({ rows }: { rows: Analysis[] }) {
  return (
    <ul className="divide-y divide-line">
      {rows.map((r) => (
        <li key={r.id}>
          <Link href={r.state === 'done' ? `/history?id=${r.id}` : `/analysis/${r.kind}?analysis=${r.id}`} className="flex items-center gap-3 px-3.5 py-2.5 hover:bg-hover/60">
            <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-line bg-panel-2 text-fg-3">
              {r.kind === 'video' ? <FileVideo className="size-4" aria-hidden /> : <ImageIcon className="size-4" aria-hidden />}
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13px] font-medium text-fg">{r.filename}</div>
              <div className="truncate text-[11px] text-fg-3">
                {r.state === 'done' ? `${r.people ?? 0} people · ${r.uniformed ?? 0} uniformed · ${ago(r.created_at)}` : r.state === 'failed' ? `failed · ${r.error ?? ''}` : `${r.state} · ${Math.round((r.progress ?? 0) * 100)}%`}
              </div>
            </div>
            {r.state === 'done' ? <StatusBadgeC status={r.status} /> : <Badge tone={r.state === 'failed' ? 'crit' : 'info'} dot>{r.state}</Badge>}
          </Link>
        </li>
      ))}
    </ul>
  )
}

export function ModelStatus({ h, id }: { h: Health | undefined; id?: string }) {
  const rows: [string, string, 'ok' | 'warn' | 'crit' | 'neutral', string?][] = h ? [
    ['Person detector', h.person_detector.state === 'ready' ? 'Ready' : 'Missing', h.person_detector.state === 'ready' ? 'ok' : 'crit', `YOLO · ${h.person_detector.model}`],
    ['Uniform classifier', h.uniform_classifier.state === 'ready' ? 'Ready' : 'Not trained', h.uniform_classifier.state === 'ready' ? 'ok' : 'warn', h.uniform_classifier.label],
    ['Tracking', 'Ready', 'ok', h.tracking.tracker],
    ['GPU', h.gpu.available ? 'Available' : 'Not available', h.gpu.available ? 'ok' : 'neutral', `running on ${h.gpu.device.toUpperCase()}`],
    ['Live CCTV', 'Coming soon', 'neutral', 'uploads only in this version'],
  ] : []
  return (
    <section id={id}>
      <Panel title="Model status" icon={Cpu}>
        {!h ? <div className="text-[12px] text-fg-3">Checking…</div> : (
          <ul className="flex flex-col gap-2">
            {rows.map(([k, v, tone, hint]) => (
              <li key={k} className="flex items-center gap-2 text-[13px]">
                <span className="w-36 shrink-0 text-fg-2">{k}</span>
                <Dot tone={tone} pulse={tone === 'ok'} />
                <span className={cx('font-medium', tone === 'ok' ? 'text-ok' : tone === 'warn' ? 'text-warn' : tone === 'crit' ? 'text-crit' : 'text-fg-3')}>{v}</span>
                {hint && <span className="ml-auto truncate pl-2 text-[11px] text-fg-3" title={hint}>{hint}</span>}
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </section>
  )
}
