'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useState } from 'react'
import { AlertTriangle, Check, CheckCheck, CircleHelp, Eye, RotateCcw, ShieldAlert } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Alert } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { ago, cx, dateTime, pct, time } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, KeyValue, PageHeader, Panel, Segmented, StatusBadge } from '../../components/ui/primitives'
import { Drawer } from '../../components/ui/overlay'
import { EmptyState, QueryView } from '../../components/ui/states'
import { clock } from '../analysis/AnalysisView'

type StateF = 'new' | 'acknowledged' | 'resolved' | 'all'
type KindF = 'all' | 'no_uniform' | 'low_confidence'

export default function AlertsPage() {
  const params = useSearchParams()
  const router = useRouter()
  const [state, setState] = useState<StateF>((params.get('state') as StateF) || 'all')
  const [kind, setKind] = useState<KindF>('all')
  const openId = params.get('id') ? Number(params.get('id')) : null
  const q = useQuery(`alerts:list:${state}:${kind}`, () => Api.alerts({ state: state === 'all' ? undefined : state, kind: kind === 'all' ? undefined : kind, limit: 200 }), { poll: 15000 })
  const open = (id: number | null) => router.replace(id ? `/alerts?id=${id}` : '/alerts', { scroll: false })

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Uniform Alerts"
        description="An alert is created only when a violation is confirmed: sustained absence of a uniform while people are inside, or a final NO UNIFORM result. A single uncertain frame never creates an alert."
        badge={q.data ? <Badge tone={q.data.new ? 'crit' : 'neutral'} className="ml-1">{q.data.new} new</Badge> : undefined} />
      <div className="flex flex-wrap items-center gap-2">
        <Segmented label="State" value={state} onChange={setState} options={[
          { value: 'all', label: 'All' }, { value: 'new', label: 'New' }, { value: 'acknowledged', label: 'Acknowledged' }, { value: 'resolved', label: 'Resolved' }]} />
        <Segmented label="Type" value={kind} onChange={setKind} options={[
          { value: 'all', label: 'All types' }, { value: 'no_uniform', label: 'No uniform', icon: ShieldAlert }, { value: 'low_confidence', label: 'Low confidence', icon: CircleHelp }]} />
        {q.data && <span className="ml-auto text-[12px] text-fg-3">{q.data.total} alert{q.data.total === 1 ? '' : 's'}</span>}
      </div>
      <QueryView q={q} isEmpty={(d) => d.alerts.length === 0}
        empty={<Panel><EmptyState icon={CheckCheck} title="No alerts" body="No uniform violation matches these filters." /></Panel>}>
        {(d) => (
          <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
            {d.alerts.map((a) => <AlertCard key={a.id} a={a} onOpen={() => open(a.id)} />)}
          </div>
        )}
      </QueryView>
      <EvidenceDrawer id={openId} onClose={() => open(null)} />
    </div>
  )
}

function AlertCard({ a, onOpen }: { a: Alert; onOpen: () => void }) {
  const crit = a.severity === 'crit'
  return (
    <article className={cx('flex flex-col overflow-hidden rounded-lg border bg-panel', crit ? 'border-crit/30' : 'border-warn/30')}>
      <button onClick={onOpen} className="relative block aspect-video bg-video" aria-label="View evidence">
        {a.evidence_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url(a.evidence_url)} alt="" loading="lazy" className="size-full object-cover" />
        ) : <span className="flex size-full items-center justify-center text-[12px] text-fg-3">No evidence frame</span>}
        <span className={cx('absolute left-2 top-2 flex items-center gap-1.5 rounded px-2 py-0.5 text-[11px] font-semibold', crit ? 'bg-crit text-white' : 'bg-warn text-black')}>
          {crit ? <ShieldAlert className="size-3" aria-hidden /> : <CircleHelp className="size-3" aria-hidden />}{a.title}
        </span>
      </button>
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="flex items-center justify-between gap-2">
          <span className="tabular font-mono text-[13px] text-fg">{time(a.created_at)}</span>
          <StatusBadge status={a.state} />
        </div>
        <p className="line-clamp-2 text-[12px] text-fg-2">{a.message}</p>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-fg-3">
          {a.people_inside != null && <span><b className="text-fg">{a.people_inside}</b> people inside</span>}
          {a.uniformed != null && <span><b className="text-fg">{a.uniformed}</b> uniformed</span>}
          {a.media_time_sec != null && <span>at <b className="font-mono text-fg">{clock(a.media_time_sec)}</b></span>}
          <span className="truncate">{a.filename}</span>
        </div>
        <div className="mt-auto flex items-center justify-between pt-1">
          <span className="text-[11px] text-fg-3">{ago(a.created_at)}</span>
          <Button size="sm" icon={Eye} onClick={onOpen}>View evidence</Button>
        </div>
      </div>
    </article>
  )
}

function EvidenceDrawer({ id, onClose }: { id: number | null; onClose: () => void }) {
  const { toast } = useApp()
  const list = useQuery(id ? `alerts:one:${id}` : null, () => Api.alerts({ limit: 500 }))
  const a = list.data?.alerts.find((x) => x.id === id)
  const setState = async (s: Alert['state']) => {
    if (!a) return
    await Api.setAlertState(a.id, s)
    invalidate('alerts')
    toast({ tone: 'ok', title: `Alert #${a.id} ${s}` })
  }
  return (
    <Drawer open={id !== null} onClose={onClose} width="w-[640px]" title={a ? `Alert #${a.id} · ${a.title}` : 'Alert'}
      subtitle={a ? `${dateTime(a.created_at)} · ${a.filename}` : undefined}
      footer={a && <>
        <Link href={`/history?id=${a.analysis_id}`}><Button variant="ghost">Open analysis</Button></Link>
        {a.state !== 'new' && <Button icon={RotateCcw} onClick={() => setState('new')}>Reopen</Button>}
        {a.state === 'new' && <Button icon={Check} onClick={() => setState('acknowledged')}>Acknowledge</Button>}
        {a.state !== 'resolved' && <Button variant="primary" icon={CheckCheck} onClick={() => setState('resolved')}>Resolve</Button>}
      </>}>
      {!a ? <div className="py-10 text-center text-[13px] text-fg-3">{list.loading ? 'Loading…' : 'Alert not found'}</div> : (
        <div className="flex flex-col gap-4">
          <div className={cx('flex items-center gap-2 rounded-lg border p-3', a.severity === 'crit' ? 'border-crit/35 bg-crit-bg text-crit' : 'border-warn/35 bg-warn-bg text-warn')}>
            <AlertTriangle className="size-5 shrink-0" aria-hidden />
            <div><div className="font-semibold tracking-wide">{a.kind === 'no_uniform' ? 'NO UNIFORMED PERSON DETECTED' : 'LOW CONFIDENCE'}</div>
              <div className="text-[12px] text-fg-2">{a.message}</div></div>
          </div>
          {a.evidence_url ? (
            <a href={url(a.evidence_url)} target="_blank" rel="noreferrer">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={url(a.evidence_url)} alt="Annotated evidence frame" className="w-full rounded-md border border-line bg-video" />
            </a>
          ) : <EmptyState compact title="No evidence frame" />}
          <KeyValue cols={3} items={[
            ['Timestamp', dateTime(a.created_at)],
            ['Position in video', a.media_time_sec != null ? clock(a.media_time_sec) : '— (image)'],
            ['State', <StatusBadge key="s" status={a.state} />],
            ['People inside', a.people_inside ?? '—'],
            ['Uniformed', a.uniformed ?? '—'],
            ['Confidence', a.confidence != null ? pct(a.confidence, 0) : '—'],
            ['Person', a.track_id != null ? `Person #${String(a.track_id).padStart(2, '0')}` : '—'],
            ['File', a.filename],
            ['Analysis', <Link key="l" href={`/history?id=${a.analysis_id}`} className="text-info hover:underline">{a.analysis_id}</Link>],
          ]} />
          <p className="text-[11px] text-fg-3">The evidence frame shows the selected camera area, the store ROI, every person box with its tracking ID, uniform label and confidence, exactly as the model saw it. Clothing only — no identity is determined.</p>
        </div>
      )}
    </Drawer>
  )
}
