'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useMemo, useState } from 'react'
import { BarChart3, History, Upload } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Analysis, AnalyticsAgg } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { dateTime, duration, num, pct, todayLocal, dayStartIso } from '../../lib/format'
import { Button, PageHeader, Panel, Segmented } from '../../components/ui/primitives'
import { Columns, Donut, Legend, StackBar } from '../../components/ui/charts'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState, LoadingState, QueryView } from '../../components/ui/states'
import { STATUS_META, StatusBadgeC } from '../../components/ui/status'
import { clock } from '../analysis/AnalysisView'

type Range = 'today' | '7' | '30' | 'all'

export default function AnalyticsPage() {
  const [range, setRange] = useState<Range>('30')
  const start = range === 'all' ? undefined : dayStartIso(todayLocal(range === 'today' ? 0 : -(Number(range) - 1)))
  const q = useQuery(`analytics:${range}`, () => Api.analytics(start), { poll: 30000 })
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Analytics" description="Computed only from the images and videos you have analysed — no sample or estimated data."
        actions={<Segmented label="Range" value={range} onChange={setRange} options={[
          { value: 'today', label: 'Today' }, { value: '7', label: '7 days' }, { value: '30', label: '30 days' }, { value: 'all', label: 'All time' }]} />} />
      <QueryView q={q} isEmpty={(d) => d.analyses === 0}
        empty={<Panel><EmptyState icon={BarChart3} title="No analyses in this period" body="Analytics appear after you analyse CCTV images or videos."
          action={<Link href="/upload"><Button variant="primary" icon={Upload}>Upload CCTV media</Button></Link>} /></Panel>}>
        {(d) => (
          <>
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <ComplianceCard title={range === 'today' ? "Today's uniform compliance" : 'Uniform compliance'} a={d} />
              <ComplianceCard title="Today's compliance" a={d.today} small />
            </div>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
              <MetricCard label="Analyses" value={num(d.analyses)} hint={`${d.images} images · ${d.videos} videos`} />
              <MetricCard label="People observed" value={num(d.people)} hint="unique per analysis" />
              <MetricCard label="Uniformed" value={num(d.uniformed)} tone="ok" />
              <MetricCard label="Non-uniform" value={num(d.non_uniform)} tone={d.non_uniform ? 'crit' : undefined} />
              <MetricCard label="Uncertain" value={num(d.uncertain)} tone={d.uncertain ? 'warn' : undefined} />
              <MetricCard label="Alerts" value={num(d.alerts)} tone={d.alerts ? 'crit' : undefined} />
              <MetricCard label="Avg uniform confidence" value={d.avg_uniform_confidence != null ? pct(d.avg_uniform_confidence, 1) : '—'} hint="confirmed uniformed people" />
              <MetricCard label="Footage analysed" value={duration(d.footage_sec)} hint={`in ${duration(d.processing_sec)}`} />
            </div>
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
              <Panel title="Compliance by day" subtitle="% of analyses with people inside that were UNIFORM COMPLIANT">
                <DayChart d={d.by_day} />
              </Panel>
              <Panel title="Results & observations">
                <div className="flex items-center gap-5">
                  <Donut ariaLabel="Result distribution" size={128}
                    data={(['compliant', 'no_uniform', 'uncertain', 'no_person'] as const).map((s) => ({
                      label: STATUS_META[s].short, color: STATUS_META[s].color,
                      value: { compliant: d.compliant, no_uniform: d.no_uniform, uncertain: d.uncertain_status, no_person: d.no_person }[s],
                    }))}
                    center={<><span className="tabular text-[20px] font-semibold text-fg">{d.analyses}</span><span className="text-[10px] text-fg-3">analyses</span></>} />
                  <Legend items={(['compliant', 'no_uniform', 'uncertain', 'no_person'] as const).map((s) => ({
                    name: `${STATUS_META[s].short} · ${{ compliant: d.compliant, no_uniform: d.no_uniform, uncertain: d.uncertain_status, no_person: d.no_person }[s]}`, color: STATUS_META[s].color }))} />
                </div>
                <div className="mt-4">
                  <div className="mb-1 flex justify-between text-[11px] text-fg-3"><span>Per-frame classifications of people inside the store area</span>
                    <span className="font-mono">{num(d.uniform_obs + d.non_uniform_obs + d.uncertain_obs)}</span></div>
                  <StackBar parts={[
                    { label: 'Uniform observations', value: d.uniform_obs, color: 'var(--ok)' },
                    { label: 'Non-uniform observations', value: d.non_uniform_obs, color: 'var(--crit)' },
                    { label: 'Uncertain observations', value: d.uncertain_obs, color: 'var(--warn)' }]} />
                  <div className="mt-1.5 flex flex-wrap gap-3 text-[11px] text-fg-3">
                    <span><b className="text-ok">{num(d.uniform_obs)}</b> uniform</span>
                    <span><b className="text-crit">{num(d.non_uniform_obs)}</b> non-uniform</span>
                    <span><b className="text-warn">{num(d.uncertain_obs)}</b> uncertain</span>
                  </div>
                </div>
              </Panel>
            </div>
          </>
        )}
      </QueryView>
      <HistoryTable />
    </div>
  )
}

function ComplianceCard({ title, a, small }: { title: string; a: AnalyticsAgg; small?: boolean }) {
  const rate = a.compliance_rate
  return (
    <section className="rounded-lg border border-line bg-panel p-4">
      <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-fg-3">{title}</div>
      <div className="mt-1 flex items-end gap-3">
        <span className={small ? 'tabular text-[28px] font-semibold leading-9 text-fg' : 'tabular text-[40px] font-semibold leading-[48px] text-fg'}
          style={{ color: rate == null ? undefined : rate >= 0.9 ? 'var(--ok)' : rate >= 0.6 ? 'var(--warn)' : 'var(--crit)' }}>
          {rate == null ? '—' : `${(rate * 100).toFixed(1)}%`}
        </span>
        <span className="pb-1.5 text-[12px] text-fg-3">{rate == null ? 'no analysis with people inside yet' : `${a.compliant} of ${a.with_people} analyses with people inside`}</span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-valuenow={rate == null ? 0 : Math.round(rate * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full rounded-full bg-ok" style={{ width: `${(rate ?? 0) * 100}%` }} />
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-fg-3">
        <span>People observed <b className="text-fg">{a.people}</b></span>
        <span>Uniformed <b className="text-ok">{a.uniformed}</b></span>
        <span>Non-uniform <b className="text-crit">{a.non_uniform}</b></span>
        <span>Alerts <b className="text-fg">{a.alerts}</b></span>
      </div>
    </section>
  )
}

function DayChart({ d }: { d: { day: string; analyses: number; compliant: number; with_people: number }[] }) {
  if (!d.length) return <EmptyState compact title="No data" />
  return (
    <Columns ariaLabel="Compliance percentage per day" height={180} format={(v) => `${v.toFixed(0)}%`}
      data={d.map((x) => ({ label: x.day.slice(5), value: x.with_people ? (x.compliant / x.with_people) * 100 : 0,
        color: !x.with_people ? 'var(--fg-3)' : x.compliant / x.with_people >= 0.9 ? 'var(--ok)' : x.compliant / x.with_people >= 0.6 ? 'var(--warn)' : 'var(--crit)' }))} />
  )
}

/** Analysis history: every analysed file with its result. Also used by /history. */
export function HistoryTable({ compact }: { compact?: boolean }) {
  const router = useRouter()
  const [kind, setKind] = useState<'all' | 'image' | 'video'>('all')
  const [status, setStatus] = useState<'all' | Analysis['status'] & string>('all')
  const q = useQuery(`analyses:${kind}:${status}`, () => Api.analyses({ limit: 200, kind: kind === 'all' ? undefined : kind, status: status === 'all' ? undefined : status }), { poll: 10000 })
  const rows = useMemo(() => q.data?.analyses ?? [], [q.data])
  return (
    <Panel title="Analysis history" icon={History} noPad
      actions={<div className="flex flex-wrap gap-1.5">
        <Segmented size="sm" label="Type" value={kind} onChange={setKind} options={[{ value: 'all', label: 'All' }, { value: 'image', label: 'Images' }, { value: 'video', label: 'Videos' }]} />
        {!compact && <Segmented size="sm" label="Status" value={status} onChange={(v) => setStatus(v)} options={[
          { value: 'all', label: 'Any result' }, { value: 'compliant', label: 'Compliant' }, { value: 'no_uniform', label: 'No uniform' }, { value: 'uncertain', label: 'Uncertain' }, { value: 'no_person', label: 'No person' }]} />}
      </div>}>
      <DataTable rows={rows} rowKey={(r) => r.id} dense pageSize={compact ? 10 : 25}
        onRowClick={(r) => router.push(r.state === 'done' ? `/history?id=${r.id}` : `/analysis/${r.kind}?analysis=${r.id}`)}
        empty={q.data ? <EmptyState compact title="No analyses yet" /> : q.error ? <EmptyState compact title="Backend not reachable" /> : <LoadingState rows={4} className="p-3" />}
        columns={[
          { key: 'date', header: 'Date', cell: (r) => <span className="whitespace-nowrap text-fg-2">{dateTime(r.created_at)}</span>, sortValue: (r) => r.created_at },
          { key: 'file', header: 'File', cell: (r) => <span className="font-medium text-fg">{r.filename}</span>, sortValue: (r) => r.filename },
          { key: 'kind', header: 'Type', cell: (r) => <span className="capitalize text-fg-2">{r.kind}</span> },
          { key: 'area', header: 'Camera', cell: (r) => <span className="text-fg-2">{r.area?.label ?? '—'}{r.area?.roi ? ' + ROI' : ''}</span> },
          { key: 'status', header: 'Result', cell: (r) => r.state === 'done' ? <StatusBadgeC status={r.status} /> : <span className="text-[12px] capitalize text-fg-3">{r.state}</span>, sortValue: (r) => r.status ?? r.state },
          { key: 'people', header: 'People', align: 'right', cell: (r) => <span className="tabular">{r.people ?? '—'}</span>, sortValue: (r) => r.people },
          { key: 'uni', header: 'Uniformed', align: 'right', cell: (r) => <span className="tabular text-ok">{r.uniformed ?? '—'}</span>, sortValue: (r) => r.uniformed },
          { key: 'dur', header: 'Duration', align: 'right', cell: (r) => r.kind === 'video' ? <span className="tabular font-mono text-[12px]">{clock(r.duration_sec)}</span> : '—' },
          { key: 'alerts', header: 'Alerts', align: 'right', cell: (r) => <span className={r.alerts ? 'tabular text-crit' : 'tabular text-fg-3'}>{r.alerts}</span>, sortValue: (r) => r.alerts },
        ]} />
    </Panel>
  )
}
