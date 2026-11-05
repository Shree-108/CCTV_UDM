'use client'

import { useMemo, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { CheckCircle2, Circle, CloudUpload, Download, FileVideo, Loader2, LogIn, LogOut, RotateCcw, XCircle } from 'lucide-react'
import { ApiError, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { S3Upload, S3UploadState } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { useApp } from '../../stores/app'
import { cx, dateTime } from '../../lib/format'
import { Badge, Button, Field, Input, KeyValue, PageHeader, Panel, Select, type Tone } from '../../components/ui/primitives'
import { ErrorState, LoadingState } from '../../components/ui/states'
import { DropZone } from '../analysis/common'

/**
 * S3 Upload (admin only): one CCTV video for a Card ID -> S3 input_data -> the existing quick sampled
 * analysis -> CSV report in S3 output_data (backend/extraction/s3_upload.py). The browser never sees AWS
 * credentials: the file goes to this app's backend, which writes to S3.
 */
const VIDEO_EXT = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.mpg', '.mpeg', '.ts']
const MAX_MB = 1024                                    // server default MAX_VIDEO_UPLOAD_MB; the server enforces its own
const NVR_NAME = /^\d{8}_(\d{4})(\d{2})(\d{2})\d{6}_\d{14}_/   // 20261104_20261104215126_20261104215614_192126.mp4
const DAY = /^\d{2}-\d{2}-\d{4}$/

const STEPS: { state: S3UploadState; label: string }[] = [
  { state: 'queued', label: 'Queued' },
  { state: 'uploading', label: 'Uploading video to S3' },
  { state: 'processing', label: 'Uniform detection' },
  { state: 'generating_report', label: 'Generating CSV' },
  { state: 'uploading_output', label: 'Uploading CSV to S3' },
  { state: 'completed', label: 'Completed (CSV verified in S3)' },
]
const STATE_TONE: Record<S3UploadState, Tone> = {
  queued: 'info', uploading: 'info', processing: 'info', generating_report: 'info', uploading_output: 'info',
  completed: 'ok', failed: 'crit',
}
const label = (s: string) => s.replace(/_/g, ' ')
const mb = (b: number) => `${(b / 2 ** 20).toFixed(1)} MB`

function dayFromName(name: string): string | null {
  const m = NVR_NAME.exec(name)
  return m ? `${m[3]}-${m[2]}-${m[1]}` : null
}

function validate(f: File): string | null {
  const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
  if (!VIDEO_EXT.includes(ext)) return `Not a video file (${VIDEO_EXT.join(', ')})`
  if (f.size === 0) return 'The file is empty'
  if (f.size > MAX_MB * 2 ** 20) return `The file is larger than ${MAX_MB} MB`
  return null
}

export default function S3UploadPage() {
  const me = useQuery('auth:me', Api.me)
  if (me.loading && !me.data && !me.error) return <LoadingState label="Checking login" />
  if (me.error instanceof ApiError && me.error.status === 401) return <Login onDone={() => void me.refetch()} />
  if (me.error) return <ErrorState error={me.error} title="Admin login unavailable" onRetry={() => void me.refetch()} />
  return <Workspace user={me.data!.username} onLogout={async () => { await Api.logout(); invalidate('auth:'); invalidate('s3:') }} />
}

// ---------------------------------------------------------------- login
function Login({ onDone }: { onDone: () => void }) {
  const [user, setUser] = useState('admin')
  const [pw, setPw] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setBusy(true); setErr(null)
    try {
      await Api.login(user, pw)
      setPw('')
      onDone()
    } catch (x) {
      setErr((x as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="mx-auto mt-10 w-full max-w-sm">
      <Panel title="Administrator login" icon={LogIn}>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <p className="text-[12px] text-fg-3">Uploading videos to S3 requires the administrator account.</p>
          <Field label="User name" htmlFor="u"><Input id="u" autoComplete="username" value={user} onChange={(e) => setUser(e.target.value)} /></Field>
          <Field label="Password" htmlFor="p"><Input id="p" type="password" autoComplete="current-password" value={pw} onChange={(e) => setPw(e.target.value)} /></Field>
          {err && <p role="alert" className="text-[12px] text-crit">{err}</p>}
          <Button type="submit" variant="primary" icon={LogIn} loading={busy} disabled={!user || !pw}>Log in</Button>
        </form>
      </Panel>
    </div>
  )
}

// ---------------------------------------------------------------- upload + jobs
function Workspace({ user, onLogout }: { user: string; onLogout: () => void }) {
  const router = useRouter()
  const params = useSearchParams()
  const jobId = params.get('job')
  const open = (id: string | null) => router.replace(id ? `/s3-upload?job=${id}` : '/s3-upload')
  return (
    <>
      <PageHeader title="S3 Upload"
        description="Upload a CCTV video for a card: it is stored in the card's S3 input_data folder, checked by the uniform model (4 random screenshots) and the CSV report is uploaded to output_data."
        badge={<Badge tone="ok" dot>S3 connected via server</Badge>}
        actions={<><Badge>{user}</Badge><Button size="sm" variant="subtle" icon={LogOut} onClick={onLogout}>Log out</Button></>} />
      <div className="grid gap-3 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <UploadForm onJob={open} />
        {jobId ? <JobPanel id={jobId} onNew={() => open(null)} /> : <RecentUploads onOpen={open} />}
      </div>
      {jobId && <div className="mt-3"><RecentUploads onOpen={open} /></div>}
    </>
  )
}

function UploadForm({ onJob }: { onJob: (id: string) => void }) {
  const { toast } = useApp()
  const cards = useQuery('s3:cards', Api.s3Cards)
  const [card, setCard] = useState<number | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [fileErr, setFileErr] = useState<string | null>(null)
  const [day, setDay] = useState('')
  const [progress, setProgress] = useState<number | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)
  const cardId = card ?? cards.data?.cards[0]?.card_id ?? null
  const derived = file ? dayFromName(file.name) : null
  const dayOk = day === '' ? !!derived : DAY.test(day)

  const pick = (f: File) => {
    setErr(null)
    const v = validate(f)
    setFileErr(v)
    setFile(v ? null : f)
    setDay('')
  }
  const submit = async () => {
    if (!file || cardId === null) return
    setErr(null); setProgress(0)
    abort.current = new AbortController()
    try {
      const job = await Api.s3Upload(file, cardId, day || undefined, (s, t) => setProgress(s / t), abort.current.signal)
      toast(job.how === 'duplicate'
        ? { tone: 'info', title: 'Already uploaded', body: `This video was already uploaded for card ${job.card_id}: showing that job.` }
        : { tone: 'ok', title: job.how === 'retried' ? 'Retrying the earlier failed job' : 'Upload received', body: file.name })
      setFile(null); setDay('')
      invalidate('s3:uploads')
      onJob(job.id)
    } catch (x) {
      if ((x as Error).name !== 'AbortError') setErr((x as Error).message)
    } finally {
      setProgress(null)
      abort.current = null
    }
  }
  const busy = progress !== null
  return (
    <Panel title="Upload video" icon={CloudUpload}>
      <div className="flex flex-col gap-3">
        <Field label="Card ID" htmlFor="card" hint={cards.data && cardId !== null
          ? `Store: ${cards.data.cards.find((c) => c.card_id === cardId)?.store_name || '—'}` : undefined}>
          {cards.error ? <p className="text-[12px] text-crit">{cards.error.message}</p> : (
            <Select id="card" value={cardId ?? ''} disabled={busy || !cards.data} onChange={(e) => setCard(Number(e.target.value))}>
              {cards.data?.cards.map((c) => <option key={c.card_id} value={c.card_id}>Card {c.card_id} · {c.cart}</option>)}
            </Select>
          )}
        </Field>
        <DropZone compact busy={busy} accept={VIDEO_EXT.join(',')} onFile={pick} label="Drop a CCTV video here or click to browse"
          hint={`One video, up to ${MAX_MB} MB`} formats={VIDEO_EXT.map((e) => e.slice(1).toUpperCase())} />
        {fileErr && <p role="alert" className="text-[12px] text-crit">{fileErr}</p>}
        {file && (
          <div className="flex items-center gap-2 rounded-md border border-line bg-panel-2 px-2.5 py-2">
            <FileVideo className="size-4 shrink-0 text-info" aria-hidden />
            <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-fg" title={file.name}>{file.name}</span>
            <span className="tabular shrink-0 text-[12px] text-fg-3">{mb(file.size)}</span>
          </div>
        )}
        {file && (
          <Field label="S3 date folder" htmlFor="day" hint={derived
            ? `From the file name: ${derived}. Leave empty to use it.`
            : 'The file name has no recording date: enter the day folder (DD-MM-YYYY).'}>
            <Input id="day" placeholder={derived ?? 'DD-MM-YYYY'} value={day} disabled={busy} onChange={(e) => setDay(e.target.value.trim())} />
          </Field>
        )}
        {busy && (
          <div aria-live="polite">
            <div className="mb-1 flex justify-between text-[12px] text-fg-2"><span>Sending to server…</span><span className="tabular">{Math.round(progress * 100)} %</span></div>
            <Bar value={progress} />
          </div>
        )}
        {err && <p role="alert" className="text-[12px] text-crit">{err}</p>}
        <div className="flex gap-2">
          <Button variant="primary" icon={CloudUpload} loading={busy} disabled={!file || cardId === null || !dayOk} onClick={submit}>
            Upload and Process
          </Button>
          {busy && <Button variant="subtle" onClick={() => abort.current?.abort()}>Cancel</Button>}
        </div>
      </div>
    </Panel>
  )
}

function Bar({ value, tone = 'info' }: { value: number; tone?: 'info' | 'ok' }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className={cx('h-full rounded-full transition-[width]', tone === 'ok' ? 'bg-ok' : 'bg-info')} style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  )
}

function JobPanel({ id, onNew }: { id: string; onNew: () => void }) {
  const { toast } = useApp()
  const q = useQuery(`s3:upload:${id}`, () => Api.s3UploadJob(id), { poll: 1500 })
  const [retrying, setRetrying] = useState(false)
  const j = q.data
  if (!j) return q.error ? <ErrorState error={q.error} onRetry={() => void q.refetch()} /> : <LoadingState />
  const failedAt = j.state === 'failed' ? STEPS.findIndex((s) => s.state === j.failed_stage) : -1
  const current = j.state === 'failed' ? failedAt : STEPS.findIndex((s) => s.state === j.state)
  const retry = async () => {
    setRetrying(true)
    try {
      await Api.s3Retry(j.id)
      toast({ tone: 'info', title: 'Retrying', body: 'Steps already done are not repeated.' })
      void q.refetch(); invalidate('s3:uploads')
    } catch (x) {
      toast({ tone: 'crit', title: 'Retry refused', body: (x as Error).message })
    } finally {
      setRetrying(false)
    }
  }
  return (
    <Panel title="Processing status" subtitle={j.id}
      actions={<><Badge tone={STATE_TONE[j.state]} dot>{label(j.state)}</Badge><Button size="sm" variant="subtle" onClick={onNew}>Close</Button></>}>
      <div className="flex flex-col gap-4">
        <KeyValue items={[
          ['Video', <span key="v" title={j.filename} className="font-mono">{j.filename}</span>],
          ['Size', mb(j.size_bytes)],
          ['Card ID', `${j.card_id} · ${j.cart}`],
          ['Store', j.store_name || '—'],
          ['Date folder', `${j.day}${j.day_source === 'file_name' ? ' (from file name)' : ' (selected)'}`],
          ['Uploaded', `${dateTime(j.created_at)}${j.created_by ? ` by ${j.created_by}` : ''}`],
        ]} />
        <ol className="flex flex-col gap-1.5">
          {STEPS.map((s, i) => {
            const done = j.state === 'completed' || i < current
            const failed = i === failedAt
            const active = !done && !failed && i === current
            const Icon = failed ? XCircle : done ? CheckCircle2 : active ? Loader2 : Circle
            return (
              <li key={s.state} className="flex items-center gap-2 text-[13px]">
                <Icon aria-hidden className={cx('size-4 shrink-0', failed ? 'text-crit' : done ? 'text-ok' : active ? 'animate-spin text-info' : 'text-fg-3')} />
                <span className={cx(done || active || failed ? 'text-fg' : 'text-fg-3')}>{s.label}</span>
                {active && s.state === 'processing' && j.analysis && (
                  <span className="tabular text-[12px] text-fg-3">
                    {j.analysis.step}{(j.analysis.detection_progress ?? j.analysis.extraction_progress) !== null
                      ? ` · ${Math.round(100 * (j.analysis.detection_progress ?? j.analysis.extraction_progress ?? 0))} %` : ''}
                  </span>
                )}
              </li>
            )
          })}
        </ol>
        {j.state === 'failed' && (
          <div role="alert" className="rounded-md border border-crit/40 bg-crit-bg p-2.5 text-[12.5px] text-crit">
            <p className="font-semibold">Failed while {label(j.failed_stage ?? 'processing')}{j.attempts > 1 ? ` (attempt ${j.attempts})` : ''}</p>
            <p className="mt-0.5 break-words">{j.error}</p>
            <Button className="mt-2" size="sm" variant="secondary" icon={RotateCcw} loading={retrying} onClick={retry}>Retry</Button>
          </div>
        )}
        <KeyValue cols={1} items={[
          ['S3 input', <span key="i" className="font-mono text-[12px]" title={j.s3_input}>{j.s3_input} {j.input_verified && <Badge tone="ok">verified</Badge>}</span>],
          ['S3 output', <span key="o" className="font-mono text-[12px]" title={j.s3_output}>{j.s3_output} {j.state === 'completed' && <Badge tone="ok">verified</Badge>}</span>],
        ]} />
        {j.state === 'completed' && j.summary && <Result job={j} />}
      </div>
    </Panel>
  )
}

function Result({ job }: { job: S3Upload }) {
  const s = job.summary!
  const rows = job.report_rows ?? []
  const cols = useMemo(() => (rows[0] ? Object.keys(rows[0]) : []), [rows])
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone="ok" dot>{s.screenshots} screenshots</Badge>
        <Badge tone="ok">{s.uniform} wearing uniform</Badge>
        <Badge tone="crit">{s.not_uniform} not wearing</Badge>
        {s.manual_review > 0 && <Badge tone="warn">{s.manual_review} need manual review</Badge>}
        {s.unreadable_frames > 0 && <Badge tone="warn">{s.unreadable_frames} unreadable frames</Badge>}
        <a href={url(job.report_url!)} className="ml-auto"><Button size="sm" variant="primary" icon={Download}>Download CSV</Button></a>
      </div>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full text-[12px]">
          <thead className="bg-panel-2 text-left text-fg-3">
            <tr>{cols.map((c) => <th key={c} className="whitespace-nowrap px-2 py-1.5 font-medium">{c.trim()}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t border-line">
                {cols.map((c) => <td key={c} className="px-2 py-1.5 align-top text-fg">{r[c] === '' ? '—' : r[c]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-fg-3">Indicative results from the experimental uniform model; people marked for manual review were not judged.</p>
    </div>
  )
}

function RecentUploads({ onOpen }: { onOpen: (id: string) => void }) {
  const q = useQuery('s3:uploads', () => Api.s3Uploads(20), { poll: 5000 })
  return (
    <Panel title="Recent uploads" noPad>
      {q.error ? <div className="p-3.5"><ErrorState compact error={q.error} /></div>
        : !q.data ? <div className="p-3.5"><LoadingState rows={2} /></div>
        : q.data.uploads.length === 0 ? <p className="p-3.5 text-[12px] text-fg-3">No uploads yet.</p>
        : (
          <ul className="divide-y divide-line">
            {q.data.uploads.map((u) => (
              <li key={u.id}>
                <button onClick={() => onOpen(u.id)} className="flex w-full items-center gap-2 px-3.5 py-2 text-left hover:bg-hover">
                  <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-fg" title={u.filename}>{u.filename}</span>
                  <span className="shrink-0 text-[12px] text-fg-3">Card {u.card_id} · {u.day}</span>
                  <Badge tone={STATE_TONE[u.state]} dot>{label(u.state)}</Badge>
                </button>
              </li>
            ))}
          </ul>
        )}
    </Panel>
  )
}
