'use client'

import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { ImageOff } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { EvidenceLinks, FrameOutcome, ScreenshotEvidence } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { Badge } from '../../components/ui/primitives'
import { Modal } from '../../components/ui/overlay'
import { ErrorState, LoadingState, Skeleton } from '../../components/ui/states'

export const OUTCOME: Record<FrameOutcome, { label: string; tone: 'ok' | 'neutral' | 'warn' | 'crit'; help: string }> = {
  UNIFORM_DETECTED: { label: 'APPROVED UNIFORM DETECTED', tone: 'ok', help: 'the model found at least one target uniform shirt above the threshold' },
  NO_UNIFORM_DETECTED: { label: 'NO APPROVED UNIFORM DETECTED', tone: 'neutral', help: 'no target shirt passed the threshold — not proof that anyone is out of uniform' },
  REVIEW_REQUIRED: { label: 'REVIEW REQUIRED', tone: 'warn', help: 'someone could not be judged (too small, cut off, blurred or unclear) — a person should look' },
  PROCESSING_FAILED: { label: 'PROCESSING FAILED', tone: 'crit', help: 'the image could not be read or analysed — no prediction was made' },
}
const PERSON: Record<string, string> = { uniform_shirt: 'uniform', not_uniform: 'not uniform', uncertain: 'needs review' }
const PREVIEW_W = 360
const OPEN_DELAY = 250
const CLOSE_DELAY = 150

/**
 * The `img name` cell of a business-CSV table. Hover / focus shows a small preview of THIS row's screenshot,
 * click opens the evidence viewer. Both use the row's own links (job id + image id + run id from the API),
 * so sorting, filtering or identical file names in other videos cannot point a row at another image.
 */
export function ImageNameCell({ name, links }: { name: string; links: EvidenceLinks }) {
  const ref = useRef<HTMLButtonElement>(null)
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const [anchor, setAnchor] = useState<DOMRect | null>(null)
  const [open, setOpen] = useState(false)
  const later = (fn: () => void, ms: number) => { clearTimeout(timer.current); timer.current = setTimeout(fn, ms) }
  const show = () => later(() => ref.current && setAnchor(ref.current.getBoundingClientRect()), OPEN_DELAY)
  const hide = () => later(() => setAnchor(null), CLOSE_DELAY)
  useEffect(() => () => clearTimeout(timer.current), [])
  useEffect(() => {
    if (!anchor) return
    const close = () => setAnchor(null)
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && close()
    window.addEventListener('scroll', close, true)
    window.addEventListener('keydown', esc)
    return () => { window.removeEventListener('scroll', close, true); window.removeEventListener('keydown', esc) }
  }, [anchor])

  if (!name) return <span className="text-crit">not extracted</span>
  return (
    <>
      <button ref={ref} type="button" aria-haspopup="dialog" title={`${name} — click to open the evidence`}
        onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}
        onClick={() => { clearTimeout(timer.current); setAnchor(null); setOpen(true) }}
        className="block max-w-44 truncate text-left font-mono text-[11px] text-fg underline decoration-dotted underline-offset-2 hover:decoration-solid focus-visible:outline focus-visible:outline-1 focus-visible:outline-info">
        {name}
      </button>
      {anchor && <HoverPreview anchor={anchor} name={name} src={links.thumb_url}
        onEnter={() => clearTimeout(timer.current)} onLeave={hide} />}
      {open && <EvidenceViewer evidenceUrl={links.evidence_url} onClose={() => setOpen(false)} />}
    </>
  )
}

/** Fixed-position popup next to the file name, kept inside the viewport. */
function HoverPreview({ anchor, name, src, onEnter, onLeave }: {
  anchor: DOMRect; name: string; src: string | null; onEnter: () => void; onLeave: () => void
}) {
  const [state, setState] = useState<'loading' | 'ok' | 'missing'>(src ? 'loading' : 'missing')
  if (typeof document === 'undefined') return null
  const vw = window.innerWidth, vh = window.innerHeight
  const w = Math.min(PREVIEW_W, vw - 16)
  const h = Math.round(w * 9 / 16) + 30                                   // image + caption
  const left = Math.max(8, Math.min(anchor.left, vw - w - 8))
  const below = anchor.bottom + 6 + h <= vh - 8 || anchor.top - 6 - h < 8
  const style: CSSProperties = { left, width: w, ...(below ? { top: Math.min(anchor.bottom + 6, vh - h - 8) } : { bottom: vh - anchor.top + 6 }) }
  return createPortal(
    <div role="tooltip" style={style} onMouseEnter={onEnter} onMouseLeave={onLeave}
      className="fixed z-[55] rounded-md border border-line bg-elev p-1.5 shadow-[var(--shadow)]">
      <div className="relative aspect-video w-full overflow-hidden rounded bg-video">
        {state === 'loading' && <Skeleton className="absolute inset-0" />}
        {state === 'missing' ? <Missing text="Screenshot not available" />
          : /* eslint-disable-next-line @next/next/no-img-element */
          <img src={url(src!)} alt={`Screenshot ${name}`} onLoad={() => setState('ok')} onError={() => setState('missing')}
            className="absolute inset-0 h-full w-full object-contain" />}
      </div>
      <p className="mt-1 truncate font-mono text-[10px] text-fg-3">{name} · click for the full evidence</p>
    </div>,
    document.body,
  )
}

function Missing({ text }: { text: string }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 p-3 text-center text-[11px] text-fg-3">
      <ImageOff className="size-5" aria-hidden />{text}
    </div>
  )
}

/** Everything stored for one CSV row: both images (full size, loaded only now) and the saved result. */
export function EvidenceViewer({ evidenceUrl, onClose }: { evidenceUrl: string; onClose: () => void }) {
  const q = useQuery(`evidence:${evidenceUrl}`, () => Api.screenshotEvidence(evidenceUrl))
  const e = q.data
  return (
    <Modal open onClose={onClose} width="w-[980px]" title={e?.original.filename ?? 'Screenshot evidence'}>
      {q.error && !e ? <ErrorState compact error={q.error} title="This screenshot's evidence could not be loaded" onRetry={q.refetch} />
        : !e ? <LoadingState rows={4} label="Loading evidence" />
          : <EvidenceBody e={e} />}
    </Modal>
  )
}

function EvidenceBody({ e }: { e: ScreenshotEvidence }) {
  const o = e.prediction ? OUTCOME[e.prediction] : null
  const scored = e.people.filter((p) => p.confidence != null)
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2 text-[12px]">
        {o ? <Badge tone={o.tone} dot>{o.label}</Badge> : <Badge tone="neutral">NO PREDICTION</Badge>}
        <span className="font-mono text-fg">{e.video_timestamp_hhmmss}</span>
        <span className="truncate text-fg-3" title={e.source_video}>{e.source_video}</span>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <EvidenceImage title="Original screenshot" file={e.original} why={e.extraction_status !== 'saved'
          ? `The screenshot was not extracted${e.extraction_error ? ` (${e.extraction_error})` : ''}.`
          : 'The saved screenshot file is no longer in storage.'} />
        <EvidenceImage title="Annotated copy (detection boxes)" file={e.annotated} why={!e.run_id
          ? 'Not analysed yet — no detection run has processed this screenshot.'
          : e.annotated.filename ? 'The annotated file is no longer in storage.'
            : `No annotated copy was saved for this screenshot (${e.processing_status}).`} />
      </div>
      <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-[12px]">
        <dt className="text-fg-3">Video</dt><dd className="break-all text-fg-2">{e.source_video}</dd>
        <dt className="text-fg-3">Screenshot file</dt><dd className="break-all font-mono text-[11px] text-fg-2">{e.original.filename ?? '—'}</dd>
        <dt className="text-fg-3">Time in video</dt><dd className="font-mono text-fg-2">{e.video_timestamp_hhmmss}{e.frame_number != null ? <span className="font-sans text-fg-3"> · frame {e.frame_number}</span> : null}</dd>
        <dt className="text-fg-3">Recorded at</dt><dd className="text-fg-2">{e.recording_timestamp ? <span className="font-mono">{e.recording_timestamp.replace('T', ' ')}</span> : 'unknown — the recording start time was not supplied'}</dd>
        <dt className="text-fg-3">Store / camera</dt><dd className="text-fg-2">{e.store_id || 'unknown'} / {e.camera_id || 'unknown'}</dd>
        <dt className="text-fg-3">Classification</dt><dd className="text-fg-2">
          {e.people.length ? e.people.map((p) => `Person ${p.person}: ${PERSON[p.type] ?? p.type}`).join(' · ')
            : e.prediction && e.prediction !== 'PROCESSING_FAILED' ? 'no person detected' : 'no prediction'}</dd>
        <dt className="text-fg-3">Confidence score</dt><dd className="text-fg-2">
          {e.row['confidence score'] || '—'}{scored.length ? <span className="text-fg-3"> · uniform classifier P(uniform){e.confidence_threshold != null ? `, threshold ${e.confidence_threshold.toFixed(2)}` : ''}</span> : null}</dd>
        <dt className="text-fg-3">Remark</dt><dd className="text-fg-2">{e.row['remark (Example)']}</dd>
        <dt className="text-fg-3">Job / run</dt><dd className="break-all font-mono text-[11px] text-fg-2">{e.job_id} / {e.run_id ?? 'not analysed'}</dd>
        <dt className="text-fg-3">Image ID</dt><dd className="break-all font-mono text-[11px] text-fg-2">{e.image_id}</dd>
        {e.model_version && <><dt className="text-fg-3">Model</dt><dd className="font-mono text-[11px] text-fg-2">{e.model_version}</dd></>}
      </dl>
    </div>
  )
}

function EvidenceImage({ title, file, why }: { title: string; file: { url: string | null; available: boolean }; why: string }) {
  const [broken, setBroken] = useState(false)
  const ok = file.available && file.url && !broken
  return (
    <figure className="min-w-0">
      <div className="relative aspect-video w-full overflow-hidden rounded-md border border-line bg-video">
        {ok ? (
          <a href={url(file.url!)} target="_blank" rel="noreferrer" title="Open at full size">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={url(file.url!)} alt={title} onError={() => setBroken(true)} className="absolute inset-0 h-full w-full object-contain" />
          </a>
        ) : <Missing text={broken ? 'The image could not be loaded.' : why} />}
      </div>
      <figcaption className="mt-0.5 text-[11px] text-fg-3">{title}</figcaption>
    </figure>
  )
}
