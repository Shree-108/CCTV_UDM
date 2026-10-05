'use client'

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Check, CheckCircle2, Eraser, MousePointer2, MousePointerClick, PenLine, Save, SquareDashed, Undo2, UploadCloud } from 'lucide-react'
import { url } from '../../api/client'
import type { Area, Layout, Media, Point, Rect } from '../../api/types'
import { cx, duration } from '../../lib/format'
import { Badge, Button, Segmented } from '../../components/ui/primitives'
import { AspectBox, toSvg } from '../../components/vision/Overlay'

// ---------------------------------------------------------------- upload
export function DropZone({ accept, onFile, label, hint, formats, compact, busy }: {
  accept: string; onFile: (f: File) => void; label: string; hint: ReactNode; formats?: string[]; compact?: boolean; busy?: boolean
}) {
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const pick = useCallback((f?: File | null) => f && !busy && onFile(f), [onFile, busy])
  return (
    <div
      role="button" tabIndex={0} aria-label={label} aria-busy={busy}
      onClick={() => !busy && input.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !busy && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files?.[0]) }}
      className={cx('flex cursor-pointer flex-col items-center justify-center gap-2.5 rounded-lg border border-dashed text-center transition-colors',
        compact ? 'p-6' : 'min-h-[340px] p-10', over ? 'border-info bg-info-bg' : 'border-line-strong bg-panel-2/40 hover:border-info/60 hover:bg-info-bg/40',
        busy && 'pointer-events-none opacity-70')}>
      <div className="flex size-12 items-center justify-center rounded-xl bg-info-bg text-info ring-1 ring-info/30">
        <UploadCloud className={cx('size-6', busy && 'animate-pulse')} aria-hidden />
      </div>
      <p className="text-[15px] font-semibold tracking-wide text-fg">{busy ? 'Uploading…' : label}</p>
      <p className="text-[12px] text-fg-3">{hint}</p>
      {formats && (
        <div className="mt-1 flex flex-wrap justify-center gap-1">
          {formats.map((f) => <span key={f} className="rounded-sm border border-line bg-panel px-1.5 py-px font-mono text-[10px] text-fg-3">{f}</span>)}
        </div>
      )}
      <input ref={input} type="file" accept={accept} className="hidden" onChange={(e) => { pick(e.target.files?.[0]); e.target.value = '' }} />
    </div>
  )
}

export function MediaInfo({ media }: { media: Media }) {
  const items: [string, string][] = [
    ['File', media.filename],
    ['Size', `${(media.size_bytes / 2 ** 20).toFixed(1)} MB`],
    ['Resolution', `${media.width} × ${media.height}`],
  ]
  if (media.kind === 'video') {
    items.push(['Duration', duration(media.duration_sec)], ['Frame rate', media.fps ? `${media.fps} fps` : '—'],
      ['Frames', media.frames ? media.frames.toLocaleString() : '—'])
  }
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3 xl:grid-cols-6">
      {items.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <dt className="text-[10px] uppercase tracking-wide text-fg-3">{k}</dt>
          <dd className="truncate text-[13px] text-fg" title={v}>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

// ---------------------------------------------------------------- steps
export function StepBar({ steps, current }: { steps: string[]; current: number }) {
  return (
    <ol className="flex flex-wrap items-center gap-1.5" aria-label="Analysis steps">
      {steps.map((s, i) => {
        const done = i < current, on = i === current
        return (
          <li key={s} className="flex items-center gap-1.5">
            <span className={cx('flex h-6 items-center gap-1.5 rounded-full border px-2.5 text-[11px] font-medium',
              done ? 'border-ok/40 bg-ok-bg text-ok' : on ? 'border-info bg-info-bg text-info' : 'border-line text-fg-3')}
              aria-current={on ? 'step' : undefined}>
              {done ? <Check className="size-3" aria-hidden /> : <span className="tabular font-mono">{i + 1}</span>}{s}
            </span>
            {i < steps.length - 1 && <span className={cx('h-px w-3', done ? 'bg-ok/50' : 'bg-line')} aria-hidden />}
          </li>
        )
      })}
    </ol>
  )
}

// ---------------------------------------------------------------- grid selection
const LAYOUT_SHORT: Record<string, string> = { '1x1': 'Single', '2x2': '2×2', '1x2': '1×2', '2x1': '2×1', '2x3': '2×3', '3x3': '3×3', '4x4': '4×4', custom: 'Custom' }

export interface GridState { layout: string; cell: number | null; rect: Rect; label: string }

export function initialGrid(media: Media, layouts: Layout[]): GridState {
  const l = layouts.find((x) => x.id === media.suggested_layout) ?? layouts[0]
  return { layout: l.id, cell: 0, rect: l.cells[0], label: l.cells.length > 1 ? 'Camera 1' : 'Full frame' }
}

/**
 * Pick the camera tile that shows the store interior: click a tile of the detected layout,
 * or switch to Custom and drag a rectangle. Only this area is sent to the model.
 */
export function GridSelector({ media, layouts, value, onChange, confirmed }: {
  media: Media; layouts: Layout[]; value: GridState; onChange: (g: GridState) => void; confirmed: boolean
}) {
  const W = media.width, H = media.height
  const svg = useRef<SVGSVGElement>(null)
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null)
  const layout = layouts.find((l) => l.id === value.layout)
  const cells = layout?.cells ?? []
  const custom = value.layout === 'custom'

  const setLayout = (id: string) => {
    if (id === 'custom') return onChange({ layout: 'custom', cell: null, rect: value.rect, label: 'Custom area' })
    const l = layouts.find((x) => x.id === id)!
    onChange({ layout: id, cell: 0, rect: l.cells[0], label: l.cells.length > 1 ? 'Camera 1' : 'Full frame' })
  }
  const norm = (e: React.PointerEvent) => {
    const p = toSvg(svg.current!, e.clientX, e.clientY)
    return [Math.min(1, Math.max(0, p.x / W)), Math.min(1, Math.max(0, p.y / H))] as const
  }
  const [x1, y1, x2, y2] = drag
    ? [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)]
    : value.rect
  const stroke = Math.max(2, W / 320), font = Math.max(12, Math.round(W / 55))

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <Segmented label="Screen layout" size="sm" value={value.layout} onChange={setLayout}
          options={[...layouts.map((l) => ({ value: l.id, label: LAYOUT_SHORT[l.id] ?? l.id, title: l.label })), { value: 'custom', label: 'Custom', icon: SquareDashed, title: 'Draw a rectangle' }]} />
        {media.suggested_layout !== '1x1' && <Badge tone="info">Auto-detected: {LAYOUT_SHORT[media.suggested_layout]} grid</Badge>}
        <span className="text-[11px] text-fg-3">
          {custom ? <><MousePointer2 className="mr-1 inline size-3" aria-hidden />Drag on the frame to draw the store-interior camera area</>
            : cells.length > 1 ? <><MousePointerClick className="mr-1 inline size-3" aria-hidden />Click the camera that shows the store interior</> : 'Single camera view — the whole frame is analysed'}
        </span>
      </div>
      <AspectBox width={W} height={H} className="overflow-hidden rounded-md border border-line bg-video">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={url(media.preview_url)}
          alt="CCTV preview" className="absolute inset-0 size-full select-none object-contain" draggable={false} />
        <svg ref={svg} viewBox={`0 0 ${W} ${H}`} className={cx('absolute inset-0 size-full touch-none', custom ? 'cursor-crosshair' : cells.length > 1 && 'cursor-pointer')}
          onPointerDown={(e) => { if (!custom) return; const [x, y] = norm(e); (e.target as Element).setPointerCapture?.(e.pointerId); setDrag({ x0: x, y0: y, x1: x, y1: y }) }}
          onPointerMove={(e) => { if (drag) { const [x, y] = norm(e); setDrag({ ...drag, x1: x, y1: y }) } }}
          onPointerUp={() => {
            if (!drag) return
            if (x2 - x1 > 0.03 && y2 - y1 > 0.03) onChange({ layout: 'custom', cell: null, rect: [x1, y1, x2, y2].map((v) => Math.round(v * 1e4) / 1e4) as Rect, label: 'Custom area' })
            setDrag(null)
          }}>
          <defs>
            <mask id="sel-mask">
              <rect width={W} height={H} fill="white" />
              <rect x={x1 * W} y={y1 * H} width={(x2 - x1) * W} height={(y2 - y1) * H} fill="black" />
            </mask>
          </defs>
          <rect width={W} height={H} fill="rgba(3,12,16,0.62)" mask="url(#sel-mask)" />
          {!custom && cells.length > 1 && cells.map((c, i) => {
            const sel = i === value.cell
            return (
              <g key={i} onClick={() => onChange({ layout: value.layout, cell: i, rect: c, label: `Camera ${i + 1}` })}>
                <rect x={c[0] * W} y={c[1] * H} width={(c[2] - c[0]) * W} height={(c[3] - c[1]) * H} fill="transparent"
                  stroke={sel ? 'var(--info)' : 'rgba(255,255,255,0.35)'} strokeWidth={sel ? stroke * 1.6 : 1} />
                <rect x={c[0] * W + 8} y={c[1] * H + 8} width={font * (sel ? 13.6 : 9.4)} height={font * 1.7} rx={4}
                  fill={sel ? 'var(--info)' : 'rgba(6,21,29,0.85)'} />
                <text x={c[0] * W + 8 + font * 0.5} y={c[1] * H + 8 + font * 1.2} fontSize={font} fontWeight={600} fontFamily="var(--font-mono)"
                  fill={sel ? '#04161c' : 'rgba(230,242,246,0.8)'}>{`CAMERA ${i + 1}${sel ? ' · SELECTED ✓' : ' · ignored'}`}</text>
              </g>
            )
          })}
          {(custom || cells.length <= 1) && (
            <rect x={x1 * W} y={y1 * H} width={(x2 - x1) * W} height={(y2 - y1) * H} fill="transparent" stroke="var(--info)" strokeWidth={stroke * 1.6}
              strokeDasharray={drag ? '10 6' : undefined} />
          )}
        </svg>
      </AspectBox>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[12px]">
        <span className="text-fg-3">Selected camera:</span>
        <span className="font-semibold uppercase tracking-wide text-info">{value.label}</span>
        {confirmed ? <Badge tone="ok" dot>✓ Store interior selected</Badge> : <Badge tone="warn" dot>Not confirmed</Badge>}
        {!custom && cells.length > 1 && (
          <span className="flex flex-wrap gap-1.5 text-[11px] text-fg-3">
            {cells.map((_, i) => <span key={i} className={i === value.cell ? 'text-info' : ''}>Camera {i + 1} → {i === value.cell ? 'selected' : 'ignored'}</span>)}
          </span>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- ROI inside the selected tile
/**
 * Optional store-interior polygon inside the selected camera tile. Points are normalised to
 * the tile. Only people whose feet fall inside it count towards the compliance result.
 */
export function RoiEditor({ media, rect, roi, onChange, onSave, onClear, savedPreset }: {
  media: Media; rect: Rect; roi: Point[] | null; onChange: (r: Point[] | null) => void
  onSave: (r: Point[]) => Promise<void>; onClear: () => Promise<void>; savedPreset: Point[] | null
}) {
  const W = media.width, H = media.height
  const [x1, y1, x2, y2] = rect
  const tw = (x2 - x1) * W, th = (y2 - y1) * H
  const svg = useRef<SVGSVGElement>(null)
  const [mode, setMode] = useState<'view' | 'define' | 'edit'>('view')
  const [draft, setDraft] = useState<Point[]>(roi ?? [])
  const [dragIdx, setDragIdx] = useState<number | null>(null)
  const [saving, setSaving] = useState(false)
  useEffect(() => { if (mode === 'view') setDraft(roi ?? []) }, [roi, mode])
  const pts = mode === 'view' ? roi ?? [] : draft
  const src = url(media.preview_url)

  const toTile = (e: React.PointerEvent | React.MouseEvent): Point => {
    const p = toSvg(svg.current!, e.clientX, e.clientY)
    return [Math.round(Math.min(1, Math.max(0, p.x / tw)) * 1e4) / 1e4, Math.round(Math.min(1, Math.max(0, p.y / th)) * 1e4) / 1e4]
  }
  const r = Math.max(4, tw / 110), sw = Math.max(2, tw / 320)
  const commit = () => { onChange(draft.length >= 3 ? draft : null); setMode('view') }

  return (
    <div className="flex flex-col gap-2.5">
      <AspectBox width={tw} height={th} className="overflow-hidden rounded-md border border-line bg-video">
        <svg ref={svg} viewBox={`0 0 ${tw} ${th}`} className={cx('absolute inset-0 size-full touch-none', mode === 'define' && 'cursor-crosshair')}
          onClick={(e) => { if (mode === 'define') setDraft((d) => [...d, toTile(e)]) }}
          onPointerMove={(e) => { if (mode === 'edit' && dragIdx !== null) { const p = toTile(e); setDraft((d) => d.map((q, i) => (i === dragIdx ? p : q))) } }}
          onPointerUp={() => setDragIdx(null)}>
          {/* the selected camera tile, cropped from the full frame */}
          <image href={src} x={-x1 * W} y={-y1 * H} width={W} height={H} preserveAspectRatio="none" />
          {pts.length >= 2 && (
            <polygon points={pts.map(([x, y]) => `${x * tw},${y * th}`).join(' ')} fill={pts.length >= 3 ? 'rgba(34,211,238,0.14)' : 'none'}
              stroke="var(--info)" strokeWidth={sw} strokeLinejoin="round" strokeDasharray={mode === 'view' ? undefined : '8 5'} />
          )}
          {pts.length >= 3 && mode === 'view' && (
            <text x={Math.min(...pts.map((p) => p[0])) * tw + 8} y={Math.min(...pts.map((p) => p[1])) * th + 20} fontSize={Math.max(12, tw / 45)}
              fill="var(--info)" fontWeight={600} fontFamily="var(--font-mono)">STORE INTERIOR</text>
          )}
          {mode !== 'view' && pts.map(([x, y], i) => (
            <circle key={i} cx={x * tw} cy={y * th} r={r} fill={mode === 'edit' ? 'var(--info)' : 'var(--warn)'} stroke="#031014" strokeWidth={1.5}
              className={mode === 'edit' ? 'cursor-grab' : undefined}
              onPointerDown={(e) => { if (mode !== 'edit') return; e.stopPropagation(); (e.target as Element).setPointerCapture?.(e.pointerId); setDragIdx(i) }} />
          ))}
        </svg>
        {mode !== 'view' && (
          <div className="pointer-events-none absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/70 px-2 py-1 text-[12px] text-white">
            <MousePointerClick className="size-3.5" aria-hidden />
            {mode === 'define' ? `Click the corners of the store interior · ${draft.length} point${draft.length === 1 ? '' : 's'}` : 'Drag the corners to adjust the ROI'}
          </div>
        )}
      </AspectBox>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[12px] text-fg-2">ROI:</span>
        {roi && roi.length >= 3 ? <Badge tone="ok" dot>Store interior ROI active · {roi.length} corners</Badge> : <Badge tone="neutral" dot>Not set — the whole camera area counts</Badge>}
        <div className="ml-auto flex flex-wrap gap-1.5">
          {mode === 'view' ? <>
            <Button size="sm" icon={PenLine} onClick={() => { setDraft([]); setMode('define') }}>Define ROI</Button>
            <Button size="sm" icon={MousePointer2} disabled={!roi} onClick={() => { setDraft(roi ?? []); setMode('edit') }}>Edit ROI</Button>
            <Button size="sm" variant="ghost" icon={Eraser} disabled={!roi && !savedPreset} onClick={async () => { onChange(null); await onClear() }}>Clear ROI</Button>
            <Button size="sm" variant="primary" icon={Save} disabled={!roi} loading={saving}
              onClick={async () => { if (!roi) return; setSaving(true); try { await onSave(roi) } finally { setSaving(false) } }}>Save ROI</Button>
          </> : <>
            {mode === 'define' && <Button size="sm" variant="ghost" icon={Undo2} disabled={!draft.length} onClick={() => setDraft((d) => d.slice(0, -1))}>Undo point</Button>}
            <Button size="sm" variant="ghost" onClick={() => setMode('view')}>Cancel</Button>
            <Button size="sm" variant="primary" icon={CheckCircle2} disabled={draft.length < 3} onClick={commit}>Apply ROI</Button>
          </>}
        </div>
      </div>
      <p className="text-[11px] text-fg-3">The bottom-centre (feet) of each person is tested against the ROI. People outside it are ignored and never affect the result. “Save ROI” remembers this camera area and ROI for your next upload.</p>
    </div>
  )
}

export const areaFrom = (g: GridState, roi: Point[] | null): Area => ({ rect: g.rect, roi, label: g.label, layout: g.layout, cell: g.cell })
