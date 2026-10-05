'use client'

import { useMemo, useState, type ReactNode } from 'react'
import { cx } from '../../lib/format'

// Lightweight SVG charts. Every chart takes real data only; callers render an
// EmptyState when there is nothing to show. Colours use semantic CSS tokens.

export const SERIES = ['var(--info)', 'var(--ok)', 'var(--warn)', 'var(--exp)', 'var(--crit)', 'var(--fg-3)',
  '#38bdf8', '#f472b6', '#84cc16', '#fb923c', '#22d3ee', '#e879f9', '#a3a3a3']

export function Sparkline({ values, color = 'var(--info)', height = 24, className }: { values: number[]; color?: string; height?: number; className?: string }) {
  if (values.length < 2) return <div style={{ height }} className={className} aria-hidden />
  const max = Math.max(...values, 1), w = 100
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${height - 2 - (v / max) * (height - 4)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" className={cx('w-full', className)} style={{ height }} aria-hidden>
      <polyline points={`0,${height} ${pts} ${w},${height}`} fill={color} opacity={0.12} stroke="none" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

function niceMax(v: number): number {
  if (v <= 0) return 1
  const p = 10 ** Math.floor(Math.log10(v))
  const n = v / p
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p
}

export interface Series { name: string; color?: string; values: (number | null)[]; dashed?: boolean }

/** Multi-series line chart over categorical x labels. */
export function LineChart({ labels, series, height = 200, yMax, yFormat = (v) => String(Math.round(v * 100) / 100), area, markerX, ariaLabel }: {
  labels: string[]; series: Series[]; height?: number; yMax?: number; yFormat?: (v: number) => string; area?: boolean; markerX?: number; ariaLabel: string
}) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 640, H = height, L = 40, R = 10, T = 10, B = 24
  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null))
  const max = yMax ?? niceMax(Math.max(...all, 0))
  const x = (i: number) => L + (labels.length <= 1 ? (W - L - R) / 2 : (i / (labels.length - 1)) * (W - L - R))
  const y = (v: number) => T + (1 - v / max) * (H - T - B)
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * max)
  const step = Math.max(1, Math.ceil(labels.length / 8))
  return (
    <div className="relative w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }} role="img" aria-label={ariaLabel}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          const px = ((e.clientX - r.left) / r.width) * W
          const i = Math.round(((px - L) / (W - L - R)) * (labels.length - 1))
          setHover(i >= 0 && i < labels.length ? i : null)
        }}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth={1} />
            <text x={L - 6} y={y(t) + 3} textAnchor="end" fontSize={10} fill="var(--fg-3)">{yFormat(t)}</text>
          </g>
        ))}
        {labels.map((l, i) => i % step === 0 && (
          <text key={i} x={x(i)} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--fg-3)">{l}</text>
        ))}
        {markerX !== undefined && markerX >= 0 && <line x1={x(markerX)} x2={x(markerX)} y1={T} y2={H - B} stroke="var(--warn)" strokeDasharray="3 3" />}
        {series.map((s, si) => {
          const color = s.color ?? SERIES[si % SERIES.length]
          const pts = s.values.map((v, i) => (v === null ? null : `${x(i)},${y(v)}`)).filter(Boolean) as string[]
          if (!pts.length) return null
          return (
            <g key={s.name}>
              {area && <polygon points={`${x(0)},${y(0)} ${pts.join(' ')} ${x(s.values.length - 1)},${y(0)}`} fill={color} opacity={0.1} />}
              <polyline points={pts.join(' ')} fill="none" stroke={color} strokeWidth={1.6} strokeDasharray={s.dashed ? '4 3' : undefined} />
              {labels.length <= 40 && s.values.map((v, i) => v !== null && <circle key={i} cx={x(i)} cy={y(v)} r={hover === i ? 3.2 : 1.8} fill={color} />)}
            </g>
          )
        })}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={T} y2={H - B} stroke="var(--line-strong)" />}
      </svg>
      {hover !== null && (
        <div className="pointer-events-none absolute top-1 rounded-md border border-line bg-elev px-2 py-1 text-[11px] shadow-[var(--shadow)]"
          style={{ left: `${Math.min(80, (x(hover) / W) * 100)}%` }}>
          <div className="mb-0.5 font-medium text-fg">{labels[hover]}</div>
          {series.map((s, si) => (
            <div key={s.name} className="flex items-center gap-1.5 text-fg-2">
              <span className="size-2 rounded-full" style={{ background: s.color ?? SERIES[si % SERIES.length] }} />
              {s.name}: <span className="tabular font-mono text-fg">{s.values[hover] === null ? '—' : yFormat(s.values[hover] as number)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/** XY line chart on [0,1]×[0,1] (e.g. precision–recall curve). */
export function XYChart({ series, height = 240, xLabel, yLabel, marker, ariaLabel }: {
  series: { name: string; color: string; points: { x: number; y: number; label?: string }[] }[]
  height?: number; xLabel: string; yLabel: string; marker?: { x: number; y: number; label: string }; ariaLabel: string
}) {
  const W = 420, H = height, L = 40, R = 12, T = 10, B = 30
  const x = (v: number) => L + v * (W - L - R)
  const y = (v: number) => T + (1 - v) * (H - T - B)
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: H }} role="img" aria-label={ariaLabel}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--line)" />
          <line x1={x(t)} x2={x(t)} y1={T} y2={H - B} stroke="var(--line)" />
          <text x={L - 6} y={y(t) + 3} textAnchor="end" fontSize={10} fill="var(--fg-3)">{t}</text>
          <text x={x(t)} y={H - B + 12} textAnchor="middle" fontSize={10} fill="var(--fg-3)">{t}</text>
        </g>
      ))}
      <text x={(L + W - R) / 2} y={H - 3} textAnchor="middle" fontSize={10} fill="var(--fg-2)">{xLabel}</text>
      <text x={10} y={(T + H - B) / 2} textAnchor="middle" fontSize={10} fill="var(--fg-2)" transform={`rotate(-90 10 ${(T + H - B) / 2})`}>{yLabel}</text>
      {series.map((s) => {
        const pts = [...s.points].sort((a, b) => a.x - b.x)
        return (
          <g key={s.name}>
            <polyline points={pts.map((p) => `${x(p.x)},${y(p.y)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={1.8} />
            {pts.map((p, i) => <circle key={i} cx={x(p.x)} cy={y(p.y)} r={2.2} fill={s.color}><title>{p.label ?? `${p.x.toFixed(2)}, ${p.y.toFixed(2)}`}</title></circle>)}
          </g>
        )
      })}
      {marker && <g><circle cx={x(marker.x)} cy={y(marker.y)} r={5} fill="none" stroke="var(--warn)" strokeWidth={2} /><title>{marker.label}</title></g>}
    </svg>
  )
}

export function Legend({ items }: { items: { name: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1">
      {items.map((i) => (
        <span key={i.name} className="flex items-center gap-1.5 text-[11px] text-fg-2">
          <span className="size-2 rounded-sm" style={{ background: i.color }} aria-hidden />{i.name}
        </span>
      ))}
    </div>
  )
}

/** Horizontal bar list: best for categories (class, camera). */
export function BarList({ data, format = (v) => String(v), color = 'var(--info)', max: maxIn, onClick }: {
  data: { label: string; value: number; color?: string; hint?: ReactNode }[]; format?: (v: number) => string; color?: string; max?: number; onClick?: (label: string) => void
}) {
  const max = maxIn ?? Math.max(...data.map((d) => d.value), 1)
  return (
    <ul className="flex flex-col gap-1.5">
      {data.map((d) => (
        <li key={d.label}>
          <button disabled={!onClick} onClick={() => onClick?.(d.label)} className="group flex w-full items-center gap-2 text-left disabled:cursor-default">
            <span className="w-32 shrink-0 truncate text-[12px] text-fg-2 group-enabled:group-hover:text-fg" title={d.label}>{d.label}</span>
            <span className="relative h-4 flex-1 overflow-hidden rounded-sm bg-panel-2">
              <span className="absolute inset-y-0 left-0 rounded-sm" style={{ width: `${(d.value / max) * 100}%`, background: d.color ?? color, opacity: 0.85 }} />
            </span>
            <span className="tabular w-14 shrink-0 text-right font-mono text-[12px] text-fg">{format(d.value)}</span>
          </button>
          {d.hint}
        </li>
      ))}
    </ul>
  )
}

/** Vertical columns for ordered bins (hour of day, histogram). */
export function Columns({ data, height = 140, color = 'var(--info)', format = (v) => String(v), ariaLabel }: {
  data: { label: string; value: number; color?: string }[]; height?: number; color?: string; format?: (v: number) => string; ariaLabel: string
}) {
  const [hover, setHover] = useState<number | null>(null)
  const max = niceMax(Math.max(...data.map((d) => d.value), 0))
  const step = Math.max(1, Math.ceil(data.length / 12))
  return (
    <div className="w-full" role="img" aria-label={ariaLabel}>
      <div className="relative flex items-end gap-[2px]" style={{ height }} onMouseLeave={() => setHover(null)}>
        {data.map((d, i) => (
          <div key={d.label} className="relative flex h-full flex-1 items-end" onMouseEnter={() => setHover(i)}>
            <div className="w-full rounded-t-[2px] transition-opacity" title={`${d.label}: ${format(d.value)}`}
              style={{ height: `${(d.value / max) * 100}%`, minHeight: d.value > 0 ? 2 : 0, background: d.color ?? color, opacity: hover === null || hover === i ? 0.9 : 0.45 }} />
          </div>
        ))}
        {hover !== null && (
          <div className="pointer-events-none absolute -top-1 rounded-md border border-line bg-elev px-2 py-0.5 text-[11px] shadow-[var(--shadow)]"
            style={{ left: `${Math.min(85, (hover / data.length) * 100)}%` }}>
            {data[hover].label}: <span className="tabular font-mono text-fg">{format(data[hover].value)}</span>
          </div>
        )}
      </div>
      <div className="mt-1 flex gap-[2px]">
        {data.map((d, i) => <span key={d.label} className="flex-1 truncate text-center text-[10px] text-fg-3">{i % step === 0 ? d.label : ''}</span>)}
      </div>
    </div>
  )
}

export function Donut({ data, size = 132, center, ariaLabel }: { data: { label: string; value: number; color: string }[]; size?: number; center?: ReactNode; ariaLabel: string }) {
  const total = data.reduce((a, d) => a + d.value, 0)
  const arcs = useMemo(() => {
    let acc = 0
    return data.map((d) => {
      const start = acc / (total || 1)
      acc += d.value
      return { ...d, start, end: acc / (total || 1) }
    })
  }, [data, total])
  const r = 42, c = 2 * Math.PI * r
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }} role="img" aria-label={ariaLabel}>
      <svg viewBox="0 0 100 100" className="size-full -rotate-90">
        <circle cx={50} cy={50} r={r} fill="none" stroke="var(--panel-2)" strokeWidth={11} />
        {total > 0 && arcs.map((a) => (
          <circle key={a.label} cx={50} cy={50} r={r} fill="none" stroke={a.color} strokeWidth={11}
            strokeDasharray={`${Math.max(0, (a.end - a.start) * c - 0.8)} ${c}`} strokeDashoffset={-a.start * c}>
            <title>{`${a.label}: ${a.value}`}</title>
          </circle>
        ))}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">{center}</div>
    </div>
  )
}

/** Stacked horizontal proportion bar. */
export function StackBar({ parts }: { parts: { label: string; value: number; color: string }[] }) {
  const total = parts.reduce((a, p) => a + p.value, 0)
  return (
    <div className="flex h-2 w-full overflow-hidden rounded-full bg-panel-2" role="img" aria-label={parts.map((p) => `${p.label} ${p.value}`).join(', ')}>
      {total > 0 && parts.map((p) => <span key={p.label} title={`${p.label}: ${p.value}`} style={{ width: `${(p.value / total) * 100}%`, background: p.color }} />)}
    </div>
  )
}
