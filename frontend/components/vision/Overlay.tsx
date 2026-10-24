'use client'

import { useMemo, useRef, type ReactNode } from 'react'
import type { Point, Rect } from '../../api/types'
import { cx } from '../../lib/format'

export interface OverlayBox {
  key: string | number
  bbox: Rect
  label?: string
  sub?: string
  color: string
  dashed?: boolean
  thin?: boolean
  onClick?: () => void
  selected?: boolean
}

/** Normalised (0..1) polygon drawn over the image. */
export interface OverlayPolygon { key: string | number; points: Point[]; color?: string; label?: string; dashed?: boolean }

/**
 * Image + SVG overlay in the image's own pixel space (viewBox), so boxes stay
 * aligned at any display size. Labels avoid overlapping by stacking.
 */
export function ImageOverlay({ src, width, height, boxes = [], polygons, showLabels = true, className, children, fit = 'contain', onImageClick, alt }: {
  src: string | null; width: number; height: number; boxes?: OverlayBox[]; polygons?: OverlayPolygon[]; showLabels?: boolean
  className?: string; children?: ReactNode; fit?: 'contain' | 'cover'; onImageClick?: (x: number, y: number) => void; alt: string
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const fontSize = Math.max(11, Math.round(Math.max(width, height) / 70))
  const labels = useMemo(() => {
    const placed: [number, number, number, number][] = []
    const hit = (r: number[], p: number[]) => !(r[2] <= p[0] || r[0] >= p[2] || r[3] <= p[1] || r[1] >= p[3])
    return boxes.map((b, bi) => {
      if (!b.label) return null
      const text = b.sub ? `${b.label}  ${b.sub}` : b.label
      const w = text.length * fontSize * 0.58 + 8, h = fontSize + 6
      const [x1, y1, x2, y2] = b.bbox
      const others = boxes.filter((o, oi) => oi !== bi && o.label).map((o) => o.bbox)
      const tries = [[x1, y1 - h], [x1, y2 + 1], [x2 - w, y1 - h], [x2 - w, y2 + 1], [x2 + 2, y1], [x1 - w - 2, y1], [x1, y1 - 2 * h], [x1, y2 + h + 1]]
      for (const [tx, ty] of tries) {
        const x = Math.max(0, Math.min(tx, width - w)), y = Math.max(0, Math.min(ty, height - h))
        const r: [number, number, number, number] = [x, y, x + w, y + h]
        if (!placed.some((p) => hit(r, p)) && !others.some((o) => hit(r, o))) {
          placed.push(r)
          return { x, y, w, h, text }
        }
      }
      return null
    })
  }, [boxes, fontSize, width, height])

  return (
    <div className={cx('overflow-hidden bg-video', className ?? 'relative')}>
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt={alt} draggable={false} className={cx('absolute inset-0 size-full select-none', fit === 'cover' ? 'object-cover' : 'object-contain')} />
      ) : null}
      {width > 0 && height > 0 && (
        <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio={fit === 'cover' ? 'xMidYMid slice' : 'xMidYMid meet'}
          className={cx('absolute inset-0 size-full', onImageClick ? 'cursor-crosshair' : 'pointer-events-none')}
          onClick={(e) => {
            if (!onImageClick || !svgRef.current) return
            const p = toSvg(svgRef.current, e.clientX, e.clientY)
            const tol = Math.max(width, height) * 0.01
            if (p.x >= -tol && p.y >= -tol && p.x <= width + tol && p.y <= height + tol) {
              onImageClick(Math.min(1, Math.max(0, p.x / width)), Math.min(1, Math.max(0, p.y / height)))
            }
          }}>
          {polygons?.map((z) => (
            <polygon key={z.key} points={z.points.map(([x, y]) => `${x * width},${y * height}`).join(' ')}
              fill="rgba(255,255,255,0.12)" stroke={z.color ?? 'var(--info)'} strokeWidth={Math.max(2, width / 320)} strokeLinejoin="round"
              strokeDasharray={z.dashed ? '8 5' : undefined}>
              {z.label && <title>{z.label}</title>}
            </polygon>
          ))}
          {showLabels && labels.map((l, i) => l && (
            <g key={i}>
              <rect x={l.x} y={l.y} width={l.w} height={l.h} fill={boxes[i].color} opacity={0.92} rx={2} />
              <text x={l.x + 4} y={l.y + l.h - 4} fontSize={fontSize} fontFamily="var(--font-mono)" fill="#0a0a0a" fontWeight={600}>{l.text}</text>
            </g>
          ))}
          {boxes.map((b) => (
            <rect key={b.key} x={b.bbox[0]} y={b.bbox[1]} width={Math.max(1, b.bbox[2] - b.bbox[0])} height={Math.max(1, b.bbox[3] - b.bbox[1])}
              fill={b.selected ? 'rgba(255,255,255,0.15)' : 'transparent'} stroke={b.color}
              strokeWidth={(b.thin ? 1 : 2) * Math.max(1, width / 900) * (b.selected ? 1.8 : 1)} strokeDasharray={b.dashed ? '5 3' : undefined}
              className={b.onClick ? 'pointer-events-auto cursor-pointer' : undefined} onClick={(e) => { if (b.onClick) { e.stopPropagation(); b.onClick() } }}>
              {b.label && <title>{b.sub ? `${b.label} ${b.sub}` : b.label}</title>}
            </rect>
          ))}
        </svg>
      )}
      {children}
    </div>
  )
}

/** Screen -> SVG user-space coordinates. */
export function toSvg(svg: SVGSVGElement, clientX: number, clientY: number) {
  const pt = svg.createSVGPoint()
  pt.x = clientX
  pt.y = clientY
  return pt.matrixTransform(svg.getScreenCTM()!.inverse())
}

/** Keeps a media container at the source aspect ratio. */
export function AspectBox({ width, height, className, children }: { width: number; height: number; className?: string; children: ReactNode }) {
  return <div className={cx('relative w-full', className)} style={{ aspectRatio: `${width || 16} / ${height || 9}` }}>{children}</div>
}
