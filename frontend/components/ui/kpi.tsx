'use client'

import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { cx } from '../../lib/format'
import { Sparkline } from './charts'
import { Skeleton } from './states'
import type { Tone } from './primitives'

const TONE_TEXT: Record<Tone, string> = { ok: 'text-ok', warn: 'text-warn', crit: 'text-crit', info: 'text-info', exp: 'text-exp', neutral: 'text-fg-3' }

/**
 * KPI tile. `value` of null renders an em dash (no data) — never a made-up number.
 */
export function KpiCard({ label, value, sub, icon: Icon, tone = 'neutral', spark, loading, onClick, status }: {
  label: string; value: ReactNode | null; sub?: ReactNode; icon?: LucideIcon; tone?: Tone; spark?: number[]; loading?: boolean; onClick?: () => void; status?: ReactNode
}) {
  const Comp = onClick ? 'button' : 'div'
  return (
    <Comp onClick={onClick}
      className={cx('flex min-w-0 flex-col gap-1 rounded-lg border border-line bg-panel px-3.5 py-3 text-left', onClick && 'transition-colors hover:border-line-strong hover:bg-panel-2')}>
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[11px] font-medium uppercase tracking-[0.06em] text-fg-3">{label}</span>
        {Icon && <Icon className={cx('size-3.5 shrink-0', TONE_TEXT[tone])} aria-hidden />}
      </div>
      {loading ? <Skeleton className="h-7 w-16" /> : (
        <div className="tabular truncate text-[22px] font-semibold leading-7 tracking-tight text-fg">{value ?? '—'}</div>
      )}
      <div className="flex min-h-4 items-center justify-between gap-2">
        <span className="truncate text-[11px] text-fg-3">{sub}</span>
        {status}
      </div>
      {spark && spark.length > 1 && <Sparkline values={spark} color={tone === 'neutral' ? 'var(--info)' : `var(--${tone})`} className="mt-0.5" />}
    </Comp>
  )
}

export function MetricCard({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: Tone }) {
  return (
    <div className="min-w-0 rounded-md border border-line bg-panel-2 px-3 py-2">
      <div className="truncate text-[11px] uppercase tracking-wide text-fg-3">{label}</div>
      <div className={cx('tabular truncate text-[17px] font-semibold', tone ? TONE_TEXT[tone] : 'text-fg')}>{value}</div>
      {hint && <div className="truncate text-[11px] text-fg-3">{hint}</div>}
    </div>
  )
}
