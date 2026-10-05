'use client'

import { forwardRef, useId, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from 'react'
import { Loader2, type LucideIcon } from 'lucide-react'
import { cx } from '../../lib/format'

// ---------------------------------------------------------------- Button
type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'subtle'
const VARIANTS: Record<Variant, string> = {
  primary: 'bg-info text-[#03141a] font-semibold hover:brightness-110 border-transparent shadow-[0_0_16px_rgba(34,211,238,0.18)]',
  secondary: 'bg-panel-2 text-fg border-line hover:bg-hover hover:border-line-strong',
  ghost: 'bg-transparent text-fg-2 border-transparent hover:bg-hover hover:text-fg',
  danger: 'bg-crit-bg text-crit border-transparent hover:brightness-125',
  subtle: 'bg-transparent text-fg-2 border-line hover:bg-hover hover:text-fg',
}

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: 'sm' | 'md'
  icon?: LucideIcon
  loading?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', icon: Icon, loading, className, children, disabled, ...rest }, ref) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cx(
        'inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md border font-medium transition-colors disabled:opacity-45',
        size === 'sm' ? 'h-7 px-2 text-xs' : 'h-8 px-3 text-[13px]',
        VARIANTS[variant], className)}
      {...rest}
    >
      {loading ? <Loader2 className="size-3.5 animate-spin" aria-hidden /> : Icon ? <Icon className="size-3.5" aria-hidden /> : null}
      {children}
    </button>
  )
})

export function IconButton({ icon: Icon, label, active, className, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { icon: LucideIcon; label: string; active?: boolean }) {
  return (
    <button
      aria-label={label}
      title={label}
      aria-pressed={active}
      className={cx('inline-flex size-8 items-center justify-center rounded-md text-fg-2 transition-colors hover:bg-hover hover:text-fg disabled:opacity-40',
        active && 'bg-hover text-fg', className)}
      {...rest}
    >
      <Icon className="size-4" aria-hidden />
    </button>
  )
}

// ---------------------------------------------------------------- Badges
export type Tone = 'ok' | 'warn' | 'crit' | 'info' | 'exp' | 'neutral'
const TONES: Record<Tone, string> = {
  ok: 'bg-ok-bg text-ok', warn: 'bg-warn-bg text-warn', crit: 'bg-crit-bg text-crit',
  info: 'bg-info-bg text-info', exp: 'bg-exp-bg text-exp', neutral: 'bg-panel-2 text-fg-2 border border-line',
}
const DOTS: Record<Tone, string> = { ok: 'bg-ok', warn: 'bg-warn', crit: 'bg-crit', info: 'bg-info', exp: 'bg-exp', neutral: 'bg-fg-3' }

export function Badge({ tone = 'neutral', children, dot, className, title }: { tone?: Tone; children: ReactNode; dot?: boolean; className?: string; title?: string }) {
  return (
    <span title={title} className={cx('inline-flex items-center gap-1 rounded-sm px-1.5 py-px text-[11px] font-medium leading-4', TONES[tone], className)}>
      {dot && <span className={cx('size-1.5 rounded-full', DOTS[tone])} aria-hidden />}
      {children}
    </span>
  )
}

export function Dot({ tone, pulse, label }: { tone: Tone; pulse?: boolean; label?: string }) {
  return <span role={label ? 'img' : undefined} aria-label={label} className={cx('inline-block size-2 rounded-full', DOTS[tone], pulse && 'live-dot')} />
}

const STATUS_TONE: Record<string, Tone> = {
  online: 'ok', operational: 'ok', confirmed: 'ok', done: 'ok', resolved: 'ok', approved: 'ok', active: 'ok', ok: 'ok',
  connecting: 'info', running: 'info', queued: 'info', acknowledged: 'info', candidate: 'warn', new: 'crit',
  reconnecting: 'warn', degraded: 'warn', weak: 'warn', pending: 'warn', stopped: 'neutral', disabled: 'neutral',
  not_configured: 'neutral', not_available: 'neutral', ignored: 'neutral', rejected: 'neutral',
  error: 'crit', offline: 'crit', failed: 'crit', false_positive: 'neutral',
  HIGH: 'crit', MEDIUM: 'warn', LOW: 'info',
}
const STATUS_LABEL: Record<string, string> = { false_positive: 'rejected', not_configured: 'not configured', not_available: 'n/a' }

/** Status badge: colour + text (never colour alone). */
export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const tone = STATUS_TONE[status] ?? 'neutral'
  return <Badge tone={tone} dot className={className}>{STATUS_LABEL[status] ?? status.replace(/_/g, ' ')}</Badge>
}
export const statusTone = (s: string): Tone => STATUS_TONE[s] ?? 'neutral'

// ---------------------------------------------------------------- Panel / Card
export function Panel({ title, subtitle, actions, children, className, bodyClass, noPad, icon: Icon }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClass?: string; noPad?: boolean; icon?: LucideIcon
}) {
  return (
    <section className={cx('flex min-w-0 flex-col rounded-lg border border-line bg-panel', className)}>
      {(title || actions) && (
        <header className="flex min-h-10 items-center justify-between gap-3 border-b border-line px-3.5 py-2">
          <div className="min-w-0">
            <h2 className="flex items-center gap-1.5 truncate text-[12px] font-semibold uppercase tracking-[0.06em] text-fg-2">
              {Icon && <Icon className="size-3.5" aria-hidden />}{title}
            </h2>
            {subtitle && <p className="truncate text-[11px] text-fg-3">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={cx('min-h-0 flex-1', !noPad && 'p-3.5', bodyClass)}>{children}</div>
    </section>
  )
}

export function PageHeader({ title, description, actions, badge }: { title: string; description?: ReactNode; actions?: ReactNode; badge?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="flex items-center gap-2 text-lg font-semibold tracking-tight text-fg">{title}{badge}</h1>
        {description && <p className="mt-0.5 max-w-3xl text-[12.5px] text-fg-3">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

// ---------------------------------------------------------------- Form controls
export function Field({ label, hint, children, htmlFor }: { label: string; hint?: ReactNode; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <label htmlFor={htmlFor} className="text-[11px] font-medium uppercase tracking-wide text-fg-3">{label}</label>
      {children}
      {hint && <p className="text-[11px] text-fg-3">{hint}</p>}
    </div>
  )
}

const inputCls = 'h-8 w-full min-w-0 rounded-md border border-line bg-panel-2 px-2.5 text-[13px] text-fg placeholder:text-fg-3 hover:border-line-strong focus:border-info focus:outline-none disabled:opacity-50'

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input({ className, ...p }, ref) {
  return <input ref={ref} className={cx(inputCls, className)} {...p} />
})

export function Select({ className, children, ...p }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cx(inputCls, 'pr-7', className)} {...p}>{children}</select>
}

export function Toggle({ checked, onChange, label, disabled, description }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean; description?: string }) {
  const id = useId()
  return (
    <div className="flex items-start justify-between gap-3">
      <label htmlFor={id} className="min-w-0 cursor-pointer">
        <span className="block text-[13px] text-fg">{label}</span>
        {description && <span className="block text-[11px] text-fg-3">{description}</span>}
      </label>
      <button
        id={id} role="switch" aria-checked={checked} disabled={disabled} onClick={() => onChange(!checked)}
        className={cx('relative mt-0.5 h-[18px] w-8 shrink-0 rounded-full border transition-colors disabled:opacity-40',
          checked ? 'border-info bg-info' : 'border-line-strong bg-panel-2')}
      >
        <span aria-hidden className={cx('absolute left-0 top-[2px] size-3 rounded-full bg-white transition-transform', checked ? 'translate-x-[15px]' : 'translate-x-[2px]')} />
      </button>
    </div>
  )
}

export function Slider({ value, onChange, min, max, step, label, format = (v) => v.toFixed(2), disabled }: {
  value: number; onChange: (v: number) => void; min: number; max: number; step: number; label: string; format?: (v: number) => string; disabled?: boolean
}) {
  const id = useId()
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <label htmlFor={id} className="text-[11px] font-medium uppercase tracking-wide text-fg-3">{label}</label>
        <span className="tabular font-mono text-[12px] text-fg">{format(value)}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} className="h-1.5 w-full cursor-pointer" />
      <div className="flex justify-between font-mono text-[10px] text-fg-3"><span>{format(min)}</span><span>{format(max)}</span></div>
    </div>
  )
}

export function Segmented<T extends string>({ value, onChange, options, label, size = 'md' }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; icon?: LucideIcon; title?: string }[]; label: string; size?: 'sm' | 'md'
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-md border border-line bg-panel-2 p-0.5">
      {options.map((o) => (
        <button key={o.value} role="radio" aria-checked={value === o.value} title={o.title}
          onClick={() => onChange(o.value)}
          className={cx('inline-flex items-center gap-1 rounded-[4px] px-2 font-medium transition-colors',
            size === 'sm' ? 'h-6 text-[11px]' : 'h-7 text-xs',
            value === o.value ? 'bg-elev text-fg shadow-sm' : 'text-fg-3 hover:text-fg')}>
          {o.icon && <o.icon className="size-3.5" aria-hidden />}{o.label}
        </button>
      ))}
    </div>
  )
}

export function Tabs<T extends string>({ value, onChange, tabs }: { value: T; onChange: (v: T) => void; tabs: { value: T; label: ReactNode; count?: number | null }[] }) {
  return (
    <div role="tablist" className="flex gap-0.5 overflow-x-auto border-b border-line">
      {tabs.map((t) => (
        <button key={t.value} role="tab" aria-selected={value === t.value} onClick={() => onChange(t.value)}
          className={cx('-mb-px flex shrink-0 items-center gap-1.5 border-b-2 px-3 py-2 text-[13px] font-medium transition-colors',
            value === t.value ? 'border-info text-fg' : 'border-transparent text-fg-3 hover:text-fg-2')}>
          {t.label}
          {t.count !== undefined && t.count !== null && <span className="tabular rounded-sm bg-panel-2 px-1 text-[11px] text-fg-3">{t.count}</span>}
        </button>
      ))}
    </div>
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded-sm border border-line bg-panel-2 px-1 font-mono text-[10px] text-fg-3">{children}</kbd>
}

export function KeyValue({ items, cols = 2 }: { items: [ReactNode, ReactNode][]; cols?: 1 | 2 | 3 }) {
  return (
    <dl className={cx('grid gap-x-4 gap-y-2', cols === 1 ? 'grid-cols-1' : cols === 2 ? 'grid-cols-2' : 'grid-cols-3')}>
      {items.map(([k, v], i) => (
        <div key={i} className="min-w-0">
          <dt className="text-[11px] uppercase tracking-wide text-fg-3">{k}</dt>
          <dd className="truncate text-[13px] text-fg">{v}</dd>
        </div>
      ))}
    </dl>
  )
}

export function Unavailable({ children }: { children: ReactNode }) {
  return <span className="inline-flex items-center gap-1 rounded-sm border border-dashed border-line-strong px-1.5 py-px text-[11px] text-fg-3">{children}</span>
}
