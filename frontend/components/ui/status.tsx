'use client'

import { CheckCircle2, CircleHelp, ShieldAlert, UserX, type LucideIcon } from 'lucide-react'
import type { ComplianceStatus, PersonLabel } from '../../api/types'
import { cx } from '../../lib/format'
import { Badge, type Tone } from './primitives'

export const STATUS_META: Record<ComplianceStatus, { title: string; short: string; tone: Tone; icon: LucideIcon; color: string }> = {
  compliant: { title: 'COMPLIANT', short: 'Compliant', tone: 'ok', icon: CheckCircle2, color: 'var(--ok)' },
  no_uniform: { title: 'NON-COMPLIANT', short: 'Non-compliant', tone: 'crit', icon: ShieldAlert, color: 'var(--crit)' },
  uncertain: { title: 'UNCERTAIN — PLEASE VERIFY', short: 'Uncertain', tone: 'warn', icon: CircleHelp, color: 'var(--warn)' },
  no_person: { title: 'NO PERSON DETECTED', short: 'No person', tone: 'neutral', icon: UserX, color: 'var(--fg-3)' },
}

export const LABEL_META: Record<PersonLabel, { text: string; tone: Tone; color: string }> = {
  uniform: { text: 'COMPLIANT', tone: 'ok', color: 'var(--ok)' },
  non_uniform: { text: 'NON-COMPLIANT', tone: 'crit', color: 'var(--crit)' },
  uncertain: { text: 'UNCERTAIN — VERIFY', tone: 'warn', color: 'var(--warn)' },
}

export function StatusBadgeC({ status, className }: { status: ComplianceStatus | null | undefined; className?: string }) {
  if (!status) return <Badge className={className}>—</Badge>
  const m = STATUS_META[status]
  return <Badge tone={m.tone} dot className={className}>{m.short}</Badge>
}

export function LabelBadge({ label }: { label: PersonLabel }) {
  const m = LABEL_META[label]
  return <Badge tone={m.tone} dot>{m.text}</Badge>
}

/** Large status block: colour + icon + text (never colour alone). */
export function StatusHero({ status, message, uniformed, people, compact }: {
  status: ComplianceStatus; message?: string; uniformed?: number | null; people?: number | null; compact?: boolean
}) {
  const m = STATUS_META[status]
  const bg = { ok: 'bg-ok-bg border-ok/35', crit: 'bg-crit-bg border-crit/35', warn: 'bg-warn-bg border-warn/35', neutral: 'bg-panel-2 border-line', info: '', exp: '' }[m.tone]
  const fg = { ok: 'text-ok', crit: 'text-crit', warn: 'text-warn', neutral: 'text-fg-2', info: '', exp: '' }[m.tone]
  return (
    <div role="status" className={cx('flex items-center gap-3 rounded-lg border', compact ? 'p-3' : 'p-4', bg)}>
      <m.icon className={cx(compact ? 'size-6' : 'size-8', 'shrink-0', fg)} aria-hidden />
      <div className="min-w-0 flex-1">
        <div className={cx('font-semibold tracking-[0.04em]', compact ? 'text-[15px]' : 'text-[19px]', fg)}>{m.title}</div>
        {message && <p className="mt-0.5 text-[12px] text-fg-2">{message}</p>}
      </div>
      {(uniformed != null || people != null) && (
        <div className="flex shrink-0 gap-4 text-right">
          {uniformed != null && <div><div className="tabular text-[20px] font-semibold text-fg">{uniformed}</div><div className="text-[10px] uppercase tracking-wide text-fg-3">Uniformed</div></div>}
          {people != null && <div><div className="tabular text-[20px] font-semibold text-fg">{people}</div><div className="text-[10px] uppercase tracking-wide text-fg-3">People inside</div></div>}
        </div>
      )}
    </div>
  )
}

export function statusMessage(status: ComplianceStatus, uniformed: number, people: number): string {
  if (status === 'compliant') return `${uniformed} uniformed person${uniformed === 1 ? '' : 's'} detected inside the selected store area.`
  if (status === 'no_uniform') return uniformed
    ? 'At least one person inside the selected store area is not wearing the approved uniform.'
    : 'No uniformed person detected inside the selected store area.'
  if (status === 'uncertain') return `${people} person${people === 1 ? '' : 's'} visible, but clothing could not be judged reliably.`
  return 'No person detected inside the selected store area.'
}
