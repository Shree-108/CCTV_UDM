'use client'

import type { ReactNode } from 'react'
import { AlertTriangle, Inbox, RefreshCw, WifiOff, type LucideIcon } from 'lucide-react'
import { ApiError } from '../../api/client'
import { cx } from '../../lib/format'
import { Button } from './primitives'

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx('skeleton rounded-md', className)} aria-hidden />
}

export function LoadingState({ rows = 3, className, label = 'Loading' }: { rows?: number; className?: string; label?: string }) {
  return (
    <div role="status" aria-label={label} className={cx('flex flex-col gap-2', className)}>
      {Array.from({ length: rows }, (_, i) => <Skeleton key={i} className={cx('h-7', i === rows - 1 && 'w-2/3')} />)}
    </div>
  )
}

export function EmptyState({ icon: Icon = Inbox, title, body, action, compact }: { icon?: LucideIcon; title: string; body?: ReactNode; action?: ReactNode; compact?: boolean }) {
  return (
    <div className={cx('flex flex-col items-center justify-center text-center', compact ? 'gap-1.5 py-6' : 'gap-2 py-12')}>
      <div className="flex size-9 items-center justify-center rounded-md border border-line bg-panel-2 text-fg-3"><Icon className="size-4" aria-hidden /></div>
      <p className="text-[13px] font-medium text-fg">{title}</p>
      {body && <p className="max-w-sm text-[12px] text-fg-3">{body}</p>}
      {action && <div className="mt-1.5">{action}</div>}
    </div>
  )
}

export function ErrorState({ error, onRetry, title, compact, action }: { error: unknown; onRetry?: () => void; title?: string; compact?: boolean; action?: ReactNode }) {
  const net = error instanceof ApiError && error.isNetwork
  const msg = error instanceof Error ? error.message : String(error)
  return (
    <div role="alert" className={cx('flex flex-col items-center justify-center gap-2 text-center', compact ? 'py-5' : 'py-10')}>
      <div className="flex size-9 items-center justify-center rounded-md bg-crit-bg text-crit">
        {net ? <WifiOff className="size-4" aria-hidden /> : <AlertTriangle className="size-4" aria-hidden />}
      </div>
      <p className="text-[13px] font-medium text-fg">{title ?? (net ? 'Backend not connected' : 'Something went wrong')}</p>
      <p className="max-w-md text-[12px] text-fg-3">{msg}</p>
      <div className="mt-1 flex gap-2">
        {onRetry && <Button size="sm" icon={RefreshCw} onClick={onRetry}>Retry</Button>}
        {action}
      </div>
    </div>
  )
}

/** Renders loading / error / empty / content for a query result. */
export function QueryView<T>({ q, empty, isEmpty, children, loading, errorTitle }: {
  q: { data: T | undefined; error: unknown; loading: boolean; refetch: () => void }
  empty?: ReactNode
  isEmpty?: (d: T) => boolean
  children: (d: T) => ReactNode
  loading?: ReactNode
  errorTitle?: string
}) {
  if (q.data === undefined && q.error) return <ErrorState error={q.error} onRetry={q.refetch} title={errorTitle} compact />
  if (q.data === undefined) return <>{loading ?? <LoadingState />}</>
  if (isEmpty?.(q.data)) return <>{empty ?? <EmptyState title="No data" compact />}</>
  return <>{children(q.data)}</>
}
