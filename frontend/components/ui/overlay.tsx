'use client'

import { useEffect, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react'
import { cx } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Button, IconButton } from './primitives'

function useEscape(open: boolean, onClose: () => void) {
  useEffect(() => {
    if (!open) return
    const h = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [open, onClose])
}

/** Moves focus into the dialog on open and restores it on close. */
function useFocusTrap(open: boolean) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const prev = document.activeElement as HTMLElement | null
    const el = ref.current
    el?.querySelector<HTMLElement>('[data-autofocus], button, input, select, textarea, [tabindex="0"]')?.focus()
    const trap = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !el) return
      const f = el.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), input, select, textarea, [tabindex="0"]')
      if (!f.length) return
      const first = f[0], last = f[f.length - 1]
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
    }
    el?.addEventListener('keydown', trap)
    return () => {
      el?.removeEventListener('keydown', trap)
      prev?.focus?.()
    }
  }, [open])
  return ref
}

export function Drawer({ open, onClose, title, subtitle, children, footer, width = 'w-[520px]' }: {
  open: boolean; onClose: () => void; title: ReactNode; subtitle?: ReactNode; children: ReactNode; footer?: ReactNode; width?: string
}) {
  useEscape(open, onClose)
  const ref = useFocusTrap(open)
  if (!open || typeof document === 'undefined') return null
  return createPortal(
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} aria-hidden />
      <div ref={ref} role="dialog" aria-modal="true" aria-label={typeof title === 'string' ? title : undefined}
        className={cx('relative flex h-full max-w-full flex-col border-l border-line bg-panel shadow-[var(--shadow)] animate-in slide-in-from-right-8 duration-150', width)}>
        <header className="flex items-start justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            <h2 className="truncate text-[15px] font-semibold text-fg">{title}</h2>
            {subtitle && <div className="mt-0.5 text-[12px] text-fg-3">{subtitle}</div>}
          </div>
          <IconButton icon={X} label="Close" onClick={onClose} />
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">{children}</div>
        {footer && <footer className="flex flex-wrap items-center justify-end gap-2 border-t border-line px-4 py-3">{footer}</footer>}
      </div>
    </div>,
    document.body,
  )
}

export function Modal({ open, onClose, title, children, footer, width = 'w-[480px]' }: {
  open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; footer?: ReactNode; width?: string
}) {
  useEscape(open, onClose)
  const ref = useFocusTrap(open)
  if (!open || typeof document === 'undefined') return null
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/55" onClick={onClose} aria-hidden />
      <div ref={ref} role="dialog" aria-modal="true" aria-label={typeof title === 'string' ? title : undefined}
        className={cx('relative flex max-h-[90vh] max-w-full flex-col rounded-lg border border-line bg-panel shadow-[var(--shadow)] animate-in fade-in zoom-in-95 duration-100', width)}>
        <header className="flex items-center justify-between border-b border-line px-4 py-3">
          <h2 className="text-[14px] font-semibold text-fg">{title}</h2>
          <IconButton icon={X} label="Close" onClick={onClose} />
        </header>
        <div className="min-h-0 overflow-y-auto p-4">{children}</div>
        {footer && <footer className="flex justify-end gap-2 border-t border-line px-4 py-3">{footer}</footer>}
      </div>
    </div>,
    document.body,
  )
}

export function ConfirmDialog({ open, onClose, onConfirm, title, body, confirmLabel = 'Confirm', danger, pending }: {
  open: boolean; onClose: () => void; onConfirm: () => void; title: string; body: ReactNode; confirmLabel?: string; danger?: boolean; pending?: boolean
}) {
  return (
    <Modal open={open} onClose={onClose} title={title}
      footer={<>
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant={danger ? 'danger' : 'primary'} onClick={onConfirm} loading={pending} data-autofocus>{confirmLabel}</Button>
      </>}>
      <div className="text-[13px] text-fg-2">{body}</div>
    </Modal>
  )
}

const TOAST_ICON = { ok: CheckCircle2, warn: AlertTriangle, crit: XCircle, info: Info }
const TOAST_TONE = { ok: 'text-ok', warn: 'text-warn', crit: 'text-crit', info: 'text-info' }

export function Toaster() {
  const { toasts, dismissToast } = useApp()
  return (
    <div aria-live="polite" className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[340px] max-w-[calc(100vw-2rem)] flex-col gap-2">
      {toasts.map((t) => {
        const Icon = TOAST_ICON[t.tone]
        return (
          <div key={t.id} role="status" className="enter-up pointer-events-auto flex gap-2.5 rounded-md border border-line bg-elev p-3 shadow-[var(--shadow)]">
            <Icon className={cx('mt-px size-4 shrink-0', TOAST_TONE[t.tone])} aria-hidden />
            <div className="min-w-0 flex-1">
              <p className="text-[13px] font-medium text-fg">{t.title}</p>
              {t.body && <p className="mt-0.5 break-words text-[12px] text-fg-3">{t.body}</p>}
            </div>
            <button onClick={() => dismissToast(t.id)} aria-label="Dismiss" className="text-fg-3 hover:text-fg"><X className="size-3.5" /></button>
          </div>
        )
      })}
    </div>
  )
}
