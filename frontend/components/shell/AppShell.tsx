'use client'

import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  ArrowRight, Bell, ChevronsLeft, ChevronsRight, CornerDownLeft, Menu, Search, Settings as SettingsIcon, X,
} from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Alert, Analysis } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { ago, cx } from '../../lib/format'
import { BRAND, MODULE_NAME } from '../../lib/module'
import { useApp } from '../../stores/app'
import { Dot, IconButton, Kbd } from '../ui/primitives'
import { Drawer, Toaster } from '../ui/overlay'
import { EmptyState } from '../ui/states'
import { STATUS_META } from '../ui/status'
import { ALL_NAV, NAV, navFor, type BadgeKey } from './nav'

// ---------------------------------------------------------------- badge data (real API values)
function useBadges(): Partial<Record<BadgeKey, number>> {
  const alerts = useQuery('alerts:newcount', () => Api.alerts({ state: 'new', limit: 1 }), { poll: 15000 })
  const ov = useQuery('overview', Api.overview, { poll: 10000 })
  return {
    newAlerts: alerts.data?.new || undefined,
    running: ov.data?.running.length || undefined,
  }
}

/** Yatrikart wordmark + the active AI module (lib/module.ts). Logo is white; inverted in light theme. */
export function Brand({ collapsed }: { collapsed?: boolean }) {
  return (
    <div className={cx('flex h-16 shrink-0 items-center', collapsed ? 'justify-center px-0' : 'px-5')}>
      {collapsed ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src="/brand/yatrikart-mark.png" alt={`${BRAND} · ${MODULE_NAME}`} title={`${BRAND} · ${MODULE_NAME}`}
          className="brand-logo h-6 w-auto select-none" draggable={false} />
      ) : (
        <div className="flex min-w-0 flex-col gap-1.5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/brand/yatrikart-wordmark.png" alt={BRAND} className="brand-logo h-[19px] w-auto self-start select-none" draggable={false} />
          <div className="truncate text-[9.5px] font-semibold uppercase tracking-[0.24em] text-fg-3">{MODULE_NAME}</div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------- sidebar
function Sidebar({ collapsed, onNavigate }: { collapsed: boolean; onNavigate?: () => void }) {
  const pathname = usePathname()
  const badges = useBadges()
  const active = navFor(pathname)
  const { connection } = useApp()
  return (
    <nav aria-label="Primary" className="flex h-full flex-col">
      <Brand collapsed={collapsed} />
      {!collapsed && (
        <div className={cx('mx-4 mb-2 flex items-center gap-2 rounded-lg px-3 py-2 text-[12px]',
          connection === 'open' ? 'bg-ok-bg text-ok' : connection === 'closed' ? 'bg-crit-bg text-crit' : 'bg-warn-bg text-warn')}>
          <span className={cx('size-1.5 rounded-full', connection === 'open' ? 'live-dot bg-ok' : connection === 'closed' ? 'bg-crit' : 'bg-warn')} aria-hidden />
          {connection === 'open' ? 'System ready' : connection === 'closed' ? 'Backend offline' : 'Connecting to backend…'}
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-y-auto py-2">
        {NAV.filter((g) => !g.hidden).map((g) => (
          <div key={g.label} className="mb-2">
            {!collapsed && <div className="px-5 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-fg-3">{g.label}</div>}
            {collapsed && <div className="mx-3 my-2 border-t border-line" aria-hidden />}
            <ul>
              {g.items.map((it) => {
                const on = active?.href === it.href
                const b = it.badge ? badges[it.badge] : undefined
                const inner = (
                  <>
                    <it.icon className={cx('size-4 shrink-0', on ? 'text-info' : 'text-fg-3 group-hover:text-fg-2')} aria-hidden />
                    {!collapsed && <span className="flex-1 truncate">{it.label}</span>}
                    {!collapsed && it.comingSoon && <span className="rounded-sm border border-line px-1 text-[9px] font-semibold uppercase tracking-wide text-fg-3">Soon</span>}
                    {b !== undefined && (collapsed
                      ? <span className="absolute right-1 top-1 size-1.5 rounded-full bg-info" aria-label={`${b}`} />
                      : <span className={cx('tabular rounded-sm px-1.5 text-[11px] font-medium', it.badge === 'newAlerts' ? 'bg-crit-bg text-crit' : 'bg-info-bg text-info')}>{b}</span>)}
                  </>
                )
                const cls = cx('group relative flex items-center gap-3 rounded-lg transition-colors h-10 text-[14px]', collapsed ? 'justify-center' : 'px-3')
                return (
                  <li key={it.href} className="px-2">
                    {it.comingSoon ? (
                      <span aria-disabled="true" title={`${it.label} — coming soon. Live CCTV is not available in this version.`}
                        className={cx(cls, 'cursor-not-allowed text-fg-3 opacity-60')}>{inner}</span>
                    ) : (
                      <Link href={it.href} onClick={onNavigate} title={collapsed ? it.label : undefined} aria-current={on ? 'page' : undefined}
                        className={cx(cls, on ? 'bg-info-bg font-medium text-info shadow-[inset_2px_0_0_var(--info)]' : 'text-fg-2 hover:bg-hover hover:text-fg')}>
                        {inner}
                      </Link>
                    )}
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </div>
      {!collapsed && <p className="px-5 pb-12 text-[10px] leading-4 text-fg-3">Clothing analysis only — no face recognition or identity detection.</p>}
    </nav>
  )
}

// ---------------------------------------------------------------- command palette
type Cmd = { id: string; label: string; hint?: string; group: string; run: () => void }

function CommandPalette() {
  const { paletteOpen, setPaletteOpen } = useApp()
  const router = useRouter()
  const [q, setQ] = useState('')
  const [idx, setIdx] = useState(0)
  const [results, setResults] = useState<{ analyses: Analysis[]; alerts: Alert[] } | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!paletteOpen) return
    setQ('')
    setIdx(0)
    setResults(null)
    setTimeout(() => inputRef.current?.focus(), 10)
  }, [paletteOpen])

  useEffect(() => {
    if (!paletteOpen || q.trim().length < 2) { setResults(null); return }
    const t = setTimeout(() => { Api.search(q.trim()).then(setResults).catch(() => setResults(null)) }, 180)
    return () => clearTimeout(t)
  }, [q, paletteOpen])

  const go = (href: string) => { setPaletteOpen(false); router.push(href) }

  const cmds: Cmd[] = useMemo(() => {
    const ql = q.toLowerCase()
    const actions: Cmd[] = [
      { id: 'a-up', label: 'Upload CCTV media', group: 'Actions', run: () => go('/upload') },
      { id: 'a-img', label: 'Analyse an image', group: 'Actions', run: () => go('/analysis/image') },
      { id: 'a-vid', label: 'Analyse a video', group: 'Actions', run: () => go('/analysis/video') },
      { id: 'a-alerts', label: 'View new alerts', group: 'Actions', run: () => go('/alerts?state=new') },
      { id: 'a-hist', label: 'Open analysis history', group: 'Actions', run: () => go('/history') },
    ]
    const nav: Cmd[] = ALL_NAV.map((n) => ({ id: `n-${n.href}`, label: `Go to ${n.label}`, hint: n.description, group: 'Navigation', run: () => go(n.href) }))
    const found: Cmd[] = []
    results?.analyses.forEach((a) => found.push({ id: `an-${a.id}`, label: a.filename, hint: `${a.kind} · ${a.status ? STATUS_META[a.status].short : a.state} · ${ago(a.created_at)}`, group: 'Analyses', run: () => go(`/history?id=${a.id}`) }))
    results?.alerts.forEach((a) => found.push({ id: `al-${a.id}`, label: `Alert #${a.id} · ${a.title}`, hint: `${a.filename} · ${ago(a.created_at)}`, group: 'Alerts', run: () => go(`/alerts?id=${a.id}`) }))
    const match = (c: Cmd) => !ql || c.label.toLowerCase().includes(ql) || (c.hint ?? '').toLowerCase().includes(ql)
    return [...found, ...actions.filter(match), ...nav.filter(match)]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, results])

  useEffect(() => setIdx(0), [cmds.length])
  if (!paletteOpen) return null
  let lastGroup = ''
  return (
    <div className="fixed inset-0 z-[55] flex items-start justify-center p-4 pt-[12vh]">
      <div className="absolute inset-0 bg-black/55" onClick={() => setPaletteOpen(false)} aria-hidden />
      <div role="dialog" aria-modal="true" aria-label="Command palette" className="relative w-[620px] max-w-full overflow-hidden rounded-lg border border-line bg-panel shadow-[var(--shadow)] animate-in fade-in zoom-in-95 duration-100">
        <div className="flex items-center gap-2 border-b border-line px-3">
          <Search className="size-4 text-fg-3" aria-hidden />
          <input ref={inputRef} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search analyses and alerts by file name… or type a command"
            aria-label="Search" role="combobox" aria-expanded aria-controls="palette-list" aria-activedescendant={cmds[idx]?.id}
            className="h-11 flex-1 bg-transparent text-[14px] text-fg placeholder:text-fg-3 focus:outline-none"
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') { e.preventDefault(); setIdx((i) => Math.min(i + 1, cmds.length - 1)) }
              if (e.key === 'ArrowUp') { e.preventDefault(); setIdx((i) => Math.max(i - 1, 0)) }
              if (e.key === 'Enter') { e.preventDefault(); cmds[idx]?.run() }
              if (e.key === 'Escape') setPaletteOpen(false)
            }} />
          <Kbd>Esc</Kbd>
        </div>
        <ul id="palette-list" role="listbox" className="max-h-[52vh] overflow-y-auto p-1.5">
          {cmds.length === 0 && <li className="px-3 py-6 text-center text-[12px] text-fg-3">No results for “{q}”</li>}
          {cmds.map((c, i) => {
            const header = c.group !== lastGroup ? (lastGroup = c.group) : null
            return (
              <li key={c.id} role="none">
                {header && <div className="px-2 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-fg-3">{header}</div>}
                <button id={c.id} role="option" aria-selected={i === idx} onMouseEnter={() => setIdx(i)} onClick={c.run}
                  className={cx('flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left', i === idx ? 'bg-hover' : '')}>
                  <ArrowRight className="size-3.5 shrink-0 text-fg-3" aria-hidden />
                  <span className="truncate text-[13px] text-fg">{c.label}</span>
                  {c.hint && <span className="ml-auto truncate pl-3 text-[11px] text-fg-3">{c.hint}</span>}
                  {i === idx && <CornerDownLeft className="size-3 shrink-0 text-fg-3" aria-hidden />}
                </button>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- notification center
function NotificationCenter({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { notices, markAllRead, connection } = useApp()
  const router = useRouter()
  useEffect(() => { if (open) markAllRead() }, [open]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Drawer open={open} onClose={onClose} title="Notifications" width="w-[400px]"
      subtitle={connection === 'open' ? 'Uniform alerts from your analyses' : 'Backend not reachable'}>
      {notices.length === 0 ? (
        <EmptyState icon={Bell} title="No notifications yet" body="Confirmed NO UNIFORM results and persistent low-confidence results from your analyses appear here." />
      ) : (
        <ul className="flex flex-col gap-1.5">
          {notices.map((n) => (
            <li key={n.id}>
              <button onClick={() => { if (n.href) { onClose(); router.push(n.href) } }}
                className="flex w-full gap-2.5 rounded-md border border-line bg-panel-2 p-2.5 text-left hover:border-line-strong">
                <Dot tone={n.tone} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[13px] font-medium text-fg">{n.title}</span>
                    <span className="shrink-0 text-[11px] text-fg-3">{ago(n.ts)}</span>
                  </div>
                  <p className="truncate text-[12px] text-fg-3">{n.body}</p>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Drawer>
  )
}

// ---------------------------------------------------------------- top bar
function Topbar() {
  const pathname = usePathname()
  const item = navFor(pathname)
  const { setPaletteOpen, setMobileNavOpen, unread } = useApp()
  const health = useQuery('health', Api.health, { poll: 20000 })
  const alerts = useQuery('alerts:newcount', () => Api.alerts({ state: 'new', limit: 1 }), { poll: 15000 })
  const [notifOpen, setNotifOpen] = useState(false)
  const h = health.data
  const sysTone = health.error ? 'crit' : !h ? 'neutral' : h.status !== 'ok' ? 'crit' : h.uniform_classifier.state === 'ready' ? 'ok' : 'warn'
  const sysText = health.error ? 'Backend offline' : !h ? 'Checking…' : h.status !== 'ok' ? 'Model error'
    : h.uniform_classifier.state === 'ready' ? `Uniform model v${h.uniform_classifier.model?.version ?? '?'}`
      : h.uniform_classifier.state === 'mismatch' ? 'Wrong uniform model file' : 'Uniform model missing'

  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-3 border-b border-line bg-bg/90 px-3 backdrop-blur md:px-5">
      <IconButton icon={Menu} label="Open navigation" className="lg:hidden" onClick={() => setMobileNavOpen(true)} />
      <div className="min-w-0">
        <div className="truncate text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-3">{ALL_NAV.find((n) => n.href === item?.href)?.group ?? BRAND}</div>
        <div className="truncate text-[13px] font-semibold text-fg">{item?.label ?? MODULE_NAME}</div>
      </div>
      <button onClick={() => setPaletteOpen(true)} aria-label="Open search (Ctrl+K)"
        className="mx-auto hidden h-8 w-full max-w-md items-center gap-2 rounded-md border border-line bg-panel px-2.5 text-[12px] text-fg-3 hover:border-line-strong md:flex">
        <Search className="size-3.5" aria-hidden /> Search analyses, alerts…
        <span className="ml-auto flex gap-0.5"><Kbd>Ctrl</Kbd><Kbd>K</Kbd></span>
      </button>
      <div className="ml-auto flex items-center gap-1.5 md:ml-0">
        <IconButton icon={Search} label="Search" className="md:hidden" onClick={() => setPaletteOpen(true)} />
        <Link href="/settings#model-status" className="hidden items-center gap-1.5 rounded-md px-2 py-1 text-[12px] text-fg-2 hover:bg-hover lg:flex" title={h?.model_error ?? h?.uniform_classifier.label}>
          <Dot tone={sysTone} pulse={sysTone === 'ok'} /> {sysText}
        </Link>
        <Link href="/alerts?state=new" className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[12px] text-fg-2 hover:bg-hover" aria-label={`${alerts.data?.new ?? 0} new alerts`}>
          <span className="hidden sm:inline">Alerts</span>
          <span className={cx('tabular rounded-sm px-1.5 font-mono text-[12px]', (alerts.data?.new ?? 0) > 0 ? 'bg-crit-bg text-crit' : 'bg-panel-2 text-fg-3')}>
            {alerts.data ? String(alerts.data.new).padStart(2, '0') : '—'}
          </span>
        </Link>
        <div className="relative">
          <IconButton icon={Bell} label={`Notifications (${unread} unread)`} onClick={() => setNotifOpen(true)} />
          {unread > 0 && <span className="pointer-events-none absolute right-1 top-1 flex size-3.5 items-center justify-center rounded-full bg-crit text-[9px] font-bold text-white">{unread > 9 ? '9+' : unread}</span>}
        </div>
        <Link href="/settings" aria-label="Settings" title="Settings" className="inline-flex size-8 items-center justify-center rounded-md text-fg-2 hover:bg-hover hover:text-fg">
          <SettingsIcon className="size-4" aria-hidden />
        </Link>
      </div>
      <NotificationCenter open={notifOpen} onClose={() => setNotifOpen(false)} />
    </header>
  )
}

// ---------------------------------------------------------------- shell
export function AppShell({ children }: { children: ReactNode }) {
  const { sidebarCollapsed, setSidebarCollapsed, mobileNavOpen, setMobileNavOpen, setPaletteOpen, paletteOpen } = useApp()
  const pathname = usePathname()
  useEffect(() => setMobileNavOpen(false), [pathname]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen(!paletteOpen)
      }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [paletteOpen, setPaletteOpen])

  return (
    <div className="flex h-dvh overflow-hidden bg-bg">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-[70] focus:rounded focus:bg-info focus:px-2 focus:py-1 focus:text-on-accent">Skip to content</a>
      <aside className={cx('relative hidden shrink-0 border-r border-line bg-sidebar transition-[width] duration-150 lg:block', sidebarCollapsed ? 'w-[60px]' : 'w-[236px]')}>
        <Sidebar collapsed={sidebarCollapsed} />
        <button onClick={() => setSidebarCollapsed(!sidebarCollapsed)} aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="absolute bottom-3 right-2 inline-flex size-7 items-center justify-center rounded-md text-fg-3 hover:bg-hover hover:text-fg">
          {sidebarCollapsed ? <ChevronsRight className="size-4" /> : <ChevronsLeft className="size-4" />}
        </button>
      </aside>
      {mobileNavOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/55" onClick={() => setMobileNavOpen(false)} aria-hidden />
          <aside className="relative h-full w-[260px] border-r border-line bg-sidebar animate-in slide-in-from-left-8 duration-150">
            <IconButton icon={X} label="Close navigation" className="absolute right-2 top-2 z-10" onClick={() => setMobileNavOpen(false)} />
            <Sidebar collapsed={false} onNavigate={() => setMobileNavOpen(false)} />
          </aside>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <main id="main" className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[1800px] p-3 md:p-5">{children}</div>
        </main>
      </div>
      <CommandPalette />
      <Toaster />
    </div>
  )
}
