'use client'

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Api } from '../api/endpoints'
import type { Alert } from '../api/types'
import { invalidate } from '../hooks/useQuery'

// ---------------------------------------------------------------- toasts
export type Toast = { id: number; tone: 'ok' | 'warn' | 'crit' | 'info'; title: string; body?: string }

// ---------------------------------------------------------------- notifications (from real alerts, polled)
export type Notice = { id: string; tone: Toast['tone']; title: string; body: string; ts: string; href?: string }

type Theme = 'dark' | 'light' | 'system'
type Connection = 'connecting' | 'open' | 'closed'

interface AppCtx {
  theme: Theme
  setTheme: (t: Theme) => void
  sidebarCollapsed: boolean
  setSidebarCollapsed: (v: boolean) => void
  mobileNavOpen: boolean
  setMobileNavOpen: (v: boolean) => void
  paletteOpen: boolean
  setPaletteOpen: (v: boolean) => void
  toasts: Toast[]
  toast: (t: Omit<Toast, 'id'>) => void
  dismissToast: (id: number) => void
  notices: Notice[]
  unread: number
  markAllRead: () => void
  connection: Connection
}

const Ctx = createContext<AppCtx | null>(null)

function readPref<T>(key: string, fallback: T): T {
  try {
    const v = window.localStorage.getItem(key)
    return v === null ? fallback : (JSON.parse(v) as T)
  } catch {
    return fallback
  }
}
function writePref(key: string, v: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(v))
  } catch {
    /* ignore */
  }
}

function applyTheme(t: Theme) {
  const dark = t === 'dark' || (t === 'system' && !window.matchMedia('(prefers-color-scheme: light)').matches)
  document.documentElement.dataset.theme = dark ? 'dark' : 'light'
}

const noticeFromAlert = (a: Alert): Notice => ({
  id: `a${a.id}`, ts: a.created_at, href: `/alerts?id=${a.id}`,
  tone: a.severity === 'crit' ? 'crit' : 'warn',
  title: a.kind === 'no_uniform' ? 'No uniform detected' : 'Low-confidence uniform result',
  body: `${a.filename} · ${a.message}`,
})

export function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>('dark')
  const [sidebarCollapsed, setCollapsed] = useState(false)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [notices, setNotices] = useState<Notice[]>([])
  const [readAt, setReadAt] = useState<string>('')
  const [connection, setConnection] = useState<Connection>('connecting')
  const toastId = useRef(0)
  const lastAlertId = useRef<number | null>(null)

  useEffect(() => {
    setThemeState(readPref<Theme>('su.theme', 'dark'))
    setCollapsed(readPref('su.sidebarCollapsed', false))
    setReadAt(readPref('su.noticesReadAt', ''))
  }, [])

  useEffect(() => {
    applyTheme(theme)
    if (theme !== 'system') return
    const mq = window.matchMedia('(prefers-color-scheme: light)')
    const h = () => applyTheme('system')
    mq.addEventListener('change', h)
    return () => mq.removeEventListener('change', h)
  }, [theme])

  const toast = useCallback((t: Omit<Toast, 'id'>) => {
    const id = ++toastId.current
    setToasts((ts) => [...ts.slice(-3), { ...t, id }])
    setTimeout(() => setToasts((ts) => ts.filter((x) => x.id !== id)), t.tone === 'crit' ? 8000 : 4500)
  }, [])

  // Backend connection + notifications: poll recent alerts (they are only created by real analyses).
  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const r = await Api.alerts({ limit: 20 })
        if (stop) return
        setConnection('open')
        setNotices(r.alerts.map(noticeFromAlert))
        const newest = r.alerts[0]?.id ?? 0
        if (lastAlertId.current !== null && newest > lastAlertId.current) {
          invalidate('alerts')
          invalidate('overview')
        }
        lastAlertId.current = newest
      } catch {
        if (!stop) setConnection('closed')
      }
    }
    void tick()
    const id = setInterval(() => document.visibilityState === 'visible' && void tick(), 10000)
    return () => {
      stop = true
      clearInterval(id)
    }
  }, [])

  const value = useMemo<AppCtx>(() => ({
    theme,
    setTheme: (t) => {
      setThemeState(t)
      writePref('su.theme', t)
    },
    sidebarCollapsed,
    setSidebarCollapsed: (v) => {
      setCollapsed(v)
      writePref('su.sidebarCollapsed', v)
    },
    mobileNavOpen, setMobileNavOpen, paletteOpen, setPaletteOpen,
    toasts, toast, dismissToast: (id) => setToasts((ts) => ts.filter((t) => t.id !== id)),
    notices,
    unread: notices.filter((n) => n.ts > readAt).length,
    markAllRead: () => {
      const now = new Date().toISOString()
      setReadAt(now)
      writePref('su.noticesReadAt', now)
    },
    connection,
  }), [theme, sidebarCollapsed, mobileNavOpen, paletteOpen, toasts, toast, notices, readAt, connection])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp(): AppCtx {
  const c = useContext(Ctx)
  if (!c) throw new Error('useApp must be used inside AppProvider')
  return c
}
