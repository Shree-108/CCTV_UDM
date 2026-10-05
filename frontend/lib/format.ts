// Formatting helpers. Backend timestamps are ISO-8601 UTC.

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`

export const num = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })

export const dash = (v: unknown) => (v === null || v === undefined || v === '' ? '—' : String(v))

export function parseTs(ts: string | number): Date {
  if (typeof ts === 'number') return new Date(ts * 1000)
  return new Date(/[zZ]|[+-]\d\d:\d\d$/.test(ts) ? ts : `${ts}Z`)
}

export function time(ts: string | number | null | undefined): string {
  if (!ts) return '—'
  return parseTs(ts).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
}

export function dateTime(ts: string | number | null | undefined): string {
  if (!ts) return '—'
  const d = parseTs(ts)
  return `${d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: '2-digit' })} ${time(ts)}`
}

export function ago(ts: string | number | null | undefined, now = Date.now()): string {
  if (!ts) return '—'
  const s = Math.max(0, Math.round((now - parseTs(ts).getTime()) / 1000))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return `${Math.floor(s / 86400)} d ago`
}

export function duration(sec: number | null | undefined): string {
  if (sec === null || sec === undefined) return '—'
  if (sec < 60) return `${Math.round(sec)}s`
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`
}

export const className = (c: string) => c.replace(/_/g, ' ')

/** Local-day ISO boundaries converted to UTC ISO for API filters. */
export function dayStartIso(d: string): string {
  return new Date(`${d}T00:00:00`).toISOString()
}
export function dayEndIso(d: string): string {
  return new Date(`${d}T23:59:59.999`).toISOString()
}
export function todayLocal(offsetDays = 0): string {
  const d = new Date()
  d.setDate(d.getDate() + offsetDays)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

export function downloadBlob(blob: Blob, name: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(' ')
