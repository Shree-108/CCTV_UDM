// Centralized API client. All data shown in the UI comes through here.

export const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8010').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message)
  }
  get isNetwork() {
    return this.status === 0
  }
}

function messageFrom(detail: unknown, status: number): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (typeof d === 'object' && d && 'msg' in d ? `${(d as { loc?: unknown[] }).loc?.slice(-1)[0] ?? ''} ${(d as { msg: string }).msg}`.trim() : String(d)))
      .join('; ')
  }
  return `Request failed (${status})`
}

export interface RequestOptions {
  method?: string
  body?: unknown
  form?: FormData
  signal?: AbortSignal
}

export async function api<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = {}
  let body: BodyInit | undefined
  if (opts.form) body = opts.form
  else if (opts.body !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(opts.body)
  }
  let res: Response
  try {
    // credentials: the admin session cookie (HttpOnly; never readable here) for the S3 upload routes
    res = await fetch(`${API_BASE}${path}`, { method: opts.method ?? (body ? 'POST' : 'GET'), headers, body, signal: opts.signal,
      credentials: 'include' })
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e
    throw new ApiError(0, `Backend not reachable at ${API_BASE}. Is the API server running?`)
  }
  if (!res.ok) {
    let detail: unknown
    try {
      detail = (await res.json()).detail
    } catch {
      detail = undefined
    }
    throw new ApiError(res.status, messageFrom(detail, res.status), detail)
  }
  const ct = res.headers.get('content-type') ?? ''
  return (ct.includes('application/json') ? res.json() : res.text()) as Promise<T>
}

/** Multipart POST with upload progress (fetch cannot report it). Same errors as api(). */
export function apiUpload<T>(path: string, form: FormData, onProgress: (sent: number, total: number) => void,
  signal?: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}${path}`)
    xhr.withCredentials = true
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded, e.total)
    xhr.onerror = () => reject(new ApiError(0, `Backend not reachable at ${API_BASE}. Is the API server running?`))
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'))
    xhr.onload = () => {
      let data: unknown
      try {
        data = JSON.parse(xhr.responseText)
      } catch {
        data = undefined
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data as T)
      else {
        const detail = (data as { detail?: unknown } | undefined)?.detail
        reject(new ApiError(xhr.status, messageFrom(detail, xhr.status), detail))
      }
    }
    signal?.addEventListener('abort', () => xhr.abort())
    xhr.send(form)
  })
}

export const url = (path: string) => `${API_BASE}${path}`

/** Same URL with ?download=true so the backend sends it as an attachment with a friendly name. */
export const downloadUrl = (path: string) => `${API_BASE}${path}${path.includes('?') ? '&' : '?'}download=true`

export function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') p.set(k, String(v))
  const s = p.toString()
  return s ? `?${s}` : ''
}
