'use client'

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { ApiError } from '../api/client'

// Minimal query cache: dedupes identical in-flight requests, keeps last data per key,
// supports polling that pauses while the tab is hidden, and manual refetch/invalidate.

type Entry = { data?: unknown; error?: ApiError | Error; at: number; promise?: Promise<unknown> }
const cache = new Map<string, Entry>()
const listeners = new Map<string, Set<() => void>>()

function emit(key: string) {
  listeners.get(key)?.forEach((l) => l())
}

const fetchers = new Map<string, () => Promise<unknown>>()

/** Refetch every mounted query whose key starts with `prefix` (stale data stays visible meanwhile). */
export function invalidate(prefix: string) {
  for (const k of [...cache.keys()]) if (k.startsWith(prefix)) {
    const f = fetchers.get(k)
    if (f && listeners.get(k)?.size) void load(k, f)
    else cache.delete(k)
  }
}

async function load<T>(key: string, fetcher: () => Promise<T>): Promise<void> {
  const e = cache.get(key) ?? { at: 0 }
  if (e.promise) return e.promise.then(() => undefined)
  const p = fetcher()
    .then((data) => cache.set(key, { data, at: Date.now() }))
    .catch((error) => cache.set(key, { data: cache.get(key)?.data, error, at: Date.now() }))
    .finally(() => emit(key))
  cache.set(key, { ...e, promise: p })
  return p.then(() => undefined)
}

export interface QueryState<T> {
  data: T | undefined
  error: ApiError | Error | undefined
  loading: boolean
  refetch: () => Promise<void>
}

export function useQuery<T>(key: string | null, fetcher: () => Promise<T>, opts: { poll?: number } = {}): QueryState<T> {
  const fetchRef = useRef(fetcher)
  fetchRef.current = fetcher
  const subscribe = useCallback((cb: () => void) => {
    if (!key) return () => {}
    if (!listeners.has(key)) listeners.set(key, new Set())
    listeners.get(key)!.add(cb)
    return () => listeners.get(key)?.delete(cb)
  }, [key])
  const snap = useSyncExternalStore(subscribe, () => (key ? cache.get(key) : undefined), () => undefined)
  const [, force] = useState(0)

  const refetch = useCallback(async () => {
    if (!key) return
    await load(key, () => fetchRef.current())
    force((n) => n + 1)
  }, [key])

  useEffect(() => {
    if (!key) return
    fetchers.set(key, () => fetchRef.current())
    if (!cache.get(key)?.at) void load(key, () => fetchRef.current())
    if (!opts.poll) return
    const id = setInterval(() => {
      if (document.visibilityState === 'visible') void load(key, () => fetchRef.current())
    }, opts.poll)
    return () => clearInterval(id)
  }, [key, opts.poll])

  return {
    data: snap?.data as T | undefined,
    error: snap?.error,
    loading: !!key && !snap?.at && !snap?.error,
    refetch,
  }
}

/** Run a mutation with pending/error state. */
export function useMutation<A extends unknown[], R>(fn: (...a: A) => Promise<R>) {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const run = useCallback(async (...a: A): Promise<R | undefined> => {
    setPending(true)
    setError(null)
    try {
      return await fn(...a)
    } catch (e) {
      setError(e as Error)
      return undefined
    } finally {
      setPending(false)
    }
  }, [fn])
  return { run, pending, error, reset: () => setError(null) }
}
