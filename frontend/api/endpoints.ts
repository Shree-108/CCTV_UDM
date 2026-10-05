import { api, qs } from './client'
import type {
  Alert, Analysis, AnalysisDetail, Analytics, Area, Health, Layout, Media, Overview, Settings, SettingSchema,
} from './types'

export const Api = {
  health: () => api<Health>('/api/health'),
  overview: () => api<Overview>('/api/overview'),
  layouts: () => api<Layout[]>('/api/layouts'),

  upload: (file: File, signal?: AbortSignal) => {
    const f = new FormData()
    f.append('file', file)
    return api<Media>('/api/media', { form: f, signal })
  },
  media: (id: string) => api<Media>(`/api/media/${id}`),

  analyse: (media_id: string, area: Area) => api<AnalysisDetail>('/api/analyses', { body: { media_id, area } }),
  analysis: (id: string) => api<AnalysisDetail>(`/api/analyses/${id}`),
  analyses: (p: { limit?: number; offset?: number; kind?: string; status?: string; q?: string } = {}) =>
    api<{ analyses: Analysis[]; total: number }>(`/api/analyses${qs(p)}`),
  cancel: (id: string) => api<{ ok: boolean }>(`/api/analyses/${id}/cancel`, { method: 'POST' }),
  deleteAnalysis: (id: string) => api<{ ok: boolean }>(`/api/analyses/${id}`, { method: 'DELETE' }),

  alerts: (p: { state?: string; kind?: string; limit?: number; offset?: number } = {}) =>
    api<{ alerts: Alert[]; total: number; new: number }>(`/api/alerts${qs(p)}`),
  setAlertState: (id: number, state: Alert['state']) => api<{ ok: boolean }>(`/api/alerts/${id}/state`, { body: { state } }),

  analytics: (start?: string) => api<Analytics>(`/api/analytics${qs({ start })}`),
  search: (q: string) => api<{ analyses: Analysis[]; alerts: Alert[] }>(`/api/search${qs({ q })}`),

  settings: () => api<{ settings: Settings; schema: SettingSchema[]; roi_preset: Area | null }>('/api/settings'),
  saveSettings: (s: Partial<Settings>) => api<{ settings: Settings }>('/api/settings', { method: 'PUT', body: s }),
  resetSettings: () => api<{ settings: Settings }>('/api/settings/reset', { method: 'POST' }),
  saveRoiPreset: (a: Area) => api<{ ok: boolean }>('/api/roi-preset', { method: 'PUT', body: a }),
  clearRoiPreset: () => api<{ ok: boolean }>('/api/roi-preset', { method: 'DELETE' }),
}
