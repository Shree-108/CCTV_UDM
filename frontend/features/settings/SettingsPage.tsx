'use client'

import Link from 'next/link'
import { useEffect, useMemo, useState } from 'react'
import { Crop, Gauge, Monitor, RotateCcw, Save, ScanSearch, SlidersHorizontal, Target, Trash2, Video } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { EvaluationReport, Settings, SettingSchema } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, PageHeader, Panel, Segmented, Select, Slider, Toggle } from '../../components/ui/primitives'
import { ConfirmDialog } from '../../components/ui/overlay'
import { EmptyState, QueryView } from '../../components/ui/states'
import { ModelStatus } from '../overview/OverviewPage'

const LABELS: Partial<Record<keyof Settings, string>> = {
  person_confidence: 'Person confidence', uniform_confidence: 'Uniform confidence', analysis_fps: 'Analysis FPS',
  inference_size: 'Inference resolution', temporal_confirmation_frames: 'Temporal confirmation frames', gpu_acceleration: 'GPU acceleration',
  output_video_quality: 'Output video quality', absence_alert_seconds: 'No-uniform alert after', non_uniform_confidence: 'Non-uniform confidence',
  frame_skip: 'Frame skip', vote_window: 'Vote window', presence_gap_tolerance: 'Occlusion tolerance', min_track_observations: 'Min. track observations',
  min_person_height_px: 'Min. person height', roi_reference_point: 'ROI reference point', untrained_model_mode: 'When the uniform model is not trained',
  low_confidence_alerts: 'Low-confidence alerts',
}

export default function SettingsPage() {
  const { toast, theme, setTheme } = useApp()
  const q = useQuery('settings', Api.settings)
  const health = useQuery('health', Api.health, { poll: 20000 })
  const [draft, setDraft] = useState<Settings | null>(null)
  const [saving, setSaving] = useState(false)
  const [confirmReset, setConfirmReset] = useState(false)
  useEffect(() => { if (q.data && !draft) setDraft(q.data.settings) }, [q.data, draft])
  const dirty = useMemo(() => !!draft && !!q.data && JSON.stringify(draft) !== JSON.stringify(q.data.settings), [draft, q.data])
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setDraft((d) => (d ? { ...d, [k]: v } : d))

  const save = async () => {
    if (!draft) return
    setSaving(true)
    try {
      const r = await Api.saveSettings(draft)
      setDraft(r.settings)
      invalidate('settings'); invalidate('health')
      toast({ tone: 'ok', title: 'Settings saved', body: 'They apply to the next analysis.' })
    } catch (e) {
      toast({ tone: 'crit', title: 'Could not save settings', body: (e as Error).message })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Settings" description="Detection thresholds and video processing. Changes apply to the next analysis; finished analyses keep the settings they were run with."
        actions={<>
          <Button variant="ghost" icon={RotateCcw} onClick={() => setConfirmReset(true)}>Reset defaults</Button>
          <Button variant="primary" icon={Save} disabled={!dirty} loading={saving} onClick={save}>Save changes</Button>
        </>} />
      <QueryView q={q}>
        {(d) => !draft ? null : (
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_400px]">
            <div className="flex flex-col gap-4">
              <Panel title="Detection" icon={ScanSearch}>
                <div className="grid gap-5 md:grid-cols-2">
                  <div className="flex flex-col gap-1">
                    <Slider label="Person confidence" value={draft.person_confidence} min={0.1} max={0.9} step={0.05} onChange={(v) => set('person_confidence', v)} format={(v) => `${Math.round(v * 100)}%`} />
                    <p className="text-[11px] text-fg-3">Minimum YOLO confidence for a detection to count as a person.</p>
                  </div>
                  <div className="flex flex-col gap-1">
                    <Slider label="Uniform confidence" value={draft.uniform_confidence} min={0.5} max={0.95} step={0.05} onChange={(v) => set('uniform_confidence', v)} format={(v) => `${Math.round(v * 100)}%`} />
                    <p className="text-[11px] text-fg-3">A shirt is called UNIFORM only above this. Weaker results become UNCERTAIN — never forced into UNIFORM. Higher = fewer false uniform detections.</p>
                  </div>
                  <div className="flex flex-col gap-1">
                    <Slider label="Temporal confirmation frames" value={draft.temporal_confirmation_frames} min={1} max={10} step={1} onChange={(v) => set('temporal_confirmation_frames', v)} format={(v) => `${v}`} />
                    <p className="text-[11px] text-fg-3">Video: uniform votes a tracked person needs before being confirmed as UNIFORM.</p>
                  </div>
                  <Field label="Inference resolution" hint="The selected camera tile is resized to this. Larger finds smaller/distant people but is slower.">
                    <Select value={draft.inference_size} onChange={(e) => set('inference_size', Number(e.target.value))}>
                      {[320, 416, 480, 640, 800, 960, 1280].map((v) => <option key={v} value={v}>{v}px</option>)}
                    </Select>
                  </Field>
                </div>
              </Panel>

              <Panel title="Video processing" icon={Video}>
                <div className="grid gap-5 md:grid-cols-2">
                  <div className="flex flex-col gap-1">
                    <Slider label="Analysis FPS" value={draft.analysis_fps} min={1} max={15} step={1} onChange={(v) => set('analysis_fps', v)} format={(v) => `${v} / s`} />
                    <p className="text-[11px] text-fg-3">Frames analysed per second of video (the rest are skipped without decoding). 3–5 is enough to follow people walking through the store.</p>
                  </div>
                  <div className="flex flex-col gap-1">
                    <Slider label="No-uniform alert after" value={draft.absence_alert_seconds} min={5} max={300} step={5} onChange={(v) => set('absence_alert_seconds', v)} format={(v) => `${v}s`} />
                    <p className="text-[11px] text-fg-3">People inside but no confirmed uniform for this long → NO UNIFORM alert with an evidence frame.</p>
                  </div>
                  <Field label="GPU acceleration" hint={health.data ? (health.data.gpu.available ? 'A CUDA GPU is available.' : 'No CUDA GPU detected — processing runs on the CPU.') : undefined}>
                    <Segmented label="GPU acceleration" value={draft.gpu_acceleration} onChange={(v) => set('gpu_acceleration', v)}
                      options={[{ value: 'auto', label: 'Auto' }, { value: 'on', label: 'On' }, { value: 'off', label: 'Off (CPU)' }]} />
                  </Field>
                  <Field label="Output video quality" hint="Quality of the annotated video you can watch and download.">
                    <Segmented label="Output video quality" value={draft.output_video_quality} onChange={(v) => set('output_video_quality', v)}
                      options={[{ value: 'low', label: 'Low' }, { value: 'medium', label: 'Medium' }, { value: 'high', label: 'High' }]} />
                  </Field>
                </div>
              </Panel>

              <Panel title="Store ROI" icon={Crop}>
                {d.roi_preset ? (
                  <div className="flex flex-wrap items-center gap-3 text-[13px]">
                    <Badge tone="ok" dot>Saved</Badge>
                    <span className="text-fg">{d.roi_preset.label}</span>
                    <span className="text-fg-3">{d.roi_preset.layout === 'custom' ? 'custom area' : `${d.roi_preset.layout} layout`} · {d.roi_preset.roi ? `${d.roi_preset.roi.length}-corner ROI` : 'whole camera area'}</span>
                    <Button size="sm" variant="ghost" icon={Trash2} className="ml-auto"
                      onClick={async () => { await Api.clearRoiPreset(); invalidate('settings'); toast({ tone: 'info', title: 'Saved ROI cleared' }) }}>Clear saved ROI</Button>
                  </div>
                ) : <p className="text-[12px] text-fg-3">No saved camera area / ROI. Use “Save ROI” on the Image or Video Analysis page to pre-select it for future uploads.</p>}
              </Panel>

              <details className="group rounded-lg border border-line bg-panel">
                <summary className="flex cursor-pointer select-none items-center gap-2 px-3.5 py-2.5 text-[12px] font-semibold uppercase tracking-[0.06em] text-fg-2 hover:text-fg">
                  <SlidersHorizontal className="size-3.5" aria-hidden /> Advanced configuration
                  <span className="ml-auto text-[11px] font-normal normal-case tracking-normal text-fg-3">for tuning — defaults work for most footage</span>
                </summary>
                <div className="grid gap-4 border-t border-line p-3.5 md:grid-cols-2">
                  {d.schema.filter((s) => s.advanced).map((s) => <AdvancedField key={s.key} s={s} value={draft[s.key]} onChange={(v) => set(s.key, v as never)} />)}
                </div>
              </details>
            </div>

            <div className="flex flex-col gap-4">
              <ModelStatus h={health.data} id="model-status" />
              <EvaluationCard ev={health.data?.evaluation ?? null} />
              <Panel title="Appearance" icon={Monitor}>
                <Segmented label="Theme" value={theme} onChange={setTheme}
                  options={[{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }, { value: 'system', label: 'System' }]} />
              </Panel>
              <Panel title="Camera integration" icon={Gauge}>
                <p className="text-[12px] text-fg-3">Live CCTV / RTSP / ONVIF / NVR connections are <b className="text-fg-2">not available in this version</b>. Analyse exported footage by uploading it on the Image or Video Analysis page.</p>
              </Panel>
            </div>
          </div>
        )}
      </QueryView>
      <ConfirmDialog open={confirmReset} onClose={() => setConfirmReset(false)} title="Reset all settings?" body="All thresholds and processing options return to their defaults." confirmLabel="Reset"
        onConfirm={async () => { const r = await Api.resetSettings(); setDraft(r.settings); invalidate('settings'); setConfirmReset(false); toast({ tone: 'ok', title: 'Settings reset to defaults' }) }} />
    </div>
  )
}

function AdvancedField({ s, value, onChange }: { s: SettingSchema; value: Settings[keyof Settings]; onChange: (v: unknown) => void }) {
  const label = LABELS[s.key] ?? s.key
  if (s.type === 'bool') return <Toggle label={label} description={s.description} checked={!!value} onChange={onChange} />
  if (s.choices) {
    return (
      <Field label={label} hint={s.description}>
        <Select value={String(value)} onChange={(e) => onChange(e.target.value)}>{s.choices.map((c) => <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>)}</Select>
      </Field>
    )
  }
  return (
    <Field label={label} hint={`${s.description} (${s.min}–${s.max})`}>
      <input type="number" value={Number(value)} min={s.min ?? undefined} max={s.max ?? undefined} step={s.type === 'int' ? 1 : 0.05}
        onChange={(e) => onChange(s.type === 'int' ? parseInt(e.target.value || '0', 10) : parseFloat(e.target.value || '0'))}
        className="h-8 w-full rounded-md border border-line bg-panel-2 px-2.5 text-[13px] text-fg hover:border-line-strong focus:border-info focus:outline-none" />
    </Field>
  )
}

function EvaluationCard({ ev }: { ev: EvaluationReport | null }) {
  const set = ev ? (ev.sets.validation_split ?? Object.values(ev.sets)[0]) : null
  return (
    <Panel title="Model evaluation" icon={Target}>
      {!set ? (
        <EmptyState compact title="No evaluation report" body={<>Train and evaluate the uniform classifier to see precision, recall, F1 and the confusion matrix here. <Link href="/help#training" className="text-info hover:underline">How →</Link></>} />
      ) : (
        <div className="flex flex-col gap-3 text-[12px]">
          <div className="grid grid-cols-2 gap-2">
            <Stat k="Accuracy" v={pct(set.accuracy, 1)} />
            <Stat k="False uniform rate" v={set.uniform_false_positive_rate != null ? pct(set.uniform_false_positive_rate, 1) : '—'} warn={(set.uniform_false_positive_rate ?? 0) > 0.02} />
            <Stat k="Uniform precision" v={pct(set.per_class.uniform.precision, 1)} />
            <Stat k="Uniform recall" v={pct(set.per_class.uniform.recall, 1)} />
            <Stat k="Uniform F1" v={pct(set.per_class.uniform.f1, 1)} />
            <Stat k="Test images" v={String(set.n)} />
          </div>
          <table className="w-full text-center font-mono text-[11px]">
            <thead><tr><th className="text-left text-fg-3">true \ pred</th>{set.confusion_matrix.labels.map((l) => <th key={l} className="text-fg-3">{l.replace('_', '-')}</th>)}</tr></thead>
            <tbody>{set.confusion_matrix.rows_true_cols_pred.map((row, i) => (
              <tr key={i}><td className="text-left text-fg-2">{set.confusion_matrix.labels[i].replace('_', '-')}</td>
                {row.map((v, j) => <td key={j} className={i === j ? 'text-ok' : v ? 'text-crit' : 'text-fg-3'}>{v}</td>)}</tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}

const Stat = ({ k, v, warn }: { k: string; v: string; warn?: boolean }) => (
  <div className="rounded-md border border-line bg-panel-2 px-2.5 py-1.5"><div className="text-[10px] uppercase tracking-wide text-fg-3">{k}</div>
    <div className={warn ? 'tabular text-[15px] font-semibold text-crit' : 'tabular text-[15px] font-semibold text-fg'}>{v}</div></div>
)
