'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useMemo, useState } from 'react'
import { Camera, CheckCircle2, Crop, FileVideo, ImageIcon, Loader2, Play, RotateCcw, ScanSearch, Upload } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Area, Layout, Media, MediaKind, Point } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { useApp } from '../../stores/app'
import { Badge, Button, PageHeader, Panel } from '../../components/ui/primitives'
import { ErrorState } from '../../components/ui/states'
import { AnalysisView } from './AnalysisView'
import { DropZone, GridSelector, MediaInfo, RoiEditor, StepBar, areaFrom, initialGrid, type GridState } from './common'

const ACCEPT: Record<MediaKind | 'any', string> = {
  image: 'image/jpeg,image/png,image/bmp,image/webp,.jpg,.jpeg,.png,.bmp,.webp',
  video: 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mp4,.mov,.avi,.mkv,.m4v,.webm',
  any: 'image/*,video/*,.mkv,.avi,.m4v',
}
const FORMATS: Record<MediaKind | 'any', string[]> = {
  image: ['JPG', 'PNG', 'BMP', 'WEBP'],
  video: ['MP4', 'AVI', 'MOV', 'MKV', 'M4V', 'WEBM'],
  any: ['JPG', 'PNG', 'MP4', 'AVI', 'MOV', 'WEBM'],
}
const STEPS = ['Upload', 'Preview', 'Select camera', 'Store ROI', 'Analyze', 'Result']
const IMAGE_STEPS = ['Upload', 'Analyzing uniform', 'Result']

/**
 * Area an uploaded image can be analysed with straight away, without asking the user:
 * the saved camera/ROI preset if it fits this image, else the whole frame of a
 * single-camera image. null = a multi-camera DVR grid with no saved camera, so the
 * user must click the store-interior camera first (that click starts the analysis).
 */
function autoArea(m: Media, layouts: Layout[], preset: Area | null): Area | null {
  if (preset && (preset.layout === m.suggested_layout || preset.layout === 'custom')) return preset
  if (m.suggested_layout === '1x1') return areaFrom(initialGrid(m, layouts), null)
  return null
}

function AnalyzingPanel({ media }: { media: Media }) {
  return (
    <Panel title="Analyzing uniform…" icon={ScanSearch} subtitle={media.filename}>
      <div className="flex flex-col items-center gap-3 py-2" role="status" aria-live="polite">
        <div className="relative flex w-full items-center justify-center overflow-hidden rounded-md border border-line bg-video">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={url(media.preview_url)} alt="Uploaded image being analysed"
            className="block max-h-[min(50vh,480px)] max-w-full object-contain opacity-60" />
          <span className="absolute flex items-center gap-2 rounded-md bg-black/75 px-3 py-2 text-[14px] font-semibold text-info">
            <Loader2 className="size-4 animate-spin" aria-hidden /> Analyzing uniform…
          </span>
        </div>
        <p className="text-[12px] text-fg-3">Detecting people → checking upper-body clothing → deciding compliance</p>
      </div>
    </Panel>
  )
}

export default function AnalysisWorkspace({ mode }: { mode: MediaKind | 'any' }) {
  const { toast } = useApp()
  const router = useRouter()
  const params = useSearchParams()
  const analysisId = params.get('analysis')
  const mediaId = params.get('media')
  const layouts = useQuery('layouts', Api.layouts)
  const settings = useQuery('settings', Api.settings)
  const health = useQuery('health', Api.health)
  const mediaQ = useQuery<Media>(mediaId ? `media:${mediaId}` : null, () => Api.media(mediaId!))
  const media = mediaId && mediaQ.data?.id === mediaId ? mediaQ.data : null
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<Error | null>(null)
  const [grid, setGrid] = useState<GridState | null>(null)
  const [gridConfirmed, setGridConfirmed] = useState(false)
  const [roi, setRoi] = useState<Point[] | null>(null)
  const [starting, setStarting] = useState(false)
  const [analyzing, setAnalyzing] = useState<Media | null>(null)   // image analysis in flight
  const preset = settings.data?.roi_preset ?? null
  const setup = params.get('setup') === '1'   // image: user chose to change camera / ROI and re-analyse

  /** Image analysis is synchronous on the server: show "Analyzing uniform…" until the result exists. */
  const runImage = async (m: Media, area: Area) => {
    setAnalyzing(m)
    try {
      const a = await Api.analyse(m.id, area)
      invalidate('overview')
      router.push(`/analysis/image?analysis=${a.id}`)
    } catch (e) {
      setAnalyzing(null)
      toast({ tone: 'crit', title: 'Uniform analysis failed', body: (e as Error).message })
      router.push(`/analysis/image?media=${m.id}&setup=1`)
    }
  }
  // keep "Analyzing uniform…" up until the result route is active (no flash of the upload screen)
  useEffect(() => { setAnalyzing(null) }, [analysisId])

  const onFile = async (f: File) => {
    setUploading(true)
    setUploadError(null)
    let m: Media
    try {
      m = await Api.upload(f)   // inference only starts once the upload has completed
    } catch (e) {
      setUploadError(e as Error)
      return
    } finally {
      setUploading(false)
    }
    if (mode !== 'any' && m.kind !== mode) {
      toast({ tone: 'info', title: `That is a${m.kind === 'image' ? 'n image' : ' video'}`, body: `Opening ${m.kind} analysis.` })
    }
    if (m.kind === 'image') {
      try {
        const [ls, st] = await Promise.all([layouts.data ?? Api.layouts(), settings.data ?? Api.settings()])
        const area = autoArea(m, ls, st.roi_preset)
        if (area) return await runImage(m, area)
      } catch (e) {
        toast({ tone: 'crit', title: 'Could not start analysis', body: (e as Error).message })
      }
    }
    router.push(`/analysis/${m.kind}?media=${m.id}`)
  }

  // new media -> default camera selection (saved preset if it fits this layout, else auto-detected layout)
  useEffect(() => {
    if (!media || !layouts.data) return
    if (preset && (preset.layout === media.suggested_layout || preset.layout === 'custom')) {
      setGrid({ layout: preset.layout, cell: preset.cell, rect: preset.rect, label: preset.label })
      setRoi(preset.roi)
    } else {
      setGrid(initialGrid(media, layouts.data))
      setRoi(null)
    }
    setGridConfirmed(media.suggested_layout === '1x1' && !preset)
  }, [media?.id, layouts.data]) // eslint-disable-line react-hooks/exhaustive-deps

  const step = analysisId ? 5 : !mediaId ? 0 : !gridConfirmed ? 2 : 3
  const kind: MediaKind | 'any' = media?.kind ?? mode
  const imageFlow = kind === 'image' || (mode === 'image' && !media)
  const imageStep = analysisId ? 2 : analyzing || uploading ? 1 : 0
  const title = kind === 'video' ? 'Video Analysis' : kind === 'image' ? 'Image Analysis' : 'Upload Analysis'

  const start = async () => {
    if (!media || !grid) return
    if (media.kind === 'image') return runImage(media, areaFrom(grid, roi))
    setStarting(true)
    try {
      const a = await Api.analyse(media.id, areaFrom(grid, roi))
      invalidate('overview')
      router.push(`/analysis/${media.kind}?analysis=${a.id}`)
    } catch (e) {
      toast({ tone: 'crit', title: 'Could not start analysis', body: (e as Error).message })
    } finally {
      setStarting(false)
    }
  }

  const reset = () => { setGrid(null); setRoi(null); setGridConfirmed(false); router.push(mode === 'any' ? '/upload' : `/analysis/${mode}`) }

  const notTrained = health.data && health.data.uniform_classifier.state !== 'ready'
  const description = useMemo(() => kind === 'video'
    ? 'Upload a CCTV video → select the store-interior camera → optional ROI → Start Analysis. People are tracked across frames and their uniform is confirmed by temporal voting.'
    : kind === 'image' ? 'Upload a CCTV image — the uniform check starts automatically. Every person inside the store area is checked for the approved uniform.'
      : 'Upload a CCTV image or video for uniform compliance analysis.', [kind])

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title={title} description={description}
        actions={(mediaId || analysisId) && <Button variant="ghost" icon={RotateCcw} onClick={reset}>New analysis</Button>} />
      <div className="flex flex-wrap items-center justify-between gap-2">
        {imageFlow && !(mediaId && !analysisId) ? <StepBar steps={IMAGE_STEPS} current={imageStep} />
          : <StepBar steps={STEPS} current={step === 3 && !analysisId ? 3 : step} />}
        {notTrained && <Link href="/help#training"><Badge tone="warn" dot>{health.data!.uniform_classifier.label}</Badge></Link>}
      </div>

      {analyzing ? (
        <AnalyzingPanel media={analyzing} />
      ) : analysisId ? (
        <AnalysisView id={analysisId} headerExtra={<Button size="sm" variant="ghost" icon={Upload} onClick={reset}>Analyse another file</Button>} />
      ) : mediaId && mediaQ.error ? (
        <Panel title="Upload CCTV media"><ErrorState error={mediaQ.error} title="Uploaded file not found" action={<Button size="sm" onClick={reset}>Upload again</Button>} /></Panel>
      ) : !mediaId ? (
        <Panel title="Upload CCTV media" icon={Upload}>
          <DropZone accept={ACCEPT[mode]} onFile={onFile} busy={uploading} formats={FORMATS[mode]}
            label={mode === 'video' ? 'DROP CCTV VIDEO' : mode === 'image' ? 'DROP CCTV IMAGE' : 'DROP VIDEO / IMAGE'}
            hint={<>or click to browse · {mode === 'any' ? 'an image or a CCTV video' : `a CCTV ${mode}`} for uniform compliance analysis</>} />
          {uploadError && <div className="mt-3"><ErrorState compact error={uploadError} title="Upload failed" onRetry={() => setUploadError(null)} /></div>}
          {mode === 'any' && (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <Link href="/analysis/image"><Button className="w-full" icon={ImageIcon}>+ Upload Image</Button></Link>
              <Link href="/analysis/video"><Button className="w-full" icon={FileVideo}>+ Upload Video</Button></Link>
            </div>
          )}
        </Panel>
      ) : !media || !grid || !layouts.data ? (
        <Panel title="Preparing preview"><div className="py-16 text-center text-[13px] text-fg-3">Loading preview…</div></Panel>
      ) : media.kind === 'image' && !setup ? (
        // multi-camera image with no saved camera: one click on the store camera starts the analysis
        <div className="mx-auto w-full max-w-4xl">
          <Panel title="Select store interior camera" icon={Camera}
            subtitle="Only the selected camera area is processed — the other grids are ignored.">
            <div className="mb-3"><MediaInfo media={media} /></div>
            <p className="mb-2 rounded-md border border-info/30 bg-info-bg px-2.5 py-1.5 text-[12px] text-info">
              Multi-camera recording detected — click the store-interior camera (or draw a Custom area) and the uniform check starts immediately.
              Save a camera + ROI once and later uploads are analysed with no click at all.
            </p>
            <GridSelector media={media} layouts={layouts.data} value={grid} confirmed={false}
              onChange={(g) => { setGrid(g); setRoi(null) }}
              onPick={(g) => runImage(media, areaFrom(g, null))} />
          </Panel>
        </div>
      ) : (
        <div className="grid gap-3 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
          <Panel title="Select store interior camera" icon={Camera}
            subtitle="Only the selected camera area is processed — the other grids are ignored."
            actions={gridConfirmed
              ? <Button size="sm" variant="ghost" onClick={() => setGridConfirmed(false)}>Change</Button>
              : <Button size="sm" variant="primary" icon={CheckCircle2} onClick={() => setGridConfirmed(true)}>Confirm selection</Button>}>
            <div className="mb-3"><MediaInfo media={media} /></div>
            <GridSelector media={media} layouts={layouts.data} value={grid} confirmed={gridConfirmed}
              onChange={(g) => { setGrid(g); setGridConfirmed(false); setRoi(null) }} />
          </Panel>
          <div className="flex flex-col gap-3">
            <Panel title={`Store interior area · ${grid.label}`} icon={Crop} subtitle="Optional — limit the result to people inside this polygon.">
              {gridConfirmed ? (
                <RoiEditor media={media} rect={grid.rect} roi={roi} onChange={setRoi} savedPreset={preset?.roi ?? null}
                  onSave={async (r) => {
                    await Api.saveRoiPreset(areaFrom(grid, r))
                    invalidate('settings')
                    toast({ tone: 'ok', title: 'ROI saved', body: `${grid.label} and this ROI will be pre-selected for your next upload.` })
                  }}
                  onClear={async () => {
                    if (preset) { await Api.clearRoiPreset(); invalidate('settings') }
                    toast({ tone: 'info', title: 'ROI cleared', body: 'The whole camera area will be analysed.' })
                  }} />
              ) : (
                <div className="flex aspect-video flex-col items-center justify-center gap-2 rounded-md border border-dashed border-line-strong text-center">
                  <Camera className="size-5 text-fg-3" aria-hidden />
                  <p className="text-[13px] text-fg-2">Confirm the store-interior camera first</p>
                  <Button size="sm" variant="primary" icon={CheckCircle2} onClick={() => setGridConfirmed(true)}>Confirm {grid.label}</Button>
                </div>
              )}
            </Panel>
            <Panel title="Start analysis" icon={ScanSearch}>
              <ul className="mb-3 flex flex-col gap-1 text-[12px] text-fg-2">
                <li>Camera area: <b className="text-fg">{grid.label}</b> {gridConfirmed ? <Badge tone="ok">confirmed</Badge> : <Badge tone="warn">not confirmed</Badge>}</li>
                <li>Store ROI: <b className="text-fg">{roi ? `${roi.length}-corner polygon` : 'whole camera area'}</b></li>
                {settings.data && <li>Uniform threshold: <b className="text-fg">{Math.round(settings.data.settings.uniform_confidence * 100)}%</b>
                  {media.kind === 'video' && <> · analysing <b className="text-fg">{settings.data.settings.analysis_fps}</b> frames / s of video</>}
                  {' '}<Link href="/settings" className="text-info hover:underline">change</Link></li>}
              </ul>
              <Button variant="primary" icon={Play} className="h-10 w-full text-[14px]" disabled={!gridConfirmed} loading={starting} onClick={start}>
                {media.kind === 'video' ? 'START ANALYSIS' : 'RE-ANALYZE WITH THIS AREA'}
              </Button>
              {!gridConfirmed && <p className="mt-2 text-center text-[11px] text-fg-3">Confirm the store-interior camera to enable analysis.</p>}
            </Panel>
          </div>
        </div>
      )}
    </div>
  )
}
