'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { Clapperboard, ExternalLink, Zap } from 'lucide-react'
import { Button, PageHeader } from '../../components/ui/primitives'
import { cx } from '../../lib/format'
import { ProcessPanel } from '../results/ResultsDashboard'
import AnalysisWorkspace from './AnalysisWorkspace'

/**
 * Video Analysis. Default: the quick sampled analysis - the SAME component and backend service as the
 * Results Dashboard (4 random frames → existing model → CSVs). The full-video pass (tracker on every
 * analysed frame, annotated video, alerts) is kept as a separate mode; its own links (?analysis=, ?media=,
 * ?setup=) keep opening it, so existing full-video results are never shown as sampled ones or vice versa.
 */
export default function VideoAnalysisPage() {
  const router = useRouter()
  const params = useSearchParams()
  const full = params.get('mode') === 'full' || params.has('analysis') || params.has('media') || params.has('setup')
  if (full) {
    return (
      <div className="flex flex-col gap-3">
        <ModeSwitch full />
        <AnalysisWorkspace mode="video" />
      </div>
    )
  }
  const jobId = params.get('job')
  // the shared panel speaks "media=<id>"; on this page the uploaded video is ?quick=<id> (?media= = full mode)
  const go = (q: string) => router.replace(`/analysis/video${q ? `?${q.replace(/^media=/, 'quick=')}` : ''}`)
  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Video Analysis"
        description="Upload a CCTV video, confirm the store and recording time, and start. Four random frames spread across the video are checked with the existing uniform model; the results are saved and added to the master CSV." />
      <ModeSwitch full={false} />
      <p className="text-[11px] text-fg-3">
        <b className="text-fg-2">Source: manual upload.</b> Receiving videos from a CCTV recorder / API is not configured: no recorder
        vendor, API documentation or credentials have been provided, so nothing is fetched automatically. This is a quick
        sample-based report — only the four sampled frames are checked, not every frame.
      </p>
      <ProcessPanel jobId={jobId} mediaId={params.get('quick')} go={go} startLabel="START ANALYSIS"
        extra={(id) => <Link href={`/results?job=${id}`}><Button size="sm" variant="primary" icon={ExternalLink}>Open in Results Dashboard</Button></Link>} />
    </div>
  )
}

function ModeSwitch({ full }: { full: boolean }) {
  const tab = (on: boolean) => cx('flex items-start gap-2 rounded-md border px-3 py-2 text-left text-[12px]',
    on ? 'border-info/60 bg-info-bg text-fg' : 'border-line text-fg-2 hover:bg-hover')
  return (
    <div className="grid gap-2 md:grid-cols-2" role="tablist" aria-label="Analysis mode">
      <Link href="/analysis/video" role="tab" aria-selected={!full} className={tab(!full)}>
        <Zap className="mt-0.5 size-4 shrink-0 text-info" aria-hidden />
        <span><b>Quick sampled analysis</b> (default) — 4 random frames across the video. Fast; only those frames are checked.</span>
      </Link>
      <Link href="/analysis/video?mode=full" role="tab" aria-selected={full} className={tab(full)}>
        <Clapperboard className="mt-0.5 size-4 shrink-0 text-fg-3" aria-hidden />
        <span><b>Full-video annotated analysis</b> — every analysed frame with the person tracker, annotated video and alerts. Slow on long videos.</span>
      </Link>
    </div>
  )
}
