'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { ArrowLeft, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Api } from '../../api/endpoints'
import { invalidate } from '../../hooks/useQuery'
import { useApp } from '../../stores/app'
import { Button, PageHeader } from '../../components/ui/primitives'
import { ConfirmDialog } from '../../components/ui/overlay'
import { AnalysisView } from '../analysis/AnalysisView'
import { HistoryTable } from './AnalyticsPage'

export default function HistoryPage() {
  const params = useSearchParams()
  const router = useRouter()
  const { toast } = useApp()
  const id = params.get('id')
  const [confirm, setConfirm] = useState(false)
  const [deleting, setDeleting] = useState(false)

  if (!id) {
    return (
      <div className="flex flex-col gap-3">
        <PageHeader title="Analysis history" description="Every analysed image and video. Open one to see its annotated result, people, alerts and evidence." />
        <HistoryTable />
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Analysis" description={<Link href="/history" className="inline-flex items-center gap-1 text-info hover:underline"><ArrowLeft className="size-3" /> Back to history</Link>}
        actions={<Button variant="ghost" icon={Trash2} onClick={() => setConfirm(true)}>Delete analysis</Button>} />
      <AnalysisView id={id} />
      <ConfirmDialog open={confirm} onClose={() => setConfirm(false)} danger pending={deleting} confirmLabel="Delete"
        title="Delete this analysis?" body="Its annotated output, report, evidence frames and alerts are removed permanently. The uploaded source file is kept."
        onConfirm={async () => {
          setDeleting(true)
          try {
            await Api.deleteAnalysis(id)
            invalidate('analyses'); invalidate('overview'); invalidate('alerts'); invalidate('analytics')
            toast({ tone: 'ok', title: 'Analysis deleted' })
            router.push('/history')
          } catch (e) {
            toast({ tone: 'crit', title: 'Could not delete', body: (e as Error).message })
          } finally {
            setDeleting(false)
            setConfirm(false)
          }
        }} />
    </div>
  )
}
