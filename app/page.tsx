import { Suspense } from 'react'
import Feature from '@/frontend/features/overview/OverviewPage'

export const metadata = { title: 'Overview' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}
