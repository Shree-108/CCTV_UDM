import { Suspense } from 'react'
import Feature from '@/frontend/features/s3upload/S3UploadPage'

export const metadata = { title: 'S3 Upload' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}
