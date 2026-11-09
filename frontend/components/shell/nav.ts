import {
  AlertTriangle, BarChart3, Cctv, FileVideo, Film, HelpCircle, History, ImageIcon, LayoutGrid, Settings, Table2, UploadCloud,
  type LucideIcon,
} from 'lucide-react'

export type BadgeKey = 'newAlerts' | 'running'

export interface NavItem {
  href: string; label: string; icon: LucideIcon; badge?: BadgeKey; description: string
  /** shown in the sidebar but not navigable (future feature) */
  comingSoon?: boolean
}
export interface NavGroup { label: string; items: NavItem[]; hidden?: boolean }

/** The sidebar is result-oriented: one workflow per entry. Pages under "More" are not in the sidebar but
 *  keep their routes (direct links, "Job details" buttons and the command palette still open them). */
export const NAV: NavGroup[] = [
  { label: 'Workspace', items: [
    { href: '/analysis/video', label: 'Video Analysis', icon: FileVideo, badge: 'running', description: 'Upload a CCTV video: 4 random frames checked by the uniform model (full-video mode also available)' },
    { href: '/results', label: 'Results Dashboard', icon: Table2, description: 'Upload a video, check 4 random screenshots, results history and the cumulative CSV' },
    { href: '/alerts', label: 'Alerts', icon: AlertTriangle, badge: 'newAlerts', description: 'Confirmed uniform violations and low-confidence results' },
  ] },
  { label: 'System', items: [
    { href: '/settings', label: 'Settings', icon: Settings, description: 'Detection thresholds, video processing, model status' },
    { href: '/help', label: 'Help Center', icon: HelpCircle, description: 'How the analysis works and how to train the uniform model' },
    { href: '/cameras', label: 'Camera Integration', icon: Cctv, comingSoon: true, description: 'Live CCTV / RTSP — not available in this version' },
  ] },
  { label: 'More', hidden: true, items: [
    { href: '/', label: 'Overview', icon: LayoutGrid, description: 'Latest uniform compliance result and recent analyses' },
    { href: '/analysis/image', label: 'Image Analysis', icon: ImageIcon, description: 'Upload a CCTV image and check uniform compliance' },
    { href: '/extract', label: 'Frame Extraction', icon: Film, description: 'Extraction job details, fixed-interval sampling and per-job CSV manifests' },
    { href: '/analytics', label: 'Analytics', icon: BarChart3, description: 'Compliance statistics of the full-video analyses' },
    { href: '/upload', label: 'Upload Analysis', icon: UploadCloud, description: 'Upload a CCTV image or video' },
    { href: '/history', label: 'Analysis History', icon: History, description: 'Every analysed file with its result and evidence' },
  ] },
]

export const ALL_NAV = NAV.flatMap((g) => g.items.filter((i) => !i.comingSoon).map((i) => ({ ...i, group: g.label })))

export function navFor(pathname: string) {
  return ALL_NAV.filter((i) => i.href === '/' ? pathname === '/' : pathname === i.href || pathname.startsWith(`${i.href}/`))
    .sort((a, b) => b.href.length - a.href.length)[0]
}
