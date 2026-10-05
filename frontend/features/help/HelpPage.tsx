'use client'

import Link from 'next/link'
import type { ReactNode } from 'react'
import { BookOpen, Camera, Crop, GraduationCap, ScanSearch, ShieldCheck, Timer } from 'lucide-react'
import { PageHeader, Panel } from '../../components/ui/primitives'
import { STATUS_META } from '../../components/ui/status'

export default function HelpPage() {
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Help Center" description="How Sightline decides whether the approved uniform is being worn, and how to get reliable results." />
      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Workflow" icon={BookOpen}>
          <ol className="flex list-decimal flex-col gap-1.5 pl-5 text-[13px] text-fg-2">
            <li><Link href="/analysis/image" className="text-info hover:underline">Image</Link> or <Link href="/analysis/video" className="text-info hover:underline">Video Analysis</Link> → upload the CCTV file.</li>
            <li>Check the preview and file details.</li>
            <li><b className="text-fg">Select the store-interior camera.</b> DVR exports often tile 4 cameras; the layout is auto-detected. Click the right tile (or draw a custom rectangle) and confirm. Only that tile is processed.</li>
            <li>Optional: <b className="text-fg">Define ROI</b> — click the corners of the store floor inside that camera. People outside it are ignored.</li>
            <li>Start the analysis. Videos show live progress with the annotated frame.</li>
            <li>Review the annotated result, per-person table, alerts and download the annotated media and report.</li>
          </ol>
        </Panel>
        <Panel title="Final status" icon={ShieldCheck}>
          <ul className="flex flex-col gap-2 text-[13px]">
            <Rule s="compliant">At least one person inside the selected store area is confidently wearing the approved uniform.</Rule>
            <Rule s="no_uniform">People are inside, their clothing was judged, and none wears the approved uniform. Message: “No uniformed person detected inside the selected store area.”</Rule>
            <Rule s="uncertain">People are inside but there is not enough visual evidence (too small, hidden, blurred, facing away, or below the confidence threshold).</Rule>
            <Rule s="no_person">Nobody was detected inside the selected store area.</Rule>
          </ul>
          <p className="mt-3 text-[11px] text-fg-3">The system checks clothing only. It never identifies who a person is and never claims someone is the owner, captain or a specific employee.</p>
        </Panel>
        <Panel title="How a person is judged" icon={ScanSearch}>
          <ul className="flex list-disc flex-col gap-1.5 pl-5 text-[13px] text-fg-2">
            <li>YOLO detects every person in the selected camera tile.</li>
            <li>Only the <b className="text-fg">upper body</b> (chest, shoulders, torso, sleeves) is cropped and classified — shelves, posters, packaging and black-and-white backgrounds are never looked at.</li>
            <li>The classifier returns UNIFORM / NON-UNIFORM / UNCERTAIN with a confidence. Below the uniform threshold (Settings) the result is UNCERTAIN, never forced into UNIFORM.</li>
            <li>People who are too small, cut off, blurred or hidden behind someone are UNCERTAIN.</li>
          </ul>
        </Panel>
        <Panel title="Video: tracking & temporal logic" icon={Timer}>
          <ul className="flex list-disc flex-col gap-1.5 pl-5 text-[13px] text-fg-2">
            <li>ByteTrack keeps the same ID for the same person — someone seen in 100 frames is one person.</li>
            <li>A person is confirmed as UNIFORM after several uniform votes (“Temporal confirmation frames”); one blurred or turned-away frame does not flip them.</li>
            <li>If a uniformed person disappears briefly (occlusion, shelf, detection miss) the uniform presence continues — no false violation.</li>
            <li>A NO UNIFORM alert with an evidence frame is raised only when people are inside and no uniform is confirmed for the configured time, or the whole video ends as NO UNIFORM.</li>
          </ul>
        </Panel>
        <Panel title="Camera & ROI tips" icon={Crop}>
          <ul className="flex list-disc flex-col gap-1.5 pl-5 text-[13px] text-fg-2">
            <li>Choose the tile that shows the sales floor, not the entrance, parking or street.</li>
            <li>Use an ROI to exclude windows, doorways and mirrors where outside people or reflections appear.</li>
            <li>“Save ROI” remembers the camera and ROI for the next upload with the same layout.</li>
            <li>Higher “Inference resolution” helps with distant, small people in 4-camera exports.</li>
          </ul>
        </Panel>
        <Panel title="Camera integration" icon={Camera}>
          <p className="text-[13px] text-fg-2">Live CCTV (RTSP, IP cameras, ONVIF, DVR/NVR) is intentionally not part of this version — only uploaded images and videos are analysed. The media layer is modular so a live source can be added later.</p>
        </Panel>
      </div>
      <section id="training">
        <Panel title="Training the uniform classifier" icon={GraduationCap}>
          <div className="grid gap-4 text-[13px] text-fg-2 lg:grid-cols-2">
            <div className="flex flex-col gap-2">
              <p>Until a trained model exists the app shows <b className="text-warn">MODEL NOT TRAINED</b>. By default it then uses a simple stripe-pattern baseline whose results are indicative only; in Settings → Advanced you can disable it so everyone is reported as UNCERTAIN instead.</p>
              <p>The reference photographs are a starting point, not a dataset. Collect upper-body crops from <b className="text-fg">your own cameras</b>:</p>
              <ul className="list-disc pl-5">
                <li><b className="text-ok">uniform</b> — front, side, back; near and far; different people, body sizes, lighting; sitting, standing, walking; partly hidden; low-resolution CCTV.</li>
                <li><b className="text-crit">non_uniform</b> — plain black / white / blue shirts, checks, horizontal stripes, <i>other</i> vertical stripes, prints, T-shirts, jackets, hoodies, customers in black &amp; white.</li>
                <li><b className="text-warn">uncertain</b> — torso mostly hidden, heavy blur, too small to judge.</li>
              </ul>
            </div>
            <div className="flex flex-col gap-2">
              <pre className="overflow-x-auto rounded-md border border-line bg-panel-2 p-3 font-mono text-[12px] text-fg">{`# 1. crops exactly as production sees them
python -m training.prepare_dataset extract --source path/to/cctv
# 2. sort data/unlabeled/*.jpg into
#    data/uniform  data/non_uniform  data/uncertain
# 3. train + evaluate
python -m training.train
python -m training.evaluate`}</pre>
              <p>The trained model is saved to <code className="font-mono text-fg">models/uniform/best.pt</code> and used automatically. Evaluation (precision, recall, F1, confusion matrix, false-uniform rate) appears in Settings → Model evaluation. Prioritise a low <b className="text-fg">false uniform</b> rate: a customer mistaken for staff makes the result unreliable.</p>
            </div>
          </div>
        </Panel>
      </section>
    </div>
  )
}

function Rule({ s, children }: { s: keyof typeof STATUS_META; children: ReactNode }) {
  const m = STATUS_META[s]
  return (
    <li className="flex gap-2.5">
      <m.icon className="mt-0.5 size-4 shrink-0" style={{ color: m.color }} aria-hidden />
      <div><b style={{ color: m.color }}>{m.title}</b> — <span className="text-fg-2">{children}</span></div>
    </li>
  )
}
