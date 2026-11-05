// npm run dev:all - starts the FastAPI backend (port 8010) and the dashboard (port 3010) together, so the
// S3 Upload page works without starting the API separately. Ctrl+C stops both; if one stops, so does the other.
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

const root = join(import.meta.dirname, '..')
const python = [join(root, '.venv', 'Scripts', 'python.exe'), join(root, '.venv', 'bin', 'python')].find(existsSync)
if (!python) {
  console.error('No .venv found: create it first (see README.md).')
  process.exit(1)
}

const procs = [
  ['api', python, ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8010']],
  ['web', process.execPath, [join(root, 'node_modules', 'next', 'dist', 'bin', 'next'), 'dev', '-p', '3010']],
].map(([name, cmd, args]) => {
  const p = spawn(cmd, args, { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] })
  for (const s of [p.stdout, p.stderr]) s.on('data', (d) => process.stdout.write(String(d).replace(/^(?=.)/gm, `[${name}] `)))
  p.on('exit', (code) => {
    console.log(`[${name}] stopped (exit ${code})`)
    stopAll()
  })
  return p
})

let stopping = false
function stopAll() {
  if (stopping) return
  stopping = true
  for (const p of procs) if (p.exitCode === null) p.kill()
  setTimeout(() => process.exit(0), 1500)
}
process.on('SIGINT', stopAll)
process.on('SIGTERM', stopAll)
console.log('Starting API on http://127.0.0.1:8010 and dashboard on http://localhost:3010 (S3 Upload: /s3-upload)')
