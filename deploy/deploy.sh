#!/usr/bin/env bash
#
# deploy.sh - release a commit of BRANCH on a server prepared by deploy/server-setup.sh.
# Run it by hand for every release; nothing here runs automatically.
#
# Safety checks - the deploy STOPS (nothing changed) when one fails:
#   - the environment from setup is missing (.venv, Node >= 20.9, pm2, rsync) or .venv has another Python
#   - the checkout's origin is not REPO_URL, the checkout is not on BRANCH, BRANCH is not on origin, or
#     REF is not a commit of origin/BRANCH
#   - tracked files were edited on this server (FORCE_RESET=1 discards them; a patch is saved first)
#   - the release changes models/ or config/uniform_config.yaml (ALLOW_MODEL_CHANGE=1 accepts)
#   - the current model file does not match its SHA-256 (models are never changed by this script)
#   - an untracked file would be replaced, or runtime data (storage, .env, .env.local) is tracked by git
#   - NEXT_PUBLIC_API_URL in .env.local is missing, empty, not http(s) or localhost
#   - jobs are queued or running (FORCE_RESTART=1 accepts: analyses are marked failed, extractions and
#     detection runs interrupted)
#   - a running app does not answer its health check (ALLOW_UNHEALTHY=1 accepts)
#   - the backup cannot be made completely
# Then: backup -> git reset --hard REF -> pinned dependencies -> import / CPU / model checks -> npm ci + build
#   -> restart both apps (re-registered if the start command changed) -> pm2 save -> health check -> last logs.
#
# Restart, not zero-downtime: the API is ONE process (in-process job queue + SQLite), so it is restarted -
# a few seconds without API; `npm ci` / `npm run build` also replace the files the running dashboard serves,
# so the dashboard can show errors until it restarts. Deploy outside business hours.
#
# Usage (as the deploy user):
#   bash deploy/deploy.sh
#   REF=<commit> bash deploy/deploy.sh      # roll the CODE back to an earlier commit of BRANCH
#
# Configuration (environment variables, defaults in brackets)
#   APP_DIR              project checkout [the parent of this deploy/ folder]
#   REPO_URL             repository the checkout's origin must point at [https://github.com/Shree-108/CCTV_UDM.git]
#   BRANCH               branch to deploy; must be the branch checked out [publish/cctv-udm]
#   REF                  commit to deploy; must be on origin/BRANCH [origin/$BRANCH]
#   PM2_APP_NAME         pm2 name prefix: <name>-api and <name>-web [yk-cctv-udm]
#   APP_PORT             FastAPI port, always bound to 127.0.0.1 [8010]
#   WEB_PORT             Next.js dashboard port [3010]
#   WEB_HOST             dashboard bind address; keep 127.0.0.1 behind Nginx [127.0.0.1]
#   PYTHON_BIN           the Python setup built .venv with - only compared, never used to rebuild [python3]
#   TORCH_VARIANT        cpu | cuda - must match setup [cpu]
#   INSTALL_DEV_DEPS     1 = also install pytest, httpx, scikit-learn [0]
#   BACKUP_DIR           backup location, outside APP_DIR and the storage folder [$HOME/yk-cctv-udm-backups]
#   BACKUP_KEEP          number of complete backups to keep; older ones made by these scripts are removed [5]
#   FORCE_RESET          1 = discard tracked-file edits made on this server (saved as a patch first) [0]
#   ALLOW_MODEL_CHANGE   1 = accept a release that changes models/ or config/uniform_config.yaml [0]
#   FORCE_RESTART        1 = restart even while jobs are queued or running [0]
#   ALLOW_UNHEALTHY      1 = deploy although a running app does not answer its health check [0]
#   ALLOW_DEFAULT_API_URL 1 = build with a missing / localhost NEXT_PUBLIC_API_URL (SSH-tunnel use only) [0]
#   ALLOW_ROOT           1 = allow running as root [0]
#
set -euo pipefail
set -o errtrace            # the ERR trap below also fires inside functions

# ------------------------------------------------------------------------------------------- config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(realpath -m "${APP_DIR:-$(dirname "$SCRIPT_DIR")}")"
REPO_URL="${REPO_URL:-https://github.com/Shree-108/CCTV_UDM.git}"
BRANCH="${BRANCH:-publish/cctv-udm}"
REF="${REF:-origin/$BRANCH}"
PM2_APP_NAME="${PM2_APP_NAME:-yk-cctv-udm}"
APP_PORT="${APP_PORT:-8010}"
WEB_PORT="${WEB_PORT:-3010}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VARIANT="${TORCH_VARIANT:-cpu}"
INSTALL_DEV_DEPS="${INSTALL_DEV_DEPS:-0}"
BACKUP_DIR="$(realpath -m "${BACKUP_DIR:-$HOME/yk-cctv-udm-backups}")"
BACKUP_KEEP="${BACKUP_KEEP:-5}"
FORCE_RESET="${FORCE_RESET:-0}"
ALLOW_MODEL_CHANGE="${ALLOW_MODEL_CHANGE:-0}"
FORCE_RESTART="${FORCE_RESTART:-0}"
ALLOW_UNHEALTHY="${ALLOW_UNHEALTHY:-0}"
ALLOW_DEFAULT_API_URL="${ALLOW_DEFAULT_API_URL:-0}"
ALLOW_ROOT="${ALLOW_ROOT:-0}"

API_NAME="$PM2_APP_NAME-api"
WEB_NAME="$PM2_APP_NAME-web"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
STORAGE_DIR=""; DB_PATH=""; UNIFORM_CONFIG=""; LAST_BACKUP=""
PREV_COMMIT=""

# ------------------------------------------------------------------------------------------ helpers
say()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; if [[ -n "$PREV_COMMIT" ]]; then rollback_hint; fi; exit 1; }
rollback_hint() {
  echo "    The code is now at $(git -C "$APP_DIR" rev-parse --short HEAD); before this deploy it was ${PREV_COMMIT:0:7}." >&2
  echo "    Roll the code back:  REF=$PREV_COMMIT bash deploy/deploy.sh" >&2
  echo "    Data backup of this run: ${LAST_BACKUP:-none} (restore steps in its MANIFEST.txt)" >&2
}
trap 'echo "ERROR: \"${BASH_COMMAND}\" failed (line ${LINENO})." >&2; if [[ -n "$PREV_COMMIT" ]]; then rollback_hint; fi' ERR

# --- shared with deploy.sh: keep these functions IDENTICAL in both scripts ----------------------------

pyexec() {   # the app's Python when the venv exists, else the system python3 (snippets use the stdlib only)
  if [[ -x "$APP_DIR/.venv/bin/python" ]]; then "$APP_DIR/.venv/bin/python" "$@"; else python3 "$@"; fi
}

require_tools() {
  local t
  for t in "$@"; do
    command -v "$t" >/dev/null 2>&1 || die "'$t' is not installed. Install it: sudo apt-get install -y $t"
  done
}

check_settings() {
  [[ "$BACKUP_KEEP" =~ ^[1-9][0-9]*$ ]] || die "BACKUP_KEEP must be a whole number >= 1 (got '$BACKUP_KEEP')"
  [[ "$TORCH_VARIANT" == "cpu" || "$TORCH_VARIANT" == "cuda" ]] || die "TORCH_VARIANT must be cpu or cuda"
  [[ "$APP_PORT" =~ ^[0-9]+$ && "$WEB_PORT" =~ ^[0-9]+$ ]] || die "APP_PORT and WEB_PORT must be numbers"
  case "$BACKUP_DIR/" in
    "$APP_DIR"/*) die "BACKUP_DIR ($BACKUP_DIR) must be outside APP_DIR ($APP_DIR)" ;;
  esac
}

repo_id() {   # git URL -> host/owner/repo (https and ssh forms compare equal)
  local u="$1"
  u="${u%.git}"; u="${u%/}"; u="${u#*://}"; u="${u#*@}"; u="${u/:/\/}"
  echo "${u,,}"
}

python_ok() {   # $1 = interpreter; the pinned packages (deploy/constraints.txt) need Python 3.12 - 3.14
  "$1" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] <= (3, 14) else 1)' 2>/dev/null
}

load_node() {
  # nvm is not loaded in non-interactive shells: load it when present
  if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    set +u; # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh"; nvm use --silent default >/dev/null 2>&1 || true; set -u
  fi
}

node_ok() {
  command -v node >/dev/null 2>&1 || return 1
  node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>20||(a===20&&b>=9)?0:1)'
}

require_pm2() {
  command -v pm2 >/dev/null 2>&1 || die "pm2 is not installed. Install it for this user, then re-run:
    npm install -g pm2
  (with nvm, no sudo is needed). Then: pm2 startup   (prints the command that starts pm2 on boot)"
  if [[ "$(dirname "$(command -v pm2)")" != "$(dirname "$(command -v node)")" ]]; then
    warn "pm2 ($(command -v pm2)) belongs to a different Node.js than the active one ($(command -v node)).
  After changing the Node version: npm install -g pm2 && pm2 update, then run 'pm2 startup' again."
  fi
}

resolve_paths() {   # where the app really keeps its data: process env first, then .env, then the defaults
  local line
  STORAGE_DIR=""; DB_PATH=""; UNIFORM_CONFIG=""
  while IFS= read -r line; do
    line="${line%$'\r'}"                     # tolerate CRLF output
    case "$line" in
      STORAGE_DIR=*)    STORAGE_DIR="${line#*=}" ;;
      DB_PATH=*)        DB_PATH="${line#*=}" ;;
      UNIFORM_CONFIG=*) UNIFORM_CONFIG="${line#*=}" ;;
    esac
  done < <(pyexec - "$APP_DIR" <<'PY'
import os, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
env = {}
dotenv = root / ".env"
if dotenv.exists():                               # same precedence as python-dotenv (override=False)
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        else:
            v = v.split(" #", 1)[0].strip()
        env[k] = v
get = lambda k: os.environ.get(k) or env.get(k) or ""
path = lambda v, default: (lambda p: p if p.is_absolute() else root / p)(Path(v) if v else default).resolve()
storage = path(get("SU_STORAGE_DIR"), root / "storage")    # relative paths: the apps run with cwd = APP_DIR
print(f"STORAGE_DIR={storage.as_posix()}")
print(f"DB_PATH={path(get('SU_DB_PATH'), storage / 'sightline_uniform.db').as_posix()}")
print(f"UNIFORM_CONFIG={path(get('SU_UNIFORM_CONFIG'), root / 'config' / 'uniform_config.yaml').as_posix()}")
PY
)
  [[ -n "$STORAGE_DIR" && -n "$DB_PATH" && -n "$UNIFORM_CONFIG" ]] \
    || die "Could not work out the storage / database / model-config paths (check $APP_DIR/.env)."
  case "$BACKUP_DIR/" in
    "$STORAGE_DIR"/*) die "BACKUP_DIR ($BACKUP_DIR) must be outside the storage folder ($STORAGE_DIR)" ;;
  esac
}

ensure_runtime_dirs() {
  # mkdir -p only: existing folders and their contents are never deleted, emptied or recreated
  local d
  for d in "$STORAGE_DIR" "$STORAGE_DIR/uploads" "$STORAGE_DIR/analyses" "$STORAGE_DIR/extractions" \
           "$STORAGE_DIR/exports" "$(dirname "$DB_PATH")" "$APP_DIR/models/person" "$APP_DIR/models/uniform/eval"; do
    mkdir -p "$d"
  done
}

install_python_deps() {
  local py="$APP_DIR/.venv/bin/python" cons="$APP_DIR/deploy/constraints.txt" req pins
  [[ -f "$cons" ]] || die "deploy/constraints.txt is missing: it pins the package versions the tests passed with."
  req="$(mktemp)"; pins="$(mktemp)"
  if [[ "$INSTALL_DEV_DEPS" == "1" ]]; then
    cp "$APP_DIR/requirements.txt" "$req"
  else  # the backend never imports these: tests / training only
    grep -Ev '^[[:space:]]*(pytest|httpx|scikit-learn)([^A-Za-z0-9_-]|$)' "$APP_DIR/requirements.txt" > "$req"
  fi
  "$py" -m pip install --quiet --upgrade pip
  if [[ "$TORCH_VARIANT" == "cpu" ]]; then
    say "Installing CPU-only PyTorch (no CUDA libraries) at the pinned version"
    "$py" -m pip install --quiet --index-url "$TORCH_CPU_INDEX" -c "$cons" torch torchvision
    # pin the exact +cpu builds so the requirements below cannot swap them for the CUDA builds on PyPI
    "$py" -m pip freeze | grep -E '^(torch|torchvision)==' > "$pins"
    "$py" -m pip install --quiet -r "$req" -c "$cons" -c "$pins" --extra-index-url "$TORCH_CPU_INDEX"
  else
    "$py" -m pip install --quiet -r "$req" -c "$cons"
  fi
  rm -f "$req" "$pins"
}

verify_model() {   # read-only: the model file must match the SHA-256 recorded in the model config
  (cd "$APP_DIR" && .venv/bin/python -m training.register_model --check) \
    || die "The uniform model does not match the SHA-256 in the model config (or is missing).
  Restore the approved models/uniform_best.pt - these scripts never replace or re-register a model."
}

verify_python_stack() {
  say "Checking that the backend packages import"
  EXPECT_CPU="$([[ "$TORCH_VARIANT" == "cpu" ]] && echo 1 || echo 0)" "$APP_DIR/.venv/bin/python" - <<'PY'
import os, sys
import cv2, fastapi, torch, torchvision, ultralytics, uvicorn
print(f"    python {sys.version.split()[0]} | torch {torch.__version__} (CUDA build: {torch.version.cuda or 'no'})"
      f" | torchvision {torchvision.__version__} | ultralytics {ultralytics.__version__} | opencv {cv2.__version__}")
if os.environ.get("EXPECT_CPU") == "1" and torch.version.cuda:
    sys.exit("torch is a CUDA build although TORCH_VARIANT=cpu")
PY
  say "Checking the uniform model against its config (read-only)"
  verify_model
}

build_frontend() {
  say "Installing dashboard dependencies (npm ci, from package-lock.json)"
  (cd "$APP_DIR" && npm ci --include=dev --no-audit --no-fund)
  say "Building the dashboard (npm run build)"
  (cd "$APP_DIR" && NEXT_TELEMETRY_DISABLED=1 npm run build)
}

env_value() {   # last value of KEY ($2) in FILE ($1), quotes removed; empty when unset. Never printed.
  [[ -f "$1" ]] || return 0
  sed -n "s/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}$2[[:space:]]*=[[:space:]]*//p" "$1" | tail -n 1 \
    | sed -e 's/[[:space:]]*$//' -e "s/^[\"']//" -e "s/[\"']\$//"
}

check_env() {   # $1 = strict (stop on problems) | warn ; prints key names and resolved paths, never values
  local url problem=""
  url="$(env_value "$APP_DIR/.env.local" NEXT_PUBLIC_API_URL)"
  if [[ -z "$url" ]]; then
    problem="NEXT_PUBLIC_API_URL is missing or empty in $APP_DIR/.env.local"
  elif [[ ! "$url" =~ ^https?://[^/[:space:]]+(/[^[:space:]]*)?$ ]]; then
    problem="NEXT_PUBLIC_API_URL in .env.local is not an http(s):// URL"
  elif [[ "$url" =~ ^https?://(localhost|127\.0\.0\.1|\[::1\])([:/]|$) ]]; then
    problem="NEXT_PUBLIC_API_URL in .env.local points at this machine (localhost): browsers elsewhere cannot reach it"
  fi
  if [[ -n "$problem" && "$ALLOW_DEFAULT_API_URL" != "1" ]]; then
    local msg="$problem. It is baked into the dashboard at BUILD time. Set the public origin Nginx serves /api/ on:
    NEXT_PUBLIC_API_URL=https://uniform.example.com      (in $APP_DIR/.env.local)
  ALLOW_DEFAULT_API_URL=1 accepts it (SSH-tunnel use only)."
    if [[ "$1" == "strict" ]]; then die "$msg"; else warn "$msg"; fi
  fi
  if [[ ! -f "$APP_DIR/.env" ]]; then
    warn "$APP_DIR/.env is missing (backend). The API starts without it (every key has a default). Keys:
    CORS_ORIGINS          dashboard origin(s) when it is served from another origin than the API
    MAX_IMAGE_UPLOAD_MB / MAX_VIDEO_UPLOAD_MB   upload limits (25 / 1024; Nginx client_max_body_size must allow them)
    SU_DEVICE             cpu | cuda | auto      SU_CAPTURE_SCHEDULER   off disables the capture scheduler
    SU_STORAGE_DIR SU_DB_PATH SU_MODELS_DIR SU_UNIFORM_CONFIG SU_UNIFORM_MODEL   data / model locations
  Template: cp .env.example .env"
  fi
  say "Data locations: storage $STORAGE_DIR | database $DB_PATH | model config $UNIFORM_CONFIG"
}

active_jobs() {   # "<count> <details>" of queued / running jobs, read from the database (read-only)
  if [[ ! -f "$DB_PATH" ]]; then echo "0"; return 0; fi
  pyexec - "$DB_PATH" <<'PY'
import sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=30)
active = {"analyses": "state IN ('queued','running')", "extractions": "state IN ('queued','running')",
          "frame_detection_runs": "state IN ('queued','running')", "captures": "status = 'running'"}
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
total, parts = 0, []
for table, where in active.items():
    if table in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]
        total += n
        if n:
            parts.append(f"{n} {table}")
print(total, ", ".join(parts))
PY
}

check_no_active_jobs() {
  local out n
  out="$(active_jobs)" || die "Could not read the job states from $DB_PATH - not restarting blind."
  out="${out%$'\r'}"; n="${out%% *}"
  [[ "$n" =~ ^[0-9]+$ ]] || die "Could not read the job states from $DB_PATH - not restarting blind."
  if (( n == 0 )); then say "No queued or running jobs"; return 0; fi
  if curl -fsS -m 5 -o /dev/null "http://127.0.0.1:$APP_PORT/api/health" 2>/dev/null; then
    [[ "$FORCE_RESTART" == "1" ]] || die "Jobs are queued or running: ${out#* }. A restart stops them (analyses are
  marked failed, extractions / detection runs interrupted; results saved so far are kept).
  Wait until they finish (pm2 logs $API_NAME), or re-run with FORCE_RESTART=1."
    warn "Restarting with active jobs (${out#* }) because FORCE_RESTART=1"
  else
    warn "The database lists ${out#* } as active, but the API is not running: they are left over from an earlier
  stop and the API marks them failed / interrupted when it starts."
  fi
}

check_running_health() {   # before changing anything: do the apps that pm2 runs answer?
  local bad=()
  if [[ "$(pm2_status "$API_NAME")" == "online" ]] \
     && ! curl -fsS -m 10 -o /dev/null "http://127.0.0.1:$APP_PORT/api/health"; then
    bad+=("$API_NAME (http://127.0.0.1:$APP_PORT/api/health)")
  fi
  if [[ "$(pm2_status "$WEB_NAME")" == "online" ]] \
     && ! curl -fsS -m 10 -o /dev/null "http://127.0.0.1:$WEB_PORT/"; then
    bad+=("$WEB_NAME (http://127.0.0.1:$WEB_PORT/)")
  fi
  if (( ${#bad[@]} )); then
    [[ "$ALLOW_UNHEALTHY" == "1" ]] || die "Running but not answering: ${bad[*]}. Look at 'pm2 logs' first;
  ALLOW_UNHEALTHY=1 continues anyway (e.g. to ship the fix)."
    warn "Continuing although ${bad[*]} did not answer (ALLOW_UNHEALTHY=1)"
  fi
}

pm2_clean() {   # pm2 records the caller's environment: keep these scripts' own settings out of the apps
  env -u FORCE_RESET -u FORCE_RESTART -u ALLOW_MODEL_CHANGE -u ALLOW_UNHEALTHY -u ALLOW_DEFAULT_API_URL \
      -u ALLOW_ROOT -u REF -u BACKUP_KEEP -u BACKUP_DIR pm2 "$@"
}

api_spec() { echo "$APP_DIR/.venv/bin/python|-m uvicorn backend.main:app --host 127.0.0.1 --port $APP_PORT|$APP_DIR|none"; }
web_spec() { echo "$APP_DIR/node_modules/next/dist/bin/next|start --hostname $WEB_HOST --port $WEB_PORT|$APP_DIR|$(command -v node)"; }

pm2_start_api() {   # ONE process, fork mode, loopback only - never add -i / cluster mode (in-process job queue + SQLite)
  pm2_clean start "$APP_DIR/.venv/bin/python" --name "$API_NAME" --cwd "$APP_DIR" --interpreter none \
    --kill-timeout 30000 --exp-backoff-restart-delay 1000 --time \
    -- -m uvicorn backend.main:app --host 127.0.0.1 --port "$APP_PORT"
}

pm2_start_web() {
  pm2_clean start "$APP_DIR/node_modules/next/dist/bin/next" --name "$WEB_NAME" --cwd "$APP_DIR" \
    --interpreter "$(command -v node)" --kill-timeout 10000 --exp-backoff-restart-delay 1000 --time \
    -- start --hostname "$WEB_HOST" --port "$WEB_PORT"
}

pm2_field() {   # $1 = name, $2 = status | spec ; "missing" when the app is not registered
  pm2 jlist 2>/dev/null | node -e '
    let s = ""; process.stdin.on("data", d => s += d).on("end", () => {
      let list = []; try { list = JSON.parse(s.slice(s.indexOf("["))) } catch (e) {}
      const p = list.find(x => x.name === process.argv[1]);
      if (!p) return console.log("missing");
      const e = p.pm2_env, args = Array.isArray(e.args) ? e.args.join(" ") : (e.args || "");
      console.log(process.argv[2] === "status" ? e.status
        : [e.pm_exec_path, args, e.pm_cwd, e.exec_interpreter].join("|")) })' "$1" "$2"
}

pm2_status() { pm2_field "$1" status; }

ensure_app() {   # $1 = name, $2 = start function, $3 = wanted spec. Restart = a short interruption.
  local status
  status="$(pm2_status "$1")"
  if [[ "$status" == "missing" ]]; then
    say "Starting $1"; "$2"
  elif [[ "$(pm2_field "$1" spec)" != "$3" ]]; then
    say "$1: start command changed (path, port or Node version) - registering it again"
    pm2 delete "$1" >/dev/null; "$2"
  elif [[ "$status" == "online" ]]; then
    say "Restarting $1 (one process: a few seconds without it, not a zero-downtime reload)"
    pm2_clean restart "$1" >/dev/null
  else
    say "$1 is '$status' - starting it again"
    pm2 delete "$1" >/dev/null; "$2"
  fi
}

wait_healthy() {
  local i
  for i in $(seq 1 90); do
    if curl -fsS -o /dev/null "http://127.0.0.1:$APP_PORT/api/health" 2>/dev/null \
       && curl -fsS -o /dev/null "http://127.0.0.1:$WEB_PORT/" 2>/dev/null; then
      say "Healthy: API http://127.0.0.1:$APP_PORT/api/health and dashboard http://127.0.0.1:$WEB_PORT/"; return 0
    fi
    sleep 2
  done
  return 1
}

backup_list() {   # complete backups made by these scripts, newest first (names start with a timestamp)
  local d
  shopt -s nullglob
  for d in "$BACKUP_DIR"/*/; do
    if [[ -f "$d/.yk-backup" && -f "$d/COMPLETE" ]]; then echo "${d%/}"; fi
  done | sort -r
  shopt -u nullglob
}

prune_backups() {   # keep the newest BACKUP_KEEP complete backups; only folders marked by these scripts
  local d i=0 all
  all="$(backup_list)"
  while IFS= read -r d; do
    [[ -n "$d" ]] || continue
    i=$((i + 1))
    if (( i > BACKUP_KEEP )) && [[ "$d" == "$BACKUP_DIR"/* && -f "$d/.yk-backup" ]]; then
      say "Removing old backup $(basename "$d") (BACKUP_KEEP=$BACKUP_KEEP)"
      rm -rf -- "$d"
    fi
  done <<< "$all"
  shopt -s nullglob
  for d in "$BACKUP_DIR"/*/; do
    if [[ -f "$d/.yk-backup" && ! -f "$d/COMPLETE" ]]; then
      warn "Incomplete backup kept for inspection (not counted, not removed): ${d%/}"
    fi
  done
  shopt -u nullglob
}

backup_data() {
  # Everything that is not in git: the storage folder (rsync; files unchanged since the previous backup
  # are hard links to it, so they take no extra space), the database (SQLite backup API = consistent copy
  # while the API writes, then PRAGMA integrity_check), the model config and the env files.
  # Reads application data only - never changes or deletes it. Any copy error stops the run.
  local stamp dest prev need=0 free rc=0 files=0 env_names="" f db_line="none (no database yet)"
  require_tools rsync
  stamp="$(date +%Y%m%d-%H%M%S)-$(git -C "$APP_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
  dest="$BACKUP_DIR/$stamp"
  mkdir -p "$BACKUP_DIR"; chmod 700 "$BACKUP_DIR"
  [[ ! -e "$dest" ]] || die "Backup folder $dest already exists"
  prev="$(backup_list)"; prev="${prev%%$'\n'*}"
  local -a rs=(-a --numeric-ids)
  if [[ -n "$prev" && -d "$prev/storage" ]]; then rs+=("--link-dest=$prev/storage"); fi
  if [[ "$DB_PATH" == "$STORAGE_DIR"/* ]]; then        # the live database is copied with the backup API below
    local db_rel="${DB_PATH#"$STORAGE_DIR"/}"
    rs+=("--exclude=/$db_rel" "--exclude=/$db_rel-wal" "--exclude=/$db_rel-shm" "--exclude=/$db_rel-journal")
  fi
  if [[ -d "$STORAGE_DIR" ]]; then                     # only new / changed files need space
    need=$(rsync "${rs[@]}" --dry-run --stats "$STORAGE_DIR/" "$dest/storage/" \
           | awk -F': ' '/^Total transferred file size/ {gsub(/[^0-9]/, "", $2); print $2}')
    need=$(( ${need:-0} / 1024 ))
  fi
  if [[ -f "$DB_PATH" ]]; then need=$(( need + $(du -sk "$DB_PATH" | cut -f1) )); fi
  free=$(df -Pk "$BACKUP_DIR" | awk 'NR==2 {print $4}')
  (( free > need + 1048576 )) || die "Not enough disk for the backup: ~$((need / 1024)) MB needed + 1 GB margin, \
$((free / 1024)) MB free in $BACKUP_DIR. Free space, lower BACKUP_KEEP or set BACKUP_DIR; nothing was changed."

  mkdir -p "$dest"
  echo "Backup made by deploy/server-setup.sh or deploy/deploy.sh" > "$dest/.yk-backup"
  if [[ -d "$STORAGE_DIR" ]]; then
    say "Backing up $STORAGE_DIR (~$((need / 1024)) MB new or changed${prev:+ since $(basename "$prev")})"
    rsync "${rs[@]}" "$STORAGE_DIR/" "$dest/storage/" || rc=$?
    if (( rc == 24 )); then
      warn "Some files vanished while being copied (deleted by the app meanwhile) - they are not in the backup"
    elif (( rc != 0 )); then
      die "rsync failed (exit $rc): the backup $dest is INCOMPLETE. Nothing was changed; fix the cause and re-run."
    fi
    files=$(find "$dest/storage" -type f | wc -l)
  fi
  if [[ -f "$DB_PATH" ]]; then
    mkdir -p "$dest/database"
    pyexec - "$DB_PATH" "$dest/database/$(basename "$DB_PATH")" <<'PY' || die "The database backup failed - nothing was changed."
import sqlite3, sys
src = sqlite3.connect(sys.argv[1], timeout=30)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)                          # consistent snapshot, including the WAL, while the API writes
result = dst.execute("PRAGMA integrity_check").fetchone()[0]
dst.close(); src.close()
if result != "ok":
    sys.exit(f"integrity_check of the copy: {result}")
PY
    db_line="$DB_PATH -> database/$(basename "$DB_PATH") (integrity_check ok)"
  fi
  mkdir -p "$dest/config"
  cp -p "$UNIFORM_CONFIG" "$dest/config/" 2>/dev/null || warn "Model config $UNIFORM_CONFIG not found - not backed up"
  for f in .env .env.local; do
    if [[ -f "$APP_DIR/$f" ]]; then cp -p "$APP_DIR/$f" "$dest/config/$f"; env_names+="$f "; fi
  done
  {
    echo "created       $(date -Is)"
    echo "app commit    $(git -C "$APP_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "storage       $STORAGE_DIR -> storage/ ($files files; unchanged files hard-linked to ${prev:-nothing})"
    echo "database      $db_line"
    echo "model config  $UNIFORM_CONFIG -> config/$(basename "$UNIFORM_CONFIG")"
    echo "env files     ${env_names:-none}-> config/"
    echo "restore       stop both pm2 apps; rsync -a storage/ <storage>/ ; copy database/* to the database path;"
    echo "              copy config/* back; start the apps. Nothing is restored automatically."
  } > "$dest/MANIFEST.txt"
  touch "$dest/COMPLETE"
  chmod -R go-rwx "$dest"
  LAST_BACKUP="$dest"
  say "Backup complete: $dest"
  sed 's/^/    /' "$dest/MANIFEST.txt"
  prune_backups
}

# --- end of shared functions ------------------------------------------------------------------------

no_tracked_runtime_data() {
  local tracked rel_storage=""
  if [[ "$STORAGE_DIR" == "$APP_DIR"/* ]]; then rel_storage="${STORAGE_DIR#"$APP_DIR"/}"; fi
  tracked="$(git ls-files -- ${rel_storage:+"$rel_storage"} .env .env.local)"
  [[ -z "$tracked" ]] || die "Runtime data is tracked by git and would be overwritten by the reset:
$tracked
  Remove it from git in the repository (git rm --cached) before deploying."
}

review_reset() {
  local edited added f collide=()
  edited="$(git status --porcelain --untracked-files=no)"
  if [[ -n "$edited" ]]; then
    warn "Tracked files were edited on this server; git reset --hard would discard these changes:"
    echo "$edited" >&2
    [[ "$FORCE_RESET" == "1" ]] || die "Not deploying. Review them (git diff), keep what is needed in the repository,
  then re-run - or re-run with FORCE_RESET=1 to discard them (a patch is saved in the backup first)."
  fi
  if git diff --name-only HEAD "$REF" -- models config/uniform_config.yaml | grep -q .; then
    warn "This release changes model files or the model configuration:"
    git diff --stat HEAD "$REF" -- models config/uniform_config.yaml >&2
    [[ "$ALLOW_MODEL_CHANGE" == "1" ]] || die "Not deploying. Review the change (git diff HEAD $REF -- models config/uniform_config.yaml);
  re-run with ALLOW_MODEL_CHANGE=1 to accept it."
  fi
  added="$(git diff --name-only --diff-filter=A HEAD "$REF")"
  while IFS= read -r f; do
    if [[ -n "$f" && -e "$f" ]] && ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then collide+=("$f"); fi
  done <<< "$added"
  (( ${#collide[@]} == 0 )) || die "These untracked files would be replaced by files of the release:
$(printf '    %s\n' "${collide[@]}")
  Move them elsewhere first."
}

# ---------------------------------------------------------------------------------------------- main
[[ "$EUID" -ne 0 || "$ALLOW_ROOT" == "1" ]] || die "Run as the deploy user, not root (ALLOW_ROOT=1 overrides)."
check_settings
[[ -d "$APP_DIR/.git" ]] || die "$APP_DIR is not a git checkout. Run deploy/server-setup.sh first."
cd "$APP_DIR"
[[ -x .venv/bin/python ]] || die "$APP_DIR/.venv is missing. Run deploy/server-setup.sh first."
python_ok .venv/bin/python || die ".venv uses $(.venv/bin/python --version 2>&1); Python 3.12 - 3.14 is needed. Run deploy/server-setup.sh."
have="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
want="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "$have")"
[[ "$have" == "$want" ]] || die ".venv was built with Python $have but $PYTHON_BIN is now $want. Run deploy/server-setup.sh (it rebuilds .venv)."
load_node
node_ok || die "Node.js >= 20.9 not found (nvm in $NVM_DIR?). Run deploy/server-setup.sh first."
require_pm2
require_tools rsync curl git
[[ "$WEB_HOST" == "127.0.0.1" ]] || warn "WEB_HOST=$WEB_HOST: keep port $WEB_PORT closed in the EC2 security group."

say "Checking repository, branch and commit"
[[ "$(repo_id "$(git remote get-url origin)")" == "$(repo_id "$REPO_URL")" ]] \
  || die "The checkout's origin is $(git remote get-url origin), not $REPO_URL. Not changing remotes - check APP_DIR / REPO_URL."
current="$(git rev-parse --abbrev-ref HEAD)"
[[ "$current" == "$BRANCH" ]] || die "The checkout is on '$current', not '$BRANCH'. Switch with: git checkout $BRANCH (or set BRANCH)."
git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null \
  || die "Branch $BRANCH does not exist on origin ($(git remote get-url origin)) or origin is not reachable. Push it first."
git fetch origin "$BRANCH"
git rev-parse --verify --quiet "$REF^{commit}" >/dev/null || die "REF=$REF is not a known commit."
git merge-base --is-ancestor "$REF^{commit}" "origin/$BRANCH" \
  || die "REF=$REF is not a commit of origin/$BRANCH - only commits of the deployment branch are deployed."
target="$(git rev-parse "$REF^{commit}")"
echo "    origin   $(git remote get-url origin)"
echo "    branch   $BRANCH"
echo "    current  $(git log -1 --format='%h %s' HEAD)"
echo "    target   $(git log -1 --format='%h %s' "$target")"
if [[ "$(git rev-parse HEAD)" == "$target" ]]; then
  say "Already at the target commit - rebuilding and restarting anyway"
elif git merge-base --is-ancestor HEAD "$target"; then
  echo "    new commits:"; git log --format='      %h %s' "HEAD..$target" | head -n 30
else
  warn "The target is not ahead of the current commit (roll-back or rewritten history): these commits go away:"
  git log --format='      %h %s' "$target..HEAD" | head -n 30 >&2
fi

say "Running the safety checks"
resolve_paths
no_tracked_runtime_data
review_reset
verify_model                       # the model in use now matches its checksum (nothing changed it unnoticed)
check_env strict
check_no_active_jobs
check_running_health

backup_data
if [[ "$FORCE_RESET" == "1" && -n "$(git status --porcelain --untracked-files=no)" ]]; then
  git diff HEAD > "$LAST_BACKUP/local-changes.patch"
  chmod go-rwx "$LAST_BACKUP/local-changes.patch"
  say "Saved the discarded server edits as local-changes.patch in the backup"
fi

PREV_COMMIT="$(git rev-parse HEAD)"
say "Resetting the code to $(git rev-parse --short "$target") (git reset --hard; untracked and ignored files stay)"
git reset --hard "$target"

ensure_runtime_dirs
say "Updating backend dependencies (pinned: deploy/constraints.txt, TORCH_VARIANT=$TORCH_VARIANT)"
install_python_deps
verify_python_stack
build_frontend

ensure_app "$API_NAME" pm2_start_api "$(api_spec)"
ensure_app "$WEB_NAME" pm2_start_web "$(web_spec)"
pm2 save

if ! wait_healthy; then
  pm2 logs "$API_NAME" --lines 40 --nostream || true
  pm2 logs "$WEB_NAME" --lines 40 --nostream || true
  die "The apps did not become healthy within 3 minutes - see the logs above."
fi

say "Last 20 log lines"
pm2 logs "$API_NAME" --lines 20 --nostream || true
pm2 logs "$WEB_NAME" --lines 20 --nostream || true
say "Deployed $(git rev-parse --short HEAD) of $BRANCH (previous: ${PREV_COMMIT:0:7}; backup: $LAST_BACKUP)"
echo "    Roll the code back: REF=$PREV_COMMIT bash deploy/deploy.sh"
