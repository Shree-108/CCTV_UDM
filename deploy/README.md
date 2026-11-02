# Deploying on EC2 (Ubuntu 24.04 + pm2)

Two scripts, both run **on the server** as the `ubuntu` user (not root):

| Script | When | What you get |
|---|---|---|
| `deploy/server-setup.sh` | once (safe to re-run) | system packages, Node 22 (nvm), pm2, `.venv`, build, two pm2 apps, pm2 boot service, `pm2 list` |
| `deploy/deploy.sh` | every release | checks + backup, `git reset --hard` to the new commit, rebuild, restart, logs, `pm2 list` |

pm2 apps created:

| pm2 name | What | Listens on |
|---|---|---|
| `yk-cctv-udm-api` | FastAPI (one process) | `127.0.0.1:8010` |
| `yk-cctv-udm-web` | Next.js dashboard | `127.0.0.1:3010` |

Both are loopback-only; Nginx (step 5) makes them reachable from a browser.

## 1. Launch the instance

- AMI **Ubuntu Server 24.04 LTS** (Python 3.12 - required; 22.04 is refused).
- Size: at least 2 vCPU / **4 GB RAM** (e.g. `t3.medium`; `t3.large` is more comfortable). Disk: **30 GB+** gp3.
- Security group: SSH (22) from your IP only; HTTP (80) / HTTPS (443) once Nginx is set up.
  **Do not open 8010 or 3010.**

## 2. Get the code onto the server

```bash
ssh -i <key.pem> ubuntu@<server-ip>
sudo apt-get update && sudo apt-get install -y git
git clone --branch publish/cctv-udm https://github.com/Shree-108/CCTV_UDM.git ~/yk-cctv-udm
cd ~/yk-cctv-udm
```

Private repository: add an SSH deploy key to GitHub and clone `git@github.com:Shree-108/CCTV_UDM.git`
instead (then also run the scripts with `REPO_URL=git@github.com:Shree-108/CCTV_UDM.git`). Never put a token in the URL.

## 3. Create the settings files (before setup, because the dashboard bakes the API address into its build)

Replace `<host>` with the server's domain or public IP - the address people will type in the browser.

```bash
cat > .env <<'EOF'
CORS_ORIGINS=http://<host>
MAX_VIDEO_UPLOAD_MB=1024
EOF
cat > .env.local <<'EOF'
NEXT_PUBLIC_API_URL=http://<host>
NEXT_PUBLIC_APP_MODULE=uniform
EOF
chmod 600 .env .env.local
```

All other keys are optional; see `.env.example`.

## 4. Run the setup

```bash
bash deploy/server-setup.sh
```

Takes 10-20 minutes the first time (PyTorch CPU, npm, Next build). It asks for your sudo password if needed
(apt and the pm2 boot service). At the end it prints a summary and `pm2 list` with both apps `online`.

Afterwards, in a **new** SSH session (so nvm puts `node` / `pm2` on PATH):

```bash
pm2 list                         # both apps
pm2 logs yk-cctv-udm-api         # live logs (Ctrl+C to leave)
pm2 restart yk-cctv-udm-api      # manual restart
systemctl status pm2-ubuntu      # the boot service that restores the apps after a reboot
curl -s http://127.0.0.1:8010/api/health
```

Options (prefix the command): `INSTALL_PM2=0` (do not install pm2, only check), `PM2_STARTUP=0` (no boot
service), `BRANCH=...`, `APP_DIR=...`. Every option is listed at the top of each script.

## 5. Nginx (makes the dashboard reachable; not done by the scripts)

```bash
sudo apt-get install -y nginx
sudo tee /etc/nginx/sites-available/yk-cctv-udm >/dev/null <<'EOF'
server {
    listen 80;
    server_name <host>;
    client_max_body_size 1024m;          # >= MAX_VIDEO_UPLOAD_MB

    location /api/ {
        proxy_pass http://127.0.0.1:8010;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 600s;
        proxy_request_buffering off;
    }
    location / {
        proxy_pass http://127.0.0.1:3010;
        proxy_set_header Host $host;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/yk-cctv-udm /etc/nginx/sites-enabled/yk-cctv-udm
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

Open `http://<host>`. **The app has no login** - restrict the security group to your office IPs (or add
Nginx basic auth / HTTPS with certbot) before real use.

## 6. Every release

On your PC: commit and push to `publish/cctv-udm`. On the server:

```bash
cd ~/yk-cctv-udm && bash deploy/deploy.sh
```

It stops without changing anything if a check fails (jobs running, edited files on the server, model
changed, unhealthy app, backup failed) and tells you what to do. Roll the code back with the
`REF=<commit> bash deploy/deploy.sh` line it prints. Backups are in `~/yk-cctv-udm-backups`
(each has a `MANIFEST.txt` with restore steps; restoring data is manual).
