# Deployment guide — AI microservice on AWS EC2 (plan)

Status: **plan.** The service mode, deploy files and S3 code do not exist yet (IMPLEMENTATION_PLAN.md).
Nothing has been deployed. This guide fixes the target so the implementation can be checked against it.

## 1. Sizing (from measurements on 2026-10-30)

Measured on a 2-core / 4-thread laptop CPU without GPU: model load 10.6 s, ~370 MB RAM peak, 0.12–0.53 s
per 1080p screenshot, ~0.35 s to seek one frame. CCTV footage is ~0.9–1.4 GB per camera-hour.

Estimated work for one store-camera-hour with 6 screenshots: reading frames from S3 (seconds if the MP4 can be
seeked remotely; ~5–15 s if the whole file must be downloaded in-region) + ~3 s inference ≈ **10–20 s**. To be
re-measured on the instance before sizing for production.

| Load (camera-hours per hour) | Instance | Workers |
|---|---|---|
| Pilot, ≤ 50 | `t3.large` (2 vCPU, 8 GB) | 1 |
| ≤ 200 | `c7i.xlarge` (4 vCPU, 8 GB) | 2 |
| More | larger `c7i`, or several instances behind a queue (then move SQLite → Postgres) | |

**GPU is not needed** for sampled screenshots: inference is a few seconds per hour of footage; a GPU instance
(`g5`/`g6`) costs several times more and only helps the full-video mode, which stays out of the service.
Disk: 50 GB gp3 (OS, venv ~4 GB, temporary videos, SQLite, logs).

## 2. Network and security

- Private subnet, **no public IP**. Admin backend reaches the service over the VPC.
- Security group `ai-service-sg`: inbound TCP 443 **only from the Admin backend's security group**; no SSH
  from the internet (use SSM Session Manager). Outbound 443 to S3 / AWS APIs.
- **S3 gateway VPC endpoint** for `ap-south-1` so S3 traffic stays inside AWS (no NAT data charges).
- **IAM instance role** (no access keys on the instance):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "ListInputAndOutput", "Effect": "Allow", "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::yk-cctv-prediction",
      "Condition": { "StringLike": { "s3:prefix": ["*/input_data/*", "*/output_data/*"] } } },
    { "Sid": "ReadInput", "Effect": "Allow", "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::yk-cctv-prediction/*/input_data/*" },
    { "Sid": "WriteAndVerifyOutput", "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::yk-cctv-prediction/*/output_data/*" }
  ]
}
```

  No `DeleteObject`, no bucket-policy or ACL permissions. The prefix patterns must be adjusted to the
  confirmed folder layout.
- Bucket: Block Public Access on, default encryption (SSE-S3 or SSE-KMS), bucket policy denying non-TLS
  (`aws:SecureTransport = false`).
- **Credentials shared earlier must be deactivated in IAM and replaced**; the service never needs keys on EC2.
- API keys for the Admin backend: generated per environment, stored in AWS Secrets Manager or SSM Parameter
  Store, given to the service as SHA-256 hashes in its env file.

## 3. Instance setup (Ubuntu 24.04 LTS, Python 3.12)

```bash
sudo apt update && sudo apt install -y python3.12-venv python3-pip git nginx
sudo useradd --system --create-home --home-dir /opt/yk-ai yk-ai
sudo -u yk-ai git clone https://github.com/Shree-108/CCTV_UDM.git /opt/yk-ai/app
cd /opt/yk-ai/app
sudo -u yk-ai python3.12 -m venv .venv
# CPU-only PyTorch first (avoids ~2 GB of CUDA wheels), then the rest
sudo -u yk-ai .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
sudo -u yk-ai .venv/bin/pip install -r requirements.txt -r requirements-service.txt
sudo -u yk-ai .venv/bin/python -m training.register_model --check     # must print MATCH
```

The frontend is not installed on the service instance.

## 4. Configuration — `/etc/yk-ai/ai-service.env` (root:yk-ai, mode 640)

```ini
SU_MODE=service
SU_STORAGE_DIR=/opt/yk-ai/data
SU_DEVICE=cpu
AWS_REGION=ap-south-1
S3_BUCKET=yk-cctv-prediction
S3_ALLOWED_PREFIX_PATTERN=^[0-9]{2}-[0-9]{4}/[0-9]{2}-[0-9]{2}-[0-9]{4}/[A-Za-z0-9._-]+/(input|output)_data/$
SERVICE_API_KEY_HASHES=<sha256-of-key-1>,<sha256-of-key-2>
SERVICE_WORKERS=1
SERVICE_ANALYSIS_TIMEOUT_S=900
SERVICE_TIMEZONE=Asia/Kolkata
```

No AWS keys: boto3 uses the instance role.

## 5. Process management — systemd

`/etc/systemd/system/yk-ai.service`:

```ini
[Unit]
Description=Yatrikart uniform AI service
After=network-online.target

[Service]
User=yk-ai
WorkingDirectory=/opt/yk-ai/app
EnvironmentFile=/etc/yk-ai/ai-service.env
ExecStart=/opt/yk-ai/app/.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8010 --workers 1 --proxy-headers
Restart=on-failure
RestartSec=5
TimeoutStopSec=60
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

`--workers 1` is deliberate: the job queue and loaded models live in the process; concurrency is
`SERVICE_WORKERS` threads inside it. Scale by instance size or by more instances, not Uvicorn workers.

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now yk-ai
sudo journalctl -u yk-ai -f
curl -s http://127.0.0.1:8010/health
```

## 6. HTTPS — nginx in front

nginx listens on 443 with a certificate (ACM certificate on an internal ALB is the alternative), proxies to
`127.0.0.1:8010`, limits the request body (16 KB for `/api/v1`), sets timeouts, and forwards `X-Request-ID`.

## 7. Monitoring and logs

- Logs: journald → CloudWatch agent (log group `/yk-ai/service`); structured JSON lines with `request_id`
  and `analysis_id`; secrets and presigned URLs never logged.
- Health: ALB / Route 53 / CloudWatch Synthetics on `GET /health` (200 = ok or degraded).
- Alarms: CPU > 80 % for 15 min, disk > 80 %, memory (CloudWatch agent), `failed` analyses per hour.

## 8. Deploy and roll back

```bash
cd /opt/yk-ai/app && sudo -u yk-ai git fetch && sudo -u yk-ai git checkout <release-tag>
sudo -u yk-ai .venv/bin/pip install -r requirements.txt -r requirements-service.txt
sudo systemctl restart yk-ai && curl -s http://127.0.0.1:8010/health
# roll back
sudo -u yk-ai git checkout <previous-tag> && sudo systemctl restart yk-ai
```

Back up `/opt/yk-ai/data/sightline_uniform.db` (daily snapshot of the EBS volume, or copy to S3) before each
deployment. Model rollback is separate: `python -m training.register_model` (activate / rollback) — never
overwrite `models/uniform_best.pt` in place.

## 9. Local development (unchanged)

```bash
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
npm run dev
```

The development server and `npm run dev` are for local use only, not for EC2.
