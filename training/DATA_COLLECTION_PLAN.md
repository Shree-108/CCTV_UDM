# Uniform classifier — data collection plan

**Goal:** collect real, labelled CCTV examples so the uniform classifier can learn the difference between the
approved black-and-white broad-striped shirt and everything else, and so its accuracy can be measured honestly.

## Why this is needed

Measured on 2026-10-28:

- The current model (v0.2.0, experimental) learned from about **4 uniform people and 4 non-uniform groups**.
  138 of its 141 uniform crops are frames of **one** staff member; 78 of its 83 non-uniform crops are frames of
  **one** customer.
- The "9,000-image" dataset is 900–1,800 augmented copies each of **6 phone photos of the same 3 uniform
  wearers**. It has no non-uniform examples at all.
- As a result the model leans towards "uniform": on its own non-uniform training crops it is right only 39
  times out of 83. In a sample of 40 real store crops it called "uniform", about half were light-blue
  patterned or plain white shirts, mostly scored 0.94–1.00.
- There is **no independent test set**, so no accuracy number can be trusted yet.

More copies of the same few people will not fix this. **Many different real people** are needed — above all
customers and others who are *not* in uniform.

## What to collect

### Footage

| Requirement | Minimum | Why |
|---|---|---|
| Recording sessions | 20 sessions of 30–60 min | Each session gives different people, positions and lighting |
| Different days | 10 days | Same day = same people and light; days must differ |
| Cameras / stores | every camera that will be monitored (at least 2 if possible) | The model must work on each camera's angle |
| Time of day | morning, afternoon, evening (lights on) | Light changes the stripe contrast |
| Busy and quiet periods | both | Crowds cause overlap and hidden people |

Use the store's normal CCTV recordings. **Do not** record anything specially without the store's usual
authorisation and signage.

### People

| Class | Minimum distinct people | Notes |
|---|---|---|
| **Uniform** (approved striped shirt) | every staff member who wears it, on at least 5 different days each; ideally 10+ people | front, back, side, bending, sitting, near and far |
| **Non-uniform** | 100+ different people | customers and anyone not in the approved shirt |
| — of which look-alikes | 20+ people | plain white, grey, light blue, plain black, narrow stripes, horizontal stripes, checks, other black-and-white patterns |
| **Uncertain** | 100+ crops | shirt hidden by a counter or another person, heavy blur, too small, very dark |

If the store does not naturally have enough look-alike shirts, a **"look-alike session"** is the fastest
fix: with their consent, ask staff or volunteers to wear similar-looking non-approved shirts and walk
through each camera's view. This is real footage, not synthetic data.

### Held-out test set (decide before labelling)

Choose **at least 3 sessions, recorded on days not used for anything else**, as the test set. Freeze them:
they are never used for training or for choosing thresholds. Aim for at least 50 uniform and 150
non-uniform people across them, including look-alikes. Without this, there is no way to prove that a new
model is better.

## How to do it

### 1. Export the footage

- The CP Plus client (KVMSProLite) is installed on this PC. Export recordings from its playback / backup
  screen. The app accepts MP4, AVI, MOV, MKV, M4V and WEBM. If the export is in CP Plus's own format
  (for example `.dav`), export as MP4 instead, or ask for help converting it.
- Name each file `STORE_CAMERA_YYYYMMDD_HHMM.mp4`, for example `STORE-001_CAM-01_20261105_1100.mp4`.
- Keep a session log (`data/incoming/sessions.csv`): file name, store, camera, date, start time, lighting,
  busy/quiet, notes, and whether the session is **train**, **validation** or **test**.

### 2. Cut out one crop per person

The existing tool runs the production person detector and tracker over a whole video and saves the
upper-body crop exactly as the app sees it. Crops of the same person share a track number, so they stay
grouped:

```bash
python -m training.prepare_dataset extract --source "path/to/STORE-001_CAM-01_20261105_1100.mp4" --every 25 --out data/incoming/STORE-001_CAM-01_20261105_1100
```

`--every 25` uses one frame per second of 25 fps footage. Use one output folder per session.

The 172 crops already prepared from analysed screenshots (`runs/label_review/20261028-192350/`) can be
labelled too, but they come from only 4 videos, so they can only go into training, never the test set.

### 3. Label

Inside each session folder, sort every crop into `uniform/`, `non_uniform/`, `uncertain/` or `exclude/`.

| Label | Use when |
|---|---|
| `uniform` | the approved black-and-white **broad vertical** stripe pattern is clearly visible on this person's own shirt |
| `non_uniform` | this person's shirt is clearly visible and is **not** the approved shirt |
| `uncertain` | you cannot honestly tell: hidden, blurred, too small, too dark, or only a sliver visible |
| `exclude` | not a person, two people in one crop, a duplicate, or footage that should not be used |

Rules:

- Judge **only the person the crop is centred on**. A neighbour's stripes at the edge do not count.
- Never label from the model's prediction. Look at the image.
- When unsure between uniform and non-uniform, choose `uncertain`.
- A second person should re-check all look-alike crops and a random 10% of the rest. Note disagreements in
  `sessions.csv`; they show where the uniform policy needs to be clearer.

### 4. Check before training

```bash
python -m training.audit_dataset --roots data/incoming
```

It counts images per class, finds duplicates and near-duplicates, and checks that no session or person
crosses between train, validation and test. Training starts only after this report has been reviewed.

## What happens next (not started)

When the labelled data passes the audit:

1. The training scripts get a fixed test split by session (today they only make train/validation) and a
   versioned model folder (`models/uniform/versions/v0.3.0/`) with its class map, thresholds, dataset
   version and metrics. The current v0.2.0 file is never overwritten.
2. v0.2.0 and the new candidate are both measured on the same frozen test set: precision, recall and F1 per
   class, false-uniform and false-non-uniform rates, review rate, results on look-alike shirts and per
   camera / lighting.
3. The thresholds are chosen on the **validation** sessions, based on how costly a false "uniform"
   approval is compared with a false violation.
4. The candidate is switched on only after you have reviewed those results. Switching back is a one-line
   change in `config/uniform_config.yaml`.

## Rough effort

| Step | Estimate |
|---|---|
| Export 20 sessions | 2–4 hours, depending on the recorder |
| Crop extraction | runs unattended, roughly real-time or faster on this PC |
| Labelling | grouped by track, roughly a few hundred crops per hour; 5,000 crops ≈ 1–2 days for one person |
| Second check | about a quarter of the labelling time |

These are estimates; the first session will show the real pace.

## Privacy and authorisation

- Customer footage is personal data. Use only footage the business is allowed to use for this purpose, and
  check with whoever handles privacy for the business before collecting (in India this falls under the
  Digital Personal Data Protection Act, 2023).
- Keep footage and crops on this PC or approved storage. Do not upload them to any outside service.
- Decide how long the crops are kept and delete them when they are no longer needed.

## Questions the business must answer

1. **Exact uniform definition:** is it only this one shirt? Are there approved variants (short sleeve, long
   sleeve, different stripe widths, with a logo, jacket over it)? Reference photos of every approved variant
   are needed.
2. **What counts as compliant for a store:** at least one staff member in uniform (`any_uniform`, current
   setting), or nobody out of uniform (`all_uniform`)? Customers are not staff, so `all_uniform` would need
   a way to tell staff from customers.
3. **Which mistake is worse:** approving someone who is not in uniform, or flagging staff who are? This
   decides the thresholds.
