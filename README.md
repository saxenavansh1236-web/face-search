# Face Search Tool

A FastAPI service that indexes known faces (with consent, from a
closed dataset) and lets you search a new photo against them. Given a
new photo, it tells you whether it matches someone already indexed —
even if the new photo is a different picture of that same person (a
different angle, lighting, or day) — and rejects photos of people who
aren't in the dataset.

The project includes these surfaces:
- **The API** (`/index-face/`, `/search-face/`, etc.) — gated behind
  user login/registration, role-based permissions (RBAC), and enforced
  via a real auth dependency (session cookie OR JWT bearer token — see
  "Authentication" below).
- **A public login/register system** — anyone can create an account to
  use the API docs; new accounts start at the lowest privilege level
  (`viewer`) and must be promoted by the admin to add or modify data.
- **An admin portal** (`/admin/`) — a separate, single-account area
  (protected by MFA) for managing indexed faces, viewing users,
  changing user roles, reviewing activity, and reviewing consent
  records and match-threshold calibration.
- **Consent tracking** — every indexed face requires a recorded
  consent (who gave it, how, and why) before it can be stored. This
  tool is scoped to closed, consenting datasets — see "Scope note."
- **An AI/edit-detection layer** — every uploaded image is screened for
  signs it was AI-generated or edited/composited, using a combination
  of filename, metadata, and pixel-level forensic checks.

## How it works

1. A known face is indexed via `/index-face/`, which runs it through
   DeepFace to generate a 512-dimensional embedding and stores it in
   ChromaDB, tagged with a `person_id` (for grouping multiple photos of
   the same person) and a thumbnail preview. Indexing requires proof
   of consent (see "Consent" below) and the `analyst` role or higher.
2. A new/query photo is sent to `/search-face/`, which generates its
   embedding the same way and compares it against everything stored,
   using cosine distance. Results are grouped by `person_id`, keeping
   each person's best (lowest-distance) match — "best-of-N" matching.
3. Each result gets a confidence label (`high` / `possible` / `none`)
   using two tunable thresholds, plus a `match_found` flag against a
   configurable per-request `threshold`.
4. Before indexing, the system auto-checks for near-duplicate faces
   already in the database and warns you (still indexes unless you
   pass `force=false`).
5. Every uploaded image — on both `/index-face/` and `/search-face/` —
   is also run through `ai_detection.py`, which flags likely
   AI-generated or edited/composited images (see its own section
   below). This never blocks the request; it just adds a warning field
   to the response.
6. Every index/search/reindex action is logged with the acting user's
   username, viewable in the admin panel.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** DeepFace/TensorFlow currently require **Python 3.11 or 3.12**
— they do not yet support Python 3.14. If your default `python`/`py`
points to 3.14, create the venv explicitly with a supported version:

```bash
py -3.12 -m venv venv        # Windows
python3.12 -m venv venv      # macOS/Linux
```

If you hit `ModuleNotFoundError` for `tensorflow` or dependency
resolution errors during install, it almost always means the venv is
still running on an unsupported Python version — check with
`python --version` after activating.

`requirements.txt` uses `~=` version ranges (patch/minor updates only,
no surprise breaking major-version bumps). For a fully reproducible
environment, after a successful install run:
```bash
pip freeze > requirements.lock.txt
```
and install from that lock file in any new environment instead.

### Environment variables

Copy `.env.example` to `.env` and fill in real values before running
this beyond your own machine:

```bash
cp .env.example .env
```

At minimum, set `ADMIN_PASSWORD`, `SESSION_SECRET_KEY`, and
`JWT_SECRET_KEY` to strong, unique values. Generate a random secret
with:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Never commit your actual `.env` file — only `.env.example` (which has
no real secrets) belongs in version control.

### Required folders

These must physically exist before the server starts (Python won't
create them for you):
app/static/css/ — holds admin.css
app/templates/ — holds every .html page below


Templates needed:

login.html — admin login (step 1: username/password)
admin_mfa_setup.html — admin MFA enrollment (first login only)
admin_mfa_verify.html — admin MFA code entry (every login after)
dashboard.html — admin dashboard (now shows consent status)
activity_log.html — admin activity log
users_page.html — admin: registered users, roles, activity
calibration.html — admin: FAR/FRR threshold calibration
user_login.html — public user login
register.html — public user registration
Missing `.html` files in `app/templates/` cause
`jinja2.exceptions.TemplateNotFound` errors at the exact moment that
page is visited, not at startup — so a missing file might not surface
until you click into a specific admin/login page.

**For threshold calibration** (see its own section below), you'll also
need a `calibration_data/` folder — this is NOT auto-created and is
separate from `app/tests/` (the pytest unit test suite already in this
project). See "Threshold calibration" for the expected layout.

**A note on large file pastes:** several files in this project are
long enough that copy-pasting them into an editor can silently
truncate partway through. Always verify with:
```bash
python -m py_compile app/services/ai_detection.py
(Get-Content app\services\ai_detection.py | Measure-Object -Line).Lines   # PowerShell
```

## Run

```bash
uvicorn app.main:app --reload
```

- **API docs (Swagger UI):** http://localhost:8000/docs
  — gated behind login; visiting it while logged out redirects to
  `/login`. Register an account at `/register` if you don't have one.
- **Admin portal:** http://localhost:8000/admin/login
  (default credentials: `admin` / `change-me` — change these via
  `.env`/environment variables before using this beyond your own
  machine — see "Security" below). First login triggers one-time MFA
  enrollment (see "Admin MFA" below).

**Double-check the port** — if `uvicorn` prints
`Uvicorn running on http://127.0.0.1:8000`, that's the port to use in
your browser.

**On startup**, if any of `admin_password`, `session_secret_key`, or
`jwt_secret_key` are still on their insecure development defaults,
you'll see a `SECURITY WARNING` printed to the console. This doesn't
block the server from running.

**If you forget the admin password:** there's no "forgot password"
flow by design. Open `.env` (or `app/core/config.py`), set
`ADMIN_PASSWORD`, and restart. **If you lose access to your MFA
authenticator app**, delete `face_db/admin_mfa.json` to force
re-enrollment on next login.

## Authentication

Every request to `/index-face/`, `/bulk-index/`, and `/search-face/`
requires **either**:
- a valid browser session cookie (from logging in at `/login`), or
- a valid JWT bearer token, sent as `Authorization: Bearer <token>`.

A bare `curl` request with no cookie and no token gets a
`401 Unauthorized`.

### Roles (RBAC)

Every user has a role, lowest to highest privilege:
`viewer` → `analyst` → `investigator` → `admin`

- **New self-registered accounts always start as `viewer`.** A viewer
  can search (`/search-face/`) but cannot add or modify indexed data.
- **`analyst` or higher** is required to call `/index-face/`,
  `PUT /index-face/{id}`, and `/bulk-index/`.
- Only the admin panel can promote a user's role
  (`POST /admin/users/{username}/role`) — nobody can elevate their own
  privileges.
- A role change made via the session-cookie path takes effect
  immediately on the user's next request. A role embedded in an
  already-issued JWT access token only updates once that token expires
  and is refreshed via `/token/refresh` (see `jwt_expiry_minutes`) —
  a deliberate stateless-JWT tradeoff.

If a role-gated endpoint is called by an authenticated user without
enough privilege, it returns `403 Forbidden` (not `401`) with a
message naming the required role.

### Getting and using tokens

```bash
# Sign in and get an access token + refresh token
curl -X POST http://localhost:8000/token \
  -d "username=youruser&password=yourpassword"
# → {"access_token": "...", "refresh_token": "...", "token_type": "bearer"}

# Use the access token to call a protected endpoint
curl -X POST http://localhost:8000/search-face/ \
  -H "Authorization: Bearer <access_token>" \
  -F "image=@photo.jpg"
```

**Access tokens** are short-lived (`jwt_expiry_minutes`, default 24
hours) and embed the user's role at issue time. When one expires, use
the refresh token:

```bash
curl -X POST http://localhost:8000/token/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<refresh_token>"}'
# → {"access_token": "<new access token>", "token_type": "bearer"}
```

**Refresh tokens** are long-lived (`jwt_refresh_expiry_days`, default
30 days). To invalidate one early, revoke it:

```bash
curl -X POST http://localhost:8000/token/revoke \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<refresh_token>"}'
# → {"status": "revoked"}
```
A revoked refresh token can no longer be exchanged for new access
tokens. Access tokens themselves are not individually revocable;
keeping their lifetime short limits exposure from a leaked one.

Passwords for regular users are hashed with PBKDF2-SHA256 (100,000
iterations, per-user salt) — never stored in plaintext.

### Admin MFA

The admin account requires a second factor (TOTP, e.g. Google
Authenticator/Authy) in addition to the username/password, controlled
by `admin_mfa_enabled` in `config.py` (default `true`).

- **First successful password login ever** redirects to a one-time QR
  enrollment screen (`/admin/mfa-setup`). Scan the code, enter the
  6-digit code it shows, and MFA is active from then on.
- **Every login after that** goes through `/admin/mfa-verify` instead
  — enter the current 6-digit code from your authenticator app.
- The TOTP secret is stored server-side in `admin_mfa.json` (next to
  `db_path`), never in `config.py`. Treat this file like a password —
  don't commit it to version control.
- There is currently no self-service MFA recovery flow if you lose
  your authenticator device — see "If you forget the admin password"
  above for the manual reset.

## Security

The following protections are in place:

- **Admin MFA** — see above.
- **Rate limiting** (via `slowapi`) — `/login`, `/register`,
  `/admin/login`, `/admin/mfa-setup`, `/admin/mfa-verify`, `/token`,
  `/token/refresh`, and `/token/revoke` are each limited to
  **5 requests per minute per IP address**. Exceeding this returns a
  `429 Too Many Requests`.
- **Per-account lockout** — a specific username is locked for **15
  minutes** after **5 consecutive failed login attempts**, independent
  of the IP-based rate limit above.
- **Timing-safe credential comparison** — both the admin login check
  and the regular-user password check use `secrets.compare_digest()`
  instead of `==`.
- **Role-based access control (RBAC)** — see "Roles" above.
- **Startup warnings for insecure defaults** — see above.
- **Hashed user passwords** (PBKDF2-SHA256, salted, 100k iterations).
- **Refresh token revocation** — tracked server-side in a SQLite table
  (`revoked_tokens`) keyed by each token's unique ID.
- **Admin routes hidden from the public OpenAPI schema**, separate
  session keys for admin vs. regular users (`admin_authenticated` vs.
  `user_authenticated`).

**Still on you before a real deployment:**
- Set `ADMIN_PASSWORD`, `SESSION_SECRET_KEY`, and `JWT_SECRET_KEY` in
  `.env` to strong, unique values.
- Access tokens themselves can't be individually revoked — keep
  `jwt_expiry_minutes` short if this matters for your use case.
- MFA has no self-service recovery flow — plan for the manual reset
  path if you deploy this for someone other than yourself.

## Consent

This tool is scoped to closed, consenting datasets (see "Scope note").
Every face indexed via `/index-face/`, `PUT /index-face/{id}`, or
`/bulk-index/` requires proof of consent, submitted as required form
fields:

- `consent_given` — must be `true`, or the request is rejected with a
  `422` before the image is ever processed or embedded.
- `consent_given_by` — who gave consent (usually the subject
  themself).
- `consent_method` — one of: `written_form`, `verbal_recorded`,
  `self_registered`, `institutional`.
- `purpose` — why this face is being indexed.

For `/bulk-index/`, one consent record covers the whole batch (all
photos in a batch are treated as different pictures of the same
already-consented person) — index individually via `/index-face/` if
different photos need different consent records.

Consent records are stored separately from the face embeddings
(`consent.db`), keyed by `face_id`. Deleting a face via the admin
panel also deletes its consent record — no orphaned consent data is
left behind. The admin dashboard shows each indexed face's consent
status (✅ method, or ⚠️ Missing for any pre-existing face indexed
before this feature was added).

## Endpoints

- `POST /index-face/` — form fields `face_id`, `source_url`, optional
  `person_id`, consent fields (see "Consent" above), and an uploaded
  `image` file. Stores the face's embedding + thumbnail. Runs
  duplicate detection and AI/edit detection first; pass `force=false`
  to skip indexing when a near-duplicate is found. Response includes
  `ai_generated_warning` and `ai_confidence`. **Requires the `analyst`
  role or higher.**
- `PUT /index-face/{face_id}` — re-indexes an existing face with a new
  photo, without changing its ID. Consent must be re-confirmed for the
  new photo. **Requires the `analyst` role or higher.**
- `POST /bulk-index/` — form field `person_id`, consent fields, +
  multiple `images`. Indexes several photos of the same person in one
  request, each auto-numbered (`person_id_01`, `person_id_02`, ...).
  **Requires the `analyst` role or higher.**
- `POST /search-face/` — an uploaded `image` file, plus optional query
  params:
  - `top_k` — how many raw candidates to fetch before grouping by
    person (default 20)
  - `threshold` — cosine distance cutoff for a confident match
    (default 0.4, overridable per-request)

  Returns each person's best match with `distance`, `confidence`
  (`high`/`possible`/`none`), and `match_found`, plus a top-level
  `best_match_found` for the closest result overall, and
  `query_image_ai_warning` / `query_image_ai_confidence` for the
  uploaded query image itself. **Requires the `viewer` role or
  higher** (i.e. any authenticated, registered user).
- `GET /health` — returns status + how many faces are indexed. No auth
  required.

### Supported image formats

Standard formats (JPEG, PNG, etc.) are read directly. **HEIC/HEIF**
files (the default format for iPhone photos) are also accepted —
`upload_utils.py` transparently converts them to JPEG on upload before
any further processing, since DeepFace's underlying image loader
(OpenCV) has no native HEIC support. This conversion is silent (JPEG,
quality 95); the original HEIC bytes are not retained. If a file has a
`.heic`/`.heif` extension but isn't actually a valid HEIC file, this
conversion step returns a `422` before the file reaches quality checks
or DeepFace.

Images of any dimension or aspect ratio are accepted as-is. Max upload
size is capped at `max_upload_size_mb` in `app/core/config.py`
(default 15 MB).

### Public auth routes (gate the API docs)

- `GET /register`, `POST /register` — create an account (5/min per
  IP). New accounts always start at the `viewer` role.
- `GET /login`, `POST /login` — sign in (5/min per IP, plus
  per-account lockout after 5 consecutive failures)
- `GET /logout` — sign out
- `POST /token` — exchange username/password for an access + refresh
  token pair (5/min per IP, plus per-account lockout)
- `POST /token/refresh` — exchange a valid refresh token for a new
  access token (5/min per IP)
- `POST /token/revoke` — revoke a refresh token early (5/min per IP)

### Admin portal routes (separate single-account area, MFA-protected)

- `GET /admin/login`, `POST /admin/login` — admin sign in step 1:
  username/password (5/min per IP)
- `GET /admin/mfa-setup`, `POST /admin/mfa-setup` — one-time MFA
  enrollment (QR code + confirmation code)
- `GET /admin/mfa-verify`, `POST /admin/mfa-verify` — MFA code entry
  on every subsequent login
- `GET /admin/` — dashboard: stat cards, searchable/paginated table of
  every indexed face, thumbnails, consent status, inline edit, bulk
  delete
- `POST /admin/faces/{face_id}/delete` — delete one face (also deletes
  its consent record)
- `POST /admin/faces/bulk-delete` — delete multiple selected faces
  (also deletes their consent records)
- `POST /admin/faces/{face_id}/edit` — edit a face's `person_id` /
  `source_url` without re-uploading a photo
- `GET /admin/export?format=json|csv` — export all indexed faces
- `GET /admin/activity` — full activity log (who did what, when)
- `GET /admin/users` — every registered user, their role, and recent
  activity
- `POST /admin/users/{username}/role` — change a user's role
  (admin-only)
- `GET /admin/calibration` — FAR/FRR threshold calibration report (see
  below)
- `GET /admin/logout` — admin sign out

## AI-generated / edited image detection

`app/services/ai_detection.py` screens every uploaded image (on both
`/index-face/` and `/search-face/`) and returns a verdict with a
confidence tier — it never blocks the request, only adds a warning.

| Check | What it catches | Confidence | Easily bypassed by |
|---|---|---|---|
| Filename pattern | "ChatGPT_Image...", "Midjourney...", etc. (normalized so spaces/hyphens/underscores all match) | High | Renaming the file |
| Embedded metadata | Stable Diffusion "parameters" field, EXIF Software tag naming a known AI tool | High | Re-saving/screenshotting |
| Error Level Analysis | Spliced/pasted regions at a different JPEG compression level than the rest of the image | Medium | Only works on JPEGs; careful re-compression can hide it |
| Noise-consistency check | Same idea as ELA, but format-agnostic (works on PNGs) — flags inconsistent noise/texture across regions, e.g. double-exposure blends | Medium | Careful post-processing/blur |
| Frequency-domain artifacts | Periodic patterns sometimes left by GAN/diffusion upsampling layers | Low | Post-processing; also false-positives on real photos with repetitive textures (fabric, brick) |
| Missing EXIF | No camera metadata at all | Low | Very weak either way — common in legitimately edited/exported real photos too |

**This is explicitly not a trained deepfake detector.** A real
deepfake/AI-image classifier requires a neural network trained on
labeled real/fake datasets (e.g. FaceForensics++) — a fundamentally
different, much larger undertaking than the heuristics here.
- Confirmed working: catches images with intact AI-tool metadata or
  filenames, and catches composited/double-exposure images via the
  noise-consistency check.
- Confirmed gaps: a well-made deepfake, or any edit carefully
  re-compressed/re-exported to erase these signals, will not be
  caught. Photoshop composites are only caught if the noise/
  compression inconsistency check happens to trigger.

## Threshold calibration (FAR/FRR)

`GET /admin/calibration` runs a False Accept Rate / False Reject Rate
analysis against a labeled test photo set, so `match_threshold` isn't
picked arbitrarily.

**Setup:** create a `calibration_data/` folder (path configurable via
`calibration_data_path` in `config.py`) laid out as:
calibration_data/
person_001/
photo1.jpg
photo2.jpg
person_002/
photo1.jpg
person_003/
photo1.jpg
photo2.jpg
Each subfolder is one real person with 2+ different photos of them.
The harness computes embeddings for every photo, then:
- **Genuine pairs** — every pair of photos within the same person's
  folder — used to measure False Reject Rate.
- **Impostor pairs** — every pair of photos across different people's
  folders — used to measure False Accept Rate.

The report shows genuine vs. impostor distance distributions, where
your current `match_threshold` lands (its FAR/FRR), a full FAR/FRR
curve across candidate thresholds, and the Equal Error Rate point
(where FAR ≈ FRR) as a reasonable starting point if you have no other
preference. This is a standalone, read-only analysis — it never
touches or depends on your live indexed (production) data.

This is separate from `app/tests/` (the pytest unit test suite already
in this project) — `calibration_data/` holds photos, not test code,
and isn't created automatically.

## Tuning the match threshold

`match_threshold` in `app/core/config.py` (single cutoff used by
`/search-face/`'s `match_found` flag) and the separate
`high_confidence_threshold` / `possible_match_threshold` pair (used
for the `confidence` label) both live in the same config file.
`duplicate_warning_threshold` controls how close a new face's
embedding must be to an existing one before `/index-face/` flags it as
a likely duplicate.

Based on testing so far:
- Same person, different photo: distances have landed roughly in the
  0.18–0.36 range
- Different people: distances have landed roughly in the 0.56–1.19
  range
- Heavily edited/composited photos of an already-indexed person can
  push the distance up to ~0.43 — past the confident-match threshold,
  landing in the "possible" tier instead

Use `/admin/calibration` (above) to validate these against your own
dataset rather than relying on these general figures alone.

## Project structure
app/
api/routes/
index_face.py — POST /index-face/, PUT /index-face/{id}
(consent required, analyst+ role)
bulk_index.py — POST /bulk-index/ (consent required, analyst+ role)
search_face.py — POST /search-face/ (viewer+ role)
auth.py — public register/login/logout, /token,
/token/refresh, /token/revoke, RBAC
dependencies (require_role)
admin.py — admin portal: login+MFA, dashboard,
users/roles, activity, calibration
services/
embedding_service.py — DeepFace wrapper (shared by index + search)
vector_store.py — ChromaDB add/query/delete/list/edit
upload_utils.py — shared upload handling, size limit,
and HEIC→JPEG conversion
thumbnail_utils.py — generates per-face preview thumbnails
ai_detection.py — AI-generated/edited image screening
activity_log.py — JSONL append-only action log (per user)
face_quality.py — pre-indexing image quality gates
user_store.py — SQLite-backed user accounts +
roles (RBAC), lockout tracking
jwt_utils.py — access/refresh JWT creation,
decoding, revocation, role claim
token_store.py — SQLite-backed revoked
refresh-token tracking
consent_store.py — SQLite-backed consent records
per indexed face
admin_mfa_store.py — TOTP secret storage +
verification for admin MFA
calibration.py — FAR/FRR calibration harness
core/
config.py — all settings + startup
security warnings
templates/ — see "Required folders" above
static/css/
admin.css — shared styling for all pages
schemas.py — Pydantic request/response models
main.py — FastAPI entrypoint: session
middleware, rate limiter
setup, static/template
mounts, custom-themed /docs,
admin routes hidden from the
public OpenAPI schema
tests/ — pytest suite (ai_detection,
face_quality, jwt_utils,
user_store, vector_store)
calibration_data/ — NOT auto-created; see
"Threshold calibration"
## The custom /docs page

`/docs` is no longer FastAPI's default Swagger UI look — `main.py`
overrides it with:
- A dark theme matching the admin portal (Inter/JetBrains Mono fonts,
  color-coded HTTP methods)
- A 3-column navbar (brand left, tool name center, version + admin
  link right)
- The "Schemas" section removed (`defaultModelsExpandDepth: -1`, with
  a CSS fallback in case that setting is ignored)
- All `/admin/*` paths stripped from the generated OpenAPI schema
  directly (`custom_openapi()` in `main.py`)
- A login gate: visiting `/docs` while not signed in redirects to
  `/login`

## Known limitations

- Face detection can fail on photos where the face is small, angled,
  obscured, or poorly lit — these return a `422` error rather than a
  bad embedding.
- Distances near 0 typically indicate the exact same file was used for
  both indexing and searching, not a genuine different-photo match.
- Rate limiting is **in-memory and per-process** — it resets on
  restart and won't be shared across multiple server instances/workers
  behind a load balancer. Would need a shared store (e.g. Redis) for
  multi-worker setups.
- Access tokens cannot be individually revoked before expiry (only
  refresh tokens can).
- A role change made via a JWT bearer session doesn't take effect
  until the caller's access token expires and is refreshed — see
  "Roles" above.
- Admin MFA has no self-service recovery flow if the authenticator
  device is lost — see "Admin MFA" above.
- Existing indexed faces from before certain features were added
  (thumbnails, `person_id`, timestamps, consent records) won't show
  that data until re-indexed — old entries display placeholders
  instead of erroring.
- `Jinja2Templates.TemplateResponse()` requires `request` as the first
  positional argument on current Starlette versions.
- AI/edit detection is heuristic, not a trained model.
- HEIC uploads are converted to JPEG at quality 95 before indexing/
  search — a lossy re-encode, so embeddings are generated from the
  converted JPEG, not the original HEIC bytes.
- Consent for `/bulk-index/` is recorded once per batch, not per
  individual photo — use `/index-face/` individually if different
  photos in a session require different consent records.
- Threshold calibration (`/admin/calibration`) recomputes embeddings
  on every page load with no caching — fine for a small test set, but
  will get slow if `calibration_data/` grows large.

## Scope note

Intended for closed, consenting datasets (e.g. a class project or a
controlled test set) — not for identifying people from open web
images. Every face indexed requires a recorded consent (see
"Consent"); this is enforced in code, not just documented as a policy.
