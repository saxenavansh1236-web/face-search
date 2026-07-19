# Face Search Tool

A FastAPI service that indexes known faces (with consent, from a
closed dataset) and lets you search a new photo against them. Given a
new photo, it tells you whether it matches someone already indexed —
even if the new photo is a different picture of that same person (a
different angle, lighting, or day) — and rejects photos of people who
aren't in the dataset.

The project includes four separate surfaces:
- **The API** (`/index-face/`, `/search-face/`, etc.) — gated behind
  user login/registration, and enforced via a real auth dependency
  (session cookie OR JWT bearer token — see "Authentication" below).
- **A public login/register system** — anyone can create an account to
  use the API docs; the admin can see who's registered and what
  they've done.
- **An admin portal** (`/admin/`) — a separate, single-account area for
  managing indexed faces, viewing users, and reviewing activity.
- **An AI/edit-detection layer** — every uploaded image is screened for
  signs it was AI-generated or edited/composited, using a combination
  of filename, metadata, and pixel-level forensic checks.

## How it works

1. A known face is indexed via `/index-face/`, which runs it through
   DeepFace to generate a 512-dimensional embedding and stores it in
   ChromaDB, tagged with a `person_id` (for grouping multiple photos of
   the same person) and a thumbnail preview.
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

```
app/static/css/     — holds admin.css
app/templates/      — holds every .html page below
```

Missing `.html` files in `app/templates/` cause
`jinja2.exceptions.TemplateNotFound` errors at the exact moment that
page is visited, not at startup — so a missing file might not surface
until you click into a specific admin/login page. If you see that
error, double check the file exists at the exact path shown in the
traceback and that the whole file was saved (not cut off partway
through a paste).

Templates needed:
```
login.html          — admin login
dashboard.html       — admin dashboard
activity_log.html     — admin activity log
users_page.html        — admin: registered users + their activity
user_login.html          — public user login
register.html              — public user registration
```

**A note on large file pastes:** several files in this project
(especially `ai_detection.py`) are long enough that copy-pasting them
into an editor can silently truncate partway through, leaving a file
that looks complete but ends mid-function. Symptoms: `SyntaxError:
expected ':'`, `IndentationError: expected an indented block`, or a
line count noticeably lower than expected. Always verify with:
```bash
python -m py_compile app/services/ai_detection.py
(Get-Content app\services\ai_detection.py | Measure-Object -Line).Lines   # PowerShell
```
If a paste keeps getting cut off at the same point, use PowerShell's
`Add-Content` with a here-string (`@'...'@`) to append the missing
portion directly, rather than re-pasting the whole file through an
editor.

## Run

```bash
uvicorn app.main:app --reload
```

- **API docs (Swagger UI):** http://localhost:8000/docs
  — gated behind login; visiting it while logged out redirects to
  `/login`. Register an account at `/register` if you don't have one.
- **Admin portal:** http://localhost:8000/admin/login
  (default credentials: `admin` / `change-me` — change these via
  `.env`/environment variables, not by editing `config.py` directly,
  before using this beyond your own machine — see "Security" below)

**Double-check the port** — if `uvicorn` prints
`Uvicorn running on http://127.0.0.1:8000`, that's the port to use in
your browser. A different port (e.g. 5000) will just refuse to connect.

**On startup**, if any of `admin_password`, `session_secret_key`, or
`jwt_secret_key` are still on their insecure development defaults,
you'll see a `SECURITY WARNING` printed to the console. This doesn't
block the server from running — local dev works fine on defaults —
it's just a reminder before deploying anywhere beyond your own machine.

**If you forget the admin password:** there's no "forgot password"
flow for the single admin account by design. Just open `.env` (or
`app/core/config.py` if you haven't set up `.env` yet), find/set
`ADMIN_PASSWORD`, and restart the server.

## Authentication

Every request to `/index-face/`, `/bulk-index/`, and `/search-face/`
requires **either**:
- a valid browser session cookie (from logging in at `/login`), or
- a valid JWT bearer token, sent as `Authorization: Bearer <token>`.

A bare `curl` request with no cookie and no token gets a
`401 Unauthorized` — these endpoints cannot be called anonymously.

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
hours). When one expires, use the refresh token instead of logging in
again:

```bash
curl -X POST http://localhost:8000/token/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<refresh_token>"}'
# → {"access_token": "<new access token>", "token_type": "bearer"}
```

**Refresh tokens** are long-lived (`jwt_refresh_expiry_days`, default
30 days). To invalidate one early — e.g. logging out a script, or a
refresh token you believe leaked — revoke it:

```bash
curl -X POST http://localhost:8000/token/revoke \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "<refresh_token>"}'
# → {"status": "revoked"}
```
A revoked refresh token can no longer be exchanged for new access
tokens. Any access tokens already issued from it remain valid until
their own (short) expiry — only the ability to mint *new* ones is cut
off. Access tokens themselves are not individually revocable; keeping
their lifetime short is what limits exposure from a leaked one.

Passwords for regular users are hashed with PBKDF2-SHA256 (100,000
iterations, per-user salt) — never stored in plaintext.

## Security

The following protections are in place:

- **Rate limiting** (via `slowapi`) — `/login`, `/register`,
  `/admin/login`, `/token`, `/token/refresh`, and `/token/revoke` are
  each limited to **5 requests per minute per IP address**. Exceeding
  this returns a `429 Too Many Requests`.
- **Per-account lockout** — independent of the IP-based rate limit
  above, a specific username is locked for **15 minutes** after **5
  consecutive failed login attempts**, so an attacker spreading
  requests across many IPs still can't brute-force one targeted
  account. A successful login resets the counter.
- **Timing-safe credential comparison** — both the admin login check
  and the regular-user password check use `secrets.compare_digest()`
  instead of `==`, so failed attempts don't leak timing information
  about how much of the password was correct.
- **Startup warnings for insecure defaults** — see above.
- **Hashed user passwords** (PBKDF2-SHA256, salted, 100k iterations).
- **Refresh token revocation** — see "Authentication" above. Tracked
  server-side in a small SQLite table (`revoked_tokens`) keyed by each
  token's unique ID, so a compromised or logged-out refresh token can
  be cut off before it naturally expires.
- **Admin routes hidden from the public OpenAPI schema** (see
  `custom_openapi()` below), separate session keys for admin vs.
  regular users (`admin_authenticated` vs. `user_authenticated`), so
  one can't be mistaken for the other even though they share one
  `SessionMiddleware` secret (that part is normal — one signed cookie
  per app is standard).

**Still on you before a real deployment:**
- Set `ADMIN_PASSWORD`, `SESSION_SECRET_KEY`, and `JWT_SECRET_KEY` in
  `.env` to strong, unique values — never rely on the shipped defaults.
- Access tokens themselves can't be individually revoked (only refresh
  tokens can) — this is a standard JWT tradeoff; keep
  `jwt_expiry_minutes` short if this matters for your use case.

## Endpoints

- `POST /index-face/` — form fields `face_id`, `source_url`,
  optional `person_id`, and an uploaded `image` file. Stores the
  face's embedding + thumbnail. Runs duplicate detection and
  AI/edit detection first; pass `force=false` to skip indexing when a
  near-duplicate is found. Response includes `ai_generated_warning`
  and `ai_confidence`. **Requires auth** (session or bearer token).
- `PUT /index-face/{face_id}` — re-indexes an existing face with a new
  photo, without changing its ID. **Requires auth.**
- `POST /bulk-index/` — form field `person_id` + multiple `images`.
  Indexes several photos of the same person in one request, each
  auto-numbered (`person_id_01`, `person_id_02`, ...). **Requires auth.**
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
  uploaded query image itself. **Requires auth.**
- `GET /health` — returns status + how many faces are indexed. No auth
  required.

Images of any dimension or aspect ratio (tall, wide, square) are
accepted as-is. Max upload size is capped at `max_upload_size_mb` in
`app/core/config.py` (default 15 MB).

### Public auth routes (gate the API docs)

- `GET /register`, `POST /register` — create an account (5/min per IP)
- `GET /login`, `POST /login` — sign in (5/min per IP, plus per-account
  lockout after 5 consecutive failures)
- `GET /logout` — sign out
- `POST /token` — exchange username/password for an access + refresh
  token pair (5/min per IP, plus per-account lockout)
- `POST /token/refresh` — exchange a valid refresh token for a new
  access token (5/min per IP)
- `POST /token/revoke` — revoke a refresh token early (5/min per IP)

### Admin portal routes (separate single-account area)

- `GET /admin/login`, `POST /admin/login` — admin sign in (5/min per IP)
- `GET /admin/` — dashboard: stat cards, searchable/paginated table of
  every indexed face, thumbnails, inline edit, bulk delete
- `POST /admin/faces/{face_id}/delete` — delete one face
- `POST /admin/faces/bulk-delete` — delete multiple selected faces
- `POST /admin/faces/{face_id}/edit` — edit a face's `person_id` /
  `source_url` without re-uploading a photo
- `GET /admin/export?format=json|csv` — export all indexed faces
- `GET /admin/activity` — full activity log (who did what, when)
- `GET /admin/users` — every registered user + their recent activity
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
labeled real/fake datasets (e.g. FaceForensics++) — that's a
fundamentally different, much larger undertaking than the heuristics
here. What's implemented are classical, well-established forensic
techniques that catch a real but limited share of cases:
- Confirmed working: catches images with intact AI-tool metadata or
  filenames, and catches composited/double-exposure images via the
  noise-consistency check (verified against a real double-exposure
  poster during testing).
- Confirmed gaps: a well-made deepfake, or any edit that's been
  carefully re-compressed/re-exported to erase these signals, will not
  be caught. Manually edited (but not AI-generated) images — like
  Photoshop composites — are only caught if the noise/compression
  inconsistency check happens to trigger; there's no dedicated
  "human-edited" detector.

## Project structure

```
app/
  api/routes/
    index_face.py      — POST /index-face/, PUT /index-face/{id}
    bulk_index.py        — POST /bulk-index/
    search_face.py         — POST /search-face/
    auth.py                  — public register/login/logout, /token,
                                 /token/refresh, /token/revoke
    admin.py                   — admin portal (all routes above)
  services/
    embedding_service.py      — DeepFace wrapper (shared by index + search)
    vector_store.py             — ChromaDB add/query/delete/list/edit
    upload_utils.py               — shared upload handling + size limit
    thumbnail_utils.py              — generates per-face preview thumbnails
    ai_detection.py                   — AI-generated/edited image screening
    activity_log.py                     — JSONL append-only action log (per user)
    face_quality.py                       — pre-indexing image quality gates
    user_store.py                           — SQLite-backed user accounts
                                             (PBKDF2-hashed passwords,
                                             per-account lockout tracking)
    jwt_utils.py                            — access/refresh JWT creation,
                                               decoding, and revocation
    token_store.py                            — SQLite-backed revoked
                                                 refresh-token tracking
  core/
    config.py                               — all settings + startup
                                               security warnings
  templates/                               — see "Required folders" above
  static/css/
    admin.css                                 — shared styling for all pages
  schemas.py                                   — Pydantic request/response models
  main.py                                       — FastAPI entrypoint: session
                                                   middleware, rate limiter
                                                   setup, static/template
                                                   mounts, custom-themed /docs,
                                                   admin routes hidden from the
                                                   public OpenAPI schema
tests/                                           — pytest suite (ai_detection,
                                                   face_quality, jwt_utils,
                                                   user_store, vector_store)
```

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
  directly (`custom_openapi()` in `main.py`), so they can never leak
  into the docs even if a route forgets `include_in_schema=False`
- A login gate: visiting `/docs` while not signed in redirects to
  `/login`

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
- Heavily edited/composited photos of an already-indexed person
  (e.g. a double-exposure poster effect) can push the distance up to
  ~0.43 — still recognizably close, but past the confident-match
  threshold, landing in the "possible" tier instead

These defaults sit in that gap but should be re-validated with more
test photos as the dataset grows — a few data points isn't enough to
fully trust a threshold.

## Known limitations

- Face detection can fail on photos where the face is small, angled,
  obscured, or poorly lit — these return a `422` error rather than a
  bad embedding. Retake or choose a clearer photo in that case.
- Distances near 0 typically indicate the exact same file was used for
  both indexing and searching, not a genuine different-photo match —
  worth keeping in mind when interpreting test results.
- Rate limiting is **in-memory and per-process** — it resets on
  restart and won't be shared across multiple server instances/workers
  behind a load balancer. Fine for a single-instance deployment; would
  need a shared store (e.g. Redis) for multi-worker setups.
- Access tokens cannot be individually revoked before expiry (only
  refresh tokens can, via `/token/revoke`) — a standard tradeoff for
  stateless JWTs; keep access-token lifetime short if this matters.
- Existing indexed faces from before certain features were added
  (thumbnails, `person_id`, timestamps) won't show that data until
  re-indexed — old entries display placeholders (`?` thumbnail, `—`
  date) instead of erroring.
- `Jinja2Templates.TemplateResponse()` requires `request` as the first
  positional argument on current Starlette versions — the older
  `(name, {"request": request, ...})` signature will raise
  `TypeError: unhashable type: 'dict'` if used with a newer Starlette
  install.
- AI/edit detection is heuristic, not a trained model — see the
  dedicated section above for exactly what it does and doesn't catch.

## Scope note

Intended for closed, consenting datasets (e.g. a class project or a
controlled test set) — not for identifying people from open web images.