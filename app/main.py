from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.routes import index_face, search_face, admin, bulk_index, auth
from app.api.routes.auth import is_user_logged_in
from fastapi import Request
from fastapi.responses import RedirectResponse
from app.services.vector_store import count_faces
from app.core.config import settings

tags_metadata = [
    {"name": "ingestion"},
    {"name": "search"},
]

app = FastAPI(
    title="Face Search Tool",
    description="",
    version="1.1.0",
    openapi_tags=tags_metadata,
    docs_url=None,
)

# --- Rate limiting (slowapi) ---
# Shared limiter instance used by login/token routes in admin.py and
# auth.py. Keyed by client IP; in-memory, resets on restart.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(SessionMiddleware, secret_key=settings.session_secret_key)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(index_face.router, tags=["ingestion"])
app.include_router(bulk_index.router, tags=["ingestion"])
app.include_router(search_face.router, tags=["search"])
app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(auth.token_router)


@app.get("/health")
def health():
    return {"status": "ok", "faces_indexed": count_faces()}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=tags_metadata,
    )
    schema["paths"] = {
        path: methods
        for path, methods in schema.get("paths", {}).items()
        if not path.startswith("/admin")
    }
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


_CUSTOM_SWAGGER_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d1117; --panel: #151b26; --panel-2: #1a2130; --border: #262e3f;
    --text: #e6e9ef; --text-dim: #8b96ab; --accent: #5b8cff; --accent-soft: rgba(91,140,255,0.10);
    --get: #5b8cff; --post: #2fd889; --put: #ffb020; --delete: #ff5b7a;
  }

  * { font-family: 'Inter', system-ui, sans-serif !important; }
  code, .swagger-ui .microlight, .swagger-ui table code {
    font-family: 'JetBrains Mono', monospace !important;
  }

  html, body { background: var(--bg) !important; }
  .swagger-ui { color: var(--text); }
  .swagger-ui .topbar { display: none; }

  .swagger-ui .info { margin: 32px 0 24px; }
  .swagger-ui .info .title {
    color: var(--text) !important; font-weight: 700 !important; font-size: 2rem !important;
  }
  .swagger-ui .info .title small.version-stamp {
    background: var(--accent) !important; border-radius: 6px;
  }
  .swagger-ui .info .title small.version-stamp pre { background: transparent !important; }
  .swagger-ui .info a.link, .swagger-ui .info a { color: var(--accent) !important; }
  .swagger-ui .info .description, .swagger-ui .info .description p,
  .swagger-ui .markdown p, .swagger-ui .markdown li {
    color: var(--text-dim) !important; font-size: 0.95rem !important; line-height: 1.6 !important;
  }
  .swagger-ui .info .description strong { color: var(--text) !important; }

  .swagger-ui .opblock-tag {
    color: var(--text) !important; border-bottom: 1px solid var(--border) !important;
    font-weight: 600 !important; font-size: 1.15rem !important;
  }
  .swagger-ui .opblock-tag small { color: var(--text-dim) !important; font-weight: 400 !important; }
  .swagger-ui .opblock-tag:hover { background: transparent !important; }

  .swagger-ui .opblock {
    background: var(--panel) !important; border-radius: 10px !important;
    border: 1px solid var(--border) !important; box-shadow: none !important;
    margin: 0 0 14px !important;
  }
  .swagger-ui .opblock .opblock-summary { border: none !important; padding: 6px; }
  .swagger-ui .opblock .opblock-summary-path, .swagger-ui .opblock .opblock-summary-path__deprecated {
    color: var(--text) !important; font-weight: 600 !important;
  }
  .swagger-ui .opblock .opblock-summary-description { color: var(--text-dim) !important; }
  .swagger-ui .opblock .opblock-summary-method {
    border-radius: 6px !important; min-width: 74px !important; text-align: center !important;
    font-weight: 700 !important;
  }
  .swagger-ui .opblock.opblock-get { border-color: var(--get) !important; background: rgba(91,140,255,0.05) !important; }
  .swagger-ui .opblock.opblock-get .opblock-summary-method { background: var(--get) !important; }
  .swagger-ui .opblock.opblock-post { border-color: var(--post) !important; background: rgba(47,216,137,0.05) !important; }
  .swagger-ui .opblock.opblock-post .opblock-summary-method { background: var(--post) !important; }
  .swagger-ui .opblock.opblock-put { border-color: var(--put) !important; background: rgba(255,176,32,0.05) !important; }
  .swagger-ui .opblock.opblock-put .opblock-summary-method { background: var(--put) !important; }

  .swagger-ui .opblock-body { background: var(--panel-2) !important; border-top: 1px solid var(--border) !important; }
  .swagger-ui .opblock-description-wrapper p { color: var(--text-dim) !important; }
  .swagger-ui .opblock-section-header { background: transparent !important; box-shadow: none !important; border-bottom: 1px solid var(--border) !important; }
  .swagger-ui .opblock-section-header h4, .swagger-ui .opblock-section-header > label { color: var(--text) !important; }

  .swagger-ui table thead tr th, .swagger-ui table tbody tr td {
    color: var(--text) !important; border-bottom: 1px solid var(--border) !important;
  }
  .swagger-ui .parameter__name { color: var(--text) !important; }
  .swagger-ui .parameter__type, .swagger-ui .parameter__in { color: var(--text-dim) !important; }
  .swagger-ui .response-col_status { color: var(--text) !important; }
  .swagger-ui .response-col_description__inner div.markdown p { color: var(--text-dim) !important; }

  .swagger-ui input, .swagger-ui select, .swagger-ui textarea {
    background: var(--panel) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
  }
  .swagger-ui .btn {
    border-radius: 6px !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; font-weight: 600 !important;
  }
  .swagger-ui .btn.execute {
    background: var(--accent) !important; border-color: var(--accent) !important; color: white !important;
  }
  .swagger-ui .btn.try-out__btn { background: var(--panel) !important; }

  .swagger-ui .scheme-container { background: transparent !important; box-shadow: none !important; }
  .swagger-ui section.models { background: var(--panel) !important; border-color: var(--border) !important; border-radius: 10px !important; }
  .swagger-ui section.models .model-box { background: var(--panel-2) !important; }
  .swagger-ui .model, .swagger-ui .model-title { color: var(--text) !important; }
  .swagger-ui .prop-type { color: var(--accent) !important; }

  .swagger-ui .highlight-code, .swagger-ui .microlight {
    background: #0a0e16 !important; border-radius: 8px !important; color: var(--text) !important;
  }

  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }

  .swagger-ui section.models { display: none !important; }
</style>

<div style="background:#0d1117; padding:16px 32px; display:grid; grid-template-columns:1fr auto 1fr; align-items:center; border-bottom:1px solid #262e3f;">

  <div style="display:flex; align-items:center; gap:10px; justify-self:start;">
    <span style="width:32px; height:32px; border-radius:9px; background:linear-gradient(135deg,#5b8cff,#7c5cff); display:flex; align-items:center; justify-content:center; font-size:16px;">🧿</span>
    <span style="font-family:'Inter',sans-serif; font-weight:600; color:#8b96ab; font-size:0.82rem; letter-spacing:0.03em; text-transform:uppercase;">API Reference</span>
  </div>

  <div style="justify-self:center; font-family:'Inter',sans-serif; font-weight:700; color:#e6e9ef; font-size:1.15rem; white-space:nowrap;">
    Face Search Tool
  </div>

  <div style="display:flex; align-items:center; gap:10px; justify-self:end;">
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#8b96ab; border:1px solid #262e3f; padding:5px 10px; border-radius:6px;">v1.1.0</span>
    <a href="/logout" style="font-family:'Inter',sans-serif; font-size:0.85rem; color:#8b96ab; text-decoration:none; border:1px solid #262e3f; padding:7px 14px; border-radius:8px;">
      Log out
    </a>
    <a href="/admin/login" style="font-family:'Inter',sans-serif; font-size:0.85rem; font-weight:600; color:#0d1117; text-decoration:none; background:linear-gradient(135deg,#5b8cff,#7c5cff); padding:8px 16px; border-radius:8px;">
      Admin Portal →
    </a>
  </div>

</div>

"""


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    if not is_user_logged_in(request):
        return RedirectResponse(url="/login", status_code=303)
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Face Search Tool — API Docs",
        swagger_favicon_url="",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "docExpansion": "list",
            "displayRequestDuration": True,
            "persistAuthorization": True,
        },
    )
    body = html.body.decode("utf-8")
    body = body.replace("<body>", f"<body>{_CUSTOM_SWAGGER_HEAD}")
    return HTMLResponse(body)