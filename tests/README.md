# tests/

Three-layer test suite covering the full stack — unit, integration, and end-to-end. Each layer is scoped to a different class of failure and runs at a different point in the CI pipeline.

---

## Structure

```
tests/
├── test_api.py          # Unit tests — FastAPI TestClient
├── test_integration.py  # Integration tests — real uvicorn subprocess + httpx
└── test_e2e.py          # E2E tests — Playwright + Chromium against live deployments
```

---

## Test Layers

### Unit — `test_api.py`

Uses FastAPI's `TestClient` (backed by `httpx`). No real server is started — requests are handled in-process.

**Covers:**
- `GET /health` and `HEAD /health`
- `POST /predict` — 200 response shape, field types, probability range `[0, 1]`
- 8 validation cases — 422 responses for missing fields, wrong types, out-of-range values

**Run:**
```bash
uv run pytest tests/test_api.py -v
```

---

### Integration — `test_integration.py`

Spawns a real uvicorn server on port 8001 via `subprocess`. Uses a socket-based readiness check before tests run.

**Covers:**
- CORS headers for allowed origins (Vercel production, localhost)
- CORS rejection for disallowed origins
- OPTIONS preflight response
- Response time assertions
- `Content-Type` header correctness

**Why not TestClient?** TestClient abstracts away the HTTP layer — CORS middleware, actual headers, and network-level behaviour are not exercised. A real server is required to test these.

**Run:**
```bash
uv run pytest tests/test_integration.py -v
```

---

### E2E — `test_e2e.py`

Uses Playwright with Chromium. Runs against live deployed infrastructure — the Vercel preview frontend and the staging Render backend. Intended for PR pipelines only.

**Covers:**
- Page load and static content
- Form field presence and dropdown options
- Happy path: survived passenger
- Happy path: perished passenger
- Loading state visibility during request
- Survival probability display
- Backend down error message
- 422 validation error message
- Negative fare input
- Out-of-range age input

**Key implementation detail:** The Vercel deployment protection bypass header (`x-vercel-protection-bypass`) is injected only to Vercel requests via `page.route()`, not via `extra_http_headers`, to avoid sending it to third-party requests.

**Run locally** (requires frontend and backend both running):
```bash
uv run pytest tests/test_e2e.py -v
```

**In CI:** E2E tests run automatically on PRs after staging deployment and Vercel preview URL resolution.

---

## CI Integration

| Test layer | Runs on |
|-----------|---------|
| Unit | Every push and PR |
| Integration | Every push and PR |
| E2E | PRs only (requires live staging environment) |

See [`.github/workflows/test.yml`](../.github/workflows/test.yml) for the full pipeline definition.