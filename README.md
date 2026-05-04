# Film Licensing Price Estimator

A production-grade FastAPI pricing engine for film and television licensing deals. Compare multiple deal scenarios side by side, powered by AI reasoning with deterministic fallback, and export a professional PDF deal memo for each estimate.

---

## What it does

- **Multi-deal comparison** — run up to N deal scenarios in parallel (different territories, rights types, platforms) against the same title and compare results side by side
- **AI-powered reasoning** — Gemini → Groq → deterministic fallback chain; each estimate includes a plain-English analyst rationale
- **Series season selector** — interactive per-season tick UI; supports full series, partial season packages, and incremental acquisitions (buyer already holds earlier seasons)
- **Per-season episode accuracy** — IMDb per-season episode counts fetched at auto-fill time; episode overrides per season for mid-season or holdback deals
- **PDF deal memo export** — one-click download of a formatted A4 memo per deal: cover, pricing summary, season context banner, deal parameters, pricing components, analyst reasoning
- **SQLite caching** — deal key hashed from all pricing inputs; cache invalidates automatically when model version changes
- **Rate limiting** — in-memory per-IP sliding window
- **Metrics endpoint** — request counts, cache hit/miss ratio, error counts

---

## Project structure

```
pricing_module/
├── main.py           # FastAPI app — models, pricing logic, AI chain, endpoints
├── index.html        # Single-file frontend — served at GET /
├── requirements.txt  # Trimmed dependency list
├── .env              # Environment variables (not committed)
└── pricing_cache.db  # SQLite cache (auto-created on first run)
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pricing_module
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file:
   ```env
   # Database
   DATABASE_URL=sqlite:///pricing_cache.db
   CACHE_TTL_SECONDS=86400

   # Auth (set REQUIRE_API_KEY=true to enforce)
   SERVICE_API_KEY=your_service_key_here
   REQUIRE_API_KEY=false

   # Rate limiting
   RATE_LIMIT_WINDOW_SECONDS=60
   RATE_LIMIT_MAX_REQUESTS=60

   # AI providers (at least one required)
   GEMINI_API_KEY1=your_gemini_key
   GROQ_API_KEY=your_groq_key

   # OMDb (for IMDb auto-fill)
   OMDB_API_KEY=your_omdb_key
   ```

---

## Running the app

```bash
python main.py
```

Open `http://localhost:8000/` in your browser.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Request metrics |
| `GET` | `/countries` | Supported country list |
| `GET` | `/title-metadata` | Fetch IMDb/OMDb metadata for a title (includes per-season episode breakdown) |
| `POST` | `/estimate` | Generate a pricing estimate |
| `POST` | `/export-memo` | Generate and download a PDF deal memo |

---

## `/estimate` payload

```json
{
  "title": "Squid Game",
  "imdb_link": "https://www.imdb.com/title/tt10919420/",
  "tmdb_link": "https://www.themoviedb.org/tv/93405",
  "content_type": "series",
  "season_count": 3,
  "episode_count": 5,
  "included_seasons": "1,2,3",
  "already_acquired_seasons": "1,2",
  "season_episode_counts": { "1": 9, "2": 7, "3": 6 },
  "episode_overrides": { "3": 5 },
  "region": "India",
  "rights_type": "Exclusive",
  "license_duration": "1 year",
  "language_rights": "Dubbed + Subtitled",
  "platforms": ["OTT / Streaming"]
}
```

### Key series fields

| Field | Description |
|-------|-------------|
| `included_seasons` | Seasons in scope for this deal (e.g. `"1,2,3"` or `"1-3"`) |
| `already_acquired_seasons` | Seasons the buyer already holds — triggers incremental pricing |
| `season_episode_counts` | Per-season IMDb episode counts — auto-populated by frontend after auto-fill |
| `episode_overrides` | Per-season custom counts — overrides IMDb breakdown (e.g. holdback on finale) |

### Incremental pricing logic

When `already_acquired_seasons` is set, the engine:
1. Subtracts already-held seasons from the included set to find net-new seasons
2. Uses only net-new seasons' episode counts (from `episode_overrides` → `season_episode_counts` → proportional average) for package sizing
3. Applies a continuation premium instead of a cold standalone discount

---

## `/export-memo` payload

```json
{
  "deal": { <same as /estimate payload> },
  "result": { <response from /estimate> }
}
```

Returns a PDF file as `application/pdf` with `Content-Disposition: attachment`.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///pricing_cache.db` | SQLAlchemy connection string |
| `CACHE_TTL_SECONDS` | `86400` | Cache TTL in seconds |
| `SERVICE_API_KEY` | — | API key for endpoint authentication |
| `REQUIRE_API_KEY` | `false` | Enforce API key on `/estimate` |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window |
| `RATE_LIMIT_MAX_REQUESTS` | `60` | Max requests per window per IP |
| `GEMINI_API_KEY1` | — | Google Gemini API key (primary AI) |
| `GROQ_API_KEY` | — | Groq API key (fallback AI) |
| `OMDB_API_KEY` | — | OMDb key for IMDb metadata fetch |

---

## Testing

```bash
python -m pytest tests/
```

---

## Author

Pratham R
email - pratham.r.108@gmail.com