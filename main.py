from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path
from typing import List, Optional
from google import genai
from google.genai import types
from groq import Groq
from datetime import datetime, timedelta
import os
import httpx
import re
import json
import time
import hashlib
import logging
import pycountry
from math import ceil, exp, log1p
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from collections import defaultdict, deque
import uuid

# ── Environment ────────────────────────────────────────────────────────────────
env_path = Path(__file__).resolve().with_name('.env')
load_dotenv(dotenv_path=env_path)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI()
templates = Jinja2Templates(directory="templates")
DB_PATH = str(Path(__file__).resolve().with_name("pricing_cache.db"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
MODEL_VERSION = "deterministic_v3"
ENGINE: Optional[Engine] = None
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").strip().lower() in {"1", "true", "yes", "on"}
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60"))
METRICS = {
    "requests_total": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "estimate_errors": 0,
    "rate_limited": 0,
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pricing_engine")
RATE_LIMIT_BUCKETS: dict[str, deque] = defaultdict(deque)

# ── AI Clients (initialized once at startup) ───────────────────────────────────
try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY1"))
except Exception:
    gemini_client = None
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    groq_client = None

# ── Data Model ─────────────────────────────────────────────────────────────────
class DealRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    imdb_link: str = Field(min_length=10, max_length=500)
    tmdb_link: Optional[str] = Field(default="", max_length=500)
    country_code: Optional[str] = Field(default="", max_length=2)
    region: str = Field(min_length=1, max_length=100)
    content_type: str = Field(default="movie", min_length=1, max_length=20)
    duration: str = Field(default="N/A", max_length=100)
    runtime_minutes: Optional[int] = Field(default=None, ge=1, le=5000)
    season_count: Optional[int] = Field(default=None, ge=1, le=200)
    episodes_per_season: Optional[int] = Field(default=None, ge=1, le=200)
    episode_count: Optional[int] = Field(default=None, ge=1, le=20000)
    included_seasons: Optional[str] = Field(default="", max_length=200)
    already_acquired_seasons: Optional[str] = Field(default="", max_length=200)
    # Per-season episode breakdown: {season_number: episode_count} — from IMDb via frontend
    season_episode_counts: Optional[dict] = Field(default=None)
    # Per-season episode overrides: {season_number: custom_count} — user-specified
    episode_overrides: Optional[dict] = Field(default=None)
    license_duration: str = Field(min_length=1, max_length=100)
    rights_type: str = Field(min_length=1, max_length=50)
    language_rights: str = Field(min_length=1, max_length=100)
    platforms: List[str] = Field(min_length=1, max_length=10)

    @field_validator("title", "region", "license_duration", "rights_type", "language_rights")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty")
        return cleaned

    @field_validator("duration")
    @classmethod
    def normalize_duration(cls, value: str) -> str:
        cleaned = (value or "").strip()
        return cleaned or "N/A"

    @field_validator("imdb_link")
    @classmethod
    def validate_imdb_link(cls, value: str) -> str:
        cleaned = value.strip()
        if "imdb.com/title/" not in cleaned:
            raise ValueError("imdb_link must be a valid IMDb title URL")
        return cleaned

    @field_validator("tmdb_link")
    @classmethod
    def validate_tmdb_link(cls, value: Optional[str]) -> str:
        cleaned = (value or "").strip()
        if cleaned and "themoviedb.org/" not in cleaned:
            raise ValueError("tmdb_link must be a valid TMDB URL")
        return cleaned

    @field_validator("platforms")
    @classmethod
    def validate_platforms(cls, value: List[str]) -> List[str]:
        cleaned = [platform.strip() for platform in value if platform and platform.strip()]
        if not cleaned:
            raise ValueError("At least one platform is required")
        return cleaned

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, value: Optional[str]) -> str:
        cleaned = (value or "").strip().upper()
        if cleaned and (len(cleaned) != 2 or not cleaned.isalpha()):
            raise ValueError("country_code must be a valid ISO-2 code")
        return cleaned

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"movie", "series"}:
            raise ValueError("content_type must be either 'movie' or 'series'")
        return normalized

    @model_validator(mode="after")
    def validate_structured_content_fields(self):
        if self.content_type == "series":
            has_episode_shape = bool(
                self.episode_count
                or self.season_count
                or (self.season_count and self.episodes_per_season)
            )
            if not has_episode_shape:
                raise ValueError(
                    "For content_type='series', provide episode_count or season_count"
                )
        return self


PLATFORM_MULTIPLIERS = {
    "OTT / Streaming": 1.0,
    "Pay TV": 0.85,
    "Free-to-Air TV": 0.5,
    "FAST channels": 0.2,
    "YouTube": 0.1,
}

LICENSE_MULTIPLIERS = {
    "6 months": 0.45,
    "1 year": 1.0,
    "2 years": 1.7,
    "3 years": 2.2,
    "5 years": 3.0,
    "Perpetual / Permanent": 5.0,
}

MAJOR_MARKETS = {"US", "UK", "India", "Europe", "LATAM", "Australia"}
MID_TIER_MARKETS = {"Southeast Asia", "Middle East", "Eastern Europe"}
SMALL_MARKET_HINTS = [
    "caribbean",
    "pacific islands",
    "sub-saharan africa",
    "central asia",
]

REGION_ALIASES = {
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "u.s.": "US",
    "u.s": "US",
    "united kingdom": "UK",
    "great britain": "UK",
    "britain": "UK",
    "india": "India",
    "latam": "LATAM",
    "latin america": "LATAM",
    "middle east": "Middle East",
    "mena": "Middle East",
    "southeast asia": "Southeast Asia",
    "sea": "Southeast Asia",
    "eastern europe": "Eastern Europe",
    "eu": "Europe",
    "european union": "Europe",
    "australia": "Australia",
}

COUNTRY_MARKET_FACTORS = {
    "US": {"tier": "major", "factor": 1.25},
    "UK": {"tier": "major", "factor": 1.15},
    "India": {"tier": "major", "factor": 1.10},
    "Canada": {"tier": "major", "factor": 1.08},
    "Australia": {"tier": "major", "factor": 1.08},
    "Germany": {"tier": "major", "factor": 1.07},
    "France": {"tier": "major", "factor": 1.07},
    "Japan": {"tier": "major", "factor": 1.10},
    "South Korea": {"tier": "major", "factor": 1.10},
    "Brazil": {"tier": "mid", "factor": 0.95},
    "Mexico": {"tier": "mid", "factor": 0.93},
    "Indonesia": {"tier": "mid", "factor": 0.90},
    "UAE": {"tier": "mid", "factor": 0.95},
    "Saudi Arabia": {"tier": "mid", "factor": 0.98},
    "Turkey": {"tier": "mid", "factor": 0.90},
    "South Africa": {"tier": "mid", "factor": 0.88},
    "Nigeria": {"tier": "small", "factor": 0.75},
    "Kenya": {"tier": "small", "factor": 0.72},
    "Sri Lanka": {"tier": "small", "factor": 0.70},
    "Nepal": {"tier": "small", "factor": 0.68},
    "Pakistan": {"tier": "small", "factor": 0.72},
}
COUNTRY_NAME_TO_CODE: dict[str, str] = {}
COUNTRY_CODE_TO_NAME: dict[str, str] = {}
COUNTRY_CODE_FACTORS: dict[str, dict] = {
    "US": {"tier": "major", "factor": 1.25},
    "GB": {"tier": "major", "factor": 1.15},
    "IN": {"tier": "major", "factor": 1.10},
    "CA": {"tier": "major", "factor": 1.08},
    "AU": {"tier": "major", "factor": 1.08},
    "DE": {"tier": "major", "factor": 1.07},
    "FR": {"tier": "major", "factor": 1.07},
    "JP": {"tier": "major", "factor": 1.10},
    "KR": {"tier": "major", "factor": 1.10},
    "BR": {"tier": "mid", "factor": 0.95},
    "MX": {"tier": "mid", "factor": 0.93},
    "ID": {"tier": "mid", "factor": 0.90},
    "AE": {"tier": "mid", "factor": 0.95},
    "SA": {"tier": "mid", "factor": 0.98},
    "TR": {"tier": "mid", "factor": 0.90},
    "ZA": {"tier": "mid", "factor": 0.88},
    "NG": {"tier": "small", "factor": 0.75},
    "KE": {"tier": "small", "factor": 0.72},
    "LK": {"tier": "small", "factor": 0.70},
    "NP": {"tier": "small", "factor": 0.68},
    "PK": {"tier": "small", "factor": 0.72},
}

# Approximate market signals used to calibrate deterministic market factors.
MARKET_SIGNALS: dict[str, dict[str, float]] = {
    "US": {"gdp_per_capita": 80000.0, "ott_subscribers_millions": 250.0, "arpu_usd": 16.0},
    "GB": {"gdp_per_capita": 52000.0, "ott_subscribers_millions": 35.0, "arpu_usd": 13.0},
    "IN": {"gdp_per_capita": 2700.0, "ott_subscribers_millions": 120.0, "arpu_usd": 2.5},
    "CA": {"gdp_per_capita": 55000.0, "ott_subscribers_millions": 16.0, "arpu_usd": 14.0},
    "AU": {"gdp_per_capita": 65000.0, "ott_subscribers_millions": 12.0, "arpu_usd": 14.0},
    "DE": {"gdp_per_capita": 53000.0, "ott_subscribers_millions": 38.0, "arpu_usd": 11.0},
    "FR": {"gdp_per_capita": 47000.0, "ott_subscribers_millions": 33.0, "arpu_usd": 10.0},
    "JP": {"gdp_per_capita": 39000.0, "ott_subscribers_millions": 45.0, "arpu_usd": 10.0},
    "KR": {"gdp_per_capita": 36000.0, "ott_subscribers_millions": 20.0, "arpu_usd": 10.0},
    "BR": {"gdp_per_capita": 10000.0, "ott_subscribers_millions": 50.0, "arpu_usd": 5.0},
    "MX": {"gdp_per_capita": 13000.0, "ott_subscribers_millions": 27.0, "arpu_usd": 5.0},
    "AE": {"gdp_per_capita": 51000.0, "ott_subscribers_millions": 4.0, "arpu_usd": 11.0},
    "SA": {"gdp_per_capita": 28000.0, "ott_subscribers_millions": 9.0, "arpu_usd": 8.0},
    "ZA": {"gdp_per_capita": 7000.0, "ott_subscribers_millions": 7.0, "arpu_usd": 5.0},
    "NG": {"gdp_per_capita": 2200.0, "ott_subscribers_millions": 10.0, "arpu_usd": 2.0},
}


def parse_money(value: Optional[str]) -> int:
    if not value or value == "N/A":
        return 0
    digits = re.sub(r"[^\d]", "", value)
    return int(digits) if digits else 0


def parse_votes(value: Optional[str]) -> int:
    if not value or value == "N/A":
        return 0
    digits = re.sub(r"[^\d]", "", value)
    return int(digits) if digits else 0


def parse_release_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"\d{4}", value)
    return int(match.group(0)) if match else None


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, "", "N/A"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def log_event(event: str, **fields: object) -> None:
    payload = {"event": event, "ts": datetime.utcnow().isoformat() + "Z", **fields}
    logger.info(json.dumps(payload, default=str))


def init_country_catalog() -> None:
    for country in pycountry.countries:
        code = country.alpha_2.upper()
        name = country.name
        COUNTRY_CODE_TO_NAME[code] = name
        COUNTRY_NAME_TO_CODE[name.lower()] = code
        if hasattr(country, "official_name"):
            COUNTRY_NAME_TO_CODE[country.official_name.lower()] = code
    COUNTRY_NAME_TO_CODE.update(
        {
            "usa": "US",
            "u.s.": "US",
            "u.s": "US",
            "uk": "GB",
            "uae": "AE",
            "south korea": "KR",
            "north korea": "KP",
            "russia": "RU",
            "vietnam": "VN",
            "laos": "LA",
            "trinidad": "TT",
            "trinidad and tobago": "TT",
        }
    )


def resolve_country(country_code: str, region: str) -> tuple[str, str]:
    code = (country_code or "").strip().upper()
    if code and code in COUNTRY_CODE_TO_NAME:
        return code, COUNTRY_CODE_TO_NAME[code]
    lookup = region.strip().lower()
    mapped = COUNTRY_NAME_TO_CODE.get(lookup)
    if mapped:
        return mapped, COUNTRY_CODE_TO_NAME.get(mapped, region.strip())
    return "ZZ", region.strip() or "Unknown"


def normalize_region(region: str) -> str:
    cleaned = region.strip()
    if not cleaned:
        return "Unknown"
    canonical = REGION_ALIASES.get(cleaned.lower())
    return canonical if canonical else cleaned


def get_market_factor(country_code: str, region: str) -> dict:
    factor = COUNTRY_CODE_FACTORS.get(country_code)
    if not factor:
        factor = COUNTRY_MARKET_FACTORS.get(region)
    if not factor:
        if region in MAJOR_MARKETS:
            factor = {"tier": "major", "factor": 1.0}
        elif region in MID_TIER_MARKETS:
            factor = {"tier": "mid", "factor": 0.9}
        elif is_small_or_emerging_market(region):
            factor = {"tier": "small", "factor": 0.75}
        else:
            factor = {"tier": "unknown", "factor": 0.85}

    signal = MARKET_SIGNALS.get(country_code)
    if not signal:
        return factor

    # Blend static mapping with normalized market signals.
    gdp_norm = min(log1p(signal["gdp_per_capita"]) / log1p(80_000.0), 1.2)
    ott_norm = min(log1p(signal["ott_subscribers_millions"]) / log1p(250.0), 1.2)
    arpu_norm = min(signal["arpu_usd"] / 16.0, 1.2)
    dynamic_score = (gdp_norm * 0.4) + (ott_norm * 0.3) + (arpu_norm * 0.3)
    dynamic_factor = 0.7 + (dynamic_score * 0.7)
    blended_factor = round((factor["factor"] * 0.55) + (dynamic_factor * 0.45), 3)

    if blended_factor >= 1.05:
        tier = "major"
    elif blended_factor >= 0.9:
        tier = "mid"
    else:
        tier = "small"
    return {"tier": tier, "factor": blended_factor}


def license_multiplier(license_duration: str) -> float:
    normalized = license_duration.strip().lower()
    for key, value in LICENSE_MULTIPLIERS.items():
        if normalized == key.lower():
            return value

    if "perpetual" in normalized or "permanent" in normalized or "lifetime" in normalized:
        return 5.0

    number_match = re.search(r"(\d+(?:\.\d+)?)", normalized)
    if not number_match:
        return 1.0

    quantity = float(number_match.group(1))
    years = quantity / 12.0 if "month" in normalized else quantity

    if years <= 0:
        return 1.0

    # Diminishing-return curve: first years carry most of the economic value.
    curve = 1.0 + (1.35 * (1.0 - exp(-0.55 * years)))
    if years > 8:
        curve += min((years - 8) * 0.05, 0.2)
    return round(min(curve, 5.0), 3)


def normalize_platforms(platforms: List[str]) -> List[str]:
    aliases = {
        "ott": "OTT / Streaming",
        "streaming": "OTT / Streaming",
        "paytv": "Pay TV",
        "pay tv": "Pay TV",
        "free to air tv": "Free-to-Air TV",
        "fta": "Free-to-Air TV",
        "fast": "FAST channels",
        "youtube": "YouTube",
    }
    normalized = []
    for platform in platforms:
        cleaned = platform.strip()
        mapped = aliases.get(cleaned.lower(), cleaned)
        normalized.append(mapped)
    return normalized


def score_content(omdb_data: dict, tmdb_data: dict) -> float:
    rating = max(0.0, min(safe_float(omdb_data.get("imdb_rating"), 0.0) / 10.0, 1.0))
    votes = parse_votes(omdb_data.get("imdb_votes"))
    votes_norm = min(log1p(votes) / log1p(2_000_000), 1.0)
    popularity = max(0.0, safe_float(tmdb_data.get("popularity"), 0.0))
    popularity_norm = min(log1p(popularity) / log1p(100.0), 1.0)
    box_office = parse_money(omdb_data.get("box_office"))
    box_office_norm = min(log1p(box_office) / log1p(2_000_000_000), 1.0)

    weighted = (rating * 0.3) + (votes_norm * 0.2) + (box_office_norm * 0.3) + (popularity_norm * 0.2)
    return round(max(0.0, min(weighted * 100.0, 100.0)), 2)


def base_price_from_score(score: float) -> tuple[int, int]:
    if score < 20:
        return 1_000, 10_000
    if score < 40:
        return 10_000, 50_000
    if score < 60:
        return 50_000, 150_000
    if score < 80:
        return 150_000, 500_000
    return 500_000, 2_000_000


def platform_multiplier(platforms: List[str]) -> float:
    if not platforms:
        return 0.5
    return max(PLATFORM_MULTIPLIERS.get(platform, 0.5) for platform in platforms)


def rights_multiplier(rights_type: str) -> float:
    return 2.5 if rights_type.lower() == "exclusive" else 1.0


def language_multiplier(language_rights: str) -> float:
    has_dubbed = "dubbed" in language_rights.lower()
    has_subtitled = "subtitled" in language_rights.lower()
    if has_dubbed and has_subtitled:
        return 1.4
    if has_dubbed:
        return 1.3
    return 1.0


def age_multiplier(
    release_year: Optional[int], current_year: int, is_perpetual: bool, box_office: int
) -> float:
    if not release_year:
        return 1.0

    age = current_year - release_year
    if age <= 1:
        factor = 1.4
    elif age <= 3:
        factor = 1.0
    elif age <= 7:
        factor = 0.5
    elif age <= 15:
        factor = 0.25
    else:
        factor = 0.1

    if is_perpetual:
        factor = (factor + 1.0) / 2.0

    # Older breakout hits decay slower, but still decay.
    if box_office >= 500_000_000:
        factor = max(factor, 0.35 if age > 15 else 0.5)
    elif box_office >= 100_000_000:
        factor = max(factor, 0.2 if age > 15 else 0.35)

    return factor


def is_small_or_emerging_market(region: str) -> bool:
    region_lower = region.lower()
    return any(hint in region_lower for hint in SMALL_MARKET_HINTS)


def should_trigger_low_data_rule(
    deal: DealRequest, normalized_region: str, tmdb_data: dict, box_office: int, votes: int
) -> bool:
    popularity = safe_float(tmdb_data.get("popularity"), 0.0)
    is_series = (
        deal.content_type == "series"
        or "episodes" in deal.duration.lower()
        or "season" in deal.title.lower()
    )
    missing_or_low_box_office = (box_office < 5_000_000) and not is_series
    return (
        missing_or_low_box_office
        or votes < 10_000
        or popularity < 3.0
        or is_small_or_emerging_market(normalized_region)
    )


def parse_included_seasons(s: str) -> list[int]:
    """Parse '1,2,3' or '1-3' or '2' into a sorted deduplicated list of ints."""
    seasons: list[int] = []
    if not s or not s.strip():
        return seasons
    for part in re.split(r"[,;]", s):
        part = part.strip()
        m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
        if m:
            seasons.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif re.match(r"^\d+$", part):
            seasons.append(int(part))
    return sorted(set(seasons))


def resolve_net_new_episodes(deal: DealRequest) -> int:
    """
    Return the episode count that should drive package sizing.

    Priority chain (highest to lowest):
    1. episode_overrides for net-new seasons  — user explicitly specified per-season counts
    2. season_episode_counts breakdown        — IMDb per-season counts
    3. episode_count                          — total from IMDb or frontend
    4. season_count * episodes_per_season     — manual fallback
    5. season_count * 10                      — last resort default
    """
    included = parse_included_seasons(deal.included_seasons or "")
    acquired = parse_included_seasons(deal.already_acquired_seasons or "")
    net_new  = [s for s in included if s not in acquired] if acquired else included

    breakdown: dict[int, int] = {}
    if deal.season_episode_counts:
        try:
            breakdown = {int(k): int(v) for k, v in deal.season_episode_counts.items()}
        except (ValueError, TypeError):
            breakdown = {}

    overrides: dict[int, int] = {}
    if deal.episode_overrides:
        try:
            overrides = {int(k): int(v) for k, v in deal.episode_overrides.items()}
        except (ValueError, TypeError):
            overrides = {}

    # Which seasons to price on
    target_seasons = net_new if net_new else (
        [s for s in included if 1 <= s <= (deal.season_count or 9999)] if included else []
    )

    if target_seasons:
        total = 0
        for s in target_seasons:
            if s in overrides:
                total += overrides[s]           # user override wins
            elif s in breakdown:
                total += breakdown[s]           # IMDb breakdown
            elif deal.season_count and deal.season_count > 0:
                avg = (deal.episode_count or deal.season_count * 10) / deal.season_count
                total += round(avg)
            else:
                total += 10
        return max(1, total)

    # No season restriction — full series
    if deal.episode_count:
        return deal.episode_count
    if deal.season_count and deal.episodes_per_season:
        return deal.season_count * deal.episodes_per_season
    if deal.season_count:
        return deal.season_count * 10
    return 10


def compute_season_ratio(deal: DealRequest) -> Optional[float]:
    """
    Fraction of the show's total seasons being newly licensed.
    Returns None for full-series deals (no discount applied).
    """
    if deal.content_type != "series":
        return None
    included = parse_included_seasons(deal.included_seasons or "")
    if not included:
        return None
    total = deal.season_count
    if not total or total <= 0:
        return None
    acquired = parse_included_seasons(deal.already_acquired_seasons or "")
    if acquired:
        net_new = [s for s in included if s not in acquired and 1 <= s <= total]
        return min(len(net_new) / total, 1.0) if net_new else None
    licensed = [s for s in included if 1 <= s <= total]
    return min(len(licensed) / total, 1.0) if licensed else None


def series_package_multiplier(deal: DealRequest) -> float:
    if deal.content_type != "series":
        return 1.0

    net_episodes    = resolve_net_new_episodes(deal)
    runtime_minutes = deal.runtime_minutes or 45
    package_minutes = net_episodes * runtime_minutes

    episodes_factor = 1.0 + min(log1p(net_episodes) / 5.0, 0.9)
    runtime_factor  = 1.0 + min(log1p(package_minutes) / 10.0, 0.5)
    base = round(min(episodes_factor * runtime_factor, 2.2), 3)

    acquired     = parse_included_seasons(deal.already_acquired_seasons or "")
    season_ratio = compute_season_ratio(deal)

    if season_ratio is None:
        return base  # Full series — no adjustment

    if acquired:
        # Incremental: continuation premium over cold standalone single-season value
        standalone = round(1.0 + min(log1p(net_episodes) / 5.0, 0.9), 3)
        return round(min(standalone * 1.10, base), 3)
    else:
        # Partial fresh deal: non-linear retained-value curve
        retained = 0.4 + (season_ratio * 0.6)
        return round(base * retained, 3)


def compute_confidence(
    market_tier: str, omdb_data: dict, tmdb_data: dict, low_data_rule: bool
) -> str:
    if low_data_rule:
        return "Low"

    votes = parse_votes(omdb_data.get("imdb_votes"))
    box_office = parse_money(omdb_data.get("box_office"))
    popularity = safe_float(tmdb_data.get("popularity"), 0.0)
    has_tmdb = bool(tmdb_data)
    has_omdb = bool(omdb_data)

    if (
        box_office > 5_000_000
        and votes > 50_000
        and popularity > 10
        and market_tier == "major"
    ):
        return "High"

    if (
        (5_000_000 <= box_office <= 100_000_000)
        or (10_000 <= votes <= 50_000)
        or (market_tier == "mid")
        or (has_tmdb != has_omdb)
    ):
        return "Medium"

    return "Low"


def compute_revenue_share_range(platforms: List[str]) -> str:
    if "YouTube" in platforms:
        return "40% - 60%"
    if "FAST channels" in platforms:
        return "30% - 50%"
    return "10% - 25%"


def format_usd_range(min_value: int, max_value: int) -> str:
    return f"USD {min_value:,} - USD {max_value:,}"


def fetch_json_with_retries(url: str, timeout: int = 5, retries: int = 2) -> dict:
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = httpx.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
    log_event("external_api_fetch_failed", url=url, error=str(last_error))
    return {}


def init_db() -> None:
    global ENGINE
    ENGINE = create_engine(DATABASE_URL, future=True)
    with ENGINE.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS pricing_estimates (
                    deal_key TEXT PRIMARY KEY,
                    request_payload TEXT NOT NULL,
                    response_payload TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS pricing_audit (
                    id TEXT PRIMARY KEY,
                    deal_key TEXT NOT NULL,
                    request_payload TEXT NOT NULL,
                    response_payload TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        )
        # Backward-compatible migration for existing SQLite DBs.
        try:
            conn.execute(text("ALTER TABLE pricing_estimates ADD COLUMN expires_at TEXT"))
        except Exception:
            pass


def generate_deal_key(deal: DealRequest, normalized_region: str, normalized_platforms: List[str]) -> str:
    canonical_payload = {
        "model_version": MODEL_VERSION,
        "title": deal.title.strip().lower(),
        "imdb_link": deal.imdb_link.strip().lower(),
        "tmdb_link": (deal.tmdb_link or "").strip().lower(),
        "region": normalized_region.strip().lower(),
        "content_type": deal.content_type.strip().lower(),
        "duration": deal.duration.strip().lower(),
        "runtime_minutes": deal.runtime_minutes or 0,
        "season_count": deal.season_count or 0,
        "episodes_per_season": deal.episodes_per_season or 0,
        "episode_count": deal.episode_count or 0,
        "included_seasons": (deal.included_seasons or "").strip().lower(),
        "already_acquired_seasons": (deal.already_acquired_seasons or "").strip().lower(),
        "episode_overrides": json.dumps(deal.episode_overrides or {}, sort_keys=True),
        "license_duration": deal.license_duration.strip().lower(),
        "rights_type": deal.rights_type.strip().lower(),
        "language_rights": deal.language_rights.strip().lower(),
        "platforms": sorted([p.strip().lower() for p in normalized_platforms]),
    }
    encoded = json.dumps(canonical_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_cached_estimate(deal_key: str) -> Optional[dict]:
    if ENGINE is None:
        return None
    with ENGINE.begin() as conn:
        row = conn.execute(
            text(
                "SELECT response_payload FROM pricing_estimates "
                "WHERE deal_key = :deal_key AND model_version = :model_version "
                "AND expires_at > :now_ts"
            ),
            {
                "deal_key": deal_key,
                "model_version": MODEL_VERSION,
                "now_ts": datetime.utcnow().isoformat() + "Z",
            },
        ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row[0])
        payload["cache_hit"] = True
        return payload
    except json.JSONDecodeError:
        return None


def store_estimate(
    deal_key: str, request_payload: dict, response_payload: dict, model_version: str
) -> None:
    if ENGINE is None:
        return
    with ENGINE.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pricing_estimates
                (deal_key, request_payload, response_payload, model_version, created_at, expires_at)
                VALUES (:deal_key, :request_payload, :response_payload, :model_version, :created_at, :expires_at)
                ON CONFLICT(deal_key) DO UPDATE SET
                    request_payload=excluded.request_payload,
                    response_payload=excluded.response_payload,
                    model_version=excluded.model_version,
                    created_at=excluded.created_at,
                    expires_at=excluded.expires_at
                """
            ),
            {
                "deal_key": deal_key,
                "request_payload": json.dumps(request_payload, sort_keys=True),
                "response_payload": json.dumps(response_payload),
                "model_version": model_version,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "expires_at": (datetime.utcnow() + timedelta(seconds=CACHE_TTL_SECONDS)).isoformat() + "Z",
            },
        )


def store_audit_record(
    deal_key: str,
    request_payload: dict,
    response_payload: dict,
    model_version: str,
    cache_hit: bool,
    latency_ms: int,
) -> None:
    if ENGINE is None:
        return
    with ENGINE.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pricing_audit
                (id, deal_key, request_payload, response_payload, model_version, cache_hit, latency_ms, created_at)
                VALUES (:id, :deal_key, :request_payload, :response_payload, :model_version, :cache_hit, :latency_ms, :created_at)
                """
            ),
            {
                "id": uuid.uuid4().hex,
                "deal_key": deal_key,
                "request_payload": json.dumps(request_payload, sort_keys=True),
                "response_payload": json.dumps(response_payload),
                "model_version": model_version,
                "cache_hit": 1 if cache_hit else 0,
                "latency_ms": latency_ms,
                "created_at": datetime.utcnow().isoformat() + "Z",
            },
        )


def deterministic_reasoning(
    deal: DealRequest,
    confidence: str,
    score: float,
    min_price: int,
    max_price: int,
    market_tier: str,
    low_data_rule: bool,
) -> str:
    rights_bias = "upward" if deal.rights_type.lower() == "exclusive" else "neutral"
    data_note = (
        "Data coverage is limited in this case, so the recommendation should be treated as a conservative anchor."
        if low_data_rule
        else "Data coverage is adequate for directional negotiation planning."
    )
    strategy_note = (
        f"Open negotiations near {format_usd_range(int(max_price * 0.9), max_price)} and protect a floor near "
        f"{format_usd_range(min_price, int(min_price * 1.1))}."
    )
    return (
        f"{deal.title} is priced at {format_usd_range(min_price, max_price)} for {deal.region} based on a "
        f"{score:.1f}/100 content score, {deal.rights_type.lower()} rights, {deal.license_duration.lower()} term, "
        f"and distribution across {', '.join(deal.platforms)}. Confidence is {confidence.lower()} in a "
        f"{market_tier.lower()} market, with deal structure biasing valuation {rights_bias}. {data_note} {strategy_note}"
    )


def generate_reasoning_with_ai(
    deal: DealRequest,
    confidence: str,
    score: float,
    min_price: int,
    max_price: int,
    market_tier: str,
    low_data_rule: bool,
) -> str:
    data_context = "limited" if low_data_rule else "adequate"
    prompt = f"""
    You are a senior film licensing consultant. Explain the deterministic pricing output in exactly 4 concise business sentences.
    Do not change numbers.
    Keep the tone executive and practical, not generic.
    Include:
    1) Main valuation drivers (rights, term, territory, platform scope),
    2) Why confidence is {confidence} using market/data context,
    3) What this implies for risk in negotiation,
    4) A clear negotiation posture with opening-anchor and protected floor language.

    Deal:
    - Title: {deal.title}
    - Region: {deal.region}
    - Market tier: {market_tier}
    - Data coverage: {data_context}
    - Platforms: {", ".join(deal.platforms)}
    - Rights Type: {deal.rights_type}
    - License Duration: {deal.license_duration}
    - Language Rights: {deal.language_rights}

    Deterministic output:
    - Content score: {score:.1f}/100
    - Confidence: {confidence}
    - Flat fee range: {format_usd_range(min_price, max_price)}

    Hard constraints:
    - Keep all numeric values exactly as provided.
    - Do not mention "AI", "model", or "deterministic" in the final answer.
    - Return plain text only.
    """
    try:
        return call_gemini(prompt, expect_json=False)
    except Exception:
        try:
            return call_groq(prompt, expect_json=False)
        except Exception:
            return deterministic_reasoning(
                deal,
                confidence,
                score,
                min_price,
                max_price,
                market_tier,
                low_data_rule,
            )

# ── Enrichment Functions ───────────────────────────────────────────────────────
def fetch_tmdb_data(tmdb_link: str) -> dict:
    try:
        match = re.search(r'/(movie|tv)/(\d+)', tmdb_link)
        if not match:
            return {}
        media_type = match.group(1)
        tmdb_id    = match.group(2)
        api_key    = os.getenv("TMDB_API_KEY")
        url        = f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}?api_key={api_key}"
        data       = fetch_json_with_retries(url, timeout=5, retries=2)
        if not data:
            return {}
        return {
            "popularity":   data.get("popularity"),
            "vote_average": data.get("vote_average"),
            "vote_count":   data.get("vote_count"),
            "genres":       [g["name"] for g in data.get("genres", [])],
            "budget":       data.get("budget"),
            "revenue":      data.get("revenue"),
            "status":       data.get("status"),
        }
    except Exception:
        return {}


def fetch_omdb_data(imdb_link: str) -> dict:
    try:
        match = re.search(r'(tt\d+)', imdb_link)
        if not match:
            return {}
        imdb_id  = match.group(1)
        api_key  = os.getenv("OMDB_API_KEY")
        url      = f"https://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
        data     = fetch_json_with_retries(url, timeout=5, retries=2)
        if not data or data.get("Response") == "False":
            return {}
        return {
            "imdb_rating":    data.get("imdbRating"),
            "imdb_votes":     data.get("imdbVotes"),
            "box_office":     data.get("BoxOffice"),
            "awards":         data.get("Awards"),
            "metascore":      data.get("Metascore"),
            "release_year":   data.get("Year"),
            "rotten_tomatoes": next(
                (r["Value"] for r in data.get("Ratings", [])
                 if r["Source"] == "Rotten Tomatoes"), None
            ),
        }
    except Exception:
        return {}


def fetch_title_metadata(imdb_link: str) -> dict:
    try:
        match = re.search(r"(tt\d+)", imdb_link or "")
        if not match:
            return {"found": False}

        imdb_id = match.group(1)
        api_key = os.getenv("OMDB_API_KEY")
        base_url = f"https://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
        data = fetch_json_with_retries(base_url, timeout=5, retries=2)
        if not data or data.get("Response") == "False":
            return {"found": False}

        runtime_match = re.search(r"(\d+)", data.get("Runtime", ""))
        runtime_minutes = int(runtime_match.group(1)) if runtime_match else None

        content_type = "series" if data.get("Type", "").lower() == "series" else "movie"
        total_seasons = int(data.get("totalSeasons", "0")) if str(data.get("totalSeasons", "")).isdigit() else None

        released_episode_count = None
        season_episode_counts: dict[int, int] = {}
        if content_type == "series" and total_seasons and total_seasons > 0:
            released_count = 0
            for season_number in range(1, total_seasons + 1):
                season_url = f"{base_url}&Season={season_number}"
                season_data = fetch_json_with_retries(season_url, timeout=5, retries=1)
                episodes = season_data.get("Episodes", []) if season_data else []
                count = len(episodes)
                if count > 0:
                    season_episode_counts[season_number] = count
                released_count += count
            released_episode_count = released_count if released_count > 0 else None

        tmdb_link = ""
        tmdb_api_key = os.getenv("TMDB_API_KEY")
        if tmdb_api_key:
            find_url = (
                f"https://api.themoviedb.org/3/find/{imdb_id}"
                f"?api_key={tmdb_api_key}&external_source=imdb_id"
            )
            tmdb_find_data = fetch_json_with_retries(find_url, timeout=5, retries=1)
            movie_results = tmdb_find_data.get("movie_results", []) if tmdb_find_data else []
            tv_results = tmdb_find_data.get("tv_results", []) if tmdb_find_data else []
            if movie_results:
                tmdb_link = f"https://www.themoviedb.org/movie/{movie_results[0].get('id')}"
            elif tv_results:
                tmdb_link = f"https://www.themoviedb.org/tv/{tv_results[0].get('id')}"

        return {
            "found": True,
            "imdb_id": imdb_id,
            "title": data.get("Title", ""),
            "content_type": content_type,
            "runtime_minutes": runtime_minutes,
            "total_seasons": total_seasons,
            "released_episode_count": released_episode_count,
            "season_episode_counts": season_episode_counts,
            "released_label": data.get("Released", ""),
            "tmdb_link": tmdb_link,
        }
    except Exception:
        return {"found": False}

# ── AI Call Functions ──────────────────────────────────────────────────────────
def call_gemini(prompt: str, expect_json: bool = True) -> str:
    if gemini_client is None:
        raise RuntimeError("Gemini client not initialized")
    log_event("ai_provider_selected", provider="gemini", expect_json=expect_json)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=1024),
        temperature=0.3
    )
    if expect_json:
        config.response_mime_type = "application/json"

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config
    )
    return response.text.strip()


def call_groq(prompt: str, expect_json: bool = True) -> str:
    if groq_client is None:
        raise RuntimeError("Groq client not initialized")
    log_event("ai_provider_selected", provider="groq", expect_json=expect_json)
    system_prompt = (
        "You are a film licensing consultant. Always respond with valid JSON only. Never add explanations or markdown formatting outside the JSON."
        if expect_json
        else "You are a film licensing consultant. Return concise plain text only."
    )
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    result = response.choices[0].message.content.strip()
    # Strip markdown fences if Groq wraps response in markdown.
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result[4:]
        result = result.strip()
    return result

# ── Fallback Response ──────────────────────────────────────────────────────────
def unavailable_response(deal: DealRequest) -> dict:
    return {
        "title": deal.title,
        "region": deal.region,
        "pricing_estimate": {
            "flat_fee_range": "Unavailable",
            "minimum_guarantee": "Unavailable",
            "revenue_share_range": "Unavailable"
        },
        "confidence_level": "Low",
        "reasoning": "We were unable to generate an estimate at this time. Please try again."
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path in {"/health", "/"}:
        return await call_next(request)

    if REQUIRE_API_KEY and not SERVICE_API_KEY:
        return JSONResponse(status_code=503, content={"detail": "Service API key is required but not configured"})

    if SERVICE_API_KEY:
        provided_key = request.headers.get("x-api-key", "")
        if provided_key != SERVICE_API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

    client_ip = (request.client.host if request.client else "unknown") or "unknown"
    now = time.time()
    bucket = RATE_LIMIT_BUCKETS[client_ip]
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        METRICS["rate_limited"] += 1
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )
    bucket.append(now)
    return await call_next(request)


@app.get("/health")
def health():
    db_ok = False
    db_error = None
    if ENGINE is not None:
        try:
            with ENGINE.begin() as conn:
                conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:
            db_error = str(exc)

    ai_readiness = {
        "gemini_configured": gemini_client is not None,
        "groq_configured": groq_client is not None,
    }
    status = "ok" if db_ok else "degraded"
    payload = {
        "status": status,
        "service": "pricing-module",
        "model_version": MODEL_VERSION,
        "database": {
            "configured_url": DATABASE_URL.split("://")[0],
            "ready": db_ok,
            "error": db_error,
        },
        "ai": ai_readiness,
    }
    return payload


@app.get("/metrics")
def metrics():
    cache_requests = METRICS["cache_hits"] + METRICS["cache_misses"]
    cache_hit_rate = (
        round((METRICS["cache_hits"] / cache_requests) * 100, 2) if cache_requests else 0.0
    )
    return {**METRICS, "cache_hit_rate_percent": cache_hit_rate}


@app.get("/countries")
def countries():
    data = [
        {"code": code, "name": name}
        for code, name in sorted(COUNTRY_CODE_TO_NAME.items(), key=lambda x: x[1])
    ]
    return {"countries": data}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/title-metadata")
def title_metadata(imdb_link: str):
    if "imdb.com/title/" not in imdb_link:
        raise HTTPException(status_code=400, detail="Provide a valid IMDb title URL")
    metadata = fetch_title_metadata(imdb_link)
    if not metadata.get("found"):
        raise HTTPException(status_code=404, detail="Could not fetch title metadata from IMDb/OMDb")
    return metadata


@app.on_event("startup")
def on_startup():
    init_country_catalog()
    init_db()
    if not SERVICE_API_KEY:
        log_event("security_warning", message="SERVICE_API_KEY not configured; API key auth disabled")


@app.post("/estimate")
def estimate(deal: DealRequest):
    started = time.time()
    METRICS["requests_total"] += 1
    try:
        current_year = datetime.now().year
        normalized_region = normalize_region(deal.region)
        resolved_country_code, resolved_country_name = resolve_country(deal.country_code, normalized_region)
        market_info = get_market_factor(resolved_country_code, resolved_country_name)
        normalized_platforms = normalize_platforms(deal.platforms)
        title_metadata = fetch_title_metadata(deal.imdb_link)
        if deal.content_type == "series":
            released_cap = title_metadata.get("released_episode_count") if title_metadata else None
            # Sync per-season breakdown from IMDb if frontend didn't send it
            if (title_metadata and title_metadata.get("season_episode_counts")
                    and not deal.season_episode_counts):
                object.__setattr__(deal, "season_episode_counts",
                                   title_metadata["season_episode_counts"])

        deal_key = generate_deal_key(deal, resolved_country_name, normalized_platforms)
        cached_response = get_cached_estimate(deal_key)
        if cached_response:
            METRICS["cache_hits"] += 1
            cached_response["model_version"] = MODEL_VERSION
            store_audit_record(
                deal_key=deal_key,
                request_payload=deal.dict(),
                response_payload=cached_response,
                model_version=MODEL_VERSION,
                cache_hit=True,
                latency_ms=int((time.time() - started) * 1000),
            )
            log_event("estimate_served", deal_key=deal_key, cache_hit=True, market_tier=market_info["tier"])
            return cached_response
        METRICS["cache_misses"] += 1

        tmdb_data = fetch_tmdb_data(deal.tmdb_link) if deal.tmdb_link else {}
        omdb_data = fetch_omdb_data(deal.imdb_link) if deal.imdb_link else {}
        score = score_content(omdb_data, tmdb_data)
        base_min, base_max = base_price_from_score(score)

        platform_factor = platform_multiplier(normalized_platforms)
        rights_factor = rights_multiplier(deal.rights_type)
        language_factor = language_multiplier(deal.language_rights)
        license_factor = license_multiplier(deal.license_duration)
        market_factor = market_info["factor"]
        package_factor  = series_package_multiplier(deal)
        season_ratio    = compute_season_ratio(deal)
        acquired_list   = parse_included_seasons(deal.already_acquired_seasons or "")
        included_list   = parse_included_seasons(deal.included_seasons or "")
        net_new_list    = [s for s in included_list if s not in acquired_list] if acquired_list else []
        net_episodes    = resolve_net_new_episodes(deal)
        multiplier = (
            platform_factor
            * rights_factor
            * language_factor
            * license_factor
            * market_factor
            * package_factor
        )

        is_perpetual = "perpetual" in deal.license_duration.lower() or "permanent" in deal.license_duration.lower()
        release_year = parse_release_year(omdb_data.get("release_year"))
        box_office = parse_money(omdb_data.get("box_office"))
        age_factor = age_multiplier(release_year, current_year, is_perpetual, box_office)
        votes = parse_votes(omdb_data.get("imdb_votes"))
        low_data_rule = should_trigger_low_data_rule(deal, resolved_country_name, tmdb_data, box_office, votes)

        min_price = int(ceil(base_min * multiplier * age_factor))
        max_price = int(ceil(base_max * multiplier * age_factor))

        if low_data_rule:
            max_price = min(max_price, 80_000)
        min_price = max(1_000, min_price)
        max_price = max(min_price, max_price)

        mg_min = int(min_price * 0.4)
        mg_max = int(max_price * 0.6)
        if low_data_rule:
            mg_max = min(mg_max, 30_000)
            mg_min = min(mg_min, mg_max)

        confidence = compute_confidence(market_info["tier"], omdb_data, tmdb_data, low_data_rule)
        reasoning = generate_reasoning_with_ai(
            deal,
            confidence,
            score,
            min_price,
            max_price,
            market_info["tier"],
            low_data_rule,
        )

        response_payload = {
            "title": deal.title,
            "region": resolved_country_name,
            "country_code": resolved_country_code,
            "market_tier": market_info["tier"],
            "pricing_estimate": {
                "flat_fee_range": format_usd_range(min_price, max_price),
                "minimum_guarantee": format_usd_range(mg_min, mg_max),
                "revenue_share_range": compute_revenue_share_range(normalized_platforms),
            },
            "pricing_components": {
                "score": score,
                "base_price_range": format_usd_range(base_min, base_max),
                "multipliers": {
                    "platform": platform_factor,
                    "rights": rights_factor,
                    "language": language_factor,
                    "license": license_factor,
                    "market": market_factor,
                    "package": package_factor,
                    "age": age_factor,
                    "combined": round(multiplier * age_factor, 4),
                },
                "season_context": (
                    {
                        "mode": "incremental",
                        "included_seasons": deal.included_seasons,
                        "already_acquired_seasons": deal.already_acquired_seasons,
                        "net_new_seasons": net_new_list,
                        "net_new_episodes": net_episodes,
                        "note": (
                            f"Incremental acquisition — "
                            f"Season{'s' if len(net_new_list) > 1 else ''} "
                            f"{', '.join(str(s) for s in net_new_list)} · "
                            f"{net_episodes} episode{'s' if net_episodes != 1 else ''} · "
                            f"Seasons {deal.already_acquired_seasons} already held by licensee."
                        ),
                    }
                    if acquired_list and included_list
                    else {
                        "mode": "partial",
                        "included_seasons": deal.included_seasons,
                        "season_ratio_pct": round((season_ratio or 1.0) * 100),
                        "net_new_episodes": net_episodes,
                        "note": (
                            f"Partial series — "
                            f"Season{'s' if len(included_list) > 1 else ''} "
                            f"{deal.included_seasons} of {deal.season_count} total · "
                            f"{round((season_ratio or 1.0) * 100)}% coverage · "
                            f"{net_episodes} episode{'s' if net_episodes != 1 else ''} priced."
                        ),
                    }
                    if season_ratio is not None
                    else None
                ),
            },
            "confidence_level": confidence,
            "reasoning": reasoning,
            "model_version": MODEL_VERSION,
            "cache_hit": False,
        }

        store_estimate(
            deal_key=deal_key,
            request_payload=deal.dict(),
            response_payload=response_payload,
            model_version=MODEL_VERSION,
        )
        store_audit_record(
            deal_key=deal_key,
            request_payload=deal.dict(),
            response_payload=response_payload,
            model_version=MODEL_VERSION,
            cache_hit=False,
            latency_ms=int((time.time() - started) * 1000),
        )
        log_event("estimate_served", deal_key=deal_key, cache_hit=False, market_tier=market_info["tier"])
        return response_payload
    except HTTPException:
        raise
    except Exception as exc:
        METRICS["estimate_errors"] += 1
        log_event("estimate_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Unable to generate estimate at this time")
    
 
# ══════════════════════════════════════════════════════════════════════════════
# ── PDF Deal Memo ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
 
# ── Brand palette ──────────────────────────────────────────────────────────────
_DARK        = colors.HexColor("#0b1018")
_MID         = colors.HexColor("#111927")
_ACCENT      = colors.HexColor("#6f90ff")
_ACCENT_SOFT = colors.HexColor("#1a2f5a")
_TEXT        = colors.HexColor("#1a2840")
_MUTED       = colors.HexColor("#5a6a82")
_BORDER      = colors.HexColor("#c8d3e8")
_ROW_ALT     = colors.HexColor("#f4f6fb")
_GREEN       = colors.HexColor("#1d9e75")
_AMBER       = colors.HexColor("#ba7517")
_RED         = colors.HexColor("#d85a30")
_WHITE       = colors.white
 
 
def _styles() -> dict:
    base = getSampleStyleSheet()
    def s(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)
    return {
        "cover_eyebrow": s("cover_eyebrow",
            fontName="Helvetica-Bold", fontSize=8, leading=12,
            textColor=colors.HexColor("#aabbd4"),
            spaceAfter=6, letterSpacing=1.4),
        "cover_title": s("cover_title",
            fontName="Helvetica-Bold", fontSize=24, leading=30,
            textColor=_WHITE, spaceAfter=4),
        "cover_sub": s("cover_sub",
            fontName="Helvetica", fontSize=10, leading=15,
            textColor=colors.HexColor("#aabbd4"), spaceAfter=2),
        "section_head": s("section_head",
            fontName="Helvetica-Bold", fontSize=8, leading=11,
            textColor=_ACCENT, spaceBefore=20, spaceAfter=8,
            letterSpacing=0.9),
        "kv_key": s("kv_key",
            fontName="Helvetica-Bold", fontSize=9, leading=13,
            textColor=_MUTED),
        "kv_val": s("kv_val",
            fontName="Helvetica", fontSize=9, leading=13,
            textColor=_TEXT),
        "price_label": s("price_label",
            fontName="Helvetica-Bold", fontSize=7.5, leading=10,
            textColor=_MUTED, spaceAfter=3, letterSpacing=0.5),
        "price_val": s("price_val",
            fontName="Helvetica-Bold", fontSize=15, leading=20,
            textColor=_TEXT),
        "conf_high": s("conf_high",
            fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=_GREEN),
        "conf_medium": s("conf_medium",
            fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=_AMBER),
        "conf_low": s("conf_low",
            fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=_RED),
        "confidence": s("confidence",
            fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=_MUTED),
        "reasoning": s("reasoning",
            fontName="Helvetica", fontSize=9.5, leading=15,
            textColor=_TEXT),
        "banner": s("banner",
            fontName="Helvetica", fontSize=9, leading=14,
            textColor=colors.HexColor("#1a5c3a")),
        "disclaimer": s("disclaimer",
            fontName="Helvetica-Oblique", fontSize=7.5, leading=11,
            textColor=_MUTED),
    }
 
 
def _kv_table(rows: list[tuple[str, str]], S: dict, inner_w: float) -> Table:
    """Alternating-row key/value table."""
    data = [[Paragraph(k, S["kv_key"]), Paragraph(v, S["kv_val"])] for k, v in rows]
    col_w = [inner_w * 0.36, inner_w * 0.64]
    tbl = Table(data, colWidths=col_w)
    cmds = [
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 9),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 9),
        ("LINEBELOW",      (0, 0), (-1, -2), 0.3, colors.HexColor("#dde3ef")),
    ]
    for i in range(0, len(data), 2):
        cmds.append(("BACKGROUND", (0, i), (-1, i), _ROW_ALT))
    tbl.setStyle(TableStyle(cmds))
    return tbl
 
 
def _season_context_block(sc: dict, S: dict, inner_w: float):
    """Green banner for incremental / amber for partial season deals."""
    if not sc:
        return None
    if sc.get("mode") == "incremental":
        new_s = sc.get("net_new_seasons", [])
        n_ep  = sc.get("net_new_episodes", "?")
        held  = sc.get("already_acquired_seasons", "")
        text  = (f"Incremental acquisition — "
                 f"Season{'s' if len(new_s) > 1 else ''} "
                 f"{', '.join(str(x) for x in new_s)} · "
                 f"{n_ep} episode{'s' if n_ep != 1 else ''} · "
                 f"Seasons {held} already held by licensee.")
        bg, border = colors.HexColor("#e8f7f1"), colors.HexColor("#5ccf9f")
        style = ParagraphStyle("sc_inc", parent=S["banner"],
                               textColor=colors.HexColor("#0d5c3a"))
    elif sc.get("mode") == "partial":
        inc = sc.get("included_seasons", "")
        pct = sc.get("season_ratio_pct", "?")
        n_ep = sc.get("net_new_episodes", "?")
        text = (f"Partial series — Seasons {inc} · "
                f"{pct}% season coverage · {n_ep} episodes priced.")
        bg, border = colors.HexColor("#fdf6e3"), colors.HexColor("#e6b84a")
        style = ParagraphStyle("sc_par", parent=S["banner"],
                               textColor=colors.HexColor("#7a4f00"))
    else:
        return None
 
    box = Table([[Paragraph(text, style)]], colWidths=[inner_w])
    box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("LINEBEFORE",    (0, 0), (0, -1),  3, border),
        ("BOX",           (0, 0), (-1, -1), 0.5, border),
        ("ROUNDEDCORNERS",(0, 0), (-1, -1), [4, 4, 4, 4]),
    ]))
    return box
 
 
def generate_deal_memo_pdf(deal: DealRequest, result: dict) -> BytesIO:
    buf = BytesIO()
    page_w, _ = A4
    margin     = 14 * mm
    inner_w    = page_w - 2 * margin
 
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=16 * mm,
        title=f"Deal Memo — {deal.title}",
        author="Film Licensing Pricing Engine",
    )
 
    S    = _styles()
    pc   = result.get("pricing_components", {})
    pe   = result.get("pricing_estimate", {})
    sc   = pc.get("season_context")
    mults = pc.get("multipliers", {})
    conf  = result.get("confidence_level", "—")
    conf_color = {
        "High": _GREEN, "Medium": _AMBER, "Low": _RED
    }.get(conf, _MUTED)
 
    story = []
 
    # ══════════════════════════════════════════════════════════
    # COVER BAND
    # ══════════════════════════════════════════════════════════
    generated_str = datetime.now(tz=__import__('datetime').timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    cover_content = [
        Paragraph("FILM LICENSING — DEAL MEMO", S["cover_eyebrow"]),
        Paragraph(deal.title, S["cover_title"]),
        Paragraph(f"{result.get('region', deal.region)}  ·  {deal.rights_type}  ·  {deal.license_duration}", S["cover_sub"]),
        Paragraph(f"Generated {generated_str}", S["cover_sub"]),
    ]
    cover_tbl = Table([[cover_content]], colWidths=[inner_w])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ("LEFTPADDING",   (0, 0), (-1, -1), 18),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 18),
        ("ROUNDEDCORNERS",(0, 0), (-1, -1), [8, 8, 8, 8]),
    ]))
    story.append(cover_tbl)
    story.append(Spacer(1, 12))
 
    # ══════════════════════════════════════════════════════════
    # PRICING SUMMARY — 4-column grid
    # ══════════════════════════════════════════════════════════
    conf_style = {
        "High":   S["conf_high"],
        "Medium": S["conf_medium"],
        "Low":    S["conf_low"],
    }.get(conf, S["confidence"])
 
    conf_hex  = "%02x%02x%02x" % (
        int(conf_color.red * 255),
        int(conf_color.green * 255),
        int(conf_color.blue * 255),
    )
    conf_para = Paragraph(conf, conf_style)
 
    price_rows = [
        [Paragraph("FLAT FEE RANGE",   S["price_label"]),
         Paragraph("MIN GUARANTEE",    S["price_label"]),
         Paragraph("REVENUE SHARE",    S["price_label"]),
         Paragraph("CONFIDENCE",       S["price_label"])],
        [Paragraph(pe.get("flat_fee_range",       "—"), S["price_val"]),
         Paragraph(pe.get("minimum_guarantee",    "—"), S["price_val"]),
         Paragraph(pe.get("revenue_share_range",  "—"), S["price_val"]),
         conf_para],
    ]
    price_tbl = Table(price_rows, colWidths=[inner_w / 4] * 4)
    price_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#eef2fb")),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("LINEBEFORE",    (1, 0), (3, -1),  0.4, _BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ROUNDEDCORNERS",(0, 0), (-1, -1), [6, 6, 6, 6]),
    ]))
    story.append(KeepTogether([
        Paragraph("PRICING ESTIMATE", S["section_head"]),
        price_tbl,
    ]))
 
    # ══════════════════════════════════════════════════════════
    # SEASON CONTEXT BANNER (incremental / partial)
    # ══════════════════════════════════════════════════════════
    sc_block = _season_context_block(sc, S, inner_w)
    if sc_block:
        story.append(Spacer(1, 10))
        story.append(sc_block)
 
    # ══════════════════════════════════════════════════════════
    # DEAL PARAMETERS
    # ══════════════════════════════════════════════════════════
    platforms_str = ", ".join(deal.platforms) if deal.platforms else "—"
    deal_rows: list[tuple[str, str]] = [
        ("Territory",        result.get("region", deal.region)),
        ("Market tier",      result.get("market_tier", "—").capitalize()),
        ("Rights type",      deal.rights_type),
        ("License duration", deal.license_duration),
        ("Language rights",  deal.language_rights),
        ("Platform(s)",      platforms_str),
        ("Content type",     deal.content_type.capitalize()),
    ]
    if deal.content_type == "series":
        if deal.season_count:
            deal_rows.append(("Total seasons", str(deal.season_count)))
        if deal.included_seasons:
            deal_rows.append(("Seasons in deal", deal.included_seasons))
        if deal.already_acquired_seasons:
            deal_rows.append(("Already acquired", deal.already_acquired_seasons))
        # Per-season breakdown from IMDb
        if deal.season_episode_counts:
            breakdown_str = ", ".join(
                f"S{k}: {v} ep"
                for k, v in sorted(deal.season_episode_counts.items(), key=lambda x: int(x[0]))
            )
            deal_rows.append(("Episode breakdown", breakdown_str))
        if deal.episode_overrides:
            overrides_str = ", ".join(
                f"S{k}: {v} ep (override)"
                for k, v in sorted(deal.episode_overrides.items(), key=lambda x: int(x[0]))
            )
            deal_rows.append(("Episode overrides", overrides_str))
    if deal.runtime_minutes:
        deal_rows.append(("Avg episode runtime", f"{deal.runtime_minutes} min"))
    deal_rows.append(("IMDb", deal.imdb_link))
    if deal.tmdb_link:
        deal_rows.append(("TMDB", deal.tmdb_link))
 
    story.append(KeepTogether([
        Paragraph("DEAL PARAMETERS", S["section_head"]),
        _kv_table(deal_rows, S, inner_w),
    ]))
 
    # ══════════════════════════════════════════════════════════
    # PRICING COMPONENTS
    # ══════════════════════════════════════════════════════════
    comp_rows: list[tuple[str, str]] = [
        ("Content score",    f"{pc.get('score', '—')}/100"),
        ("Base price range", pc.get("base_price_range", "—")),
        ("Platform ×",       str(mults.get("platform",  "—"))),
        ("Rights ×",         str(mults.get("rights",    "—"))),
        ("Language ×",       str(mults.get("language",  "—"))),
        ("License ×",        str(mults.get("license",   "—"))),
        ("Market ×",         str(mults.get("market",    "—"))),
        ("Package ×",        str(mults.get("package",   "—"))),
        ("Age ×",            str(mults.get("age",       "—"))),
        ("Combined ×",       str(mults.get("combined",  "—"))),
    ]
    story.append(KeepTogether([
        Paragraph("PRICING COMPONENTS", S["section_head"]),
        _kv_table(comp_rows, S, inner_w),
    ]))
 
    # ══════════════════════════════════════════════════════════
    # ANALYST REASONING
    # ══════════════════════════════════════════════════════════
    reasoning_text = result.get("reasoning", "").strip()
    if reasoning_text:
        reasoning_box = Table(
            [[Paragraph(reasoning_text, S["reasoning"])]],
            colWidths=[inner_w],
        )
        reasoning_box.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f7f9ff")),
            ("TOPPADDING",    (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("LEFTPADDING",   (0, 0), (-1, -1), 16),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
            ("LINEBEFORE",    (0, 0), (0, -1),  4, _ACCENT),
            ("ROUNDEDCORNERS",(0, 0), (-1, -1), [4, 4, 4, 4]),
        ]))
        story.append(KeepTogether([
            Paragraph("ANALYST REASONING", S["section_head"]),
            reasoning_box,
        ]))
 
    # ══════════════════════════════════════════════════════════
    # FOOTER — disclaimer + model version
    # ══════════════════════════════════════════════════════════
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", thickness=0.4, color=_BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"This memo is a non-binding pricing estimate generated by an automated model. "
        f"All figures are indicative only and subject to negotiation. "
        f"Model version: {result.get('model_version', '—')}.",
        S["disclaimer"],
    ))
 
    doc.build(story)
    buf.seek(0)
    return buf
 
 
# ── Memo endpoint ──────────────────────────────────────────────────────────────
class MemoRequest(BaseModel):
    deal:   DealRequest
    result: dict
 
 
@app.post("/export-memo")
def export_memo(body: MemoRequest):
    """Generate and stream a PDF deal memo for a completed pricing estimate."""
    try:
        safe_title = re.sub(r"[^\w\-. ]", "_", body.deal.title)[:60].strip()
        filename   = f"deal_memo_{safe_title}.pdf".replace(" ", "_")
        buf        = generate_deal_memo_pdf(body.deal, body.result)
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        log_event("export_memo_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Unable to generate deal memo: {exc}")