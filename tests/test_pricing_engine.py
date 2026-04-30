import json
from pathlib import Path

from hypothesis import given, strategies as st

import main


def _setup_in_memory_db():
    main.DATABASE_URL = "sqlite+pysqlite:///:memory:"
    main.init_db()


def _stub_reasoning(*args, **kwargs):
    return "Deterministic reasoning stub."


def test_golden_pricing_cases(monkeypatch):
    _setup_in_memory_db()
    monkeypatch.setattr(main, "generate_reasoning_with_ai", _stub_reasoning)

    fixture_path = Path(__file__).parent / "golden_cases.json"
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))

    for case in fixtures:
        monkeypatch.setattr(main, "fetch_omdb_data", lambda _, c=case: c["omdb_data"])
        monkeypatch.setattr(main, "fetch_tmdb_data", lambda _, c=case: c["tmdb_data"])
        deal = main.DealRequest(**case["deal"])
        result = main.estimate(deal)

        expected = case["expected"]
        assert result["pricing_estimate"]["flat_fee_range"] == expected["flat_fee_range"]
        assert result["pricing_estimate"]["minimum_guarantee"] == expected["minimum_guarantee"]
        assert result["confidence_level"] == expected["confidence_level"]
        assert result["market_tier"] == expected["market_tier"]
        assert result["model_version"] == main.MODEL_VERSION


def test_cache_hit_after_first_estimate(monkeypatch):
    _setup_in_memory_db()
    monkeypatch.setattr(main, "generate_reasoning_with_ai", _stub_reasoning)
    monkeypatch.setattr(
        main,
        "fetch_omdb_data",
        lambda _: {"imdb_rating": "8.0", "imdb_votes": "100,000", "box_office": "$200,000,000", "release_year": "2024"},
    )
    monkeypatch.setattr(main, "fetch_tmdb_data", lambda _: {"popularity": 15.0})

    deal = main.DealRequest(
        title="Cache Test",
        imdb_link="https://www.imdb.com/title/tt1234567/",
        tmdb_link="https://www.themoviedb.org/movie/1-cache-test",
        region="US",
        duration="100 min",
        license_duration="1 year",
        rights_type="Exclusive",
        language_rights="Original + Dubbed + Subtitled",
        platforms=["OTT / Streaming"],
    )

    first = main.estimate(deal)
    second = main.estimate(deal)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


@given(st.floats(min_value=0.5, max_value=10.0), st.floats(min_value=0.5, max_value=10.0))
def test_license_multiplier_monotonic(years_a, years_b):
    a = min(years_a, years_b)
    b = max(years_a, years_b)
    mult_a = main.license_multiplier(f"{a:.2f} years")
    mult_b = main.license_multiplier(f"{b:.2f} years")
    assert mult_b >= mult_a


@given(st.text(min_size=1, max_size=30))
def test_unknown_region_does_not_crash(random_region):
    _setup_in_memory_db()
    deal = main.DealRequest(
        title="Unknown Region Test",
        imdb_link="https://www.imdb.com/title/tt1234567/",
        tmdb_link="",
        region=random_region,
        duration="90 min",
        license_duration="custom",
        rights_type="Non-Exclusive",
        language_rights="Original",
        platforms=["Unknown Platform"],
    )
    result = main.estimate(deal)
    assert "pricing_estimate" in result
    assert "flat_fee_range" in result["pricing_estimate"]
