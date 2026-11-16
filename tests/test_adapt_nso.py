"""fluxpype.adapt_nso: parse NSO's ADAPT listing and pick the map for a window (offline)."""

import datetime as dt

from fluxpype import adapt_nso

LISTING = """
<a href="?C=N;O=D">Name</a>
<a href="adapt40311_044012_201502010000_i00005600n1.fts.gz">adapt40311_044012_201502010000_i00005600n1.fts.gz</a>
<a href="adapt41311_044012_201502010600_i00005600n1.fts.gz">adapt41311_044012_201502010600_i00005600n1.fts.gz</a>
<a href="adapt40311_044012_201502010600_i00005600n1.fts.gz">adapt40311_044012_201502010600_i00005600n1.fts.gz</a>
<a href="adapt40311_044012_201502010800_i00010600n1.fts.gz">adapt40311_044012_201502010800_i00010600n1.fts.gz</a>
"""


def test_parse_listing_dedupes_and_reads_fields():
    entries = adapt_nso.parse_listing(LISTING)
    assert len(entries) == 4
    assert entries[0] == (
        dt.datetime(2015, 2, 1, 0, 0),
        0,
        "adapt40311_044012_201502010000_i00005600n1.fts.gz",
    )
    assert {e[1] for e in entries} == {0, 1}


def test_pick_nearest_carrington_map_in_window():
    entries = adapt_nso.parse_listing(LISTING)
    start = dt.datetime(2015, 2, 1, 6, 38)  # CR 2160 start
    hit = adapt_nso.pick(entries, start - dt.timedelta(hours=2), start + dt.timedelta(hours=2))
    assert hit[1] == "adapt40311_044012_201502010600_i00005600n1.fts.gz"


def test_pick_skips_other_lon_types_and_empty_windows():
    entries = adapt_nso.parse_listing(LISTING)
    start = dt.datetime(2015, 2, 1, 6, 0)
    assert adapt_nso.pick(entries, start, start, lon_type=1)[1].startswith("adapt41311")
    assert adapt_nso.pick(entries, dt.datetime(2015, 3, 1), dt.datetime(2015, 3, 2)) is None


def test_find_reads_each_year_once(monkeypatch):
    seen = []

    def fake_get(url, timeout=60):
        seen.append(url)
        return LISTING.encode()

    monkeypatch.setattr(adapt_nso, "_get", fake_get)
    url = adapt_nso.find(dt.datetime(2015, 2, 1, 5), dt.datetime(2015, 2, 1, 7))
    assert seen == [f"{adapt_nso.BASE_URL}/2015/"]
    assert url == f"{adapt_nso.BASE_URL}/2015/adapt40311_044012_201502010600_i00005600n1.fts.gz"
