from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

API_BASE = "https://api.reliefweb.int/v2/reports"
APPNAME = os.getenv("RELIEFWEB_APPNAME", "").strip()

DATA_DIR = Path("data/external")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CURSOR_FILE = DATA_DIR / "reliefweb_cursor.txt"

# Keep payloads tight; see fields table docs.
FIELDS_INCLUDE = [
    "title",
    "body-html",
    "date.created",
    "url",
    "source.name",
    "primary_country.name",
    "disaster_type.name",
    "language.code",
]

HTML_TAGS_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RWRecord:
    id: int
    created_at: str
    title: str
    body_text: str
    url: str
    sources: str
    primary_country: str
    disaster_type: str
    language_code: str


def _strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = HTML_TAGS_RE.sub(" ", s)
    s = WS_RE.sub(" ", s)
    return s.strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_cursor(default_from_days: int = 7) -> str:
    if CURSOR_FILE.exists():
        return CURSOR_FILE.read_text().strip()
    # default: 7 days back if first run
    dt = datetime.now(timezone.utc).replace(microsecond=0)
    from_dt = dt - timedelta(days=default_from_days)  # type: ignore[name-defined]
    # lazy import to keep top clean
    return from_dt.isoformat().replace("+00:00", "Z")


# (avoid top-level import to keep deps minimal)
from datetime import timedelta  # after function uses


def _write_cursor(iso_ts: str) -> None:
    CURSOR_FILE.write_text(iso_ts)


def _flatten_item(item: Dict[str, Any]) -> RWRecord:
    fid = int(item["id"])
    f = item.get("fields", {})
    # date.created can be "date": {"created": "..."} or directly via fields include
    created = f.get("date", {}).get("created") or f.get("date.created") or ""
    title = f.get("title") or ""
    body_html = f.get("body-html") or f.get("body") or ""
    url = f.get("url") or item.get("href") or ""

    # arrays → join with |
    def _first_or_join(obj, key):
        vals = []
        for it in obj or []:
            v = it.get(key)
            if v:
                vals.append(v)
        return "|".join(vals)

    sources = _first_or_join(f.get("source"), "name")
    country = _first_or_join(f.get("primary_country"), "name")
    dtype = _first_or_join(f.get("disaster_type"), "name")
    lang = ""
    for langobj in f.get("language") or []:
        if langobj.get("code"):
            lang = langobj["code"]
            break

    return RWRecord(
        id=fid,
        created_at=str(created),
        title=str(title).strip(),
        body_text=_strip_html(body_html),
        url=str(url),
        sources=sources,
        primary_country=country,
        disaster_type=dtype,
        language_code=lang,
    )


def fetch_reliefweb(
    since_iso: str,
    until_iso: Optional[str] = None,
    limit: int = 200,
    sleep_s: float = 0.3,
    language_code: str = "en",
) -> Iterable[RWRecord]:
    """Yield flattened ReliefWeb reports created in [since_iso, until_iso]."""
    if not APPNAME:
        raise RuntimeError(
            "RELIEFWEB_APPNAME is not set. Get a pre-approved appname and export it."
        )

    session = requests.Session()
    offset = 0
    latest_seen: Optional[str] = None

    while True:
        payload: Dict[str, Any] = {
            "preset": "latest",
            "limit": limit,
            "offset": offset,
            "fields": {"include": FIELDS_INCLUDE},
            "filter": {
                "operator": "AND",
                "conditions": [
                    {
                        "field": "date.created",
                        "value": {
                            "from": since_iso,
                            **({"to": until_iso} if until_iso else {}),
                        },
                    },
                    {"field": "language.code", "value": language_code},
                ],
            },
        }

        url = f"{API_BASE}?appname={APPNAME}"
        resp = session.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("data", [])
        if not items:
            break

        for it in items:
            rec = _flatten_item(it)
            latest_seen = max(latest_seen or "", rec.created_at)
            yield rec

        count = int(data.get("count", len(items)))
        total = int(data.get("totalCount", 0))
        offset += count
        if offset >= total:
            break

        time.sleep(sleep_s)

    # persist the max created_at we saw (coarser than per-page next link, but fine)
    if latest_seen:
        _write_cursor(latest_seen)


def _write_csv(rows: Iterable[RWRecord]) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    out = DATA_DIR / f"reliefweb_reports_{ts}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "created_at",
                "title",
                "body_text",
                "url",
                "sources",
                "primary_country",
                "disaster_type",
                "language_code",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.id,
                    r.created_at,
                    r.title,
                    r.body_text,
                    r.url,
                    r.sources,
                    r.primary_country,
                    r.disaster_type,
                    r.language_code,
                ]
            )
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Incremental fetch from ReliefWeb")
    p.add_argument("--since", help="ISO8601 from (default=cursor or 7d ago)")
    p.add_argument("--until", help="ISO8601 to (optional)")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--lang", default="en", help="language.code filter")
    args = p.parse_args(argv)

    since = args.since or _read_cursor()
    rows = list(
        fetch_reliefweb(
            since_iso=since,
            until_iso=args.until,
            limit=args.limit,
            language_code=args.lang,
        )
    )
    out = _write_csv(rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    if CURSOR_FILE.exists():
        print(f"Cursor now at: {CURSOR_FILE.read_text().strip()}")


if __name__ == "__main__":
    main(sys.argv[1:])
