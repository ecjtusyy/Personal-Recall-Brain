from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .models import DateResolution


FULL_DATE = re.compile(r"(?<!\d)(?P<y>20\d{2})[年._\-/](?P<m>0?[1-9]|1[0-2])[月._\-/](?P<d>0?[1-9]|[12]\d|3[01])日?(?!\d)")
SHORT_DATE = re.compile(r"(?<!\d)(?P<m>0?[1-9]|1[0-2])[月._\-/](?P<d>0?[1-9]|[12]\d|3[01])日?(?!\d)")


def _valid(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def resolve_event_date(
    path: Path,
    study_year: int,
    core_properties: dict[str, Any] | None = None,
    stat: Any | None = None,
) -> DateResolution:
    name = path.stem
    match = FULL_DATE.search(name)
    if match:
        parsed = _valid(int(match["y"]), int(match["m"]), int(match["d"]))
        if parsed:
            return DateResolution(parsed, "filename", 0.99)
    match = SHORT_DATE.search(name)
    if match:
        parsed = _valid(study_year, int(match["m"]), int(match["d"]))
        if parsed:
            return DateResolution(parsed, "filename", 0.92)
    props = core_properties or {}
    for key in ("created", "modified", "last_printed"):
        parsed = _as_date(props.get(key))
        if parsed:
            return DateResolution(parsed, f"document_{key}", 0.75 if key == "created" else 0.65)
    file_stat = stat or path.stat()
    timestamp = getattr(file_stat, "st_birthtime", None) or getattr(file_stat, "st_ctime", None) or file_stat.st_mtime
    return DateResolution(datetime.fromtimestamp(timestamp).date(), "filesystem", 0.45)


def parse_date_query(text: str, study_year: int) -> tuple[str | None, str | None]:
    match = FULL_DATE.search(text)
    if match:
        parsed = _valid(int(match["y"]), int(match["m"]), int(match["d"]))
        if parsed:
            iso = parsed.isoformat()
            return iso, iso
    match = SHORT_DATE.search(text)
    if match:
        parsed = _valid(study_year, int(match["m"]), int(match["d"]))
        if parsed:
            iso = parsed.isoformat()
            return iso, iso
    return None, None

