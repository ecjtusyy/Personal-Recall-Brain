from __future__ import annotations

import sqlite3

from ..models import SearchResult
from .lexical import search_memory


def hybrid_search(conn: sqlite3.Connection, query: str, start_date: str | None = None,
                  end_date: str | None = None, file_type: str | None = None,
                  limit: int = 20) -> list[SearchResult]:
    # Optional OpenVINO semantic candidates plug into this stable merge boundary.
    return search_memory(conn, query, start_date, end_date, file_type, limit)

