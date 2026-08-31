from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DateResolution:
    value: date
    source: str
    confidence: float


@dataclass(frozen=True)
class TextBlock:
    index: int
    text: str
    kind: str = "text"


@dataclass(frozen=True)
class ParsedAsset:
    order_index: int
    original_name: str
    data: bytes
    mime_type: str | None = None
    context_before: str = ""
    context_after: str = ""


@dataclass
class ParsedDocument:
    title: str | None = None
    text_blocks: list[TextBlock] = field(default_factory=list)
    assets: list[ParsedAsset] = field(default_factory=list)
    core_properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    block_index: int
    chunk_index: int
    source_kind: str
    content: str
    content_hash: str


@dataclass(frozen=True)
class SearchResult:
    document_id: int
    chunk_id: int
    title: str
    filename: str
    path: str
    file_type: str
    event_date: str | None
    date_source: str | None
    snippet: str
    score: float
    source_kind: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ScanStats:
    files_seen: int = 0
    files_changed: int = 0
    files_failed: int = 0
    files_skipped: int = 0


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    boxes: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SourceGuard:
    roots: tuple[Path, ...]

    def contains(self, path: Path) -> bool:
        resolved = path.resolve(strict=False)
        return any(resolved == root or root in resolved.parents for root in self.roots)

