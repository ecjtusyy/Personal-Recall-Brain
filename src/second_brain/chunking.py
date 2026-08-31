from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .models import Chunk, TextBlock


SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;\n])")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_long(text: str, maximum: int) -> list[str]:
    if len(text) <= maximum:
        return [text]
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > maximum:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(sentence[pos : pos + maximum] for pos in range(0, len(sentence), maximum))
        elif current and len(current) + len(sentence) + 1 > maximum:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current}\n{sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_blocks(
    blocks: Iterable[TextBlock],
    target: int = 700,
    maximum: int = 900,
    source_kind: str = "text",
) -> list[Chunk]:
    del target  # target is kept in the API for future adaptive chunking.
    chunks: list[Chunk] = []
    pending: list[str] = []
    pending_block = 0

    def flush() -> None:
        if not pending:
            return
        content = "\n".join(pending).strip()
        if content:
            chunks.append(Chunk(pending_block, len(chunks), source_kind, content, _hash(content)))
        pending.clear()

    for block in blocks:
        text = re.sub(r"[\t\r ]+", " ", block.text).strip()
        if not text:
            continue
        for piece in _split_long(text, maximum):
            if pending and sum(map(len, pending)) + len(piece) + len(pending) > maximum:
                flush()
            if not pending:
                pending_block = block.index
            pending.append(piece)
    flush()
    return chunks

