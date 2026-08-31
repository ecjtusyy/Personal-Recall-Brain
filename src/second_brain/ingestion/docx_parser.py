from __future__ import annotations

import mimetypes
import zipfile
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..models import ParsedAsset, ParsedDocument, TextBlock


def _property_dict(document: Document) -> dict[str, datetime | None]:
    props = document.core_properties
    return {
        "created": props.created,
        "modified": props.modified,
        "last_printed": props.last_printed,
    }


def _ordered_text(document: Document) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for child in document.element.body.iterchildren():
        text = ""
        if child.tag == qn("w:p"):
            text = Paragraph(child, document._body).text
        elif child.tag == qn("w:tbl"):
            table = Table(child, document._body)
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            text = "\n".join(row for row in rows if row.strip(" |"))
        text = text.strip()
        if text:
            blocks.append(TextBlock(len(blocks), text))
    return blocks


def _embedded_assets(path: Path, text_blocks: list[TextBlock]) -> list[ParsedAsset]:
    assets: list[ParsedAsset] = []
    before = text_blocks[-1].text if text_blocks else ""
    with zipfile.ZipFile(path) as archive:
        media = sorted(name for name in archive.namelist() if name.startswith("word/media/") and not name.endswith("/"))
        for order, name in enumerate(media):
            original = Path(name).name
            mime, _ = mimetypes.guess_type(original)
            assets.append(
                ParsedAsset(
                    order_index=order,
                    original_name=original,
                    data=archive.read(name),
                    mime_type=mime,
                    context_before=before,
                )
            )
    return assets


def parse_docx(path: Path, include_assets: bool = True) -> ParsedDocument:
    document = Document(path)
    text_blocks = _ordered_text(document)
    return ParsedDocument(
        title=document.core_properties.title or (text_blocks[0].text[:120] if text_blocks else path.stem),
        text_blocks=text_blocks,
        assets=_embedded_assets(path, text_blocks) if include_assets else [],
        core_properties=_property_dict(document),
    )

