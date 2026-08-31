from second_brain.chunking import chunk_blocks
from second_brain.models import TextBlock


def test_chunking_preserves_order_and_hashes():
    blocks = [TextBlock(0, "第一段" * 100), TextBlock(1, "第二段" * 100)]
    chunks = chunk_blocks(blocks, maximum=500)
    assert "第一段" in chunks[0].content
    assert "第二段" in chunks[-1].content
    assert all(len(chunk.content) <= 500 for chunk in chunks)
    assert len({chunk.content_hash for chunk in chunks}) == len(chunks)


def test_long_paragraph_is_split():
    chunks = chunk_blocks([TextBlock(0, "申论" * 600)], maximum=300)
    assert len(chunks) == 4

