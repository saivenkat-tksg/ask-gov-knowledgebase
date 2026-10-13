from kb.config import Settings
from kb.ingestion.ingestor import Ingestor
from kb.ingestion.structure import StructureChunker, breadcrumb, parse_sections, service_of

DOC = """Intro text before any heading.

# Guide

## Eligibility

### Who can apply?
Citizens aged 18 or older.

### Do I need ID?
Yes, a National ID or passport.

## Payment

### How am I paid?
By bank transfer.
"""


def words(n: int, word: str = "word") -> str:
    return " ".join(f"{word}{i}." if i % 12 == 11 else f"{word}{i}" for i in range(n))


def test_parse_sections_paths_and_offsets():
    sections, headings = parse_sections(DOC)
    assert headings == 6
    by_path = {s.path: s for s in sections}
    assert by_path[()].text == "Intro text before any heading."
    s = by_path[("Guide", "Eligibility", "Who can apply?")]
    assert s.text == "Citizens aged 18 or older."
    assert DOC[s.start:s.end] == s.text
    assert by_path[("Guide",)].text == ""  # heading directly followed by a subheading


def test_too_few_headings_falls_back():
    assert StructureChunker(min_headings=3).split("# One\ntext\n\n## Two\ntext") is None


def test_small_siblings_merge_within_parent_only():
    chunks = StructureChunker(chunk_size=200, chunk_overlap=0, min_tokens=60).split(DOC)
    texts = [c.text for c in chunks]
    eligibility = next(t for t in texts if t.startswith("[Guide > Eligibility]"))
    assert "Who can apply?" in eligibility and "Do I need ID?" in eligibility
    assert "bank transfer" not in eligibility  # never merged across the "Payment" heading
    assert any(t.startswith("[Guide > Payment > How am I paid?]") or t.startswith("[Guide > Payment]")
               for t in texts)


def test_large_section_split_recursively_with_breadcrumb():
    text = "# Manual\n\n## Births\n\n" + "\n\n".join(words(40, f"p{i}w") for i in range(10)) + "\n\n## Deaths\n\nshort\n\n## Marriage\n\nshort"
    chunks = StructureChunker(chunk_size=300, chunk_overlap=40, min_tokens=10).split(text)
    births = [c for c in chunks if c.path == ("Manual", "Births")]
    assert len(births) > 1
    for c in births:
        assert c.text.startswith("[Manual > Births]\n")
        assert len(c.text) <= 300
        assert c.parts == len(births)
    assert all("short" not in c.text for c in births)


def test_chunk_offsets_point_at_body():
    chunks = StructureChunker(chunk_size=500, chunk_overlap=0, min_tokens=0).split(DOC)
    for c in chunks:
        if not c.merged:
            body = c.text.split("\n", 1)[1] if c.path else c.text
            assert DOC[c.start:c.end] == body


def test_service_of():
    path = ("Chapter 1: MINISTRY OF HOME AFFAIRS", "Desired State:", "1. TINT Waiver Application", "Benefits")
    assert service_of(path) == "Chapter 1: MINISTRY OF HOME AFFAIRS > TINT Waiver Application"
    assert service_of(("Guide", "Eligibility")) is None
    assert breadcrumb(()) == ""


def _ingestor(strategy: str) -> Ingestor:
    return Ingestor(None, None, Settings(_env_file=None, chunk_strategy=strategy, chunk_length_unit="chars",
                                         chunk_size=300, chunk_overlap=30, chunk_min_tokens=60))


def test_ingestor_structure_metadata():
    _, doc_meta, rows = _ingestor("structure").build_chunks("guide.md", DOC.encode())
    assert doc_meta["chunk_strategy"] == "structure"
    paths = [r["extra"]["section_path"] for r in rows]
    assert ["Guide", "Eligibility"] in paths
    assert all(r["page"] is None for r in rows)


def test_ingestor_falls_back_to_recursive():
    _, doc_meta, rows = _ingestor("structure").build_chunks("note.txt", b"Just a short note with no headings.")
    assert doc_meta["chunk_strategy"] == "recursive"
    assert [r["text"] for r in rows] == ["Just a short note with no headings."]


def test_recursive_strategy_unchanged():
    _, doc_meta, rows = _ingestor("recursive").build_chunks("guide.md", DOC.encode())
    assert doc_meta["chunk_strategy"] == "recursive"
    assert not any("section_path" in r["extra"] for r in rows)
