"""Document-level extraction: read the whole uploaded document, not one chunk at a time.

The text is the loader's output for the uploaded file (document_texts): PDF pages joined, with
tables as markdown and headings as `#` lines. A document that fits in `window_tokens` goes to the
LLM in one call. A larger one is cut into parts along its own structure (chapters, then sections,
then paragraphs), because no model can return every fact of a 200-page booklet in a single reply;
every part carries the document's outline so the model knows where it is.

Extracted facts are then traced back to the chunks that state them (by locating each quote), which
keeps citations, graph search and per-chunk deletes working. Entities are also linked to every
chunk of the document that names them, so graph search can reach all the places an office or
form appears.
"""

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from kb.graph.extract import Extraction, ExtractionError, _normal_text
from kb.graph.resolve import Resolver

_HEADING = re.compile(r"^(#{1,12})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
MIN_NAME_CHARS = 4  # shorter names ("ID", "GRO") are linked by quote only, never by name search
MAX_NAME_MENTIONS = 60  # an entity named in more chunks than this is linked to its quoted chunks only


@dataclass
class Part:
    text: str
    start: int  # offset in the document text
    pages: tuple[int | None, int | None] = (None, None)


@dataclass
class ChunkText:
    chunk_id: str
    text: str  # normalised (kb.graph.extract._normal_text)


@dataclass
class DocumentFacts:
    """Validated facts for one document, with the chunks each one is traced to."""

    entities: list[tuple[str, str, str, set[str]]] = field(default_factory=list)  # type, name, quote, chunk ids
    relations: list[tuple[str, str, str, str, str | None]] = field(default_factory=list)  # type, src, dst, quote, chunk
    parts: int = 0
    dropped: int = 0


def outline(text: str, max_levels: int = 3, max_lines: int = 120) -> str:
    lines = [f"{'  ' * (len(m.group(1)) - 1)}- {m.group(2).strip()}" for m in _HEADING.finditer(text)
             if len(m.group(1)) <= max_levels]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"  ... ({len(lines) - max_lines} more headings)"]
    return "\n".join(lines)


def split_parts(text: str, window_tokens: int, length: Callable[[str], int]) -> list[Part]:
    """The whole text if it fits; otherwise parts cut at the coarsest heading level that works,
    falling back to paragraphs. Parts are packed up to the window, never split mid-paragraph."""
    if length(text) <= window_tokens:
        return [Part(text, 0)]
    for level in range(1, 7):
        cuts = [m.start() for m in _HEADING.finditer(text) if len(m.group(1)) <= level]
        pieces = _cut(text, cuts)
        if len(pieces) > 1 and all(length(p.text) <= window_tokens for p in pieces):
            return _pack(pieces, window_tokens, length)
    # Too big even per heading: cut every oversized piece at paragraph breaks.
    out: list[Part] = []
    level_one = [m.start() for m in _HEADING.finditer(text) if len(m.group(1)) <= 2]
    for piece in _cut(text, level_one) or [Part(text, 0)]:
        if length(piece.text) <= window_tokens:
            out.append(piece)
            continue
        paragraphs = [m.start() + piece.start for m in re.finditer(r"\n\s*\n", piece.text)]
        out.extend(_pack(_cut(text, paragraphs, piece.start, piece.start + len(piece.text)), window_tokens, length))
    return _pack(out, window_tokens, length)


def _cut(text: str, cuts: list[int], lo: int = 0, hi: int | None = None) -> list[Part]:
    hi = len(text) if hi is None else hi
    points = sorted({lo, *(c for c in cuts if lo < c < hi)})
    return [Part(text[a:b], a) for a, b in zip(points, [*points[1:], hi]) if text[a:b].strip()]


def _pack(pieces: list[Part], window_tokens: int, length: Callable[[str], int]) -> list[Part]:
    packed: list[Part] = []
    for p in pieces:
        if packed and length(packed[-1].text) + length(p.text) <= window_tokens:
            packed[-1] = Part(packed[-1].text + p.text, packed[-1].start)
        else:
            packed.append(p)
    return packed


def page_range(start: int, end: int, page_map: list[list[int]]) -> tuple[int | None, int | None]:
    if not page_map:
        return None, None

    def page_at(pos: int) -> int | None:
        page = page_map[0][1]
        for offset, p in page_map:
            if offset > pos:
                break
            page = p
        return page

    return page_at(start), page_at(max(start, end - 1))


def extract_document(text: str, filename: str, page_map: list[list[int]], chunks: list[ChunkText],
                     extract: Callable[[str, str], Extraction], resolver: Resolver, window_tokens: int,
                     length: Callable[[str], int], concurrency: int = 4) -> DocumentFacts:
    """Run the LLM over the whole document (in parts if needed) and trace facts to chunks."""
    doc_outline = outline(text)
    parts = split_parts(text, window_tokens, length)
    for p in parts:
        p.pages = page_range(p.start, p.start + len(p.text), page_map)

    def context(i: int, part: Part) -> str:
        where = f", pages {part.pages[0]}-{part.pages[1]}" if part.pages[0] is not None else ""
        head = f"Document: {filename}" + (f"\nPart {i + 1} of {len(parts)}{where}" if len(parts) > 1 else "")
        return f"{head}\nDocument outline (for context only):\n{doc_outline}" if doc_outline else head

    def run(item: tuple[int, Part]) -> Extraction:
        i, part = item
        return _extract_splitting(part.text, context(i, part), extract, length)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(run, enumerate(parts)))

    facts = DocumentFacts(parts=len(parts))
    seen_entities: dict[tuple[str, str], int] = {}
    seen_relations: set[tuple] = set()
    for result in results:
        facts.dropped += result.dropped
        for e in result.entities:
            key = (e.type, resolver.canonical(e.name))
            located = _chunks_with(_normal_text(e.quote), chunks)
            named = _chunks_naming(e.name, resolver.canonical(e.name), chunks)
            if key in seen_entities:
                facts.entities[seen_entities[key]][3].update(located | named)
                continue
            seen_entities[key] = len(facts.entities)
            facts.entities.append((e.type, e.name, e.quote, located | named))
        names = {e.name: e.type for e in result.entities}
        for r in result.relations:
            key = (r.type, names[r.source], resolver.canonical(r.source), names[r.target],
                   resolver.canonical(r.target))
            if key in seen_relations:
                continue
            seen_relations.add(key)
            located = sorted(_chunks_with(_normal_text(r.quote), chunks))
            facts.relations.append((r.type, r.source, r.target, r.quote, located[0] if located else None))
    return facts


def _extract_splitting(text: str, context: str, extract: Callable[[str, str], Extraction],
                       length: Callable[[str], int], depth: int = 0) -> Extraction:
    """Extract from one part; if the reply was cut off (too many facts), halve the part and retry."""
    try:
        return extract(text, context)
    except ExtractionError as exc:
        if "truncated" not in str(exc) or depth >= 3 or length(text) < 2000:
            raise
    found = text.rfind("\n\n", 0, len(text) // 2 + len(text) // 4)
    middle = found if found > len(text) // 4 else len(text) // 2
    left = _extract_splitting(text[:middle], context, extract, length, depth + 1)
    right = _extract_splitting(text[middle:], context, extract, length, depth + 1)
    return Extraction(left.entities + right.entities, left.relations + right.relations, left.dropped + right.dropped)


def _chunks_with(normal_quote: str, chunks: list[ChunkText]) -> set[str]:
    """Chunks whose text contains the quote; for a quote spanning two chunks, its first 60 characters."""
    if not normal_quote:
        return set()
    hits = {c.chunk_id for c in chunks if normal_quote in c.text}
    if not hits and len(normal_quote) > 60:
        head = normal_quote[:60]
        hits = {c.chunk_id for c in chunks if head in c.text}
    return hits


def _chunks_naming(name: str, canonical: str, chunks: list[ChunkText]) -> set[str]:
    """Chunks that name the entity (as extracted, or by its canonical name), as whole words."""
    hits: set[str] = set()
    for form in {_normal_text(name), canonical}:
        if len(form) < MIN_NAME_CHARS:
            continue
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])")
        hits |= {c.chunk_id for c in chunks if pattern.search(c.text)}
    return hits if len(hits) <= MAX_NAME_MENTIONS else set()
