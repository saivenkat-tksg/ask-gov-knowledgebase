"""Structure-aware chunking: split at headings first, recursively only inside oversized sections.

Input is text with markdown headings (`## Title`): DOCX/HTML/Markdown loaders produce them, and
PDF headings are found from fonts by `kb.ingestion.pdf_headings`.

1. Sections: each heading starts a section that runs to the next heading, across page breaks.
   A section knows its heading path, e.g. ("Chapter 4: GENERAL REGISTRAR OFFICE",
   "Agency Processes for Digitization", "1. Births and Deaths Certification Process").
2. Fitting, per section (budget = chunk_size minus the heading path line):
   - fits: one chunk;
   - too big: recursive split (paragraph > line > sentence > word) inside the section only,
     with overlap between its pieces; every piece keeps the section's heading path. Tables
     (markdown, see kb.ingestion.tables) are never split inside a row: a table that fits stays
     whole, a larger one is split between rows with its title and header repeated in each piece;
   - small (below min_tokens): merged with the following sibling sections (same parent), or with
     its own first subsection, until the chunk reaches min_tokens. Sections are never merged
     across a higher heading, so one chunk never mixes two services.
3. Every chunk starts with its heading path, `[A > B > C]`, so the embedding, the keyword index
   and the LLM all see which service and step the text belongs to.

Documents with fewer than `min_headings` headings return None, and the caller falls back to plain
recursive chunking.
"""

import re
from dataclasses import dataclass, field

from kb.ingestion.chunking import DEFAULT_SEPARATORS, LengthFunction, RecursiveChunker
from kb.ingestion.tables import split_blocks, split_table

_HEADING = re.compile(r"^(#{1,12})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_DIGIT_NUMBERED = re.compile(r"^(?:\d{1,2}\.)*\d{1,2}[.)]?\s+(.+)$")


@dataclass
class Section:
    path: tuple[str, ...]  # heading titles from the top level down to this section
    start: int  # offsets of the section's own text (heading line excluded) in the document text
    end: int
    text: str


@dataclass
class StructuredChunk:
    text: str  # heading-path line + section text, as stored and embedded
    start: int  # offsets of the body text in the document text
    end: int
    path: tuple[str, ...]  # heading path shared by everything in the chunk
    parts: int = 1  # pieces of a recursively split section
    part: int = 1
    merged: list[str] = field(default_factory=list)  # titles of sections merged into this chunk
    sections: list[Section] = field(default_factory=list, repr=False)  # empty for split pieces


def parse_sections(text: str) -> tuple[list[Section], int]:
    """Sections in document order (including text before the first heading), and the heading count."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    path: tuple[str, ...] = ()
    body_start = 0
    count = 0

    def add(end: int) -> None:
        raw = text[body_start:end]
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        sections.append(Section(path, body_start + lead, body_start + lead + len(body), body))

    for m in _HEADING.finditer(text):
        add(m.start())
        level, title = len(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = tuple(t for _, t in stack)
        body_start = m.end()
        count += 1
    add(len(text))
    return sections, count


def breadcrumb(path: tuple[str, ...]) -> str:
    return f"[{' > '.join(path)}]" if path else ""


def service_of(path: tuple[str, ...]) -> str | None:
    """The process a section belongs to: its deepest digit-numbered heading, without the number,
    under the document's top heading. '1. TINT Waiver Application' in the current-state part and
    in the desired-state part get the same value, which links them across distant pages."""
    for i in range(len(path) - 1, 0, -1):
        m = _DIGIT_NUMBERED.match(path[i])
        if m:
            return f"{path[0]} > {m.group(1).strip()}"
    return None


class StructureChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        length_function: LengthFunction = len,
        min_tokens: int = 150,
        min_headings: int = 3,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length = length_function
        self.min_tokens = min_tokens
        self.min_headings = min_headings

    def split(self, text: str) -> list[StructuredChunk] | None:
        sections, headings = parse_sections(text)
        if headings < self.min_headings:
            return None
        chunks: list[StructuredChunk] = []
        group: list[Section] = []

        def flush() -> None:
            if group:
                chunks.append(self._render(group))
                group.clear()

        for section in sections:
            if not section.text:
                continue  # heading directly followed by a subheading: it lives on in the paths
            budget = self.chunk_size - self.length(breadcrumb(section.path)) - 1
            if self.length(section.text) > budget:
                flush()
                chunks.extend(self._split_large(section, budget))
                continue
            if group and self._joins(group, section):
                if self.length(self._render([*group, section]).text) <= self.chunk_size:
                    group.append(section)
                    continue
            flush()
            group.append(section)
        flush()
        return self._absorb_tail(chunks)

    def _joins(self, group: list[Section], section: Section) -> bool:
        """Join while the group is still small, and only within one parent heading."""
        if self.length("\n".join(s.text for s in group)) >= self.min_tokens:
            return False
        first = group[0]
        return _related(first.path, section.path) or (len(group) > 1 and _related(group[-1].path, section.path))

    def _render(self, group: list[Section]) -> StructuredChunk:
        common = _common_prefix([s.path for s in group])
        lines = [breadcrumb(common)] if common else []
        merged = []
        for s in group:
            if len(group) > 1 and s.path != common:
                title = " > ".join(s.path[len(common):])
                lines.append(title)
                merged.append(title)
            lines.append(s.text)
        return StructuredChunk("\n".join(lines), group[0].start, group[-1].end, common,
                               merged=merged, sections=list(group))

    def _split_large(self, section: Section, budget: int) -> list[StructuredChunk]:
        """Text is split recursively; tables only between rows, each piece repeating the table's
        title and header. Neighbouring pieces that fit together are packed into one chunk."""
        budget = max(budget, 32)
        splitter = RecursiveChunker(budget, min(self.chunk_overlap, budget // 4), DEFAULT_SEPARATORS, self.length)
        pieces: list[tuple[str, int, int]] = []  # (text, start, end) in section.text
        for block in split_blocks(section.text):
            text = section.text[block.start:block.end]
            if block.kind == "table":
                for piece, s, e in split_table(text, budget, self.length):
                    if self.length(piece) <= budget:
                        pieces.append((piece, block.start + s, block.start + e))
                    else:  # a single row larger than a chunk
                        pieces += [(p.text, block.start + s, block.start + e) for p in splitter.split(piece)]
            else:
                pieces += [(p.text, block.start + p.start, block.start + p.end) for p in splitter.split(text)]
        packed: list[tuple[str, int, int]] = []
        for text, s, e in pieces:
            if packed and self.length(f"{packed[-1][0]}\n{text}") <= budget:
                packed[-1] = (f"{packed[-1][0]}\n{text}", packed[-1][1], e)
            else:
                packed.append((text, s, e))
        head = breadcrumb(section.path)
        return [
            StructuredChunk(f"{head}\n{text}" if head else text, section.start + s, section.start + e,
                            section.path, parts=len(packed), part=i)
            for i, (text, s, e) in enumerate(packed, start=1)
        ]

    def _absorb_tail(self, chunks: list[StructuredChunk]) -> list[StructuredChunk]:
        """A chunk left below min_tokens joins the previous chunk when it is a sibling or child of
        what that chunk holds (never across a higher heading) and the result still fits."""
        out: list[StructuredChunk] = []
        for chunk in chunks:
            prev = out[-1] if out else None
            if (prev is not None and prev.sections and chunk.sections
                    and self.length(chunk.text) < self.min_tokens
                    and _related(prev.sections[-1].path, chunk.sections[0].path)):
                merged = self._render([*prev.sections, *chunk.sections])
                if self.length(merged.text) <= self.chunk_size:
                    out[-1] = merged
                    continue
            out.append(chunk)
        return out


def _related(earlier: tuple[str, ...], later: tuple[str, ...]) -> bool:
    """`later` is a sibling of `earlier` (same parent heading) or its direct subsection."""
    if not later:
        return False
    return later[:-1] == earlier[:-1] if len(later) == len(earlier) else later[:-1] == earlier


def _common_prefix(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    common = paths[0]
    for p in paths[1:]:
        n = 0
        while n < min(len(common), len(p)) and common[n] == p[n]:
            n += 1
        common = common[:n]
    return common
