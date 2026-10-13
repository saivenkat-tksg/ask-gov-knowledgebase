"""Find section headings in a PDF from its fonts, and mark them in the extracted page text.

pypdf's `extract_text()` stays the source of the text (eval `contains` phrases are written against
it). pdfium is used only to read each line's font size and weight; lines that look like headings
are then located in the pypdf text and replaced by a markdown heading line (`### Title`), which
`kb.ingestion.structure` turns into sections.

A line is a heading candidate when it is short and either larger than the body text or bold at
body size. Levels come from three signals, in this order:

- `Chapter 2:` / `Part 1:` lines and the largest sizes open a new top-level scope;
- numbered headings nest by numbering depth under the last unnumbered heading
  (`Agency Processes` > `4. Personnel Processes` > `4.1 Leave Management`);
- an unnumbered heading that follows numbered ones belongs to the last numbered heading
  (`4.1 Leave Management` > `Impediments`), unless it starts a new list (`Desired State:` then `1. ...`);
- questions (`...?`, `Q12. User: ...`) sit one level below the heading that introduces them.
"""

import ctypes
import logging
import re
from collections import Counter
from dataclasses import dataclass

log = logging.getLogger(__name__)

MAX_HEADING_WORDS = 16
_NUMBERED = re.compile(r"^((?:\d{1,2}\.)*\d{1,2}|[A-Z])[.)]?\s+\S")  # "4.1 Leave", "1.3. Finance", "A. General"
_CHAPTER = re.compile(r"^(chapter|part|section)\s+[\w.]+\s*:?", re.IGNORECASE)  # may wrap onto the next line
_TOP = re.compile(r"^chapter\s+[\w.]+\s*:?", re.IGNORECASE)  # always a top-level heading
_QUESTION = re.compile(r"(\?\s*$)|(^Q\d+[.)]?\s)")
_BULLET = re.compile(r"^([o•▪◦·*-]||)\s")
_SENTENCE_END = re.compile(r"[.;,]\s*$")
_BOLD_FONT = re.compile(r"bold|semibold|black|heavy", re.IGNORECASE)
_TOC = re.compile(r"table\s+of\s+contents?", re.IGNORECASE)
_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "’": "'", "‘": "'",
                            "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


@dataclass
class Line:
    page: int  # 0-based
    text: str
    size: float
    bold: bool


@dataclass
class Heading:
    page: int  # 0-based
    title: str
    level: int


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_LIGATURES)).strip()


def _normalise(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.translate(_LIGATURES).lower())


# --- reading lines with font information -------------------------------------------------------

def _font_name(raw, textpage, index: int) -> str:
    buf = ctypes.create_string_buffer(128)
    flags = ctypes.c_int()
    raw.FPDFText_GetFontInfo(textpage, index, buf, 128, ctypes.byref(flags))
    return buf.value.decode("latin-1", "replace")


def _to_line(page_no: int, chars: list[tuple[str, float, bool]]) -> Line | None:
    text = clean("".join(c for c, _, _ in chars))
    visible = [(s, b) for c, s, b in chars if c.strip()]
    if not text or not visible:
        return None
    size = Counter(round(s, 1) for s, _ in visible).most_common(1)[0][0]
    bold = sum(b for _, b in visible) > len(visible) / 2
    return Line(page_no, text, size, bold)


def read_lines(data: bytes) -> list[list[Line]]:
    """Every text line of every page, with its dominant font size (in points, after scaling) and weight."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    pdf = pdfium.PdfDocument(data)
    pages: list[list[Line]] = []
    try:
        for page_no in range(len(pdf)):
            page = pdf[page_no]
            textpage = page.get_textpage()
            lines: list[Line] = []
            chars: list[tuple[str, float, bool]] = []
            for i in range(textpage.count_chars()):
                ch = chr(raw.FPDFText_GetUnicode(textpage, i))
                if ch in "\r\n":
                    if line := _to_line(page_no, chars):
                        lines.append(line)
                    chars = []
                    continue
                if not ch.strip():
                    chars.append((ch, 0.0, False))
                    continue
                matrix = raw.FS_MATRIX()
                raw.FPDFText_GetMatrix(textpage, i, ctypes.byref(matrix))
                size = raw.FPDFText_GetFontSize(textpage, i) * (abs(matrix.d) or 1.0)
                bold = raw.FPDFText_GetFontWeight(textpage, i) >= 600 or bool(
                    _BOLD_FONT.search(_font_name(raw, textpage, i)))
                chars.append((ch, size, bold))
            if line := _to_line(page_no, chars):
                lines.append(line)
            textpage.close()
            page.close()
            pages.append(lines)
    finally:
        pdf.close()
    return pages


# --- choosing heading lines and their levels ---------------------------------------------------

def _body_size(pages: list[list[Line]]) -> float:
    sizes = Counter()
    for lines in pages:
        for line in lines:
            sizes[line.size] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _is_candidate(line: Line, body: float) -> bool:
    words = line.text.split()
    if not words or len(words) > MAX_HEADING_WORDS or len(line.text) < 3:
        return False
    if not re.search(r"[A-Za-z]{2}", line.text) or _SENTENCE_END.search(line.text):
        return False
    if line.size < body * 0.95:  # captions, diagram labels, footnotes
        return False
    if _BULLET.match(line.text):
        return False
    return line.size >= body * 1.15 or line.bold


def _continues(flat: list[Line], i: int) -> bool:
    """A bold line whose sentence carries on in lowercase on the next line is wrapped text, not a heading."""
    if i + 1 >= len(flat):
        return False
    nxt = flat[i + 1]
    return nxt.page == flat[i].page and nxt.text[:1].islower() and not _BULLET.match(nxt.text)


def _merge_wrapped(cands: list[tuple[int, Line]], body: float) -> list[tuple[int, Line]]:
    """Join a heading that wrapped onto two lines: a bare `Chapter 2:` with the title below it, or two
    consecutive title-sized lines (>= 1.5x body). Section headings at smaller sizes are left alone,
    because they are often stacked (`Agency Processes` directly above `1. Births ...`)."""
    out: list[tuple[int, Line]] = []
    for idx, line in cands:
        if out:
            prev_idx, prev = out[-1]
            adjacent = idx == prev_idx + 1 and prev.page == line.page
            bare_chapter = _CHAPTER.fullmatch(prev.text.strip()) is not None
            wrapped_title = (prev.size == line.size and line.size >= body * 1.5
                             and not _NUMBERED.match(line.text) and not prev.text.endswith((":", "?")))
            if adjacent and (bare_chapter or wrapped_title):
                out[-1] = (idx, Line(prev.page, f"{prev.text} {line.text}", max(prev.size, line.size),
                                     prev.bold or line.bold))
                continue
        out.append((idx, line))
    return out


def _chapters_first(lines: list[Line]) -> list[Line]:
    out: list[Line] = []
    for line in lines:
        if _TOP.match(line.text):
            insert = len(out)
            while insert > 0 and out[insert - 1].page == line.page and not _NUMBERED.match(out[insert - 1].text):
                insert -= 1
            out.insert(insert, line)
        else:
            out.append(line)
    return out


def _numbering(title: str) -> tuple[str, int] | None:
    """('digit', 2) for '4.1 Leave', ('alpha', 1) for 'B. Follow-up', None when unnumbered."""
    m = _NUMBERED.match(title)
    if not m:
        return None
    return ("alpha" if m.group(1).isalpha() else "digit"), m.group(1).count(".") + 1


def _starts_list(title: str) -> bool:
    m = _NUMBERED.match(title)
    return bool(m) and m.group(1).split(".")[-1] in ("1", "A")


def detect_headings(pages: list[list[Line]]) -> list[Heading]:
    body = _body_size(pages)
    if not body:
        return []
    flat = [line for lines in pages for line in lines]
    toc_pages = {i for i, lines in enumerate(pages) if _TOC.search(" ".join(line.text for line in lines))}
    cands = [(i, line) for i, line in enumerate(flat)
             if line.page not in toc_pages and _is_candidate(line, body) and not _continues(flat, i)]
    cands = _merge_wrapped(cands, body)

    # Drop a repeated title: "4.1 Leave Management" followed by the figure title "Leave Management".
    kept: list[Line] = []
    for _, line in cands:
        stripped = _normalise(_NUMBERED.sub(lambda m: m.group(0)[-1], line.text, count=1))
        if kept and kept[-1].page == line.page and stripped and _normalise(kept[-1].text).endswith(stripped):
            continue
        kept.append(line)
    # A chapter title governs its whole page, even when the page's text lists it after an unnumbered
    # subheading ("Current State Overview" above "Chapter 4: ..."). Numbered headings above it on the
    # page continue the previous chapter, so it stays below those.
    kept = _chapters_first(kept)

    # Size bands: larger text = higher level; bold body-size text is the lowest band. Sizes used on a
    # single page (a cover) don't get a band, and chapter headings are always level 1.
    pages_per_size: dict[int, set[int]] = {}
    for line in kept:
        if line.size >= body * 1.15 and not _TOP.match(line.text):
            pages_per_size.setdefault(round(line.size), set()).add(line.page)
    sizes = sorted((size for size, used in pages_per_size.items() if len(used) > 1), reverse=True)
    # Level 1 is kept free for chapters, or for a one-off title larger than every repeated size.
    has_title = any(size > sizes[0] for size in pages_per_size) if sizes else False
    offset = 2 if has_title or any(_TOP.match(line.text) for line in kept) else 1
    band = {size: rank + offset for rank, size in enumerate(sizes)}
    lowest = len(sizes) + offset

    def size_band(size: float) -> int:
        size = round(size)
        if size in band:
            return band[size]
        if sizes and size > sizes[0]:
            return offset - 1  # a one-off title above every repeated heading size
        below = [lvl for s, lvl in band.items() if s <= size]  # nearest band at or below this size
        return min(below) if below and size >= body * 1.15 else lowest

    headings: list[Heading] = []
    # Open headings, outermost first: (level, kind, numbering, size_level).
    # kind: anchor (unnumbered scope opener) | numbered | sub (label under a numbered heading) | question
    stack: list[tuple[int, str, tuple[str, int] | None, int]] = []
    for pos, line in enumerate(kept):
        title = line.text
        size_level = max(size_band(line.size), 1)
        nxt = kept[pos + 1].text if pos + 1 < len(kept) else ""
        numbering = _numbering(title)
        last_numbered = next((e[0] for e in reversed(stack) if e[1] == "numbered"), None)
        anchor = next((e for e in reversed(stack) if e[1] == "anchor"), None)

        if _TOP.match(title):
            level, kind = 1, "anchor"
        elif size_level == 1:
            level, kind = 1, "anchor"
        elif _QUESTION.search(title):
            # Questions are siblings of the previous question, else children of the heading above.
            previous = next((e[0] for e in reversed(stack) if e[1] == "question"), None)
            level = previous if previous is not None else (stack[-1][0] if stack else 0) + 1
            kind = "question"
        elif numbering:
            sibling = next((e[0] for e in reversed(stack) if e[2] == numbering), None)
            if sibling is not None:  # "2. Firearms" continues "1. TINT", whatever its font size
                level = sibling
            elif stack and stack[-1][1] == "question":  # a numbered list inside an answer
                level = stack[-1][0] + 1
            else:
                level = (anchor[0] if anchor else 0) + numbering[1]
            kind = "numbered"
        elif (last_numbered is not None and not _starts_list(nxt)
              and (anchor is None or size_level >= anchor[3])):
            # "Impediments" after "4.1 Leave Management" belongs to it; a heading that starts a new
            # list ("Desired State:" then "1. ...") or is larger than the scope opener does not.
            # A smaller label directly under a larger one nests ("Desired Capabilities" > "Births").
            level = last_numbered + 1
            for entry in reversed(stack):
                if entry[1] != "sub":
                    break
                if size_level == entry[3]:  # same style as an open label: its sibling
                    level = entry[0]
                    break
                if size_level > entry[3]:  # smaller than an open label: its child
                    level = entry[0] + 1
                    break
            kind = "sub"
        else:
            level, kind = max(size_level, 1), "anchor"

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, kind, numbering, size_level))
        headings.append(Heading(line.page, title, level))
    return headings


# --- marking headings in the pypdf text ---------------------------------------------------------

def _locate(text: str, title: str, start: int) -> tuple[int, int] | None:
    """Find `title` in `text` from `start`, ignoring spacing, case and ligatures (pypdf breaks some
    headings across lines, e.g. 'Ov\\nerseas'). Returns the (start, end) span in `text`."""
    target = _normalise(title)
    if not target:
        return None
    norm_chars: list[str] = []
    index: list[int] = []
    for i in range(start, len(text)):
        for ch in _normalise(text[i]):
            norm_chars.append(ch)
            index.append(i)
    found = "".join(norm_chars).find(target)
    if found == -1:
        return None
    end = index[found + len(target) - 1] + 1
    while end < len(text) and text[end] in "?:.!)]\"' \t":  # trailing punctuation of the heading line
        end += 1
    return index[found], end


_PAGE_LABEL = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)


def strip_running_lines(page_texts: list[str]) -> list[str]:
    """Remove page furniture before sectioning: page-number lines ("Page 3", "12", "3 of 9") and lines
    repeated on most pages (running headers and footers), which would otherwise land inside
    whichever section is open at the page break."""
    counts: Counter = Counter()
    for text in page_texts:
        counts.update({_normalise(re.sub(r"\d+", "", line)) for line in text.splitlines() if line.strip()})
    pages = len(page_texts)
    repeated = {key for key, n in counts.items() if key and pages >= 4 and n >= 0.6 * pages}
    out = []
    for text in page_texts:
        kept = [line for line in text.splitlines()
                if not _PAGE_LABEL.match(line) and _normalise(re.sub(r"\d+", "", line)) not in repeated]
        out.append("\n".join(kept))
    return out


_STEP_CELL = re.compile(r"^((\d{1,2}\.)*\d{1,2}|[A-Za-z])[.)]?$")


def _split_table_at(text: str, line_start: int) -> tuple[tuple[int, int], str] | None:
    """For a heading found in a table row: if the row is a title row (one filled cell, apart from a
    step number), return the row's span and the table header to restart the table with below the
    heading. Desired-state pages put each process ("| 2. | Firearms Registration | |") as a title
    row inside one big table; splitting there gives each process its own section and table."""
    line_end = text.find("\n", line_start)
    line_end = len(text) if line_end == -1 else line_end
    cells = [c.strip() for c in text[line_start:line_end].strip().strip("|").split("|")]
    if sum(1 for c in cells if c and not _STEP_CELL.match(c)) != 1:
        return None
    # The table's header: the first two lines of the run of rows this row belongs to.
    lines_before = text[:line_start].split("\n")[:-1]
    k = len(lines_before)
    while k > 0 and lines_before[k - 1].startswith("|"):
        k -= 1
    header = lines_before[k:k + 2]
    if len(header) < 2 or "---" not in header[1]:
        return None
    rest = text[line_end + 1:]
    restart = "\n".join(header) + "\n" if rest.startswith("|") else ""
    return (line_start, line_end + 1), restart


def mark_headings(page_texts: list[str], headings: list[Heading]) -> tuple[list[str], int]:
    """Replace each heading's text in its page with a markdown heading line. Returns (texts, found).

    Headings are located in page order where possible, falling back to a search of the whole page
    (pdfium and pypdf sometimes order a page's lines differently). A level-1 heading (a chapter) is
    moved to the start of its page so the page's content falls under it."""
    by_page: dict[int, list[Heading]] = {}
    for h in headings:
        by_page.setdefault(h.page, []).append(h)
    out = list(page_texts)
    found = 0
    for page, items in by_page.items():
        if page >= len(out):
            continue
        text = out[page]
        spans: list[tuple[int, int, Heading]] = []
        cursor = 0
        for h in items:
            span = _locate(text, h.title, cursor) or _locate(text, h.title, 0)
            if span is None or any(a < span[1] and span[0] < b for a, b, *_ in spans):
                log.debug("heading not found in page %d text: %s", page + 1, h.title)
                continue
            marker = f"\n\n{'#' * h.level} {h.title}\n\n"
            line_start = text.rfind("\n", 0, span[0]) + 1
            if text.startswith("|", line_start):
                split = _split_table_at(text, line_start)
                if split is None:
                    continue  # a heading-like cell in an ordinary row: never break a row
                span, marker = split[0], marker.rstrip("\n") + "\n\n" + split[1]
            elif text.startswith("Table:", line_start):
                span = (line_start, line_start)  # the table's title: put the heading above the table
            spans.append((*span, h, marker))
            cursor = span[1]
        spans.sort(key=lambda x: x[0])
        lead, parts, cursor, numbered_above = [], [], 0, False
        for a, b, h, marker in spans:
            parts.append(text[cursor:a])
            (lead if _TOP.match(h.title) and not numbered_above else parts).append(marker)
            numbered_above = numbered_above or bool(_NUMBERED.match(h.title))
            cursor = b
            found += 1
        parts.append(text[cursor:])
        out[page] = "".join(lead + parts)
    return out, found
