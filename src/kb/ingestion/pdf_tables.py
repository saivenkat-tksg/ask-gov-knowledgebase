"""Find tables in PDF pages and rebuild them as markdown tables, so rows stay intact through chunking.

pypdf flattens a table into a stream of cells: booklet page 48 comes out as "1. ISIS Supervisor is
emailed the ... Data Entry Clerk 2. Photo and signature are ... ISIS Supervisor", and the reader
can no longer tell which role belongs to which step. Here pdfplumber only *locates* tables (ruling
lines); each table is then rebuilt from word positions:

- columns come from the header row's cells ("Step | Action | Completed by (Role) | Notes"); a
  multi-line header ("Completed" / "by (Role)") is joined per column;
- a row starts at a step number ("1.", "2.") and wrapped lines are appended to the row above; a
  table without step numbers starts a row at every line with text in its first column;
- a step-number column with no header gets one ("#").

Candidates that are not data tables are skipped: single-column boxes (bordered paragraphs and
callouts), single-row boxes (flowchart nodes) and tables nested inside another table.

A page with tables is re-read with pdfplumber so each table can replace its own region:
text above, the table as markdown, text below. Pages without tables keep pypdf's text.
"""

import logging
import re
from dataclasses import dataclass

from kb.ingestion.tables import markdown_table

log = logging.getLogger(__name__)

_STEP = re.compile(r"^(\d{1,2}|[a-z]|[ivx]{1,4})[.)]$|^\d{1,2}\.\S", re.IGNORECASE)  # "1.", "a)", "ii.", "1.Clips"
_GLYPHS = str.maketrans({"\uf0b7": "\u2022", "\uf0a7": "\u2022", "\uf076": "\u2022", "\u00a0": " "})
_BOLD = re.compile(r"bold|semibold|black|heavy", re.IGNORECASE)
COLUMN_GAP = 14.0  # points of white space that separate two header cells on one line
LINE_TOLERANCE = 3.0
MAX_TITLE_WORDS = 12
ANCHOR_TOLERANCE = 18.0  # content may start this far left of its header text


@dataclass
class Table:
    bbox: tuple[float, float, float, float]  # x0, top, x1, bottom
    title: str | None
    header: list[str]
    rows: list[list[str]]
    continued: bool = False  # no header row on this page: the table continues from the previous page

    def to_markdown(self) -> str:
        return markdown_table(self.header, self.rows, self.title)


# --- reading a table from its words -------------------------------------------------------------

def _lines(words: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= LINE_TOLERANCE:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def _clusters(line: list[dict], edges: list[float] = ()) -> list[list[dict]]:
    """Runs of words on one line, one per cell: split where the gap is at least COLUMN_GAP points or
    a vertical ruling line of the table runs between two words."""
    out: list[list[dict]] = []
    for w in line:
        if out:
            prev = out[-1][-1]
            ruled = any(prev["x1"] - 1 <= e <= w["x0"] + 1 for e in edges)
            if w["x0"] - prev["x1"] < COLUMN_GAP and not ruled:
                out[-1].append(w)
                continue
        out.append([w])
    return out


def _text(words: list[dict]) -> str:
    return " ".join(w["text"] for w in words).translate(_GLYPHS).strip()


def _join(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    return a + b if a.endswith("-") and not a.endswith(" -") else f"{a} {b}"


def _is_step(word: dict) -> bool:
    return bool(_STEP.match(word["text"]))


def _is_bold(word: dict) -> bool:
    return bool(_BOLD.search(word.get("fontname", "")))


def build_table(words: list[dict], bbox, edges: list[float] = ()) -> Table | None:
    """`edges`: x positions of the table's vertical ruling lines (cell borders)."""
    lines = _lines(words)
    if len(lines) < 2:
        return None
    # Header: the first line with two or more cells. Single-cell lines above it form the title.
    header_at = next((i for i, line in enumerate(lines[:4]) if len(_clusters(line, edges)) >= 2), None)
    if header_at is None:
        return None
    title = " ".join(_text(line) for line in lines[:header_at]) or None
    header_cells = _clusters(lines[header_at], edges)
    anchors = [c[0]["x0"] for c in header_cells]
    # A first row that starts with a step number, or mid-sentence in lowercase, is data: the table
    # continues from the previous page, where its header is.
    first = header_cells[0][0]
    continued = _is_step(first) or first["text"][:1].islower()
    header = [""] * len(anchors) if continued else [_text(c) for c in header_cells]

    def column(word: dict) -> int:
        idx = 0
        for i, a in enumerate(anchors):
            if word["x0"] + ANCHOR_TOLERANCE >= a:
                idx = i
        return idx

    body = lines[header_at:] if continued else lines[header_at + 1:]
    first_data = next((i for i, line in enumerate(body) if _is_step(line[0])), None)
    numbered = first_data is not None
    if numbered and not continued:
        # Short lines between the header and the first numbered row continue the header ("by (Role)");
        # from the first longer line on (a banner or description row) it is table content.
        header_lines = 0
        for line in body[:first_data]:
            if len(line) > 4 or len({column(w) for w in line}) > 1:
                break  # a banner, description or spanning sub-title row: table content
            for w in line:
                header[column(w)] = _join(header[column(w)], w["text"])
            header_lines += 1
        body = body[header_lines:]

    # A step number left of the first header cell gets its own column.
    step_column = numbered and any(_is_step(line[0]) and line[0]["x1"] < anchors[0] - 2 for line in body)
    width = len(anchors) + (1 if step_column else 0)
    rows: list[list[str]] = []
    for line in body:
        if not _is_step(line[0]) and len(line) <= MAX_TITLE_WORDS and all(_is_bold(w) for w in line):
            # A bold line inside the table is a sub-title spanning the columns ("Personnel: Managing
            # Employee Leave"): its own row, text in the first content column, so it stays whole.
            cells = [""] * width
            cells[1 if step_column and width > 1 else 0] = _text(line)
            rows.append(cells)
            continue
        cells = [""] * width
        for k, w in enumerate(line):
            if step_column and k == 0 and _is_step(w) and w["x1"] < anchors[0] - 2:
                col = 0
            else:
                col = column(w) + (1 if step_column else 0)
            cells[col] = _join(cells[col], w["text"].translate(_GLYPHS))
        starts_row = _is_step(line[0]) if numbered else bool(cells[0])
        if starts_row or not rows:
            rows.append(cells)
        else:
            rows[-1] = [_join(a, b) for a, b in zip(rows[-1], cells)]
    if step_column:
        header = ["#", *header]

    # Drop columns that are empty everywhere, then require a real grid.
    keep = [i for i in range(width) if header[i].strip() or any(r[i].strip() for r in rows)]
    header = [header[i] for i in keep]
    rows = [[r[i] for i in keep] for r in rows if any(r[i].strip() for i in keep)]
    filled_columns = sum(1 for i in range(len(header)) if sum(1 for r in rows if r[i].strip()) >= 1)
    if len(header) < 2 or filled_columns < 2 or not rows:
        return None
    return Table(tuple(bbox), title, header, rows, continued)


# --- finding tables on a page and rebuilding the page text --------------------------------------

def _centre_in(obj: dict, bbox) -> bool:
    x = (obj["x0"] + obj["x1"]) / 2
    y = (obj["top"] + obj["bottom"]) / 2
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _chars_where(page, keep):
    """The page restricted to characters for which `keep(char)` holds (other objects untouched).
    Characters are assigned by their centre, so one straddling a table border is neither lost
    (as with strict within/outside filters) nor counted twice (as with crop)."""
    return page.filter(lambda obj: obj.get("object_type") != "char" or keep(obj))


def _mark_bold(words: list[dict], chars: list[dict]) -> None:
    """Set word["fontname"] to its majority font. (Asking extract_words for fontname would split a
    word wherever the font subset changes mid-word: "Lic ensed".)"""
    for w in words:
        fonts = [c["fontname"] for c in chars
                 if w["x0"] - 0.5 <= (c["x0"] + c["x1"]) / 2 <= w["x1"] + 0.5
                 and w["top"] - 0.5 <= (c["top"] + c["bottom"]) / 2 <= w["bottom"] + 0.5]
        w["fontname"] = max(set(fonts), key=fonts.count) if fonts else ""


def _inside(inner, outer, slack: float = 2.0) -> bool:
    return (inner[0] >= outer[0] - slack and inner[1] >= outer[1] - slack
            and inner[2] <= outer[2] + slack and inner[3] <= outer[3] + slack)


def find_tables(page) -> list[Table]:
    candidates = []
    for found in page.find_tables():
        x0, top, x1, bottom = found.bbox
        grid = found.extract()
        columns = max((sum(1 for c in row if c and c.strip()) for row in grid), default=0)
        if bottom - top < 30 or len(grid) < 2 or columns < 2:
            continue  # flowchart node, bordered paragraph or callout
        edges = sorted({round(c[0], 1) for c in found.cells} | {round(c[2], 1) for c in found.cells})
        candidates.append((found.bbox, edges))
    boxes = [b for b, _ in candidates]
    outer = [(b, e) for b, e in candidates if not any(o != b and _inside(b, o) for o in boxes)]
    tables = []
    for bbox, edges in sorted(outer, key=lambda c: c[0][1]):
        region = _chars_where(page, lambda c, b=bbox: _centre_in(c, b))
        words = region.extract_words(x_tolerance=1.5, y_tolerance=LINE_TOLERANCE)
        _mark_bold(words, region.chars)
        inner = [e for e in edges if bbox[0] + 2 < e < bbox[2] - 2]  # the outer border splits nothing
        table = build_table(words, bbox, inner)
        if table:
            tables.append(table)
    return tables


def _continue_header(table: Table, previous: Table | None) -> None:
    """Give a header-less continuation the header of the table it continues, when the columns match."""
    if previous is not None and len(previous.header) == len(table.header):
        table.header = list(previous.header)
        table.title = table.title or (f"{previous.title} (continued)" if previous.title else "(continued)")
    else:
        table.header = ["#" if i == 0 else f"Column {i + 1}" for i in range(len(table.header))]


def page_text_with_tables(page, tables: list[Table]) -> str:
    """The page's text with each table's region replaced by the table in markdown, top to bottom."""
    def band(top: float, bottom: float) -> str:
        def keep(c) -> bool:
            y = (c["top"] + c["bottom"]) / 2
            return top <= y < bottom and not any(_centre_in(c, t.bbox) for t in tables)

        return _chars_where(page, keep).extract_text() or ""

    parts, top = [], 0.0
    for t in tables:
        if t.bbox[1] > top:
            parts.append(band(top, t.bbox[1]))
        parts.append(t.to_markdown())
        top = max(top, t.bbox[3])
    parts.append(band(top, page.height + 1))
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def tables_by_page(data: bytes, skip: set[int] = frozenset()) -> dict[int, tuple[str, int]]:
    """{0-based page index: (page text with markdown tables, number of tables)} for pages with tables."""
    import io

    import pdfplumber

    out: dict[int, tuple[str, int]] = {}
    previous: Table | None = None  # last table on the previous page, for header-less continuations
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i in skip:
                previous = None
                continue
            try:
                tables = find_tables(page)
                for t in tables:
                    if t.continued:
                        _continue_header(t, previous)
                    previous = t
                if not tables:
                    previous = None
                if tables:
                    out[i] = (page_text_with_tables(page, tables), len(tables))
            except Exception as exc:  # one malformed page shouldn't stop the document
                log.warning("table extraction failed on page %d: %s", i + 1, exc)
            finally:
                page.close()
    return out
