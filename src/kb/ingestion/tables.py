"""Tables as markdown, and splitting them between rows.

Every loader writes tables the same way, so the chunker can recognise and protect them:

    Table: ID Card Creation
    | Step | Action | Completed by (Role) | Notes |
    | --- | --- | --- | --- |
    | 1. | ISIS Supervisor is emailed the ... application | Data Entry Clerk |  |

A table that fits in a chunk is never split. A larger one is split between rows only, and every
piece repeats the title and header rows, so each chunk reads as a complete small table: a row
always arrives with the column names that give its cells meaning.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

_TITLE = re.compile(r"^Table:\s")


def markdown_table(header: list[str], rows: list[list[str]], title: str | None = None) -> str:
    def cell(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").replace("|", "/")).strip()

    width = max(len(header), *(len(r) for r in rows)) if rows else len(header)
    header = [*header, *[""] * (width - len(header))]
    lines = [f"Table: {cell(title)}"] if title else []
    lines.append("| " + " | ".join(cell(h) for h in header) + " |")
    lines.append("|" + " --- |" * width)
    for row in rows:
        row = [*row, *[""] * (width - len(row))]
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def is_row(line: str) -> bool:
    return line.startswith("|") and line.rstrip().endswith("|")


@dataclass
class Block:
    kind: str  # "text" | "table"
    start: int  # offsets in the text the block came from
    end: int


def split_blocks(text: str) -> list[Block]:
    """Text and table blocks in order. A table block is a run of `| ... |` lines, with the `Table:`
    title line directly above it."""
    blocks: list[Block] = []
    pos = 0
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        title = _TITLE.match(lines[i]) and i + 1 < len(lines) and is_row(lines[i + 1])
        if title or is_row(lines[i]):
            start = pos
            if title:
                pos += len(lines[i])
                i += 1
            while i < len(lines) and is_row(lines[i]):
                pos += len(lines[i])
                i += 1
            blocks.append(Block("table", start, pos))
            continue
        if blocks and blocks[-1].kind == "text":
            blocks[-1].end = pos + len(lines[i])
        else:
            blocks.append(Block("text", pos, pos + len(lines[i])))
        pos += len(lines[i])
        i += 1
    return blocks


def split_table(text: str, budget: int, length: Callable[[str], int]) -> list[tuple[str, int, int]]:
    """Pieces of one table block, each within `budget`: (text, start, end) with offsets of its rows in
    `text`. Pieces after the first repeat the title and header rows."""
    lines = text.rstrip("\n").split("\n")
    head_count = 0
    if lines and _TITLE.match(lines[0]):
        head_count = 1
    if len(lines) > head_count + 1 and set(lines[head_count + 1].replace("|", "").split()) <= {"---"}:
        head_count += 2  # header row + separator row
    else:
        head_count += 1
    head = "\n".join(lines[:head_count])
    if length(text.rstrip("\n")) <= budget:
        return [(text.rstrip("\n"), 0, len(text.rstrip("\n")))]

    pieces: list[tuple[str, int, int]] = []
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    current: list[int] = []

    def emit() -> None:
        if current:
            body = "\n".join(lines[i] for i in current)
            start, end = offsets[current[0]], offsets[current[-1]] + len(lines[current[-1]])
            pieces.append((f"{head}\n{body}", start, end))
            current.clear()

    for i in range(head_count, len(lines)):
        if current and length(head + "\n" + "\n".join(lines[j] for j in [*current, i])) > budget:
            emit()
        current.append(i)
    emit()
    return pieces or [(head, 0, len(head))]
