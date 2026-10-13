"""Readable chunk labels, e.g. 'cash_grant_chunk_3', stored in chunks.labelchunk.

Questions files for `kb eval` refer to chunks by this label, so ingest and evaluation share it.
"""

import re
from pathlib import Path


def doc_key(filename: str) -> str:
    """'cash_grant (1)(1).pdf' -> 'cash_grant': the stem without download-copy suffixes, as an identifier."""
    stem = re.sub(r"\(\d+\)", "", Path(filename).stem)
    return re.sub(r"\W+", "_", stem).strip("_")


def chunk_label(filename: str, chunk_index: int) -> str:
    """Stable, readable chunk id used in questions files, e.g. 'cash_grant_chunk_3'."""
    return f"{doc_key(filename)}_chunk_{chunk_index}"
