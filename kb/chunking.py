"""Recursive chunking via LangChain's RecursiveCharacterTextSplitter.

Text is split on the coarsest separator present (headings, then paragraphs,
lines, sentences, words, characters), recursing into pieces that are still too
long and merging small pieces back up to `chunk_size` with `chunk_overlap`.

Character offsets are computed here rather than with LangChain's
`add_start_index`, which mixes token overlap with character lengths and can
return -1 when chunking by tokens.
"""

from dataclasses import dataclass
from typing import Callable

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
# LangChain's markdown separators are regexes (headings, code fences, rules) ending in
# "\n\n", "\n", " ", "". Insert sentence boundaries before falling back to words.
MARKDOWN_SEPARATORS = [
    *RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)[:-2],
    r"\. ", r"\? ", "! ", "; ", ", ", " ", "",
]

LengthFunction = Callable[[str], int]


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def token_length_function(encoding: str = "cl100k_base") -> LengthFunction:
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
        length_function: LengthFunction = len,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        separators = separators or DEFAULT_SEPARATORS
        self._splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            is_separator_regex=separators is MARKDOWN_SEPARATORS,
            keep_separator="start",  # headings stay with the section they introduce
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            strip_whitespace=True,
        )

    def split(self, text: str) -> list[TextChunk]:
        if not text or not text.strip():
            return []
        return self._locate(text, [c for c in self._splitter.split_text(text) if c])

    @staticmethod
    def _locate(text: str, pieces: list[str]) -> list[TextChunk]:
        out: list[TextChunk] = []
        prev_start = -1
        for piece in pieces:
            idx = text.find(piece, prev_start + 1)
            if idx == -1:
                idx = text.find(piece)
            if idx == -1:
                idx = max(prev_start, 0)
            out.append(TextChunk(piece, idx, idx + len(piece)))
            prev_start = idx
        return out
