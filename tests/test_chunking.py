from kb.chunking import MARKDOWN_SEPARATORS, RecursiveChunker


def test_short_text_is_single_chunk():
    chunks = RecursiveChunker(100, 10).split("Hello world.")
    assert [c.text for c in chunks] == ["Hello world."]
    assert (chunks[0].start, chunks[0].end) == (0, 12)


def test_chunks_respect_size_and_offsets():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(20))
    chunker = RecursiveChunker(200, 40)
    chunks = chunker.split(text)
    assert len(chunks) > 5
    for c in chunks:
        assert len(c.text) <= 200
        assert text[c.start : c.end] == c.text


def test_overlap_carries_context():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = RecursiveChunker(50, 20).split(text)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # neighbours overlap


def test_long_word_falls_back_to_characters():
    chunks = RecursiveChunker(10, 0).split("x" * 35)
    assert [len(c.text) for c in chunks] == [10, 10, 10, 5]


def test_markdown_splits_on_headings_first():
    text = "# Title\nintro\n\n## A\n" + "alpha " * 10 + "\n\n## B\n" + "beta " * 10
    chunks = RecursiveChunker(80, 0, MARKDOWN_SEPARATORS).split(text)
    assert any(c.text.startswith("## A") for c in chunks)
    assert any(c.text.startswith("## B") for c in chunks)


def test_empty_text():
    assert RecursiveChunker(10, 0).split("   \n ") == []
