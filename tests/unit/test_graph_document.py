from kb.graph.document import ChunkText, extract_document, outline, page_range, split_parts
from kb.graph.extract import Entity, Extraction, ExtractionError, Relation, _normal_text
from kb.graph.resolve import Resolver

DOC = ("# Chapter 1: GRO\n\nThe General Register Office registers births.\n\n"
       "## 1. Birth Registration\n\nRegistrations are free of charge. Bring the medical certificate.\n\n"
       "# Chapter 2: Police\n\nThe Guyana Police Force handles firearm licences.\n")


def chunks_of(*texts) -> list[ChunkText]:
    return [ChunkText(f"c{i}", _normal_text(t)) for i, t in enumerate(texts)]


def test_small_document_is_one_part():
    assert [p.start for p in split_parts(DOC, 10_000, len)] == [0]


def test_large_document_is_cut_at_chapters_then_packed():
    parts = split_parts(DOC, 160, len)  # chapter 1 is 157 characters: chapters fit, two don't together
    assert len(parts) == 2
    assert parts[0].text.startswith("# Chapter 1") and parts[1].text.startswith("# Chapter 2")
    assert "".join(p.text for p in parts) == DOC  # nothing lost or repeated


def test_oversized_section_falls_back_to_paragraphs():
    text = "# Only chapter\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(10))
    parts = split_parts(text, 200, len)
    assert len(parts) > 1 and all(len(p.text) <= 200 for p in parts)
    assert "".join(p.text for p in parts) == text


def test_outline_and_pages():
    assert outline(DOC).splitlines() == ["- Chapter 1: GRO", "  - 1. Birth Registration", "- Chapter 2: Police"]
    assert page_range(5, 60, [[0, 1], [50, 2], [100, 3]]) == (1, 2)
    assert page_range(5, 6, []) == (None, None)


def test_facts_are_traced_to_chunks_and_names():
    chunks = chunks_of("[GRO] The General Register Office registers births.",
                       "Registrations are free of charge.",
                       "Contact the General Register Office in Georgetown.")
    calls = []

    def fake_extract(text, context):
        calls.append(context)
        return Extraction(
            [Entity("Office", "General Register Office", "The General Register Office registers births"),
             Entity("Service", "Birth registration", "registers births"),
             Entity("Fee", "Free", "free of charge")],
            [Relation("HANDLED_BY", "Birth registration", "General Register Office", "The General Register Office registers births"),
             Relation("COSTS", "Birth registration", "Free", "Registrations are free of charge")])

    facts = extract_document(DOC, "guide.pdf", [], chunks, fake_extract, Resolver(), 10_000, len)
    assert facts.parts == 1 and "Document: guide.pdf" in calls[0] and "Chapter 2: Police" in calls[0]
    office = next(e for e in facts.entities if e[0] == "Office")
    assert office[3] == {"c0", "c2"}  # quoted in c0, named in c2
    assert {(r[0], r[4]) for r in facts.relations} == {("HANDLED_BY", "c0"), ("COSTS", "c1")}


def test_cut_off_reply_splits_the_part():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 300 for i in range(4))
    sizes = []

    def fake_extract(part, context):
        sizes.append(len(part))
        if len(part) > len(text) // 2 + 100:
            raise ExtractionError("extraction reply truncated at the output limit")
        return Extraction([Entity("Fee", f"Fee {len(sizes)}", "word word")], [])

    facts = extract_document(text, "big.pdf", [], [], fake_extract, Resolver(), 100_000, len, concurrency=1)
    assert sizes[0] == len(text) and len(sizes) == 3  # whole text, then two halves
    assert len(facts.entities) == 2
