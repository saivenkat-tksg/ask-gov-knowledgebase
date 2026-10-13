from kb.ingestion.pdf_headings import Heading, Line, detect_headings, mark_headings, strip_running_lines

BODY = 11.0


def page(n: int, *lines: tuple[str, float, bool]) -> list[Line]:
    return [Line(n, text, size, bold) for text, size, bold in lines]


def body(text: str = "Ordinary body text that runs for a while and ends here.") -> tuple[str, float, bool]:
    return (text, BODY, False)


def levels(pages) -> list[tuple[int, str]]:
    return [(h.level, h.title) for h in detect_headings(pages)]


def test_booklet_hierarchy():
    pages = [
        page(0, ("Chapter 1:", 16, False), ("MINISTRY OF HOME AFFAIRS", 18, True), body(), body()),
        page(1, ("Agency Processes", 14, False), body(), ("1. TINT Waiver Application", 14, False), body(),
             ("Impediments", 14, False), body(), ("2. Firearms Registration", 14, False), body()),
        page(2, ("4.1 Leave Management", 14, False), body(), ("Impediments:", 14, False), body(), body()),
        page(3, ("Desired State:", 14, False), body(), ("1. TINT Waiver Application", 14, True), body(),
             ("2. Firearms Registration", 13, False), body()),
        page(4, ("Chapter 2:", 16, False), ("CUSTOMS ANTI-NARCOTIC UNIT", 18, True), body(), body()),
    ]
    assert levels(pages) == [
        (1, "Chapter 1: MINISTRY OF HOME AFFAIRS"),
        (2, "Agency Processes"),
        (3, "1. TINT Waiver Application"),
        (4, "Impediments"),  # belongs to the numbered process above it
        (3, "2. Firearms Registration"),
        (4, "4.1 Leave Management"),
        (5, "Impediments:"),
        (2, "Desired State:"),  # followed by "1. ...": starts a new list, so it is not a sub-label
        (3, "1. TINT Waiver Application"),
        (3, "2. Firearms Registration"),  # numbering, not its smaller font, sets the level
        (1, "Chapter 2: CUSTOMS ANTI-NARCOTIC UNIT"),
    ]


def test_faq_questions_and_lists_inside_answers():
    pages = [
        page(0, ("Cash Grant - Frequently Asked Questions", 18, True), body(),
             ("Eligibility", 13, True), ("Who is eligible?", 13, True), body(),
             ("Do I need ID?", 13, True), body(), ("Payment", 13, True), ("How am I paid?", 13, True), body()),
        page(1, ("Part 2: User Questions", 13, True), ("A. General Questions", 13, True),
             ("Q1. User: What is the GRO?", BODY, True), body(),
             ("1. Barima-Waini", 13, True), ("2. Pomeroon-Supenaam", 13, True),
             ("Q2. User: Which region is Georgetown in?", BODY, True), body(),
             ("B. Follow-up Questions", 13, True), body()),
    ]
    assert levels(pages) == [
        (1, "Cash Grant - Frequently Asked Questions"),
        (2, "Eligibility"),
        (3, "Who is eligible?"),
        (3, "Do I need ID?"),
        (2, "Payment"),
        (3, "How am I paid?"),
        (2, "Part 2: User Questions"),
        (3, "A. General Questions"),
        (4, "Q1. User: What is the GRO?"),
        (5, "1. Barima-Waini"),  # a numbered list inside an answer
        (5, "2. Pomeroon-Supenaam"),
        (4, "Q2. User: Which region is Georgetown in?"),
        (3, "B. Follow-up Questions"),  # continues "A.", not nested under the question
    ]


def test_not_headings():
    pages = [
        page(0, ("TABLE OF", 40, False), ("CONTENT", 40, True), ("05. Guyana Fire and Rescue Service", 13, False)),
        page(1, ("Real Heading", 14, False), body(), ("A bold sentence that wraps onto the", BODY, True),
             ("next line in lowercase.", BODY, False), ("o VISAs", BODY, True), ("Applicant", 6, True),
             ("Section Title", 14, False), body()),
    ]
    assert [t for _, t in levels(pages)] == ["Real Heading", "Section Title"]


def test_mark_headings_handles_broken_text_and_punctuation():
    texts = ["Intro\n3. Ov\nerseas Registrations (Births) \nBody text.\nWho is eligible?\nAnswer."]
    heads = [Heading(0, "3. Overseas Registrations (Births)", 3), Heading(0, "Who is eligible?", 4)]
    out, found = mark_headings(texts, heads)
    assert found == 2
    assert "### 3. Overseas Registrations (Births)\n\n" in out[0]
    assert "#### Who is eligible?\n\n\nAnswer." in out[0]  # the "?" went with the heading
    assert "Ov\nerseas" not in out[0]


def test_chapter_moves_to_page_start_unless_numbered_headings_precede():
    out, _ = mark_headings(["Current State Overview\nText.\nChapter 4: GRO\n"],
                           [Heading(0, "Current State Overview", 2), Heading(0, "Chapter 4: GRO", 1)])
    assert out[0].lstrip().startswith("# Chapter 4: GRO")


def test_strip_running_lines():
    pages = [f"Booklet Header\nContent {i}\nPage {i}" for i in range(1, 6)]
    assert strip_running_lines(pages) == [f"Content {i}" for i in range(1, 6)] or all(
        "Page" not in p and "Booklet Header" not in p for p in strip_running_lines(pages))
