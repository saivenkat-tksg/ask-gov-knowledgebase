import io

import pytest

from kb.config import Settings
from kb.loaders import load_pdf
from kb.ocr import PdfOcr, _layout


def test_needs_ocr_modes():
    auto = PdfOcr(Settings(ocr_mode="auto", ocr_min_chars=20))
    assert auto.needs_ocr("") and auto.needs_ocr("  12 \n ")
    assert not auto.needs_ocr("This page has a real text layer.")
    assert PdfOcr(Settings(ocr_mode="force")).needs_ocr("This page has a real text layer.")
    assert not PdfOcr(Settings(ocr_mode="off")).needs_ocr("")


def test_layout_groups_lines_and_paragraphs():
    boxes = [  # (top, bottom, left, text), deliberately unordered
        (100, 130, 400, "Office"),
        (102, 131, 100, "Register"),
        (140, 170, 100, "Georgetown"),
        (300, 330, 100, "Fees"),
    ]
    assert _layout(boxes) == "Register Office\nGeorgetown\n\nFees"


class FakeOcr(PdfOcr):
    def __init__(self):
        super().__init__(Settings(ocr_mode="auto"))
        self.calls = []

    def read_pages(self, data, page_indexes):
        self.calls.append(page_indexes)
        return {i: f"scanned text {i}" for i in page_indexes}


def test_only_pages_without_text_are_ocred():
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(100, 100)
    w.add_blank_page(100, 100)
    buf = io.BytesIO()
    w.write(buf)
    ocr = FakeOcr()
    doc = load_pdf(buf.getvalue(), ocr=ocr)
    assert ocr.calls == [[0, 1]]
    assert [s.text for s in doc.sections] == ["scanned text 0", "scanned text 1"]
    assert all(s.metadata["ocr"] for s in doc.sections)
    assert [s.metadata["page"] for s in doc.sections] == [1, 2]


def _scanned_pdf(lines: list[str]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (1700, 2200), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default(size=40)
    for n, line in enumerate(lines):
        draw.text((150, 150 + 70 * n), line, fill=0, font=font)
    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=200)
    return buf.getvalue()


def test_rapidocr_reads_scanned_page():
    pytest.importorskip("rapidocr")
    pytest.importorskip("pypdfium2")
    data = _scanned_pdf(["Application for a Birth Certificate", "The fee is 1000 dollars."])
    doc = load_pdf(data, ocr=PdfOcr(Settings(ocr_engine="rapidocr")))
    text = doc.sections[0].text
    assert doc.sections[0].metadata["ocr"] is True
    assert "Birth Certificate" in text and "1000" in text
