"""OCR for PDF pages that have no text layer (scanned or photographed documents).

Pages are rendered with pdfium and read by one of two engines:
- rapidocr: ONNX models installed with pip, no system binary needed (default).
- tesseract: needs the Tesseract binary on PATH plus `pip install pytesseract`.
"""

import logging
import threading
from functools import lru_cache
from statistics import median
from typing import Protocol

from .config import Settings


log = logging.getLogger(__name__)


class OcrUnavailable(ValueError):
    pass


class Engine(Protocol):
    def read(self, image) -> str: ...


class RapidOcrEngine:
    def __init__(self, min_confidence: float):
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise OcrUnavailable("OCR engine 'rapidocr' is not installed: pip install rapidocr onnxruntime") from exc
        self._ocr = RapidOCR(params={"Global.log_level": "warning"})  # hide model-loading chatter
        self._min_confidence = min_confidence

    def read(self, image) -> str:
        import numpy as np

        result = self._ocr(np.asarray(image.convert("RGB")))
        if result.boxes is None:
            return ""
        boxes = [
            (min(p[1] for p in box), max(p[1] for p in box), min(p[0] for p in box), text)
            for box, text, score in zip(result.boxes.tolist(), result.txts, result.scores)
            if score >= self._min_confidence and text.strip()
        ]
        return _layout(boxes)


class TesseractEngine:
    def __init__(self, lang: str):
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
        except ImportError as exc:
            raise OcrUnavailable("OCR engine 'tesseract' needs: pip install pytesseract") from exc
        except Exception as exc:
            raise OcrUnavailable("OCR engine 'tesseract' needs the Tesseract binary on PATH") from exc
        self._tesseract = pytesseract
        self._lang = lang

    def read(self, image) -> str:
        return self._tesseract.image_to_string(image, lang=self._lang)


def _layout(boxes: list[tuple[float, float, float, str]]) -> str:
    """Join (top, bottom, left, text) boxes into lines, with blank lines between paragraphs."""
    if not boxes:
        return ""
    boxes.sort(key=lambda b: (b[0], b[2]))
    lines: list[list] = []  # [top, bottom, [(left, text), ...]]
    for top, bottom, left, text in boxes:
        if lines:
            line = lines[-1]
            # Same line when the box's vertical centre falls inside the current line.
            if line[0] <= (top + bottom) / 2 <= line[1]:
                line[0], line[1] = min(line[0], top), max(line[1], bottom)
                line[2].append((left, text))
                continue
        lines.append([top, bottom, [(left, text)]])

    height = median(line[1] - line[0] for line in lines)
    out: list[str] = []
    for i, (top, _, words) in enumerate(lines):
        if i and top - lines[i - 1][1] > height:
            out.append("")
        out.append(" ".join(text for _, text in sorted(words)))
    return "\n".join(out)


class PdfOcr:
    def __init__(self, settings: Settings):
        self.mode = settings.ocr_mode
        self.min_chars = settings.ocr_min_chars
        self.dpi = settings.ocr_dpi
        self._settings = settings
        self._engine: Engine | None = None
        self._lock = threading.Lock()

    def needs_ocr(self, extracted_text: str) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "force":
            return True
        return sum(c.isalnum() for c in extracted_text) < self.min_chars

    def _get_engine(self) -> Engine:
        if self._engine is None:
            s = self._settings
            self._engine = (
                TesseractEngine(s.ocr_lang) if s.ocr_engine == "tesseract" else RapidOcrEngine(s.ocr_min_confidence)
            )
        return self._engine

    def read_pages(self, data: bytes, page_indexes: list[int]) -> dict[int, str]:
        """OCR the given 0-based pages of a PDF; returns {index: text}."""
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise OcrUnavailable("PDF OCR needs: pip install pypdfium2 Pillow") from exc

        # One page at a time across threads: the engines already use every core, and this
        # keeps memory bounded when several scanned uploads arrive together.
        with self._lock:
            engine = self._get_engine()
            pdf = pdfium.PdfDocument(data)
            try:
                texts = {}
                log.info("OCR: %d page(s) without a text layer", len(page_indexes))
                for n, i in enumerate(page_indexes, start=1):
                    page = pdf[i]
                    image = page.render(scale=self.dpi / 72).to_pil()
                    texts[i] = engine.read(image).strip()
                    page.close()
                    log.info("OCR: page %d done (%d/%d)", i + 1, n, len(page_indexes))
                return texts
            finally:
                pdf.close()


@lru_cache
def get_pdf_ocr() -> PdfOcr:
    from .config import get_settings

    return PdfOcr(get_settings())
