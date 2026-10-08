import threading
import uuid
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kb.documents import stale_reasons
from kb.drift import is_url, read_source
from kb.schemas import DocumentInfo

TODAY = date(2026, 10, 7)


def doc(**overrides) -> DocumentInfo:
    fields = dict(id=uuid.uuid4(), filename="a.pdf", file_type="pdf", size_bytes=1, num_chunks=1, metadata={},
                  created_at=datetime(2026, 9, 1, tzinfo=timezone.utc), doc_key="a.pdf", version=1, is_current=True)
    return DocumentInfo(**{**fields, **overrides})


def test_fresh_document_has_no_reasons():
    assert stale_reasons(doc(review_by=date(2027, 1, 1), source_status="ok"), 365, TODAY) == []


def test_stale_reasons():
    assert "review date 2026-10-01 has passed" in stale_reasons(doc(review_by=date(2026, 10, 1)), 365, TODAY)
    old = doc(created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert "over 365 days ago" in stale_reasons(old, 365, TODAY)[0]
    assert stale_reasons(old, 0, TODAY) == []  # age check disabled
    # A future review date overrides the age check: someone has vouched for the content.
    assert stale_reasons(doc(created_at=old.created_at, review_by=date(2027, 1, 1)), 365, TODAY) == []
    assert "expired on 2026-10-06" in stale_reasons(doc(valid_until=date(2026, 10, 6)), 365, TODAY)[0]
    assert "source no longer exists" in stale_reasons(doc(source="x.pdf", source_status="missing"), 365, TODAY)[0]
    assert "could not be checked" in stale_reasons(doc(source="x.pdf", source_status="error"), 365, TODAY)[0]


def test_read_local_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"v1")
    assert read_source(str(path)).data == b"v1"
    assert read_source(str(tmp_path / "gone.txt")).status == "missing"
    assert not is_url(str(path)) and is_url("HTTPS://example.gov/a.pdf")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/gone.pdf":
            self.send_error(404)
        elif self.path == "/broken.pdf":
            self.send_error(500)
        elif self.headers.get("If-None-Match") == '"v1"':
            self.send_response(304)
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("ETag", '"v1"')
            self.end_headers()
            self.wfile.write(b"content")

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_read_url(server):
    first = read_source(f"{server}/a.pdf")
    assert first.status == "ok" and first.data == b"content" and first.etag == '"v1"'
    again = read_source(f"{server}/a.pdf", etag=first.etag)
    assert again.status == "not_modified" and again.data is None
    assert read_source(f"{server}/gone.pdf").status == "missing"
    assert read_source(f"{server}/broken.pdf").status == "error"
    assert read_source("http://127.0.0.1:1/a.pdf", timeout=2).status == "error"
