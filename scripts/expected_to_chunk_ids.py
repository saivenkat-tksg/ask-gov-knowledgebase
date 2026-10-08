"""Rewrite a questions file's `expected` items as chunk ids, e.g. {"chunk_id": "cash_grant_chunk_3"}.

Each item with a `contains` phrase is looked up in the current chunks of its `filename`
(the schema in KB_DB_SCHEMA). One chunk holding the phrase becomes {"chunk_id": "..."};
several become a list (any of them counts) and are reported so you can trim them by hand.
With --single, the one sharing the most words with the question and reference is kept instead.
Items that are already chunk ids are kept. Chunk ids follow chunk_index, so re-run this
after re-chunking. Comment lines and blank lines are preserved.

    python scripts/expected_to_chunk_ids.py eval/questions.jsonl            # dry run
    python scripts/expected_to_chunk_ids.py eval/questions.jsonl --write
    python scripts/expected_to_chunk_ids.py eval/questions_contains.jsonl --single > eval/questions.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

from kb.deps import get_pool
from kb.evaluate import chunk_label

_CHUNKS = """SELECT d.filename, c.chunk_index, c.content
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE d.is_current ORDER BY d.filename, c.chunk_index"""


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$]+", text.lower()) if len(w) > 3}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions")
    parser.add_argument("--write", action="store_true", help="overwrite the file (default: print only)")
    parser.add_argument("--single", action="store_true",
                        help="one chunk per item: the candidate sharing most words with question + reference")
    args = parser.parse_args()

    chunks: dict[str, list[tuple[int, str]]] = {}
    with get_pool().connection() as conn:
        for filename, index, content in conn.execute(_CHUNKS):
            chunks.setdefault(filename, []).append((index, content.lower()))

    path = Path(args.questions)
    out, problems = [], 0
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue
        raw = json.loads(line)
        expected = []
        for item in raw.get("expected", []):
            if "chunk_id" in item:
                expected.append({"chunk_id": item["chunk_id"]})
                continue
            filename, phrase = item.get("filename"), item.get("contains", "").lower()
            hits = [i for i, text in chunks.get(filename, []) if phrase in text]
            if not hits:
                problems += 1
                print(f"line {n} {raw.get('id')}: no chunk of {filename!r} contains {item.get('contains')!r}",
                      file=sys.stderr)
                expected.append(item)  # left as-is so nothing is lost
                continue
            if len(hits) > 1 and args.single:
                wanted = _words(raw["question"] + " " + (raw.get("reference") or ""))
                text_of = dict(chunks[filename])
                best = max(hits, key=lambda i: (len(wanted & _words(text_of[i])), -i))
                print(f"line {n} {raw.get('id')}: phrase in chunks {hits}, kept {best}", file=sys.stderr)
                hits = [best]
            elif len(hits) > 1:
                print(f"line {n} {raw.get('id')}: phrase in chunks {hits}; review the list", file=sys.stderr)
            ids = [chunk_label(filename, i) for i in hits]
            expected.append({"chunk_id": ids[0] if len(ids) == 1 else ids})
        if "expected" in raw:
            raw["expected"] = expected
        out.append(json.dumps(raw, ensure_ascii=False))

    text = "\n".join(out) + "\n"
    if args.write:
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}; {problems} item(s) unresolved", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
