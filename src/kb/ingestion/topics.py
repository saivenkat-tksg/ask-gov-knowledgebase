"""Topic tagging at ingest: an LLM names each section's subjects, stored as `topics` on its chunks.

Sections are runs of consecutive chunks with the same heading path (structure chunking) or page
(recursive chunking). The tagger sees the knowledge base's existing topics and reuses them where they
fit, so "Leave management" in one document and in another get the same name and can be filtered on:
{"topics": {"$contains": "Leave management"}}.

Tagging is an enrichment: if the LLM call fails, chunks are stored without topics and ingest goes on.
"""

import json
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

SECTION_CHARS = 1500  # text of each section shown to the LLM
BATCH_SECTIONS = 25  # sections per LLM call
MAX_KNOWN = 200  # existing topics offered for reuse

_PROMPT = """You tag sections of government process documents with their subjects, for search filtering.

For each section give 1 to {n} short topic names (2-4 words, Title case, e.g. "Leave management", \
"Birth registration", "Firearm licensing"). Name the service, process or subject the section is about, \
not its format ("Overview", "Table") or the agency alone.

Reuse an existing topic whenever one fits, spelled exactly as given:
{known}

Return only JSON: {{"sections": [{{"id": 0, "topics": ["..."]}}, ...]}} with one entry per section id."""


@dataclass
class TopicSection:
    title: str  # heading path ("A > B") or page, for the LLM
    text: str
    rows: list[int] = field(default_factory=list)  # indexes into the chunk rows


def group_sections(rows: list[dict]) -> list[TopicSection]:
    """Consecutive chunk rows with the same heading path (or page, without one) form a section."""
    sections: list[TopicSection] = []
    last_key = None
    for i, row in enumerate(rows):
        path = row["extra"].get("section_path")
        key = tuple(path) if path else ("page", row.get("page"))
        if sections and key == last_key:
            sections[-1].rows.append(i)
            if len(sections[-1].text) < SECTION_CHARS:
                sections[-1].text += "\n" + row["text"]
            continue
        title = " > ".join(path) if path else (f"page {row['page']}" if row.get("page") is not None else "text")
        sections.append(TopicSection(title, row["text"], [i]))
        last_key = key
    return sections


def known_topics(conn, limit: int = MAX_KNOWN) -> list[str]:
    """Topics already on current chunks, most used first."""
    rows = conn.execute(
        "SELECT t, count(*) FROM chunks c JOIN documents d ON d.id = c.document_id, "
        "jsonb_array_elements_text(CASE WHEN jsonb_typeof(c.metadata->'topics') = 'array' "
        "THEN c.metadata->'topics' ELSE '[]'::jsonb END) AS t "
        "WHERE d.is_current GROUP BY t ORDER BY count(*) DESC, t LIMIT %s",
        [limit],
    ).fetchall()
    return [r[0] for r in rows]


def _clean(topics, limit: int) -> list[str]:
    out: list[str] = []
    for t in topics if isinstance(topics, list) else []:
        if isinstance(t, str) and t.strip():
            t = re.sub(r"\s+", " ", t).strip()[:60]
            if t.lower() not in {o.lower() for o in out}:
                out.append(t)
    return out[:limit]


class TopicTagger:
    def __init__(self, model: str, api_key: str | None = None, max_per_section: int = 3, timeout: float = 60.0):
        self.model = model
        self.max_per_section = max_per_section
        self._kwargs: dict = {"temperature": 0, "num_retries": 2, "timeout": timeout}
        if api_key:
            self._kwargs["api_key"] = api_key

    def tag(self, sections: list[TopicSection], known: list[str]) -> list[list[str]]:
        """Topics for each section, in order; [] for a section the LLM gave none (or a failed call)."""
        known = list(known)
        out: list[list[str]] = []
        for start in range(0, len(sections), BATCH_SECTIONS):
            batch = sections[start:start + BATCH_SECTIONS]
            tagged = self._tag_batch(batch, known)
            out.extend(tagged)
            # Topics named in this batch can be reused by the next one.
            seen = {k.lower() for k in known}
            known.extend(t for topics in tagged for t in topics if t.lower() not in seen and not seen.add(t.lower()))
        return out

    def _tag_batch(self, batch: list[TopicSection], known: list[str]) -> list[list[str]]:
        import litellm

        system = _PROMPT.format(n=self.max_per_section,
                                known="\n".join(f"- {k}" for k in known[:MAX_KNOWN]) or "(none yet)")
        user = "\n\n".join(f"### Section {i}: {s.title}\n{s.text[:SECTION_CHARS]}" for i, s in enumerate(batch))
        try:
            response = litellm.completion(model=self.model, response_format={"type": "json_object"},
                                          messages=[{"role": "system", "content": system},
                                                    {"role": "user", "content": user}], **self._kwargs)
            data = json.loads(response.choices[0].message.content or "{}")
        except Exception as exc:  # tagging is optional: never fail an ingest over it
            log.warning("Topic tagging failed, storing chunks without topics: %s", exc)
            return [[] for _ in batch]
        by_id = {}
        for item in data.get("sections", []) if isinstance(data, dict) else []:
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                by_id[item["id"]] = _clean(item.get("topics"), self.max_per_section)
        return [by_id.get(i, []) for i in range(len(batch))]
