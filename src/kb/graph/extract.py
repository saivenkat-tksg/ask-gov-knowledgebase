"""LLM extraction of entities and relations from document text, with validation against that text.

The model gets the text (a whole document, or a chapter-sized part of a long one; see
kb.graph.document), optional context (document name, part, outline) and the allowed vocabulary
(kb.graph.schema), and must return JSON. Every entity and relation carries a `quote`: the words in
the text that state it. Validation keeps only items whose quote is actually in the text, whose
types are in the vocabulary, and whose relation runs between allowed types, so a model that
invents a fact, a type or a direction cannot get it into the graph.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from kb.graph.schema import ENTITY_TYPES, RELATION_DESCRIPTIONS, RELATION_TYPES

log = logging.getLogger(__name__)

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "’": "'", "‘": "'",
                            "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


class ExtractionError(RuntimeError):
    """The LLM call failed or returned something that isn't the expected JSON."""


@dataclass(frozen=True)
class Entity:
    type: str
    name: str
    quote: str


@dataclass(frozen=True)
class Relation:
    type: str
    source: str  # entity name, as in Entity.name
    target: str
    quote: str


@dataclass
class Extraction:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    dropped: int = 0  # items removed by validation


def _normal_text(text: str) -> str:
    """Lowercase, ligatures and quotes unified, markdown table pipes and all whitespace collapsed:
    a quote matches the chunk even when the PDF broke a line inside it."""
    text = text.translate(_LIGATURES).lower().replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def build_messages(chunk_text: str, context: str | None = None) -> list[dict]:
    entity_lines = "\n".join(f"- {name}: {desc}" for name, desc in ENTITY_TYPES.items())
    relation_lines = "\n".join(
        f"- {name} ({'/'.join(sorted(src))} -> {'/'.join(sorted(dst))}): {RELATION_DESCRIPTIONS[name]}"
        for name, (src, dst) in RELATION_TYPES.items()
    )
    system = f"""You extract facts from Guyana government documents into a knowledge graph.

Entity types (use only these):
{entity_lines}

Relation types (use only these, only in the direction shown):
{relation_lines}

Rules:
- Extract only what the text states. Do not infer, guess or use outside knowledge.
- Every entity and relation needs "quote": words copied exactly from the text that state it.
- Use the fullest name the text gives (e.g. "General Register Office" rather than "GRO" when both appear),
  and the same name every time the same thing appears.
- Lines starting with # are headings; lines starting with | are table rows (read each row with its
  column headers). A line in [brackets] names the section the text belongs to.
- Extract every qualifying fact in the text, from every section and table, not only the main ones.
- Quote only from the text, never from the context or outline.
- A relation's "source" and "target" must be names of entities in your "entities" list.
- If nothing qualifies, return empty lists.

Return only JSON: {{"entities": [{{"type": "...", "name": "...", "quote": "..."}}],
"relations": [{{"type": "...", "source": "...", "target": "...", "quote": "..."}}]}}"""
    user = f"{context}\n\n--- TEXT ---\n{chunk_text}" if context else chunk_text
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate(chunk_text: str, data: object) -> Extraction:
    """Keep only well-formed items, of allowed types, whose quote is found in the chunk."""
    if not isinstance(data, dict):
        raise ExtractionError("LLM output is not a JSON object")
    haystack = _normal_text(chunk_text)
    out = Extraction()

    def quoted(item: dict) -> bool:
        quote = item.get("quote")
        return isinstance(quote, str) and len(quote.strip()) >= 3 and _normal_text(quote) in haystack

    names: dict[str, str] = {}  # entity name (lowercase) -> type
    for item in data.get("entities") or []:
        if not (isinstance(item, dict) and item.get("type") in ENTITY_TYPES
                and isinstance(item.get("name"), str) and item["name"].strip() and quoted(item)):
            out.dropped += 1
            continue
        name = re.sub(r"\s+", " ", item["name"]).strip()[:200]
        if name.lower() in names:
            continue
        names[name.lower()] = item["type"]
        out.entities.append(Entity(item["type"], name, item["quote"].strip()[:500]))

    for item in data.get("relations") or []:
        ok = isinstance(item, dict) and item.get("type") in RELATION_TYPES and quoted(item)
        if ok:
            source_type = names.get(str(item.get("source", "")).strip().lower())
            target_type = names.get(str(item.get("target", "")).strip().lower())
            allowed_src, allowed_dst = RELATION_TYPES[item["type"]]
            ok = source_type in allowed_src and target_type in allowed_dst
        if not ok:
            out.dropped += 1
            continue
        source = next(e.name for e in out.entities if e.name.lower() == item["source"].strip().lower())
        target = next(e.name for e in out.entities if e.name.lower() == item["target"].strip().lower())
        out.relations.append(Relation(item["type"], source, target, item["quote"].strip()[:500]))
    return out


class LLMExtractor:
    """Calls the extraction model through LiteLLM (any provider it supports)."""

    def __init__(self, model: str, api_key: str | None = None, timeout: float = 300.0,
                 max_output_tokens: int | None = None):
        self.model = model
        self._kwargs: dict = {"temperature": 0, "num_retries": 2, "timeout": timeout}
        if api_key:
            self._kwargs["api_key"] = api_key
        if max_output_tokens:
            self._kwargs["max_tokens"] = max_output_tokens

    def extract(self, chunk_text: str, context: str | None = None) -> Extraction:
        import litellm

        try:
            response = litellm.completion(model=self.model, messages=build_messages(chunk_text, context),
                                          response_format={"type": "json_object"}, **self._kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""
        except Exception as exc:
            raise ExtractionError(f"extraction call failed ({self.model}): {exc}") from exc
        if getattr(choice, "finish_reason", None) == "length":
            raise ExtractionError("extraction reply truncated at the output limit; the text needs splitting")
        try:
            data = json.loads(_strip_fences(content))
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"extraction returned invalid JSON: {content[:200]!r}") from exc
        result = validate(chunk_text, data)
        if result.dropped:
            log.info("dropped %d unsupported or unquoted items", result.dropped)
        return result


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return text
