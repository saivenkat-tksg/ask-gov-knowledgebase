import json

import pytest

from kb.graph import extract
from kb.graph.extract import ExtractionError, LLMExtractor, build_messages, validate
from kb.graph.resolve import Resolver, normalise
from kb.graph.schema import ENTITY_TYPES, RELATION_TYPES

CHUNK = ("[Chapter 4: GENERAL REGISTRAR OFFICE > 1. Births and Deaths Certification Process]\n"
         "Registrations for Births, Marriages and Deaths are free of charge. All registrars send the\n"
         "original registration forms to the General Register Ofﬁce for retention.\n"
         "| 1. | Data Entry Clerk types the file | Data Entry Clerk |")


def test_validate_keeps_quoted_items_and_drops_the_rest():
    data = {
        "entities": [
            {"type": "Service", "name": "Birth registration", "quote": "Registrations for Births"},
            {"type": "Fee", "name": "Free", "quote": "free of charge"},
            {"type": "Office", "name": "General Register Office", "quote": "the General Register Office"},  # ligature in text
            {"type": "Role", "name": "Data Entry Clerk", "quote": "Data Entry Clerk types the file"},  # across a table pipe
            {"type": "Fee", "name": "GY$5,000", "quote": "costs GY$5,000"},  # not in the text: invented
            {"type": "Person", "name": "Alice", "quote": "registrars"},  # type not allowed
        ],
        "relations": [
            {"type": "COSTS", "source": "Birth registration", "target": "Free", "quote": "are free of charge"},
            {"type": "HANDLED_BY", "source": "Birth registration", "target": "General Register Office",
             "quote": "send the original registration forms to the General Register Office"},  # spans a line break
            {"type": "COSTS", "source": "Free", "target": "Birth registration", "quote": "free of charge"},  # wrong direction
            {"type": "COSTS", "source": "Birth registration", "target": "GY$5,000", "quote": "free of charge"},  # dropped entity
            {"type": "OWNS", "source": "Birth registration", "target": "Free", "quote": "free of charge"},  # unknown type
        ],
    }
    out = validate(CHUNK, data)
    assert [e.name for e in out.entities] == ["Birth registration", "Free", "General Register Office", "Data Entry Clerk"]
    assert [(r.source, r.type, r.target) for r in out.relations] == [
        ("Birth registration", "COSTS", "Free"),
        ("Birth registration", "HANDLED_BY", "General Register Office"),
    ]
    assert out.dropped == 5


def test_validate_rejects_non_objects():
    with pytest.raises(ExtractionError):
        validate(CHUNK, ["not", "an", "object"])
    assert validate(CHUNK, {}).entities == []


def test_prompt_lists_the_whole_vocabulary():
    system = build_messages(CHUNK)[0]["content"]
    assert all(name in system for name in ENTITY_TYPES)
    assert all(name in system for name in RELATION_TYPES)
    assert build_messages(CHUNK)[1]["content"] == CHUNK


def test_extractor_parses_and_validates(monkeypatch):
    reply = {"entities": [{"type": "Fee", "name": "Free", "quote": "free of charge"}], "relations": []}

    class Message:
        content = "```json\n" + json.dumps(reply) + "\n```"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    calls = []
    import litellm

    monkeypatch.setattr(litellm, "completion", lambda **kw: calls.append(kw) or Response())
    out = LLMExtractor("gpt-4o-mini", api_key="k").extract(CHUNK)
    assert [e.name for e in out.entities] == ["Free"]
    assert calls[0]["temperature"] == 0 and calls[0]["response_format"] == {"type": "json_object"}

    monkeypatch.setattr(litellm, "completion", lambda **kw: (_ for _ in ()).throw(TimeoutError("slow")))
    with pytest.raises(ExtractionError):
        LLMExtractor("gpt-4o-mini").extract(CHUNK)


def test_resolver_merges_names_of_one_thing():
    r = Resolver()
    assert normalise("  The General Register Ofﬁce. ") == "general register office"
    keys = {r.key("kb_large", "Office", n) for n in ("GRO", "General Register Office", "General Registrar Office",
                                                       "the general register office")}
    assert keys == {"kb_large|Office|general register office"}
    # Same name, different type or knowledge base: different nodes.
    assert r.key("kb_large", "Region", "Region 6") != r.key("kb_large", "Office", "Region 6")
    assert r.key("kb_eval", "Office", "GRO") != r.key("kb_large", "Office", "GRO")


def test_resolver_aliases_file(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({"ISS": "Immigration Support Services Unit"}), encoding="utf-8")
    assert Resolver(str(path)).canonical("iss") == "immigration support services unit"


def test_vocabulary_is_safe_to_put_into_cypher():
    # Labels and relationship types are interpolated into query text, so they must be plain identifiers.
    for name in [*ENTITY_TYPES, *RELATION_TYPES]:
        assert name.replace("_", "").isalnum()
    for src, dst in RELATION_TYPES.values():
        assert src <= set(ENTITY_TYPES) and dst <= set(ENTITY_TYPES)


def test_normal_text_matches_across_pipes_and_breaks():
    assert extract._normal_text("A |  b\nC") == "a b c"
