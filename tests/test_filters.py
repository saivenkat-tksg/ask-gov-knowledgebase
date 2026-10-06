import pytest

from kb.filters import FilterError, compile_filters


def sql_of(filters):
    clause, params = compile_filters(filters)
    return clause.as_string(None), params


def test_no_filters():
    assert sql_of(None) == ("TRUE", [])


def test_equality_uses_containment():
    s, params = sql_of({"department": "health"})
    assert s == "c.metadata @> %s"
    assert params[0].obj == {"department": "health"}


def test_nested_path_and_range():
    s, params = sql_of({"source.year": {"$gte": 2020, "$lt": 2025}})
    assert s.count("jsonb_typeof") == 2
    assert params[0] == ["source", "year"] and params[1] == "number"
    assert s.count("%s") == len(params)


def test_in_or_not():
    s, params = sql_of({"$or": [{"file_type": {"$in": ["pdf", "docx"]}}, {"$not": {"draft": True}}]})
    assert " OR " in s and "NOT" in s
    assert s.count("%s") == len(params) == 3


def test_contains_wraps_scalar_in_list():
    _, params = sql_of({"tags": {"$contains": "covid"}})
    assert params[0].obj == {"tags": ["covid"]}


def test_exists():
    s, params = sql_of({"author": {"$exists": False}})
    assert "IS  NULL" in s and params == [["author"]]


@pytest.mark.parametrize(
    "bad",
    [
        {"a; drop table x": 1},
        {"$foo": 1},
        {"a": {"$in": []}},
        {"a": {"$gt": True}},
        {"a": {"$bogus": 1}},
        {"a": {"$eq": 1, "plain": 2}},
        {"$or": []},
    ],
)
def test_invalid_filters(bad):
    with pytest.raises(FilterError):
        compile_filters(bad)
