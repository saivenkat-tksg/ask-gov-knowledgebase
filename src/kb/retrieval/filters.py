"""Compile a Mongo-style metadata filter into a parameterised SQL WHERE clause
over the `c.metadata` JSONB column.

    {"department": "health"}                       equality (uses GIN index)
    {"year": {"$gte": 2020, "$lt": 2025}}          range (numbers or ISO date strings)
    {"file_type": {"$in": ["pdf", "docx"]}}        membership
    {"tags": {"$contains": "covid"}}               array contains value(s)
    {"author": {"$exists": true}}                  key present
    {"$or": [{...}, {...}]}, {"$and": [...]}, {"$not": {...}}
    {"source.agency": "CDC"}                       dotted path into nested objects

Operators: $eq $ne $in $nin $contains $gt $gte $lt $lte $exists.
Keys are passed as bind parameters, never interpolated.
"""

import re
from typing import Any

from psycopg import sql
from psycopg.types.json import Jsonb

_SEGMENT = re.compile(r"^[A-Za-z0-9_\-]+$")
_COMPARE = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
_CONTAINS = sql.SQL("c.metadata @> %s")


class FilterError(ValueError):
    pass


def compile_filters(filters: dict[str, Any] | None) -> tuple[sql.Composable, list[Any]]:
    if not filters:
        return sql.SQL("TRUE"), []
    params: list[Any] = []
    return _node(filters, params), params


def _node(node: Any, params: list[Any]) -> sql.Composable:
    if not isinstance(node, dict) or not node:
        raise FilterError("Each filter must be a non-empty JSON object")
    parts: list[sql.Composable] = []
    for key, value in node.items():
        if key in ("$and", "$or"):
            if not isinstance(value, list) or not value:
                raise FilterError(f"{key} expects a non-empty list of filters")
            children = [_node(v, params) for v in value]
            joiner = sql.SQL(" AND " if key == "$and" else " OR ")
            parts.append(sql.SQL("({})").format(joiner.join(children)))
        elif key == "$not":
            parts.append(sql.SQL("NOT {}").format(_node(value, params)))
        elif key.startswith("$"):
            raise FilterError(f"Unknown top-level operator '{key}'")
        else:
            parts.append(_field(key, value, params))
    return _and(parts)


def _and(parts: list[sql.Composable]) -> sql.Composable:
    if len(parts) == 1:
        return parts[0]
    return sql.SQL("({})").format(sql.SQL(" AND ").join(parts))


def _path(key: str) -> list[str]:
    segments = key.split(".")
    if not all(_SEGMENT.match(s) for s in segments):
        raise FilterError(f"Invalid metadata key '{key}'")
    return segments


def _nest(path: list[str], value: Any) -> dict:
    for segment in reversed(path):
        value = {segment: value}
    return value


def _field(key: str, condition: Any, params: list[Any]) -> sql.Composable:
    path = _path(key)
    if isinstance(condition, dict) and condition:
        ops = [k.startswith("$") for k in condition]
        if all(ops):
            return _and([_op(path, op, v, params) for op, v in condition.items()])
        if any(ops):
            raise FilterError(f"Cannot mix operators and plain keys under '{key}'")
    return _op(path, "$eq", condition, params)


def _op(path: list[str], op: str, value: Any, params: list[Any]) -> sql.Composable:
    key = ".".join(path)
    if op == "$eq":
        params.append(Jsonb(_nest(path, value)))
        return _CONTAINS
    if op == "$ne":
        params.append(Jsonb(_nest(path, value)))
        return sql.SQL("NOT ({})").format(_CONTAINS)
    if op in ("$in", "$nin"):
        if not isinstance(value, list) or not value:
            raise FilterError(f"{op} on '{key}' expects a non-empty list")
        for v in value:
            params.append(Jsonb(_nest(path, v)))
        clause = sql.SQL("({})").format(sql.SQL(" OR ").join([_CONTAINS] * len(value)))
        return clause if op == "$in" else sql.SQL("NOT {}").format(clause)
    if op == "$contains":
        values = value if isinstance(value, list) else [value]
        params.append(Jsonb(_nest(path, values)))
        return _CONTAINS
    if op in _COMPARE:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise FilterError(f"{op} on '{key}' expects a number or string")
        json_type = "string" if isinstance(value, str) else "number"
        params.extend([path, json_type, path, Jsonb(value)])
        # Type guard avoids jsonb's cross-type ordering (e.g. "2021" vs 2020).
        return sql.SQL(
            "(jsonb_typeof(c.metadata #> %s::text[]) = %s AND (c.metadata #> %s::text[]) {} %s::jsonb)"
        ).format(sql.SQL(_COMPARE[op]))
    if op == "$exists":
        if not isinstance(value, bool):
            raise FilterError(f"$exists on '{key}' expects true or false")
        params.append(path)
        return sql.SQL("(c.metadata #> %s::text[]) IS {} NULL").format(
            sql.SQL("NOT" if value else "")
        )
    raise FilterError(f"Unknown operator '{op}' on '{key}'")
