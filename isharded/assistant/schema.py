"""The part of JSON Schema the assistant needs, so it needs no extra package.

Used to check a tool's arguments before anyone is asked about them, and by a
task's mechanical checks. Keywords it does not know are ignored, as the standard
says they should be. ``pattern`` is left to the caller: this runs in-process, and
a pattern written by a model could hang it.
"""
from __future__ import annotations

from typing import Any

TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list, tuple),
    "null": (type(None),),
}


def is_type(value: Any, name: str) -> bool:
    if name not in TYPES:
        return True
    if isinstance(value, bool):
        return name == "boolean"
    if name == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, TYPES[name])


def errors(value: Any, schema: Any, where: str = "value") -> list[str]:
    """Everything wrong with value, as sentences a person can act on.

    A schema can come from a stranger's server. One that cannot be used fails closed,
    since a check that crashes must never read as a check that passed.
    """
    try:
        return _errors(value, schema, where)
    except (TypeError, ValueError, KeyError, AttributeError, RecursionError):
        return [f"{where} has a schema that cannot be used."]


def _errors(value: Any, schema: Any, where: str) -> list[str]:
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{where} is not allowed."]
    if not isinstance(schema, dict):
        return [f"{where} has a schema that is not a JSON object."]
    found: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        if not any(is_type(value, name) for name in names):
            return [f"{where} should be {' or '.join(map(str, names))}."]
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{where} should be one of {schema['enum']}.")
    if "const" in schema and value != schema["const"]:
        found.append(f"{where} should be {schema['const']!r}.")
    found += _combined(value, schema, where)
    if isinstance(value, str):
        found += _sized(len(value), schema, "minLength", "maxLength", where, "characters")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        found += _bounds(value, schema, where)
    if isinstance(value, (list, tuple)):
        found += _sized(len(value), schema, "minItems", "maxItems", where, "items")
        if "items" in schema:
            for index, item in enumerate(value):
                found += _errors(item, schema["items"], f"{where}[{index}]")
    if isinstance(value, dict):
        found += _object(value, schema, where)
    return found


def _combined(value: Any, schema: dict[str, Any], where: str) -> list[str]:
    found: list[str] = []
    for part in schema.get("allOf", []):
        found += _errors(value, part, where)
    if "anyOf" in schema and not any(not _errors(value, part, where) for part in schema["anyOf"]):
        found.append(f"{where} matches none of the allowed forms.")
    if "oneOf" in schema and sum(not _errors(value, part, where) for part in schema["oneOf"]) != 1:
        found.append(f"{where} should match exactly one of the allowed forms.")
    if "not" in schema and not _errors(value, schema["not"], where):
        found.append(f"{where} is a form that is not allowed.")
    return found


def _sized(size: int, schema: dict[str, Any], low: str, high: str, where: str, unit: str) -> list[str]:
    found = []
    if low in schema and size < schema[low]:
        found.append(f"{where} needs at least {schema[low]} {unit}.")
    if high in schema and size > schema[high]:
        found.append(f"{where} allows at most {schema[high]} {unit}.")
    return found


def _bounds(value: float, schema: dict[str, Any], where: str) -> list[str]:
    found = []
    if "minimum" in schema and value < schema["minimum"]:
        found.append(f"{where} should be at least {schema['minimum']}.")
    if "maximum" in schema and value > schema["maximum"]:
        found.append(f"{where} should be at most {schema['maximum']}.")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        found.append(f"{where} should be above {schema['exclusiveMinimum']}.")
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        found.append(f"{where} should be below {schema['exclusiveMaximum']}.")
    return found


def _object(value: dict[str, Any], schema: dict[str, Any], where: str) -> list[str]:
    found = [f"{where} is missing {name!r}." for name in schema.get("required", []) if name not in value]
    properties = schema.get("properties", {})
    extra = schema.get("additionalProperties", True)
    for name, item in value.items():
        if name in properties:
            found += _errors(item, properties[name], f"{where}.{name}")
        elif extra is False:
            found.append(f"{where} has {name!r}, which is not allowed.")
        elif isinstance(extra, dict):
            found += _errors(item, extra, f"{where}.{name}")
    return found
