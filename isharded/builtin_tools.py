"""Tools every mesh has, with no client bringing any.

A model that can call tools is offered these when a chat asks for them and sends no tools of
its own. The node that answers the chat runs each call itself and gives the model the result,
so the reply that reaches the client is the finished answer.

Each tool here is safe to run for anyone who may chat: it reads a clock, does arithmetic, or
reports what the mesh already shows on its status page. None reads a file, opens a
connection or runs a program. A tool that reaches outside the node does not belong in this
list without a rule for what it may reach.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import math
import operator
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Rounds of calls one reply may make. A model that keeps calling is told to answer.
ROUNDS = 4
# What a tool may hand back, in characters. A result is read by the model, so it costs context.
RESULT_LIMIT = 2000
EXPRESSION_LIMIT = 300


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    parameters = {"type": "object", "properties": properties, "required": required}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


DEFINITIONS: list[dict[str, Any]] = [
    _tool(
        "current_time",
        "The current date and time. Use it whenever the answer depends on today's date, the day of the week or the time.",
        {"timezone": {"type": "string", "description": "An IANA time zone such as Europe/Paris or America/Chicago. Left out, the answer is in UTC."}},
        [],
    ),
    _tool(
        "calculate",
        "Works out an arithmetic expression exactly. Use it for any sum a mistake would matter in.",
        {"expression": {"type": "string", "description": "Such as (1234.5 * 0.0725) + 19, 2 ** 32, sqrt(2) or round(22 / 7, 4)."}},
        ["expression"],
    ),
    _tool(
        "mesh_facts",
        "What this assistant is running on: the model's name, how many machines share it, and how many replies it can run at once.",
        {},
        [],
    ),
]
NAMES = [item["function"]["name"] for item in DEFINITIONS]

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log2": math.log2, "log10": math.log10, "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "radians": math.radians, "degrees": math.degrees, "factorial": math.factorial,
}  # fmt: skip
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
# A power or a factorial can ask for a number with millions of digits. These keep one small.
_MOST_BITS = 4096
_MOST_FACTORIAL = 500


class ToolRefusal(ValueError):
    """What a tool will not do, said so the model can try again or tell the user."""


def _value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_value(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int) and right > 0 and left.bit_length() * right > _MOST_BITS:
            raise ToolRefusal("that power is too large to work out here")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        arguments = [_value(item) for item in node.args]
        if node.func.id == "factorial" and (not arguments or not isinstance(arguments[0], int) or arguments[0] > _MOST_FACTORIAL):
            raise ToolRefusal(f"factorial takes a whole number up to {_MOST_FACTORIAL}")
        return _FUNCTIONS[node.func.id](*arguments)
    raise ToolRefusal("only numbers, + - * / // % **, brackets, pi, e, tau and these functions are understood: " + ", ".join(sorted(_FUNCTIONS)))


def calculate(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ToolRefusal("give an expression to work out")
    if len(expression) > EXPRESSION_LIMIT:
        raise ToolRefusal(f"an expression may be {EXPRESSION_LIMIT} characters at most")
    try:
        result = _value(ast.parse(expression.strip().replace("^", "**"), mode="eval").body)
    except ToolRefusal:
        raise
    except ZeroDivisionError as exc:
        raise ToolRefusal("that divides by zero") from exc
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError, MemoryError) as exc:
        raise ToolRefusal(f"that could not be worked out: {type(exc).__name__}") from exc
    if isinstance(result, int) and result.bit_length() > _MOST_BITS:
        raise ToolRefusal("the result is too large to write out")
    return json.dumps({"expression": expression.strip(), "result": result})


def current_time(timezone_name: str | None = None, now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    zone = "UTC"
    if timezone_name and timezone_name.upper() != "UTC":
        try:
            moment, zone = moment.astimezone(ZoneInfo(timezone_name)), timezone_name
        except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
            raise ToolRefusal(f"{timezone_name!r} is not a time zone this node knows; use a name such as Europe/Paris, or leave it out for UTC") from exc
    offset = moment.strftime("%z")
    return json.dumps(
        {
            "date": moment.strftime("%Y-%m-%d"),
            "time": moment.strftime("%H:%M:%S"),
            "weekday": moment.strftime("%A"),
            "timezone": zone,
            "utc_offset": f"{offset[:3]}:{offset[3:]}" if offset else "+00:00",
        }
    )


def run(name: str, arguments: Any, facts: Callable[[], dict[str, Any]] | None = None) -> str:
    """One call's result as text for the model. A call that cannot be answered says why, and never raises."""
    try:
        given = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else (arguments or {})
        if not isinstance(given, dict):
            raise ToolRefusal("arguments must be a JSON object")
        if name == "current_time":
            result = current_time(given.get("timezone"))
        elif name == "calculate":
            result = calculate(given.get("expression", ""))
        elif name == "mesh_facts":
            result = json.dumps(facts() if facts else {})
        else:
            raise ToolRefusal(f"there is no tool named {name}; the tools are {', '.join(NAMES)}")
    except ToolRefusal as exc:
        result = json.dumps({"error": str(exc)})
    except json.JSONDecodeError:
        result = json.dumps({"error": "the arguments were not valid JSON"})
    return result[:RESULT_LIMIT]
