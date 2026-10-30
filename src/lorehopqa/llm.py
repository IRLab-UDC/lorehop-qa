"""Error formatting shared by the structured-output parsers."""

from __future__ import annotations

from pydantic import ValidationError


def _format_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        messages = []
        for error in exc.errors():
            msg = error["msg"]
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]
            loc = ".".join(str(part) for part in error["loc"])
            messages.append(f"{loc}: {msg}" if loc else msg)
        if messages:
            return "; ".join(messages)
    return f"{type(exc).__name__}: {exc}"
