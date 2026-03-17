"""Structured logger — ANSI colored, stderr output, pipeline/step tagging."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Final

import structlog

##### ANSI COLORS (RGB) #####

_RESET: Final[str] = "\033[0m"

_STEP_COLORS: dict[str, str] = {
    # System lifecycle
    "START": "255;215;0",
    "STOP": "255;165;0",
    "OK": "50;205;50",
    "IPC": "147;112;219",
    # STT pipeline
    "STT": "144;238;144",
    "VAD": "46;139;87",
    "MODEL": "0;206;209",
    "DOWNLOAD": "100;149;237",
    # WebSocket / streaming
    "WS": "70;130;180",
    "SSE": "70;130;180",
    # Errors / warnings (overridden by level)
    "ERROR": "255;69;0",
    "WARN": "255;165;0",
}

_LEVEL_COLORS: dict[str, str] = {
    "error": "255;69;0",
    "critical": "255;69;0",
    "warning": "255;165;0",
}

_DEFAULT_COLOR: Final[str] = "192;192;192"
_MAX_TEXT_LEN: Final[int] = 120


##### RENDERER #####


class ColorRenderer:
    """Format: HH:MM:SS [STEP] message key=value — ANSI RGB colored."""

    __slots__ = ()

    def __call__(self, logger_instance: Any, method_name: str, event_dict: dict[str, Any]) -> str:
        step = str(event_dict.pop("step", "")).upper()
        event = event_dict.pop("event", "")

        color = _LEVEL_COLORS.get(method_name) or _STEP_COLORS.get(step, _DEFAULT_COLOR)
        ts = time.strftime("%H:%M:%S")

        skip = {"timestamp", "level", "logger"}
        extras = " ".join(f"{k}={v}" for k, v in event_dict.items() if k not in skip and v is not None)

        parts = [ts]
        if step:
            parts.append(f"[{step}]")
        if event:
            parts.append(str(event)[:_MAX_TEXT_LEN])
        if extras:
            parts.append(extras)

        line = " ".join(parts)
        return f"\033[38;2;{color}m{line}{_RESET}"


##### CONFIGURE #####


def configure_logging(level: str = "info") -> None:
    """Configure structlog with ANSI colored stderr output."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.UnicodeDecoder(),
            ColorRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")

    for noisy in ("websockets", "httpx", "httpcore", "hpack", "robyn", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


configure_logging()

logger = structlog.get_logger("evoice")
