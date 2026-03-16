import logging
from dataclasses import dataclass, field
from enum import StrEnum

import orjson
import structlog
from asgi_correlation_id import correlation_id
from beartype import beartype

from app.core.settings import settings


class LoggerError(Exception):
    """Exception for logger related issues."""


class LogLevel(StrEnum):
    """LogLevel types for logger configuration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    NOTSET = "NOTSET"


class LogIcon(StrEnum):
    """Icon mappings for different log categories."""

    DEFAULT = "📋"

    # Status & Results
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    CRITICAL = "🔴"
    INFO = "ℹ️"

    # Operations
    START = "🚀"
    PROCESSING = "🔄"
    DETECTION = "🔍"
    COMPLETE = "✨"

    # Components & Services
    AGENT = "🤖"
    MODEL = "🧠"
    TOOL = "🔧"
    GUARDRAIL = "🛡️"
    PROCESSOR = "⚙️"
    ADAPTER = "🔌"

    # Technical Systems
    AUTH = "🔐"
    TOKEN = "🎫"
    DATABASE = "💾"
    NETWORK = "🌐"
    STREAMING = "📡"
    CACHE = "📦"

    # Specialized Operations
    DIAGRAM = "📊"
    TRANSLATION = "🌍"
    CHAT = "💬"
    HEALTHCHECK = "❤️"
    INTROSPECTION = "🪞"
    VALIDATION = "✓"

    # Time & Performance
    TIMER = "🕒"
    TIMEOUT = "⏱️"
    LATENCY = "⚡"

    # Retry & Recovery
    RETRY = "🔁"
    RECOVERY = "🔧"
    EXHAUSTION = "😵"

    # Data & Content
    IMAGE = "🖼️"
    FILE = "📄"
    JSON = "📝"
    UPLOAD = "📤"
    DOWNLOAD = "📥"

    # Security & Compliance
    SECURITY = "🔒"
    FORBIDDEN = "🚫"
    THREAT = "⚠️"
    BLACKLIST = "🔇"


@dataclass
class LoggerConfig:
    """Logger configuration with debug-specific settings."""

    debug: bool = field(default_factory=lambda: settings.DEBUG)
    app_name: str = field(default="irius")
    log_level: LogLevel = field(default=LogLevel.INFO)


def add_correlation_id(logger, method_name: str, event_dict: dict) -> dict:
    """Add correlation_id to event_dict if present in context."""
    request_id = correlation_id.get()
    if request_id:
        event_dict["correlation_id"] = request_id
    return event_dict


class BusinessRulesProcessor:
    """
    Apply business rules to log events.

    Business Rules:
    - Rule 1: Convert event messages to uppercase for consistency.
    - Rule 2: Truncate event messages exceeding 80 characters to prevent log bloat.
    - Rule 3: Validate that icon kwarg, if provided, is a LogIcon enum member.
    - Rule 4: Prepend icons to event messages only in DEBUG mode.
    """

    def __init__(self, debug: bool) -> None:
        self.debug = debug

    @beartype
    def __call__(self, logger, name: str, event_dict: dict) -> dict:
        """Transform event with uppercase, length limit, and optional icon."""
        try:
            event = str(event_dict.get("event", ""))[:80].upper()
            icon_enum = LogIcon(event_dict.pop("icon", LogIcon.DEFAULT))

            if self.debug:
                event = f"{icon_enum.value} {event}"

            event_dict["event"] = event
            return event_dict
        except ValueError as err:
            raise LoggerError("Wrong Icon chosen, please choose a valid LogIcon enum member") from err
        except Exception as ex:
            raise LoggerError(f"Error with extra kwargs passed to the logger: {ex}") from ex


def dev_pipeline_renderer(logger, name: str, event_dict: dict) -> str:
    """Render log events in human-readable format with pipe-separated fields."""
    reserved_keys = {"timestamp", "level", "event", "filename", "lineno"}

    timestamp = event_dict.get("timestamp", "")
    level = event_dict.get("level", LogLevel.INFO.value).upper()
    event = event_dict.get("event", "")
    filename = event_dict.get("filename", "")
    lineno = event_dict.get("lineno", "")

    location = f"{filename}:{lineno}" if filename else ""

    extra_kwargs = " | ".join(f"{k}={v}" for k, v in event_dict.items() if k not in reserved_keys)

    parts = [timestamp, level, event, extra_kwargs, location]
    return " | ".join(filter(None, parts))


def setup_logging(config: LoggerConfig) -> None:
    """Configure structlog with debug-specific processors."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ],
            additional_ignores=["logger"],
        ),
        BusinessRulesProcessor(debug=config.debug),
    ]

    if config.debug:
        processors = shared_processors + [dev_pipeline_renderer]
    else:
        processors = shared_processors + [
            add_correlation_id,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(serializer=orjson.dumps),
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


# Configure logger by default
_default_config = LoggerConfig()
setup_logging(_default_config)

# Create logger instance ready to use
logger = structlog.get_logger()
