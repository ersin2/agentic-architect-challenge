"""Error types shared by all parts.

Each way a model call can fail maps to exactly one class, so a caller picks
an outcome by catching a type, not by parsing an error message.
"""


class AgentKitError(Exception):
    """Base class for every error raised by this project."""


class ConfigError(AgentKitError):
    """Configuration is missing or invalid (for example, no API key)."""


class LLMError(AgentKitError):
    """Base class for model call failures."""


class LLMUnavailable(LLMError):
    """The provider could not be reached: timeout, 5xx, 429 after retries, or the circuit is open."""


class LLMBadRequest(LLMError):
    """The provider rejected the request (4xx). Retrying will not help: wrong model, payload or key."""


class LLMBlocked(LLMError):
    """The provider refused to answer, for example because of a safety filter."""


class LLMOutputError(LLMError):
    """The model answered, but the answer is unusable: empty, cut off, or not the JSON we asked for."""


class ToolError(AgentKitError):
    """A tool rejected its input or failed. The message is safe to show to the model."""
