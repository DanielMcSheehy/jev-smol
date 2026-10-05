"""Typed errors raised by engines, mapped to HTTP responses by the API layer.

ImageError / UnsupportedError -> 400, EngineNotReadyError -> 503, each with a
body of the form {"error": {"type": ..., "message": ...}}.
"""

from __future__ import annotations


class JevError(Exception):
    """Base class for jev API errors."""

    error_type = "jev_error"
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ImageError(JevError):
    """A request image violated the decode/size/format rules."""

    error_type = "image_error"
    status_code = 400


class UnsupportedError(JevError):
    """The request uses an input modality this model does not support."""

    error_type = "unsupported"
    status_code = 400


class EngineNotReadyError(JevError):
    """The model backend failed to load or is not available yet."""

    error_type = "engine_not_ready"
    status_code = 503
