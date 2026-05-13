"""
Exception handling module

This module defines all custom exception classes and error codes used in the project.
Follows a unified exception handling specification, facilitating error tracking and debugging.
"""

from typing import Optional, Dict, Any
from core.constants.errors import ErrorCode


class CriticalError(Exception):
    """Marker base class for critical errors that must never be silently swallowed.

    Errors inheriting from this class indicate serious system-level issues
    (e.g., missing tenant context, broken invariants) that should always
    propagate to the caller and result in a request failure (HTTP 500).

    Use ``reraise_critical_errors()`` from ``common_utils.async_utils``
    to guard ``asyncio.gather(return_exceptions=True)`` result processing.
    """

    pass


class ValidationException(Exception):
    """Data validation exception

    Raised when input data validation fails.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if field:
            message = f"Field '{field}': {message}"

        super().__init__(message)
        self.code = ErrorCode.VALIDATION_ERROR.value
        self.message = message
        self.details = details or {}


class FatalError(Exception):
    """Fatal error that should not be retried."""


class BusinessLogicError(Exception):
    """Business logic error that may be resolved by retrying."""


class LongJobError(Exception):
    """Base class for long-job related errors."""


class JobNotFoundError(LongJobError):
    """Job not found error."""


class JobAlreadyExistsError(LongJobError):
    """Job already exists error."""


class JobStateError(LongJobError):
    """Invalid job state error."""


class ManagerShutdownError(LongJobError):
    """Long-job manager has been shut down."""


class MaxConcurrentJobsError(LongJobError):
    """Exceeded maximum number of concurrent jobs."""

# Export long job system error classes
__all__ = [
    # Error codes and base exception
    'ErrorCode',
    'CriticalError',
    'ValidationException',
    # Long job system error classes
    'FatalError',
    'BusinessLogicError',
    'LongJobError',
    'JobNotFoundError',
    'JobAlreadyExistsError',
    'JobStateError',
    'ManagerShutdownError',
    'MaxConcurrentJobsError',
]
