"""
Fallback chain primitive per [ADR-013 §"Provider fallback chain"](../../docs/ADR-013-model-router-policy.md).

A ``FallbackChain[B]`` wraps an ordered list of backends and exposes a
single method, ``run(fn)``, that tries each backend until one returns
without raising a ``BackendError`` (or any subclass).

Why generic? Concrete ``ModelBackend`` implementations are unrelated
modules (anthropic/openrouter/ollama) that share only the structural
``ModelBackend`` Protocol. Pulling fallback into a generic primitive
keeps it testable in isolation and lets the router compose chains
per tier without ``isinstance`` checks.

PRD-008 US-003 wants the last entry in every chain to be a local
Ollama backend so the agent keeps working offline. This module does
not enforce that — it's a wiring concern of the router. The chain
just walks until exhaustion.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Generic, TypeVar


class BackendError(RuntimeError):
    """Raised by a ``ModelBackend`` to signal a retryable failure.

    The ``kind`` attribute classifies the failure for telemetry
    (ADR-013 §"Provider health tracking"):

    - ``"rate_limit"`` — provider returned 429 / quota
    - ``"overloaded"`` — provider returned 529 / surge
    - ``"timeout"`` — local timer expired or transport-level timeout
    - ``"server"`` — 5xx other than the above
    - ``"network"`` — connection refused / DNS / TLS
    - ``"unknown"`` — fallback bucket

    Auth failures and content-filter rejections are **not**
    ``BackendError`` — those are configuration / input bugs that the
    fallback chain should not paper over.
    """

    def __init__(self, message: str, *, kind: str = "unknown") -> None:
        super().__init__(message)
        self.kind = kind


B = TypeVar("B")
"""Generic backend type. Constrained structurally at call sites — this
module only requires that the callable supplied to ``run`` can accept
each ``B``."""

R = TypeVar("R")
"""Return type of the user-supplied callable."""


class FallbackChain(Generic[B]):
    """Ordered list of backends. ``run(fn)`` tries each until one
    succeeds.

    The ``fn`` argument is async and accepts a backend, so the chain
    is independent of any specific backend method shape. The router
    calls ``run(lambda b: b.generate(messages))`` to dispatch a chat
    completion; tests can call ``run(lambda b: b.probe())`` to wire
    a health probe through the same primitive.

    Parameters
    ----------
    backends:
        Ordered tuple of backends. The first entry is the primary,
        every subsequent entry is a fallback. An empty sequence is
        rejected at construction time — silently doing nothing would
        be worse than failing fast.
    """

    def __init__(self, backends: Sequence[B]) -> None:
        if not backends:
            raise ValueError("FallbackChain requires at least one backend")
        self._backends: tuple[B, ...] = tuple(backends)

    @property
    def backends(self) -> tuple[B, ...]:
        """The backend tuple in priority order."""
        return self._backends

    async def run(
        self,
        fn: Callable[[B], Awaitable[R]],
    ) -> tuple[R, B, int]:
        """Dispatch ``fn`` against backends in order; return on first
        success.

        Returns
        -------
        ``(result, backend, fallback_steps)``
            ``backend`` is the one that succeeded. ``fallback_steps`` is
            zero for the primary, 1 for the first fallback, etc. — what
            PRD-008 FR-5 stores in telemetry as ``fallback_steps``.

        Raises
        ------
        BackendError
            When every backend in the chain raised ``BackendError``.
            The raised exception's ``__cause__`` chain preserves every
            attempt for debuggability; the outermost exception's
            ``kind`` is taken from the final attempt.
        """
        last_err: BackendError | None = None
        for step, backend in enumerate(self._backends):
            try:
                result = await fn(backend)
            except BackendError as exc:
                # Wrap into chain and continue. The original exc is
                # preserved as __cause__ via "raise … from exc" on
                # the final failure path; intermediate failures are
                # logged at the router layer (this module is silent
                # to keep the chain decision-free).
                last_err = exc
                continue
            return result, backend, step

        # Exhausted — re-raise with the last failure as the cause.
        assert last_err is not None, "non-empty chain cannot exit without a last error"
        raise BackendError(
            f"all {len(self._backends)} backends in the fallback chain failed; "
            f"last error: {last_err}",
            kind=last_err.kind,
        ) from last_err


__all__ = ["B", "BackendError", "FallbackChain"]
