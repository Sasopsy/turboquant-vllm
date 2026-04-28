"""Shared scaffold helpers for future TileLang kernel integration."""

from __future__ import annotations


class TileLangKernelNotImplementedError(NotImplementedError):
    """Raised when scaffold-only kernel wrappers are called directly."""


_KERNEL_MESSAGE = (
    "TileLang kernel scaffolding is present, but the low-level kernel "
    "implementation has not been written yet."
)


def raise_kernel_not_implemented() -> None:
    """Raise the canonical scaffold-only kernel exception."""

    raise TileLangKernelNotImplementedError(_KERNEL_MESSAGE)
