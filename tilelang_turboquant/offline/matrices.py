"""C2 matrix builders: rotation and QJL projection matrices."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch


def _cpu_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


@dataclass(frozen=True)
class RotationSpec:
    """Build spec for rotation matrix construction."""

    head_dim: int
    mode: Literal["qr", "randomized_hadamard"]
    seed: int
    allow_hadamard_padding: bool = False

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        if self.mode not in ("qr", "randomized_hadamard"):
            raise ValueError(
                "mode must be one of {'qr', 'randomized_hadamard'}; "
                f"got {self.mode!r}"
            )


@dataclass(frozen=True)
class QJLSpec:
    """Build spec for Gaussian QJL projection matrix S."""

    head_dim: int
    seed: int

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be > 0")


def _build_qr_rotation(head_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = _cpu_generator(seed)
    gaussian = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian, mode="reduced")

    # Remove sign ambiguity in QR for deterministic orientation.
    diag = torch.diag(r)
    signs = torch.sign(diag)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q.contiguous(), q.transpose(0, 1).contiguous()


def _build_normalized_hadamard(head_dim: int) -> torch.Tensor:
    if not _is_power_of_two(head_dim):
        raise ValueError(
            "Hadamard rotation requires power-of-two head_dim; "
            f"got {head_dim}."
        )
    hadamard = torch.tensor([[1.0]], dtype=torch.float32)
    while hadamard.shape[0] < head_dim:
        hadamard = torch.cat(
            [
                torch.cat([hadamard, hadamard], dim=1),
                torch.cat([hadamard, -hadamard], dim=1),
            ],
            dim=0,
        )
    return (hadamard / math.sqrt(head_dim)).contiguous()


def _build_randomized_hadamard_rotation(
    head_dim: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = _cpu_generator(seed)
    hadamard = _build_normalized_hadamard(head_dim)
    signs = torch.where(
        torch.rand(head_dim, generator=gen) < 0.5,
        torch.tensor(-1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    perm = torch.randperm(head_dim, generator=gen)

    # Row permutation + column sign flips keeps orthonormality.
    rotation = hadamard[perm, :] * signs.unsqueeze(0)
    return rotation.contiguous(), rotation.transpose(0, 1).contiguous()


@lru_cache(maxsize=128)
def _get_rotation_cached(spec: RotationSpec) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if spec.mode == "qr":
        rotation, rotation_t = _build_qr_rotation(spec.head_dim, spec.seed)
        return rotation, rotation_t, False

    if _is_power_of_two(spec.head_dim):
        rotation, rotation_t = _build_randomized_hadamard_rotation(
            spec.head_dim, spec.seed
        )
        return rotation, rotation_t, False

    warnings.warn(
        "randomized_hadamard requested for non-power-of-two head_dim; "
        "falling back to qr rotation.",
        RuntimeWarning,
        stacklevel=2,
    )
    rotation, rotation_t = _build_qr_rotation(spec.head_dim, spec.seed)
    return rotation, rotation_t, False


def get_rotation(spec: RotationSpec) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return (rotation, rotation_t, is_symmetric) as CPU float32 tensors."""

    rotation, rotation_t, is_symmetric = _get_rotation_cached(spec)
    return rotation, rotation_t, is_symmetric


@lru_cache(maxsize=128)
def _get_s_matrix_cached(spec: QJLSpec) -> torch.Tensor:
    gen = _cpu_generator(spec.seed)
    return torch.randn(
        spec.head_dim, spec.head_dim, generator=gen, dtype=torch.float32
    ).contiguous()


def get_s_matrix(spec: QJLSpec) -> torch.Tensor:
    """Return seeded dense Gaussian S matrix as CPU float32."""

    return _get_s_matrix_cached(spec)

