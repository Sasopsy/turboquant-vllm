"""C2 Lloyd-Max codebook builders for the canonical Beta prior."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch

_DEFAULT_MAX_ITER = 200
_DEFAULT_TOL = 1e-7
_DEFAULT_GRID_SIZE = 1025


@dataclass(frozen=True)
class CodebookSpec:
    """Build spec for C2 centroid/midpoint artifacts."""

    head_dim: int
    mse_bits: int
    distribution: Literal["beta"] = "beta"

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        if self.mse_bits <= 0:
            raise ValueError("mse_bits must be > 0")
        if self.distribution != "beta":
            raise ValueError(
                "Only canonical distribution='beta' is supported in C2 v1."
            )


def _beta_log_pdf(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    # f_X(x) = C_d * (1 - x^2)^((d-3)/2) for x in [-1, 1]
    # Evaluate in log-space for numerical stability.
    x64 = x.to(dtype=torch.float64)
    half = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor(float(head_dim), dtype=torch.float64)
    log_const = (
        torch.special.gammaln(d * half)
        - 0.5 * math.log(math.pi)
        - torch.special.gammaln((d - 1.0) * half)
    )
    exponent = (d - 3.0) * half
    inner = torch.clamp(1.0 - x64 * x64, min=1e-24)
    return log_const + exponent * torch.log(inner)


def _beta_pdf(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    return torch.exp(_beta_log_pdf(x, head_dim))


def _integrate_interval(
    lower: float,
    upper: float,
    head_dim: int,
    num_grid_points: int,
) -> tuple[float, float]:
    xs = torch.linspace(
        lower,
        upper,
        steps=num_grid_points,
        dtype=torch.float64,
    )
    weights = _beta_pdf(xs, head_dim)
    numerator = torch.trapz(xs * weights, xs).item()
    denominator = torch.trapz(weights, xs).item()
    return numerator, denominator


def solve_lloyd_max_beta(
    head_dim: int,
    mse_bits: int,
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
    num_grid_points: int = _DEFAULT_GRID_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd-Max for the canonical Beta prior on [-1, 1]."""

    if head_dim <= 0:
        raise ValueError("head_dim must be > 0")
    if mse_bits <= 0:
        raise ValueError("mse_bits must be > 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if num_grid_points < 16:
        raise ValueError("num_grid_points must be >= 16")

    num_levels = 2**mse_bits
    # Symmetric initialization in the support interval.
    centroids = torch.linspace(
        -1.0 + 1.0 / (num_levels + 1),
        1.0 - 1.0 / (num_levels + 1),
        steps=num_levels,
        dtype=torch.float64,
    )

    for _ in range(max_iter):
        midpoints = (centroids[:-1] + centroids[1:]) / 2.0
        edges = torch.cat(
            [
                torch.tensor([-1.0], dtype=torch.float64),
                midpoints,
                torch.tensor([1.0], dtype=torch.float64),
            ]
        )

        new_centroids = centroids.clone()
        for level_idx in range(num_levels):
            lower = edges[level_idx].item()
            upper = edges[level_idx + 1].item()
            numerator, denominator = _integrate_interval(
                lower=lower,
                upper=upper,
                head_dim=head_dim,
                num_grid_points=num_grid_points,
            )
            if denominator > 1e-20:
                new_centroids[level_idx] = numerator / denominator

        delta = torch.max(torch.abs(new_centroids - centroids)).item()
        centroids = new_centroids
        if delta < tol:
            break

    centroids, _ = torch.sort(centroids)
    midpoints = (centroids[:-1] + centroids[1:]) / 2.0

    centroids_out = centroids.to(dtype=torch.float32).contiguous()
    midpoints_out = midpoints.to(dtype=torch.float32).contiguous()
    return centroids_out, midpoints_out


@lru_cache(maxsize=128)
def _get_codebook_cached(spec: CodebookSpec) -> tuple[torch.Tensor, torch.Tensor]:
    return solve_lloyd_max_beta(spec.head_dim, spec.mse_bits)


def get_codebook(spec: CodebookSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic (centroids, midpoints) CPU float32 tensors."""

    return _get_codebook_cached(spec)

