from __future__ import annotations

import math

import torch


def create_dct_matrix(n_size: int, device=None) -> torch.Tensor:
    """Create an orthonormal DCT-II transform matrix."""

    n = torch.arange(n_size, dtype=torch.float32, device=device).reshape((1, n_size))
    k = torch.arange(n_size, dtype=torch.float32, device=device).reshape((n_size, 1))
    dct_matrix = torch.sqrt(torch.tensor(2.0 / n_size, device=device)) * torch.cos(
        math.pi * k * (2 * n + 1) / (2 * n_size)
    )
    dct_matrix[0, :] = 1 / math.sqrt(n_size)
    return dct_matrix


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Apply 2D DCT-II to the last two dimensions of a tensor."""

    h, w = x.size(-2), x.size(-1)
    dct_matrix_h = create_dct_matrix(h, device=x.device)
    dct_matrix_w = create_dct_matrix(w, device=x.device)
    return torch.matmul(dct_matrix_h, torch.matmul(x, dct_matrix_w.t()))


def idct_2d(x: torch.Tensor) -> torch.Tensor:
    """Apply inverse 2D DCT under the orthonormal convention."""

    h, w = x.size(-2), x.size(-1)
    dct_matrix_h = create_dct_matrix(h, device=x.device)
    dct_matrix_w = create_dct_matrix(w, device=x.device)
    return torch.matmul(dct_matrix_h.t(), torch.matmul(x, dct_matrix_w))


def non_dc_positions(block_size: int):
    """Return all 2D coefficient positions except the DC coefficient (0, 0)."""

    return [
        (i, j)
        for i in range(block_size)
        for j in range(block_size)
        if not (i == 0 and j == 0)
    ]


def all_block_positions(block_size: int):
    """Return all 2D coefficient positions in a block."""

    return [(i, j) for i in range(block_size) for j in range(block_size)]
