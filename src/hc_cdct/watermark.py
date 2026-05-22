from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.fft import dctn, idctn

from .dct_utils import all_block_positions, non_dc_positions
from .metrics import vote_repeat_and_coeff


@dataclass
class WatermarkConfig:
    block_size: int = 3
    watermark_length: int | None = None
    repeat_factor: int = 8
    alpha: float = 0.8
    dct_detail: float = 80
    mme_detail: float = 80
    payload_scale: float = 500
    watermark_amplitude: float = 0.001


def build_watermark(host_signal: torch.Tensor, config: WatermarkConfig, seed: int = 123):
    """Create the original and block-repeated watermark bit arrays."""

    block_size = config.block_size
    n_blocks = int(host_signal.numel() // (block_size * block_size))

    if config.watermark_length is None:
        wm_len = int(n_blocks / config.repeat_factor)
    else:
        wm_len = int(config.watermark_length)

    if wm_len <= 0:
        raise ValueError(f"Invalid watermark length: {wm_len}")

    if n_blocks % wm_len != 0:
        raise ValueError(
            f"Number of blocks {n_blocks} is not divisible by watermark length {wm_len}."
        )

    repeat_n = n_blocks // wm_len

    rng = np.random.default_rng(seed)
    water = rng.integers(0, 2, wm_len, dtype=np.int32)
    water_rep = np.repeat(water, repeat_n).astype(np.int32)
    water_amplitude = config.watermark_amplitude * water_rep

    return {
        "water": water,
        "water_rep": water_rep,
        "water_amplitude": water_amplitude,
        "wm_len": wm_len,
        "repeat_n": repeat_n,
        "n_blocks": n_blocks,
    }


def decode_bits_from_values(values, extractor, alpha: float, detail: float):
    """
    Convert an extractor output into 0/1 bits.

    The external extractor may or may not support vectorized NumPy input, so the
    function falls back to element-wise calls.
    """

    values = np.asarray(values)
    flat_values = values.reshape(-1)

    try:
        extracted = extractor(flat_values, alpha, detail)
    except Exception:
        extracted = np.array([extractor(float(v), alpha, detail) for v in flat_values])

    bits = (np.round(2 * detail * np.asarray(extracted)) % 2).astype(np.int32)
    return bits.reshape(values.shape)


def embed_hc_cdct(host_signal: torch.Tensor, water_amplitude, emb_func, config: WatermarkConfig, device: str):
    """
    Embed the watermark into DCT-domain non-DC coefficients.

    The operation follows the original script: each 3x3 weight block is converted
    to the absolute-value domain, DCT is applied, the DC coefficient is skipped,
    and the same block-level watermark bit is embedded into the remaining
    coefficients. After inverse DCT, the original signs are restored.
    """

    block_size = config.block_size
    shape = host_signal.shape

    if tuple(shape[-2:]) != (block_size, block_size):
        raise ValueError(
            f"The selected embedding layer must have last two dimensions "
            f"({block_size}, {block_size}), but got {tuple(shape[-2:])}."
        )

    coeff_positions = non_dc_positions(block_size)
    host_blocks = host_signal.reshape(-1, block_size, block_size).detach().cpu().numpy().copy()

    for h in range(host_blocks.shape[0]):
        block = host_blocks[h]
        sign_block = np.where(block >= 0, 1.0, -1.0)

        abs_block = np.abs(block)
        block_dct = dctn(abs_block, type=2, axes=(-2, -1), norm="ortho")

        payload = config.payload_scale * water_amplitude[h] / config.dct_detail

        for u, v in coeff_positions:
            block_dct[u, v], _ = emb_func(
                block_dct[u, v],
                payload,
                config.alpha,
                config.dct_detail,
            )

        block_abs_rec = idctn(block_dct, type=2, axes=(-2, -1), norm="ortho")
        block_abs_rec = np.maximum(block_abs_rec, 0)
        host_blocks[h] = sign_block * block_abs_rec

    return torch.from_numpy(host_blocks).to(device).reshape(shape)


def extract_hc_cdct(
    pruned_signal: torch.Tensor,
    extractor,
    water,
    water_rep,
    wm_len: int,
    repeat_n: int,
    config: WatermarkConfig,
    beta: float = 0.5,
    eps: float = 1e-12,
):
    """Extract the DCT-domain watermark after pruning, using minimum-amplitude compensation."""

    block_size = config.block_size
    coeff_positions = non_dc_positions(block_size)

    signal_blocks = pruned_signal.reshape(-1, block_size, block_size).detach().cpu().numpy()
    signal_abs = np.abs(signal_blocks)

    nonzero_mask = signal_abs > eps
    pruned_mask = signal_abs <= eps

    if np.any(nonzero_mask):
        min_nonzero = np.min(signal_abs[nonzero_mask])
    else:
        min_nonzero = 0.0

    comp_value = beta * min_nonzero
    signal_abs_comp = signal_abs.copy()
    signal_abs_comp[pruned_mask] += comp_value

    signal_freq = np.zeros_like(signal_abs_comp)
    for h in range(signal_abs_comp.shape[0]):
        signal_freq[h] = dctn(signal_abs_comp[h], type=2, axes=(-2, -1), norm="ortho")

    extracted_values = np.stack([signal_freq[:, u, v] for u, v in coeff_positions], axis=1)
    extracted_bits = decode_bits_from_values(
        extracted_values,
        extractor,
        config.alpha,
        config.dct_detail,
    )

    ber_rep, ber_each, wm_vote, ber_vote = vote_repeat_and_coeff(
        extracted_bits,
        water,
        water_rep,
        wm_len,
        repeat_n,
    )

    return {
        "ber_rep_coeff": ber_rep,
        "ber_each_coeff": ber_each,
        "wm_vote": wm_vote,
        "ber_vote": ber_vote,
        "comp_value": float(comp_value),
    }


def embed_hc_space(host_signal: torch.Tensor, water_amplitude, emb_mme_func, config: WatermarkConfig, device: str):
    """Spatial-domain HC baseline: embed the same bit into all raw coefficients in each 3x3 block."""

    block_size = config.block_size
    shape = host_signal.shape
    positions = all_block_positions(block_size)

    host_blocks = host_signal.reshape(-1, block_size, block_size).detach().cpu().numpy().copy()

    for h in range(host_blocks.shape[0]):
        payload = config.payload_scale * water_amplitude[h] / config.mme_detail

        for u, v in positions:
            host_blocks[h, u, v], _ = emb_mme_func(
                host_blocks[h, u, v],
                payload,
                config.alpha,
                config.mme_detail,
            )

    return torch.from_numpy(host_blocks).to(device).reshape(shape)


def extract_hc_space(
    pruned_signal: torch.Tensor,
    extractor_mme,
    water,
    water_rep,
    wm_len: int,
    repeat_n: int,
    config: WatermarkConfig,
):
    """Extract the spatial-domain HC baseline watermark."""

    block_size = config.block_size
    n_coeff = block_size * block_size

    signal_blocks = pruned_signal.reshape(-1, block_size, block_size).detach().cpu().numpy()
    extracted_values = signal_blocks.reshape(signal_blocks.shape[0], n_coeff)

    extracted_bits = decode_bits_from_values(
        extracted_values,
        extractor_mme,
        config.alpha,
        config.mme_detail,
    )

    ber_rep, ber_each, wm_vote, ber_vote = vote_repeat_and_coeff(
        extracted_bits,
        water,
        water_rep,
        wm_len,
        repeat_n,
    )

    return {
        "ber_rep_coeff": ber_rep,
        "ber_each_coeff": ber_each,
        "wm_vote": wm_vote,
        "ber_vote": ber_vote,
    }
