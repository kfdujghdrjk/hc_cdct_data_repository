from __future__ import annotations

import numpy as np


def bit_error_rate(reference_bits, extracted_bits) -> float:
    reference_bits = np.asarray(reference_bits).astype(np.int32)
    extracted_bits = np.asarray(extracted_bits).astype(np.int32)
    if reference_bits.shape != extracted_bits.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference_bits.shape}, extracted {extracted_bits.shape}"
        )
    return float(np.mean(reference_bits != extracted_bits))


def vote_repeat_and_coeff(wm_extract_coeff, water, water_rep, wm_len: int, repeat_n: int):
    """
    Two-stage voting used in the original experiment.

    Parameters
    ----------
    wm_extract_coeff:
        Array with shape [n_blocks, n_coeff].
    water:
        Original watermark bits with shape [wm_len].
    water_rep:
        Block-level repeated watermark bits with shape [n_blocks].
    wm_len:
        Length of the original watermark.
    repeat_n:
        Number of blocks used to repeat each watermark bit.

    Returns
    -------
    ber_rep_coeff:
        BER computed over all block/coefficient-level repeated bits.
    ber_each_coeff:
        BER after repeat voting, computed separately for each coefficient position.
    wm_vote:
        Final watermark after repeat-level and coefficient-level voting.
    ber_vote:
        Final voted BER.
    """

    wm_extract_coeff = np.asarray(wm_extract_coeff).astype(np.int32)
    n_blocks_eff, n_coeff_eff = wm_extract_coeff.shape
    usable = wm_len * repeat_n

    if n_blocks_eff < usable:
        raise ValueError(f"Available blocks {n_blocks_eff} < wm_len * repeat_n = {usable}")

    wm_extract_coeff = wm_extract_coeff[:usable, :]
    water_rep = np.asarray(water_rep[:usable]).astype(np.int32)

    ber_rep_coeff = float(np.mean(wm_extract_coeff != water_rep[:, None]))

    wm_groups = wm_extract_coeff.reshape(wm_len, repeat_n, n_coeff_eff)
    wm_by_coeff = (np.sum(wm_groups, axis=1) >= (repeat_n / 2)).astype(np.int32)

    water = np.asarray(water).astype(np.int32)
    ber_each_coeff = np.mean(wm_by_coeff != water[:, None], axis=0)

    wm_vote = (np.sum(wm_by_coeff, axis=1) >= (n_coeff_eff / 2)).astype(np.int32)
    ber_vote = float(np.mean(water != wm_vote))

    return ber_rep_coeff, ber_each_coeff, wm_vote, ber_vote
