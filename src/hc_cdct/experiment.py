from __future__ import annotations

import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import DataConfig, get_cifar_loaders
from .model_factory import build_model, load_checkpoint_if_needed, maybe_wrap_data_parallel
from .pruning import apply_global_l1_pruning, get_parameters_to_prune
from .state_utils import copy_state_tensor, get_module_by_weight_name, get_state_tensor
from .train_eval import evaluate
from .watermark import (
    WatermarkConfig,
    build_watermark,
    embed_hc_cdct,
    embed_hc_space,
    extract_hc_cdct,
    extract_hc_space,
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_parent(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _serialize_vector(values):
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def run_pruning_experiment(config_path: str | Path):
    """
    Run the DCT-domain HC-CDCT experiment and the spatial-domain HC baseline.

    External functions `emb`, `extract`, `emb_MME`, and `extract_MME` are imported
    from the user's local `torchQim.py`, consistent with the original script.
    """

    # Keep the external dependency explicit. The original script used:
    # from torchQim import *
    try:
        from torchQim import emb, extract, emb_MME, extract_MME  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Cannot import emb/extract/emb_MME/extract_MME from torchQim.py. "
            "Copy torchQim.py into the repository root or install it as a package."
        ) from exc

    cfg = load_yaml(config_path)
    seed = int(cfg["project"].get("seed", 123))
    set_seed(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_printoptions(precision=8)

    data_cfg = DataConfig(**cfg["data"])
    model_cfg = cfg["model"]
    exp_cfg = cfg["experiment"]
    comp_cfg = cfg["compensation"]
    eval_cfg = cfg["evaluation"]

    trainloader, testloader = get_cifar_loaders(data_cfg)

    model = build_model(
        name=model_cfg["name"],
        num_classes=int(model_cfg["num_classes"]),
        device=device,
    )
    model = load_checkpoint_if_needed(
        model,
        checkpoint_path=model_cfg.get("checkpoint_path"),
        device=device,
        resume=bool(model_cfg.get("resume", True)),
    )
    model = maybe_wrap_data_parallel(
        model,
        device=device,
        data_parallel=bool(model_cfg.get("data_parallel", True)),
    )

    criterion = nn.CrossEntropyLoss()

    embedding_layer = exp_cfg["embedding_layer"]
    host_signal = get_state_tensor(model, embedding_layer)
    shape = host_signal.shape

    print(f"Using device: {device}")
    print(f"Embedding layer: {embedding_layer}, shape={tuple(shape)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    wm_config = WatermarkConfig(
        block_size=int(exp_cfg["block_size"]),
        watermark_length=exp_cfg.get("watermark_length"),
        repeat_factor=int(exp_cfg["repeat_factor"]),
        alpha=float(exp_cfg["alpha"]),
        dct_detail=float(exp_cfg["dct_detail"]),
        mme_detail=float(exp_cfg["mme_detail"]),
        payload_scale=float(exp_cfg["payload_scale"]),
        watermark_amplitude=float(exp_cfg["watermark_amplitude"]),
    )

    wm = build_watermark(host_signal, wm_config, seed=seed)
    water = wm["water"]
    water_rep = wm["water_rep"]
    water_amplitude = wm["water_amplitude"]
    wm_len = int(wm["wm_len"])
    repeat_n = int(wm["repeat_n"])

    clean_eval = None
    if bool(eval_cfg.get("compute_accuracy", False)):
        clean_eval = evaluate(model, testloader, criterion, device=device)
        print(f"Clean accuracy: {clean_eval['accuracy']:.4f}")

    rows = []

    for amount in exp_cfg["pruning_amounts"]:
        amount = float(amount)
        print(f"\nPruning amount: {amount:.3f}")

        start = time.time()
        embedded_dct = embed_hc_cdct(
            host_signal=host_signal,
            water_amplitude=water_amplitude,
            emb_func=emb,
            config=wm_config,
            device=device,
        )
        dct_embedding_time = time.time() - start

        dct_distortion_var = torch.var(embedded_dct - host_signal).detach().cpu().item()

        model_dct = copy.deepcopy(model)
        copy_state_tensor(model_dct, embedding_layer, embedded_dct)

        dct_layer_module = get_module_by_weight_name(model_dct, embedding_layer)
        apply_global_l1_pruning([(dct_layer_module, "weight")], amount=amount)

        pruned_dct_signal = get_state_tensor(model_dct, embedding_layer)
        dct_extract_result = extract_hc_cdct(
            pruned_signal=pruned_dct_signal,
            extractor=extract,
            water=water,
            water_rep=water_rep,
            wm_len=wm_len,
            repeat_n=repeat_n,
            config=wm_config,
            beta=float(comp_cfg["beta"]),
            eps=float(comp_cfg["eps"]),
        )

        dct_accuracy = None
        if bool(eval_cfg.get("compute_accuracy", False)):
            dct_accuracy = evaluate(model_dct, testloader, criterion, device=device)["accuracy"]

        start = time.time()
        embedded_hc = embed_hc_space(
            host_signal=host_signal,
            water_amplitude=water_amplitude,
            emb_mme_func=emb_MME,
            config=wm_config,
            device=device,
        )
        hc_embedding_time = time.time() - start

        hc_distortion_var = torch.var(embedded_hc - host_signal).detach().cpu().item()

        model_hc = copy.deepcopy(model)
        copy_state_tensor(model_hc, embedding_layer, embedded_hc)

        # Original script pruned all model weights for the spatial HC baseline.
        hc_parameters_to_prune = get_parameters_to_prune(model_hc)
        apply_global_l1_pruning(hc_parameters_to_prune, amount=amount)

        pruned_hc_signal = get_state_tensor(model_hc, embedding_layer)
        hc_extract_result = extract_hc_space(
            pruned_signal=pruned_hc_signal,
            extractor_mme=extract_MME,
            water=water,
            water_rep=water_rep,
            wm_len=wm_len,
            repeat_n=repeat_n,
            config=wm_config,
        )

        hc_accuracy = None
        if bool(eval_cfg.get("compute_accuracy", False)):
            hc_accuracy = evaluate(model_hc, testloader, criterion, device=device)["accuracy"]

        row = {
            "pruning_amount": amount,
            "wm_len": wm_len,
            "repeat_n": repeat_n,
            "dct_ber_rep_coeff": dct_extract_result["ber_rep_coeff"],
            "dct_ber_vote": dct_extract_result["ber_vote"],
            "dct_ber_each_coeff": _serialize_vector(dct_extract_result["ber_each_coeff"]),
            "dct_distortion_var": dct_distortion_var,
            "dct_embedding_time_sec": dct_embedding_time,
            "dct_comp_value": dct_extract_result["comp_value"],
            "hc_space_ber_rep_coeff": hc_extract_result["ber_rep_coeff"],
            "hc_space_ber_vote": hc_extract_result["ber_vote"],
            "hc_space_ber_each_coeff": _serialize_vector(hc_extract_result["ber_each_coeff"]),
            "hc_space_distortion_var": hc_distortion_var,
            "hc_space_embedding_time_sec": hc_embedding_time,
            "clean_accuracy": clean_eval["accuracy"] if clean_eval else "",
            "dct_pruned_accuracy": dct_accuracy if dct_accuracy is not None else "",
            "hc_space_pruned_accuracy": hc_accuracy if hc_accuracy is not None else "",
        }

        rows.append(row)

        print(f"HC-CDCT voted BER: {row['dct_ber_vote']:.6f}")
        print(f"HC-space voted BER: {row['hc_space_ber_vote']:.6f}")

    if bool(eval_cfg.get("save_csv", True)):
        result_csv = eval_cfg["result_csv"]
        _ensure_parent(result_csv)
        fieldnames = list(rows[0].keys()) if rows else []
        with open(result_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved results to: {result_csv}")

    return rows
