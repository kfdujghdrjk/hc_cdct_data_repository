from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import DataConfig, get_cifar_loaders
from .model_factory import build_model, load_checkpoint_if_needed, maybe_wrap_data_parallel
from .state_utils import copy_state_tensor, get_state_tensor
from .train_eval import evaluate, train_one_epoch
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


def _build_optimizer(model, attack_cfg: dict, lr: float):
    name = str(attack_cfg.get("optimizer", "SGD")).lower()
    weight_decay = float(attack_cfg.get("weight_decay", 5.0e-4))

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(attack_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported fine-tuning optimizer: {attack_cfg.get('optimizer')!r}")


def _extract_scheme(
    scheme: str,
    model,
    embedding_layer: str,
    extract,
    extract_MME,
    water,
    water_rep,
    wm_len: int,
    repeat_n: int,
    wm_config: WatermarkConfig,
    comp_cfg: dict,
):
    signal = get_state_tensor(model, embedding_layer)

    if scheme == "hc_cdct":
        result = extract_hc_cdct(
            pruned_signal=signal,
            extractor=extract,
            water=water,
            water_rep=water_rep,
            wm_len=wm_len,
            repeat_n=repeat_n,
            config=wm_config,
            beta=float(comp_cfg.get("beta", 0.5)),
            eps=float(comp_cfg.get("eps", 1.0e-12)),
        )
        return {
            "ber_rep_coeff": result["ber_rep_coeff"],
            "ber_vote": result["ber_vote"],
            "ber_each_coeff": result["ber_each_coeff"],
            "comp_value": result.get("comp_value", ""),
        }

    if scheme == "hc_space":
        result = extract_hc_space(
            pruned_signal=signal,
            extractor_mme=extract_MME,
            water=water,
            water_rep=water_rep,
            wm_len=wm_len,
            repeat_n=repeat_n,
            config=wm_config,
        )
        return {
            "ber_rep_coeff": result["ber_rep_coeff"],
            "ber_vote": result["ber_vote"],
            "ber_each_coeff": result["ber_each_coeff"],
            "comp_value": "",
        }

    raise ValueError(f"Unsupported watermarking scheme: {scheme!r}")


def _embed_scheme(
    scheme: str,
    model,
    host_signal: torch.Tensor,
    embedding_layer: str,
    water_amplitude,
    emb,
    emb_MME,
    wm_config: WatermarkConfig,
    device: str,
):
    attacked_model = copy.deepcopy(model)

    if scheme == "hc_cdct":
        embedded = embed_hc_cdct(
            host_signal=host_signal,
            water_amplitude=water_amplitude,
            emb_func=emb,
            config=wm_config,
            device=device,
        )
    elif scheme == "hc_space":
        embedded = embed_hc_space(
            host_signal=host_signal,
            water_amplitude=water_amplitude,
            emb_mme_func=emb_MME,
            config=wm_config,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported watermarking scheme: {scheme!r}")

    copy_state_tensor(attacked_model, embedding_layer, embedded)
    distortion_var = torch.var(embedded - host_signal).detach().cpu().item()
    return attacked_model, distortion_var


def run_finetuning_attack_experiment(config_path: str | Path):
    """
    Run fine-tuning robustness tests for HC-CDCT and the spatial HC baseline.

    The tested attack is ordinary supervised fine-tuning on the training split.
    For each learning rate and each watermarking scheme, the function embeds the
    watermark into a fresh copy of the checkpointed model, fine-tunes it for the
    configured number of epochs, and extracts the watermark after each epoch.
    """

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
    comp_cfg = cfg.get("compensation", {})
    eval_cfg = cfg.get("evaluation", {})
    attack_cfg = cfg.get("finetuning_attack", {})

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

    schemes = attack_cfg.get("schemes", ["hc_cdct", "hc_space"])
    learning_rates = attack_cfg.get("learning_rates", [1.0e-3])
    epochs = int(attack_cfg.get("epochs", 5))
    evaluate_each_epoch = bool(attack_cfg.get("evaluate_each_epoch", True))
    save_csv = bool(attack_cfg.get("save_csv", True))
    result_csv = attack_cfg.get("result_csv", "results/tables/finetuning_results.csv")

    print(f"Using device: {device}")
    print(f"Embedding layer: {embedding_layer}, shape={tuple(host_signal.shape)}")
    print(f"Fine-tuning epochs: {epochs}")
    print(f"Fine-tuning learning rates: {learning_rates}")

    clean_eval = None
    if bool(eval_cfg.get("compute_accuracy", False)) or evaluate_each_epoch:
        clean_eval = evaluate(model, testloader, criterion, device=device)
        print(f"Clean checkpoint accuracy: {clean_eval['accuracy']:.4f}")

    rows = []

    for lr in learning_rates:
        lr = float(lr)
        for scheme in schemes:
            print(f"\nScheme: {scheme}; fine-tuning lr={lr:g}")

            attacked_model, distortion_var = _embed_scheme(
                scheme=scheme,
                model=model,
                host_signal=host_signal,
                embedding_layer=embedding_layer,
                water_amplitude=water_amplitude,
                emb=emb,
                emb_MME=emb_MME,
                wm_config=wm_config,
                device=device,
            )

            initial_extract = _extract_scheme(
                scheme=scheme,
                model=attacked_model,
                embedding_layer=embedding_layer,
                extract=extract,
                extract_MME=extract_MME,
                water=water,
                water_rep=water_rep,
                wm_len=wm_len,
                repeat_n=repeat_n,
                wm_config=wm_config,
                comp_cfg=comp_cfg,
            )

            initial_eval = None
            if bool(eval_cfg.get("compute_accuracy", False)) or evaluate_each_epoch:
                initial_eval = evaluate(attacked_model, testloader, criterion, device=device)

            rows.append(
                {
                    "scheme": scheme,
                    "learning_rate": lr,
                    "optimizer": attack_cfg.get("optimizer", "SGD"),
                    "epoch": 0,
                    "train_loss": "",
                    "train_accuracy": "",
                    "test_loss": initial_eval["loss"] if initial_eval else "",
                    "test_accuracy": initial_eval["accuracy"] if initial_eval else "",
                    "clean_accuracy": clean_eval["accuracy"] if clean_eval else "",
                    "wm_len": wm_len,
                    "repeat_n": repeat_n,
                    "ber_rep_coeff": initial_extract["ber_rep_coeff"],
                    "ber_vote": initial_extract["ber_vote"],
                    "ber_each_coeff": _serialize_vector(initial_extract["ber_each_coeff"]),
                    "distortion_var": distortion_var,
                    "comp_value": initial_extract["comp_value"],
                }
            )
            print(f"Epoch 0 BER: {initial_extract['ber_vote']:.6f}")

            optimizer = _build_optimizer(attacked_model, attack_cfg, lr=lr)

            for epoch in range(1, epochs + 1):
                train_metrics = train_one_epoch(
                    attacked_model,
                    trainloader,
                    criterion,
                    optimizer,
                    device=device,
                    epoch=epoch,
                )

                eval_metrics = None
                if evaluate_each_epoch or bool(eval_cfg.get("compute_accuracy", False)):
                    eval_metrics = evaluate(attacked_model, testloader, criterion, device=device)

                extracted = _extract_scheme(
                    scheme=scheme,
                    model=attacked_model,
                    embedding_layer=embedding_layer,
                    extract=extract,
                    extract_MME=extract_MME,
                    water=water,
                    water_rep=water_rep,
                    wm_len=wm_len,
                    repeat_n=repeat_n,
                    wm_config=wm_config,
                    comp_cfg=comp_cfg,
                )

                row = {
                    "scheme": scheme,
                    "learning_rate": lr,
                    "optimizer": attack_cfg.get("optimizer", "SGD"),
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "test_loss": eval_metrics["loss"] if eval_metrics else "",
                    "test_accuracy": eval_metrics["accuracy"] if eval_metrics else "",
                    "clean_accuracy": clean_eval["accuracy"] if clean_eval else "",
                    "wm_len": wm_len,
                    "repeat_n": repeat_n,
                    "ber_rep_coeff": extracted["ber_rep_coeff"],
                    "ber_vote": extracted["ber_vote"],
                    "ber_each_coeff": _serialize_vector(extracted["ber_each_coeff"]),
                    "distortion_var": distortion_var,
                    "comp_value": extracted["comp_value"],
                }
                rows.append(row)

                acc_text = (
                    f", test_acc={eval_metrics['accuracy']:.4f}"
                    if eval_metrics is not None
                    else ""
                )
                print(
                    f"Epoch {epoch}: train_acc={train_metrics['accuracy']:.4f}"
                    f"{acc_text}, BER={extracted['ber_vote']:.6f}"
                )

    if save_csv:
        _ensure_parent(result_csv)
        fieldnames = list(rows[0].keys()) if rows else []
        with open(result_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved fine-tuning attack results to: {result_csv}")

    return rows
