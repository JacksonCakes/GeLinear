import argparse
import os
import logging
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.models.model_config import get_model_config
from src.models.model import GemmaForCausalLM
from src.dataloaders.loader import load_data
from src.models.linear.linear_attention import SubGemmaModel
from src.train.attention_hooks import AttentionHook


def evaluate(
    model,
    feature_map_model,
    val_loader,
    device,
    mse_loss,
    mse_factor=1000,
    attention_hook=None,
):
    feature_map_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(input_ids, train_attn=True)
            outputs = outputs["all_outputs"]

            for layer_name, tensors in attention_hook.captured_tensors.items():
                hs = tensors["hidden_states"]
            batch_loss = 0.0
            outputs_pred = feature_map_model(hs)["all_outputs"]
            outputs = [
                item.detach()
                for idx, item in enumerate(outputs["attn_scores"])
                if idx in layer_idx
            ]
            assert len(outputs) == len(
                outputs_pred["attn_scores_pred"]
            ), "Inconsistent length between teacher and student"
            total_loss = 0.0
            zipped_outputs = list(zip(outputs, outputs_pred["attn_scores_pred"]))
            n_layers = len(zipped_outputs)
            for attn_scores, attn_scores_pred in zipped_outputs:
                layer_loss = mse_loss(attn_scores_pred, attn_scores)
                batch_loss += layer_loss
                del attn_scores, attn_scores_pred
            batch_loss = batch_loss / n_layers * mse_factor
            total_loss += batch_loss
            batch_loss = 0.0

            del hs, outputs, outputs_pred, zipped_outputs
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def save_checkpoint(step, feature_map_model, directory, filename_prefix, **kwargs):
    save_path = os.path.join(directory, f"{filename_prefix}_step_{step}.pt")
    os.makedirs(directory, exist_ok=True)
    filtered_state_dict = {
        k: v for k, v in feature_map_model.state_dict().items() if "feature_map" in k
    }
    checkpoint = {
        "step": step,
        "feature_map_model_state_dict": filtered_state_dict,
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved at step {step} to {save_path}")


def train(
    model,
    feature_map_model,
    train_loader,
    val_loader,
    layer_idx,
    device,
    config_params,
    attention_hook=None,
    mse_factor=1000,
):
    optimizer = torch.optim.AdamW(
        [p for p in feature_map_model.parameters() if p.requires_grad],
        lr=float(config_params["training"]["lr"]),
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader)
    )

    num_trainable_params = sum(
        p.numel() for p in feature_map_model.parameters() if p.requires_grad
    )
    params_in_millions = num_trainable_params / 1e6

    print(f"Number of trainable parameters: {params_in_millions:.2f}M")

    mse_loss = nn.MSELoss(reduction="mean")

    num_epochs = config_params["training"]["num_epochs"]
    save_interval = config_params["training"]["save_interval"]
    eval_interval = config_params["training"]["eval_interval"]

    step = 0
    lowest_val_loss = float("inf")
    n_batches_per_step = 64

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        # torch.cuda.memory._record_memory_history(enabled='all')
        # with profile(
        # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes=True,       # record input shapes
        # profile_memory=True       # record memory usage
        # ) as prof:
        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch"
        ):
            optimizer.zero_grad()
            step += 1
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(input_ids, train_attn=True)
            outputs = outputs["all_outputs"]

            for layer_name, tensors in attention_hook.captured_tensors.items():
                hs = tensors["hidden_states"]
            outputs_pred = feature_map_model(hs)["all_outputs"]
            outputs = [
                item.detach()
                for idx, item in enumerate(outputs["attn_scores"])
                if idx in layer_idx
            ]
            assert len(outputs) == len(
                outputs_pred["attn_scores_pred"]
            ), "Inconsistent length between teacher and student"
            total_loss = 0.0
            zipped_outputs = list(zip(outputs, outputs_pred["attn_scores_pred"]))
            n_layers = len(zipped_outputs)
            for attn_scores, attn_scores_pred in zipped_outputs:
                layer_loss = mse_loss(attn_scores_pred, attn_scores)
                total_loss += layer_loss
                del attn_scores, attn_scores_pred
            total_loss = total_loss / n_layers * mse_factor
            total_loss.backward()
            optimizer.step()
            # prof.step()
            total_train_loss += total_loss.item()
            del hs, outputs, outputs_pred, zipped_outputs, total_loss
            if step % n_batches_per_step == 0:
                lr_scheduler.step()
            if step % 100 == 0:
                torch.cuda.empty_cache()
            if step % eval_interval == 0:
                val_loss = evaluate(
                    model,
                    feature_map_model,
                    val_loader,
                    device,
                    mse_loss,
                    attention_hook=attention_hook,
                )
                logging.info(f"Step {step} - Validation Loss: {val_loss:.3f}")
                writer.add_scalar("Loss/Validation", val_loss, step)

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    save_checkpoint(
                        step,
                        feature_map_model,
                        config_params["model"]["checkpoint_dir"],
                        "best_model",
                        val_loss=val_loss,
                    )
                    logging.info(f"New best validation loss: {val_loss:.3f}")

            if step % save_interval == 0:
                avg_train_loss_so_far = total_train_loss / step
                save_checkpoint(
                    step,
                    feature_map_model,
                    config_params["model"]["checkpoint_dir"],
                    "checkpoint",
                    train_loss=avg_train_loss_so_far,
                )

                logging.info(
                    f"Step {step+1} - Training Loss: {avg_train_loss_so_far:.3f} - LR: {lr_scheduler.get_last_lr()[0]:.6f}"
                )
                writer.add_scalar("Loss/Train", avg_train_loss_so_far, step + 1)
                writer.add_scalar("LR", lr_scheduler.get_last_lr()[0], step + 1)
    # prof.export_chrome_trace("trace.json")
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    # s = torch.cuda.memory._snapshot()
    # from pickle import dump
    # with open(f"snapshot.pickle", "wb") as f:
    #     dump(s, f)
    # torch.cuda.memory._record_memory_history(enabled=None)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_params = yaml.safe_load(f)

    log_level = getattr(
        logging, config_params["logging"]["level"].upper(), logging.INFO
    )
    logging.basicConfig(level=log_level, format="%(asctime)s - %(message)s")

    # disable_caching()

    writer = SummaryWriter(log_dir=config_params["logging"]["log_dir"])
    model_path = config_params["model"]["model_path"]
    model_config = get_model_config()
    model_dtype = model_config.get_dtype()
    layer_idx = config_params["model"]["distill_layer_idx"]
    model = GemmaForCausalLM.from_pretrained(
        config=model_config,
        checkpoint_path=model_path,
        device_map="sequential",
        dtype=model_dtype,
        strict=False,
    )
    model.eval()
    # model = torch.compile(model)
    for name, param in model.named_parameters():
        param.requires_grad = False

    feature_map_model = SubGemmaModel(
        original_model=model.model,
        rope_embeddings=model.rope_embeddings,
        layer_idx=layer_idx,
    ).to(config_params["model"]["device"], dtype=model_dtype)
    # feature_map_model = torch.compile(feature_map_model)
    for name, param in feature_map_model.named_parameters():
        if param.requires_grad:
            print(f"{name}: requires_grad = {param.requires_grad}")
    attention_hook = AttentionHook()
    layers_to_hook = [layer_idx[0] - 1]
    attention_hook.register_hooks(model, layers_to_hook)
    dataset = load_dataset(
        config_params["data"]["dataset_path"], split=config_params["data"]["subset"]
    )
    train_val_split = dataset.train_test_split(
        test_size=config_params["data"]["split_ratio"], seed=42
    )

    tokenizer = AutoTokenizer.from_pretrained(config_params["data"]["tokenizer_path"])

    remove_columns = (
        list(train_val_split["train"].features)
        if config_params["data"].get("remove_columns_from_train", False)
        else None
    )

    data_loaders = load_data(
        tokenizer=tokenizer,
        train_set=train_val_split["train"],
        val_set=train_val_split["test"],
        remove_columns=remove_columns,
    )

    train_loader = data_loaders["train_data"]
    val_loader = data_loaders["val_data"]

    train(
        model=model,
        feature_map_model=feature_map_model,
        train_loader=train_loader,
        val_loader=val_loader,
        layer_idx=layer_idx,
        device=config_params["model"]["device"],
        attention_hook=attention_hook,
        config_params=config_params,
    )
