import argparse
import os
import logging
import contextlib
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import AutoTokenizer
from safetensors.torch import load_file
from tqdm import tqdm

from src.models.model_config import get_model_config
from src.models.model import GemmaForCausalLM
from src.dataloaders.loader import load_data
from src.models.linear.feature_map import TrainableHedgehog


@contextlib.contextmanager
def _set_default_dtype(dtype: torch.dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def load_full_state_dict(shards, pattern):
    state_dict = {}
    for shard in shards:
        shard_file = pattern.format(shard_num=str(shard))
        state_dict.update(load_file(shard_file))
    return state_dict


def evaluate(model, feature_map_model, val_loader, device, mse_loss, layer_slices):
    model.eval()
    feature_map_model.eval()
    total_loss = 0.0

    # unpack layer slicing configuration
    # outer slice will take the odd layer (zero-indexed), since gemma is hybrid attention
    # alternate between sliding window and global attention
    # inner is the layer that we want to perform distillation
    outer_slice = slice(*layer_slices["outer_slice"])
    inner_slice = slice(*layer_slices["inner_slice"])

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(input_ids, output_attn=True)
            outputs = outputs["all_outputs"]
            outputs = {k: [v_i.detach() for v_i in v] for k, v in outputs.items()}
            zipped_outputs = zip(
                outputs["all_query"][outer_slice][inner_slice],
                outputs["all_key"][outer_slice][inner_slice],
                outputs["all_attn"][outer_slice][inner_slice],
                outputs["all_mask"][outer_slice][inner_slice],
            )

            for idx, (q, k, a_true, mask) in enumerate(zipped_outputs):
                a_pred = feature_map_model(q=q, k=k, layer_idx=idx)
                m, n = a_pred.shape[-2:]
                causal_mask = torch.ones((m, n), device=device, dtype=torch.bool).triu(
                    n - m + 1
                )
                a_pred = a_pred.masked_fill(causal_mask, 0)
                total_loss += mse_loss(a_pred, a_true).item()
                del q, k, a_true, mask, a_pred

            del outputs, zipped_outputs
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def save_checkpoint(step, feature_map_model, directory, filename_prefix, **kwargs):
    save_path = os.path.join(directory, f"{filename_prefix}_step_{step}.pt")
    os.makedirs(directory,exist_ok=True)
    checkpoint = {
        "step": step,
        "feature_map_model_state_dict": feature_map_model.state_dict(),
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved at step {step} to {save_path}")


def train(model, feature_map_model, train_loader, val_loader, device, config_params):
    optimizer = torch.optim.AdamW(
        [p for p in feature_map_model.parameters() if p.requires_grad],
        lr=float(config_params["training"]["lr"]),
    )
    mse_loss = nn.MSELoss()

    num_epochs = config_params["training"]["num_epochs"]
    save_interval = config_params["training"]["save_interval"]
    eval_interval = config_params["training"]["eval_interval"]

    outer_slice = slice(*config_params["training"]["layer_slice"]["outer_slice"])
    inner_slice = slice(*config_params["training"]["layer_slice"]["inner_slice"])
    layer_slices = {
        "outer_slice": config_params["training"]["layer_slice"]["outer_slice"],
        "inner_slice": config_params["training"]["layer_slice"]["inner_slice"],
    }

    step = 0
    lowest_val_loss = float("inf")

    for epoch in range(num_epochs):
        feature_map_model.train()
        total_train_loss = 0.0

        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch"
        ):
            step += 1
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, output_attn=True)
            outputs = outputs["all_outputs"]
            outputs = {k: [v_i.detach() for v_i in v] for k, v in outputs.items()}

            optimizer.zero_grad()

            zipped_outputs = zip(
                outputs["all_query"][outer_slice][inner_slice],
                outputs["all_key"][outer_slice][inner_slice],
                outputs["all_attn"][outer_slice][inner_slice],
                outputs["all_mask"][outer_slice][inner_slice],
            )
            
            causal_mask = None
            total_loss = 0.0
            for layer_idx, (q, k, a_true, mask) in enumerate(zipped_outputs):
                a_pred = feature_map_model(q=q, k=k, layer_idx=layer_idx)
                m, n = a_pred.shape[-2:]
                if causal_mask is None or causal_mask.shape != (m, n):
                    causal_mask = torch.ones((m, n), device=device, dtype=torch.bool).triu(n - m + 1)       
                a_pred = a_pred.masked_fill(causal_mask, 0)
                layer_loss = mse_loss(a_pred, a_true)
                total_loss += layer_loss
                del q, k, a_true, mask, a_pred

            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            del outputs, zipped_outputs, total_loss
            
            if step % 100 == 0:
                torch.cuda.empty_cache()
            if step % eval_interval == 0:
                val_loss = evaluate(
                    model, feature_map_model, val_loader, device, mse_loss, layer_slices
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
                avg_train_loss_so_far = total_train_loss / (
                    (step % len(train_loader)) or len(train_loader)
                )
                save_checkpoint(
                    step,
                    feature_map_model,
                    config_params["model"]["checkpoint_dir"],
                    "checkpoint",
                    train_loss=avg_train_loss_so_far,
                )

        avg_train_loss = total_train_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.3f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

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

    model_config = get_model_config()
    state_dict = load_full_state_dict(
        config_params["model"]["shards"], config_params["model"]["file_pattern"]
    )

    with _set_default_dtype(model_config.get_dtype()):
        model = GemmaForCausalLM(model_config)
        model.load_weights(state_dict)
        model = model.to(config_params["model"]["device"]).eval()

        feature_map_model = TrainableHedgehog(
            num_feature_maps=config_params["training"]["feature_maps"],
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.head_dim,
            feature_dim=config_params["training"]["feature_dim"],
        ).to(config_params["model"]["device"])

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
        device=config_params["model"]["device"],
        config_params=config_params,
    )
