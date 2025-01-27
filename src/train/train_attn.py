import argparse
import os
import logging
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn

from datasets import load_dataset
from transformers import AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from src.models.model_config import get_model_config
from src.models.model import GemmaForCausalLM
from src.dataloaders.loader import load_data
from src.models.linear.linear_attention import SubGemmaModel
from src.train.attention_hooks import AttentionHook


def evaluate(
    accelerator,
    model,
    feature_map_model,
    val_loader,
    mse_loss,
    layer_idx,
    mse_factor=1000,
    attention_hook=None,
):
    feature_map_model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc="Evaluating", disable=not accelerator.is_main_process
        ):
            outputs = model(batch["input_ids"], train_attn=True)
            outputs = outputs["all_outputs"]

            for _, tensors in attention_hook.captured_tensors.items():
                hs = tensors["hidden_states"]

            outputs_pred = feature_map_model(hs)["all_outputs"]

            teacher_attn_scores = [
                item.detach()
                for idx, item in enumerate(outputs["attn_scores"])
                if idx in layer_idx
            ]
            student_attn_scores = outputs_pred["attn_scores_pred"]

            assert (
                len(teacher_attn_scores) == len(student_attn_scores)
            ), "Inconsistent number of attention score tensors between teacher and student."

            batch_loss = 0.0
            for t_attn_scores, s_attn_scores in zip(
                teacher_attn_scores, student_attn_scores
            ):
                layer_loss = mse_loss(s_attn_scores, t_attn_scores)
                batch_loss += layer_loss
                del t_attn_scores, s_attn_scores

            batch_loss = (batch_loss / len(teacher_attn_scores)) * mse_factor
            del hs, outputs, outputs_pred, teacher_attn_scores, student_attn_scores
            torch.cuda.empty_cache()
            total_loss += accelerator.gather(batch_loss).sum().item()
            num_batches += 1
    total_loss = total_loss / num_batches if num_batches > 0 else 0.0
    feature_map_model.train()
    return total_loss


def save_checkpoint(
    accelerator, step, feature_map_model, directory, filename_prefix, **kwargs
):
    if not accelerator.is_main_process:
        return

    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, f"{filename_prefix}_step_{step}.pt")

    # Unwrap the feature_map_model to get its original state_dict
    unwrapped_model = accelerator.unwrap_model(feature_map_model)
    filtered_state_dict = {
        k: v for k, v in unwrapped_model.state_dict().items() if "feature_map" in k
    }

    checkpoint = {
        "step": step,
        "feature_map_model_state_dict": filtered_state_dict,
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved at step {step} to {save_path}")


def train(
    accelerator,
    model,
    feature_map_model,
    train_loader,
    val_loader,
    layer_idx,
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

    mse_loss = nn.MSELoss(reduction="mean")
    num_epochs = config_params["training"]["num_epochs"]
    save_interval = config_params["training"]["save_interval"]
    eval_interval = config_params["training"]["eval_interval"]

    feature_map_model, optimizer, train_loader, val_loader, lr_scheduler = (
        accelerator.prepare(
            feature_map_model, optimizer, train_loader, val_loader, lr_scheduler
        )
    )

    # count num of trainable parameters
    num_trainable_params = sum(
        p.numel() for p in feature_map_model.parameters() if p.requires_grad
    )
    params_in_millions = num_trainable_params / 1e6
    if accelerator.is_main_process:
        print(f"Number of trainable parameters: {params_in_millions:.2f}M")

    step = 0
    lowest_val_loss = float("inf")
    n_batches_per_step = 64
    total_train_loss = 0.0

    for epoch in range(num_epochs):
        feature_map_model.train()
        with tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        ) as pbar:
            for batch in pbar:
                step += 1
                optimizer.zero_grad()
                with torch.no_grad():
                    outputs = model(batch["input_ids"], train_attn=True)
                outputs = outputs["all_outputs"]

                # get hidden states from the hooked layer
                for _, tensors in attention_hook.captured_tensors.items():
                    hs = tensors["hidden_states"]

                outputs_pred = feature_map_model(hs)["all_outputs"]

                teacher_attn_scores = [
                    item.detach()
                    for idx, item in enumerate(outputs["attn_scores"])
                    if idx in layer_idx
                ]
                student_attn_scores = outputs_pred["attn_scores_pred"]

                loss = 0.0
                for t_attn_scores, s_attn_scores in zip(
                    teacher_attn_scores, student_attn_scores
                ):
                    loss += mse_loss(s_attn_scores, t_attn_scores)

                loss = (loss / len(teacher_attn_scores)) * mse_factor

                accelerator.backward(loss)
                optimizer.step()

                if step % 100 == 0:
                    torch.cuda.empty_cache()

                # step scheduler every n_batches_per_step
                if step % n_batches_per_step == 0:
                    lr_scheduler.step()

                total_train_loss += loss.item()
                del hs, outputs, outputs_pred, teacher_attn_scores, student_attn_scores
                pbar.set_postfix({"loss": loss.item()})

                if step % eval_interval == 0:
                    val_loss = evaluate(
                        accelerator,
                        model,
                        feature_map_model,
                        val_loader,
                        mse_loss,
                        layer_idx=layer_idx,
                        attention_hook=attention_hook,
                        mse_factor=mse_factor,
                    )

                    if accelerator.is_main_process:
                        logging.info(f"Step {step} - Validation Loss: {val_loss:.3f}")
                        accelerator.log({"valid_loss": val_loss}, step=step)
                        # If best, save checkpoint
                        if val_loss < lowest_val_loss:
                            lowest_val_loss = val_loss
                            save_checkpoint(
                                accelerator,
                                step,
                                feature_map_model,
                                config_params["model"]["checkpoint_dir"],
                                "best_model",
                                val_loss=val_loss,
                            )
                            logging.info(f"New best validation loss: {val_loss:.3f}")

                if step % save_interval == 0:
                    avg_train_loss_so_far = total_train_loss / step
                    if accelerator.is_main_process:
                        save_checkpoint(
                            accelerator,
                            step,
                            feature_map_model,
                            config_params["model"]["checkpoint_dir"],
                            "checkpoint",
                            train_loss=avg_train_loss_so_far,
                        )
                        lr = lr_scheduler.get_last_lr()[0]
                        logging.info(
                            f"Step {step} - Training Loss: {avg_train_loss_so_far:.3f} - LR: {lr:.6f}"
                        )
                        accelerator.log(
                            {"train_loss": avg_train_loss_so_far, "lr": lr}, step=step
                        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_params = yaml.safe_load(f)
    project_config = ProjectConfiguration(
        project_dir=config_params["logging"]["log_dir"],
        logging_dir=config_params["logging"]["log_dir"],
    )
    accelerator = Accelerator(log_with="mlflow", project_config=project_config)
    accelerator.init_trackers("my_project", config=config_params)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    model_config = get_model_config()
    model_dtype = model_config.get_dtype()
    layer_idx = config_params["model"]["distill_layer_idx"]
    checkpoint_path = config_params["model"]["model_path"]
    dataset_path = config_params["data"]["dataset_path"]
    dataset_split = config_params["data"]["subset"]
    split_ratio = config_params["data"]["split_ratio"]
    tokenizer_path = config_params["model"]["tokenizer_path"]
    model = GemmaForCausalLM.from_pretrained(
        config=model_config,
        checkpoint_path=checkpoint_path,
        device_map="sequential",
        dtype=model_dtype,
        strict=False,
    )
    model.eval()

    # freeze teacher model
    for name, param in model.named_parameters():
        param.requires_grad = False

    feature_map_model = SubGemmaModel(
        original_model=model.model,
        rope_embeddings=model.rope_embeddings,
        layer_idx=layer_idx,
    )
    feature_map_model = feature_map_model.to(dtype=model_dtype)

    # for debugging
    for name, param in feature_map_model.named_parameters():
        if param.requires_grad:
            print(f"{name}: requires_grad = {param.requires_grad}")

    attention_hook = AttentionHook()
    # hook the previous layer output before the distillation layer
    # eg. if layer_idx = [3], then we hook the output of layer 2
    layers_to_hook = [layer_idx[0] - 1]
    attention_hook.register_hooks(model, layers_to_hook)

    dataset = load_dataset(dataset_path, split=dataset_split)
    train_val_split = dataset.train_test_split(test_size=split_ratio, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
        accelerator=accelerator,
        model=model,
        feature_map_model=feature_map_model,
        train_loader=train_loader,
        val_loader=val_loader,
        layer_idx=layer_idx,
        config_params=config_params,
        attention_hook=attention_hook,
    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
