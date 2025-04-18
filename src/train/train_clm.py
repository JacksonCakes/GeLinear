from src.models.model_config import get_model_config
from src.dataloaders.clm_loader import load_data
from src.utils import load_model_and_tokenizer, switch_attn, init_linear_weights

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


import argparse
import os
import logging
import yaml
from tqdm import tqdm

import torch

from datasets import load_dataset, disable_caching

disable_caching()


torch.backends.cuda.matmul.allow_tf32 = True


def evaluate(
    accelerator,
    model,
    classifier,
    val_loader,
):
    total_loss = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc="Evaluating", disable=not accelerator.is_main_process
        ):
            if torch.all(batch["labels"] == -100):
                continue

            loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
            # embeddings = model.compute_embedding(batch["input_ids"])

            # loss = linear_cross_entropy(
            #     embeddings, classifier, batch["labels"], shift=True
            # )

            gathered_loss = accelerator.gather(loss)
            total_loss += gathered_loss.sum().item()
            num_batches += len(gathered_loss)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    torch.cuda.empty_cache()
    return avg_loss


def save_checkpoint(accelerator, step, model, directory, filename_prefix, **kwargs):
    if not accelerator.is_main_process:
        return

    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, f"{filename_prefix}_step_{step}.pt")

    # Unwrap the model to get its original state_dict
    unwrapped_model = accelerator.unwrap_model(model)
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
    train_loader,
    val_loader,
    config_params,
):
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config_params["training"]["lr"]),
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader)
    )
    # criterion_xent = torch.nn.CrossEntropyLoss(reduction="mean")
    num_epochs = config_params["training"]["num_epochs"]
    save_interval = config_params["training"]["save_interval"]
    eval_interval = config_params["training"]["eval_interval"]

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # count num of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_millions = num_trainable_params / 1e6
    if accelerator.is_main_process:
        print(f"Number of trainable parameters: {params_in_millions:.2f}M")

    step = 0
    lowest_val_loss = float("inf")
    total_train_loss = 0.0
    for epoch in range(num_epochs):
        with tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        ) as pbar:
            # with profile(
            #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #     record_shapes=True,       # record input shapes
            #     profile_memory=True       # record memory usage
            #     ) as prof:
            for batch in pbar:
                step += 1
                if torch.all(batch["labels"] == -100):
                    continue
                optimizer.zero_grad()
                # torch.cuda.memory._record_memory_history(enabled='all')
                # print(f"input shape: {batch['input_ids'].shape}")
                loss = model(
                    batch["input_ids"], labels=batch["labels"], compute_loss=True
                )["loss"]
                # shifted_labels = batch["labels"][:, 1:].contiguous()

                # shifted_logits = logits[:, :-1, :].contiguous()

                # loss = criterion_xent(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                # prof.export_chrome_trace("trace.json")
                # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
                # s = torch.cuda.memory._snapshot()
                # torch.cuda.memory._dump_snapshot(f"snapshot.pickle")
                # torch.cuda.memory._record_memory_history(enabled=None)

                total_train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

                if step % eval_interval == 0:
                    val_loss = evaluate(
                        accelerator,
                        model,
                        val_loader,
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
                                model,
                                config_params["model"]["checkpoint_dir"],
                                "best_model",
                                val_loss=val_loss,
                            )
                            logging.info(f"New best validation loss: {val_loss:.3f}")

                if step % save_interval == 0:
                    torch.cuda.empty_cache()
                    avg_train_loss_so_far = total_train_loss / step
                    if accelerator.is_main_process:
                        save_checkpoint(
                            accelerator,
                            step,
                            model,
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
    checkpoint_path = config_params["model"]["model_path"]
    dataset_path = config_params["data"]["dataset_path"]
    dataset_split = config_params["data"]["subset"]
    split_ratio = config_params["data"]["split_ratio"]
    prev_feature_maps_checkpoint_dir = config_params["model"][
        "prev_feature_maps_checkpoint_dir"
    ]

    feature_maps_state_dict = torch.load(
        prev_feature_maps_checkpoint_dir,
        weights_only=True,
        map_location=accelerator.device,
    )
    model, tokenizer = load_model_and_tokenizer(
        model_path=checkpoint_path, model_dtype=model_dtype
    )

    # freeze teacher model
    for name, param in model.named_parameters():
        param.requires_grad = False

    model = switch_attn(
        model,
        model_config=model_config,
        layers_to_switch=model_config.num_hidden_layers,
    )
    model = init_linear_weights(model, feature_maps_state_dict)
    model.set_start_checkpoint_idx()
    # for debugging
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: requires_grad = {param.requires_grad}")
            print(param.dtype)

    dataset = load_dataset(dataset_path, split=dataset_split)
    train_val_split = dataset.train_test_split(test_size=split_ratio, seed=42)

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
        train_loader=train_loader,
        val_loader=val_loader,
        config_params=config_params,
    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
