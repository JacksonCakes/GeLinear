from functools import partial
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader


def template_and_tokenize(sample, tokenizer, max_length=1024):
    """
    Format dataset context and answers into single-sequence prompts
    """
    messages = sample["messages"]
    input_ids = []
    labels = []
    for msg in messages:
        if msg["role"] == "user":
            cur_prompt = [{"role": "user", "content": msg["content"]}]
            cur_input_ids = tokenizer.apply_chat_template(
                cur_prompt, tokenize=True, add_generation_prompt=True
            )
            if input_ids:
                cur_input_ids = cur_input_ids[1:]
            labels += [-100] * len(cur_input_ids)
        else:
            cur_input_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
            cur_input_ids += [107, 108]  # add eos token, hard code for now
            labels += cur_input_ids

        input_ids += cur_input_ids

    attn_mask = [1] * len(input_ids)
    fill_len = max_length - len(input_ids)
    input_ids = [tokenizer.pad_token_type_id] * fill_len + input_ids
    labels = [-100] * fill_len + labels
    sample = {
        "input_ids": input_ids[:max_length],
        "attention_mask": attn_mask,
        "labels": labels[:max_length],
    }
    return sample


def load_data(
    tokenizer, train_set, val_set, remove_columns, batch_size=1, max_length=100
):
    train_set = train_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer),
        remove_columns=remove_columns,
        num_proc=16,
    )
    val_set = val_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer),
        remove_columns=remove_columns,
        num_proc=16,
    )
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    )
    train_data = DataLoader(
        train_set, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    val_data = DataLoader(
        val_set, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    return {"train_data": train_data, "val_data": val_data}
