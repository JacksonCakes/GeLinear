from functools import partial
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader


def template_and_tokenize(sample, tokenizer, max_length=1024):
    """
    Format dataset context and answers into single-sequence prompts
    """
    messages = sample["messages"]
    prompt = messages[0]["content"]
    response = messages[-1]["content"]
    prompt = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    input_ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, truncation=True, max_length=max_length
    )
    attn_mask = [1] * len(input_ids)
    sample = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,  # placeholder only for now
        "labels": [-100] * len(prompt),  # placeholder only for now
    }
    return sample


def load_data(tokenizer, train_set, val_set, remove_columns, batch_size=1):
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
        tokenizer, label_pad_token_id=-100, return_tensors="pt", padding=False
    )
    train_data = DataLoader(
        train_set, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    val_data = DataLoader(
        val_set, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    return {"train_data": train_data, "val_data": val_data}
