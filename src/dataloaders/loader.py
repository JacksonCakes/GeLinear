from functools import partial
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader


def template_and_tokenize(sample, tokenizer, test):
    """
    Format dataset context and answers into single-sequence prompts
    """
    messages = sample["messages"]
    prompt = messages[0]["content"]
    response = messages[-1]["content"]
    prompt = tokenizer.encode(
        prompt, add_special_tokens=True, max_length=512, truncation=True
    )
    answer = tokenizer.encode(
        f"{response}{tokenizer.eos_token}",
        add_special_tokens=False,
        max_length=512,
        truncation=True,
    )
    input_ids = prompt + answer
    attn_mask = [1] * len(input_ids)
    sample = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": [-100] * len(prompt) + answer,
    }
    return sample


def load_data(tokenizer, train_set, val_set, remove_columns):
    train_set = train_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, test=True),
        remove_columns=remove_columns,
        num_proc=16,
    )
    val_set = val_set.map(
        partial(template_and_tokenize, tokenizer=tokenizer, test=True),
        remove_columns=remove_columns,
        num_proc=16,
    )
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, return_tensors="pt", padding=False
    )
    train_data = DataLoader(
        train_set, shuffle=False, collate_fn=collate_fn, batch_size=1
    )
    val_data = DataLoader(val_set, shuffle=False, collate_fn=collate_fn, batch_size=1)
    return {"train_data": train_data, "val_data": val_data}
