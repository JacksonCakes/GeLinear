import torch
import contextlib
from transformers import AutoTokenizer
from src.models.model_config import get_model_config
from safetensors.torch import load_file
from src.models.model import GemmaForCausalLM
from src.models.linear.linear_attention import GemmaLinearAttention


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def load_model_and_tokenizer(model_path, feature_maps_state_dict, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_config = get_model_config()

    state_dict = {}
    for shard_num in ["1", "2", "3", "4"]:
        cur_shard = f"{model_path}model-0000{shard_num}-of-00004.safetensors"
        state_dict.update(load_file(cur_shard))

    with _set_default_tensor_type(model_config.get_dtype()):
        model = GemmaForCausalLM(model_config)
        model.load_weights(state_dict)

        for idx, layer in enumerate(model.model.decoder_layers[1::2]):
            if 5 <= idx < 10:
                layer.attention = GemmaLinearAttention(
                    hidden_size=model_config.hidden_size,
                    num_heads=model_config.num_attention_heads,
                    num_kv_heads=model_config.num_key_value_heads,
                    attn_logit_softcapping=model_config.attn_logit_softcapping,
                    head_dim=model_config.head_dim,
                    attn_type=model_config.attention_type[1],
                    sliding_window_size=model_config.sliding_window_size,
                )
                layer.attention.feature_map_q.load_state_dict(
                    {
                        "layer": feature_maps_state_dict[
                            "feature_map_model_state_dict"
                        ][f"feature_maps_q.{idx-5}.layer"]
                    }
                )
                layer.attention.feature_map_k.load_state_dict(
                    {
                        "layer": feature_maps_state_dict[
                            "feature_map_model_state_dict"
                        ][f"feature_maps_k.{idx-5}.layer"]
                    }
                )

    model = model.to(device).eval()
    return model, tokenizer


def generate_response(model, tokenizer, chat, device="cuda"):
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    input_ids = tokenizer([prompt], return_tensors="pt").to(device)
    res = model.generate(input_token_ids_tensor=input_ids["input_ids"])
    decoded_texts = [
        tokenizer.decode(token_ids, skip_special_tokens=False).strip()
        for token_ids in res
    ]
    return decoded_texts


if __name__ == "__main__":
    model_path = "/tmp2/share_data/google-gemma-9b-it/"
    feature_maps_state_dict = torch.load(
        "/home/jackson/LLM-LinAtt/checkpoints_5to9/checkpoint_step_160000.pt",
        weights_only=True,
    )

    model, tokenizer = load_model_and_tokenizer(model_path, feature_maps_state_dict)

    chat = [{"role": "user", "content": "what is greater? 9.9 or 9.11?"}]

    responses = generate_response(model, tokenizer, chat)
    print(responses)
