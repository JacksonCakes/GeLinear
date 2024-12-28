import torch
import contextlib
from transformers import AutoTokenizer
from src.models.model_config import get_model_config
from src.models.model import GemmaForCausalLM


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def load_model_and_tokenizer(model_path, feature_maps_state_dict, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_config = get_model_config()

    model = GemmaForCausalLM.from_pretrained(
        config=model_config,
        checkpoint_path=model_path,
        device_map="sequential",
        dtype=torch.bfloat16,
        strict=False,
    )
    # for idx, layer in enumerate(model.model.decoder_layers[1::2]):
    #     if idx < 7 or idx > 9:
    #         continue
    #     layer.attention = GemmaLinearAttention(
    #         hidden_size=model_config.hidden_size,
    #         num_heads=model_config.num_attention_heads,
    #         num_kv_heads=model_config.num_key_value_heads,
    #         attn_logit_softcapping=model_config.attn_logit_softcapping,
    #         head_dim=model_config.head_dim,
    #         attn_type=model_config.attention_type[1],
    #         sliding_window_size=model_config.sliding_window_size,
    #     )
    #     layer.attention.feature_map_q.load_state_dict(
    #         {
    #             "layer": feature_maps_state_dict["feature_map_model_state_dict"][
    #                 f"feature_maps_q.{idx}.layer"
    #             ]
    #         }
    #     )
    #     layer.attention.feature_map_k.load_state_dict(
    #         {
    #             "layer": feature_maps_state_dict["feature_map_model_state_dict"][
    #                 f"feature_maps_k.{idx}.layer"
    #             ]
    #         }
    #     )

    model = model.eval()
    return model, tokenizer


def generate_response(
    model, tokenizer, chat, output_len=8192, device="cuda", stream=False
):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)

    if stream:
        response_stream = model.generate(
            input_token_ids_tensor=input_ids, output_len=output_len
        )
        for token_ids in response_stream:
            decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
            yield decoded_text
    else:
        res = model.generate(input_token_ids_tensor=input_ids, output_len=output_len)
        decoded_texts = [
            tokenizer.decode(token_ids, skip_special_tokens=False).strip()
            for token_ids in res
        ]
        return decoded_texts


if __name__ == "__main__":
    model_path = "/tmp2/share_data/google-gemma-9b-it/"
    feature_maps_state_dict = torch.load(
        "/home/jackson/LLM-LinAtt/full_checkpoint.pt",
        weights_only=True,
    )

    model, tokenizer = load_model_and_tokenizer(model_path, feature_maps_state_dict)

    chat_history = []

    print("Chat with the model! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break
        if user_input.lower() == "clear":
            print("Cleared chat history.")
            chat_history = []
            continue
        chat_history.append({"role": "user", "content": user_input})

        print("Model:", end="", flush=True)
        stream = True
        assistant_response = ""
        if stream:
            for response_chunk in generate_response(
                model, tokenizer, chat_history, stream=True
            ):
                print(response_chunk, end="", flush=True)
                if response_chunk.strip() == "<eos>":
                    continue
                assistant_response += response_chunk

            print()
            chat_history.append({"role": "assistant", "content": assistant_response})
        else:
            responses = list(
                generate_response(model, tokenizer, chat_history, stream=stream)
            )
            print(responses[0])
