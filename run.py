# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import re
from transformers import AutoTokenizer
from src.models.model_config import get_model_config
from src.models.model import GemmaForCausalLM
from src.models.linear.linear_attention import GemmaLinearAttention


def extract_layer_idx_from_state_dict(state_dict):
    extracted_numbers = []
    for key in state_dict["feature_map_model_state_dict"].keys():
        matches = re.findall(r"\.(\d+)\.", key)
        if matches:
            extracted_numbers.extend(int(match) for match in matches)
    return extracted_numbers


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
    layer_switch_idx = extract_layer_idx_from_state_dict(feature_maps_state_dict)
    for layer_idx in layer_switch_idx[1::2][10:15]:
        # if layer_idx == 33:
        #     continue
        # if layer_idx == 11 or layer_idx == 15 or layer_idx == 17:
        #     continue
        # if (layer_idx >= 11 and layer_idx <=19) or (layer_idx >= 21 and layer_idx <=29) or layer_idx == 33:
        #     continue
        print(f"Layer idx: {layer_idx}")
        q_proj = model.model.layers[layer_idx].self_attn.q_proj
        k_proj = model.model.layers[layer_idx].self_attn.k_proj
        v_proj = model.model.layers[layer_idx].self_attn.v_proj
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        model.model.layers[layer_idx].self_attn = GemmaLinearAttention(
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
        )
        model.model.layers[layer_idx].self_attn = model.model.layers[
            layer_idx
        ].self_attn.to(device=device, dtype=torch.bfloat16)

        model.model.layers[layer_idx].self_attn.q_proj = q_proj
        model.model.layers[layer_idx].self_attn.k_proj = k_proj
        model.model.layers[layer_idx].self_attn.v_proj = v_proj
        model.model.layers[layer_idx].self_attn.o_proj = o_proj

        model.model.layers[layer_idx].self_attn.feature_map_q.load_state_dict(
            {
                "layer": feature_maps_state_dict["feature_map_model_state_dict"][
                    f"layers.{layer_idx}.self_attn.feature_map_q.layer"
                ]
            }
        )
        model.model.layers[layer_idx].self_attn.feature_map_k.load_state_dict(
            {
                "layer": feature_maps_state_dict["feature_map_model_state_dict"][
                    f"layers.{layer_idx}.self_attn.feature_map_k.layer"
                ]
            }
        )
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
    model_path = "/tmp2/jackson/google-gemma-9b/"
    feature_maps_state_dict = torch.load(
        "/home/jackson/LLM-LinAtt/full_checkpoint.pt",
        weights_only=True,
        map_location=torch.device("cuda"),
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
        print()
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
