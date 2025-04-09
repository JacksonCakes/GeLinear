# GeLinear

GeLinear is the implementation of [LoLCATs](https://arxiv.org/pdf/2410.10254), but with Gemma 2 model.
Unlike the original LoLCATs approach that linearizes all attention layers, I focuses on linearizing only the global attention layer, while retaining Gemma 2’s built-in Sliding Window Attention (SWA) for local context (since the time complexity for this already scaled linearly with sequence length). 

You can find the fine-tuned linearized Gemma 2 model [here](https://huggingface.co/jacksonkek/GeLinear)

Currently this repo supports both:

✅ Attention Distillation <br>
✅ Causal Language Modeling Fine-Tuning

To run the model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
model = AutoModelForCausalLM.from_pretrained("jacksonkek/GeLinear",trust_remote_code=True,torch_dtype=torch.bfloat16,device_map="sequential")
tokenizer = AutoTokenizer.from_pretrained("jacksonkek/GeLinear")

x = "tell me a joke"
input_text = [{"role":"user","content":x}]
input_ids = tokenizer.apply_chat_template(input_text, add_generation_prompt=True,return_tensors="pt").to("cuda")
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids, streamer = text_streamer, do_sample=False,max_new_tokens = 8192)

```
