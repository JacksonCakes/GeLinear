class AttentionHook:
    def __init__(self):
        self.captured_tensors = {}

    def hook_fn(self, module, inputs, outputs):
        layer_name = module.name
        self.captured_tensors[layer_name] = outputs

    def register_hooks(self, model, layers_to_hook):
        for layer_idx in layers_to_hook:
            attention_module = model.model.layers[layer_idx]
            attention_module.name = f"layer_{layer_idx}"
            attention_module.register_forward_hook(self.hook_fn)
