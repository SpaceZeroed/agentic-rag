# LoRA

LoRA freezes the pretrained weights and trains low-rank matrices for weight updates.

For a weight matrix W, the trainable update is represented as a product BA. The rank controls the number of trainable parameters. Adapter weights can be merged into the base weights for inference. LoRA reduces fine-tuning state; it does not itself quantize the frozen base model.

Primary source: https://arxiv.org/abs/2106.09685
