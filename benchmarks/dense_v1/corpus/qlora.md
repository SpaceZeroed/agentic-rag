# QLoRA

QLoRA trains low-rank adapters through a frozen base model stored in 4-bit precision.

NormalFloat4 represents normally distributed weights. Double quantization also compresses quantization constants. Paged optimizers help manage memory spikes. The method targets fine-tuning memory; four-bit storage does not mean every training operation uses four-bit arithmetic.

Primary source: https://arxiv.org/abs/2305.14314
