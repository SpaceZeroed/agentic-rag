# FlashAttention

FlashAttention computes exact attention using tiling to reduce transfers between GPU memory levels.

The algorithm reorganizes attention around blocks that fit in fast on-chip memory. It avoids materializing the full attention matrix in high-bandwidth memory. It is an IO-aware computation algorithm, not a method for allocating a serving system's persistent KV cache.

Primary source: https://arxiv.org/abs/2205.14135
