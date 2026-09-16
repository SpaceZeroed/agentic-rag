# PagedAttention

PagedAttention stores the KV cache in blocks that need not occupy contiguous physical memory.

A block table maps logical token blocks to physical cache blocks. This reduces wasted reservation and fragmentation when serving requests with different lengths. Blocks can be shared across related sequences. The cache stores attention keys and values, rather than all model weights.

Primary source: https://arxiv.org/abs/2309.06180
