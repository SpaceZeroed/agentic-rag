# Rotary Position Embedding

RoPE encodes positions by rotating query and key vectors so their dot product depends on relative position.

Pairs of vector coordinates are rotated by position-dependent angles. This introduces positional structure into attention scores. Rotation preserves vector norms. Using rotary positions alone does not guarantee reliable extrapolation to arbitrarily longer contexts than those used during training.

Primary source: https://arxiv.org/abs/2104.09864
