# Speculative Decoding

Speculative decoding lets a small draft model propose tokens that a larger target model verifies.

Several proposed tokens can be checked together. Acceptance and correction sampling preserve the target distribution under the algorithm's assumptions. Speedup depends on agreement and verification cost. This differs from simply accepting all draft tokens, which would change the output distribution.

Primary source: https://arxiv.org/abs/2211.17192
