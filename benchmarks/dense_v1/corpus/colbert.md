# ColBERT

ColBERT keeps token vectors and scores query-document matches with late interaction.

For each query token, MaxSim selects the largest similarity to a document token; the scores are summed. Document token representations can be precomputed. Keeping multiple vectors preserves fine-grained matching information but increases index storage relative to a single-vector bi-encoder.

Primary source: https://arxiv.org/abs/2004.12832
