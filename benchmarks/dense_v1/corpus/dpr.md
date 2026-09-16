# Dense Passage Retrieval

DPR encodes questions and passages separately and ranks passages by the dot product of their vectors.

Passage representations can be computed before a query arrives. Training pulls matching question-passage pairs together relative to negatives. A single vector per passage makes retrieval efficient but compresses token-level detail. This is a bi-encoder architecture for open-domain question answering.

Primary source: https://arxiv.org/abs/2004.04906
