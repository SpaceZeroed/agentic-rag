# Retrieval-Augmented Generation

RAG combines a parametric language model with non-parametric memory accessed through retrieval.

A retriever selects passages from an external corpus and a generator conditions its output on retrieved content. Updating the external index changes available evidence without requiring the same update to be stored in generator weights. Retrieval provides evidence but does not guarantee that generated claims are correct.

Primary source: https://arxiv.org/abs/2005.11401
