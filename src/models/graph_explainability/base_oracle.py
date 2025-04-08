from typing import List, Dict

import torch


class BaseOracle:
    def concept_grounding_from_embeddings(self, embeddings: torch.Tensor, num_terms_to_retrieve: int = 1,
                                          similarity_threshold: float = 0.9999) -> Dict:
        pass

    def concept_grounding_from_words(self, words: List[str], num_terms_to_retrieve: int = 1,
                                     similarity_threshold: float = 0.9999) -> Dict:
        pass
