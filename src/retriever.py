"""Historical Resolution Retriever (RAG) for Apple Support Knowledge Base."""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import RAG_TOP_K, SIMILARITY_THRESHOLD
from src.data_loader import load_historical_corpus


class HistoricalResolutionRetriever:
    """
    Retrieves the most semantically relevant historical Apple Support
    customer queries and their corresponding official resolutions.
    """

    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None):
        self.corpus = corpus if corpus is not None else load_historical_corpus()
        self.customer_texts = [item["customer_text"] for item in self.corpus]
        self.brand_replies = [item["brand_reply"] for item in self.corpus]

        # TF-IDF Vectorizer with sublinear term frequency and n-gram overlap
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=10000
        )
        self.corpus_matrix = self.vectorizer.fit_transform(self.customer_texts)

    def retrieve(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        min_similarity: float = SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k historically resolved customer dialogues similar to the query.
        """
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.corpus_matrix)[0]

        # Get indices of top candidates
        ranked_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(sims[idx])
            if score >= min_similarity or len(results) == 0:
                results.append({
                    "pair_id": self.corpus[idx].get("pair_id", idx + 1),
                    "customer_text": self.customer_texts[idx],
                    "brand_reply": self.brand_replies[idx],
                    "similarity_score": round(score, 4)
                })

        return results
