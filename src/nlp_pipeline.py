"""
Volunteer and Community Service Recommender - NLP Pipeline Module
Author: Le Van Hoai Rin (ID: 29211332113) - CS467
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Common English stop words
STANDARD_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", 
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
}

def clean_text(text: str) -> str:
    """
    Cleans raw text: converts to lower case, removes punctuation and non-alphanumeric chars,
    and strips redundant whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_filter(text: str, stop_words: set = STANDARD_STOP_WORDS) -> list[str]:
    """
    Tokenizes clean text and removes stop words and single-letter tokens.
    """
    tokens = clean_text(text).split()
    return [token for token in tokens if token not in stop_words and len(token) > 1]

class VolunteerNLPPipeline:
    """
    NLP Pipeline for Volunteer Recommender.
    Handles text feature extraction, TF-IDF vectorization, user profile modeling,
    and similarity explanation.
    """
    def __init__(self, max_features: int = 1200, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.item_ids = []
        self.item_vectors = None
        self.feature_names = np.array([])
        self.df_items = None

    def fit_items(self, df_items: pd.DataFrame) -> np.ndarray:
        """
        Builds combined enriched text for volunteer items and fits TF-IDF representation.
        Enriches text by weighting title, cause area, skill tags, and detailed description.
        """
        self.df_items = df_items.copy()
        self.item_ids = df_items["item_id"].tolist()

        # Combine text fields: title (x2), cause (x2), skill tags (x2), and description
        enriched_texts = []
        for _, row in df_items.iterrows():
            title = str(row.get("activity_title", ""))
            cause = str(row.get("cause_area", ""))
            tags = str(row.get("skill_tags", ""))
            desc = str(row.get("description", ""))
            combined = f"{title} {title} {cause} {cause} {tags} {tags} {desc}"
            enriched_texts.append(clean_text(combined))

        self.item_vectors = self.vectorizer.fit_transform(enriched_texts).toarray()
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        return self.item_vectors

    def vectorize_text(self, text: str) -> np.ndarray:
        """
        Vectorizes free-text queries or descriptions using the fitted TF-IDF vocabulary.
        Returns a 1D numpy array.
        """
        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned]).toarray()[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def build_user_interest_profile(
        self,
        user_id: str,
        df_ratings: pd.DataFrame,
        min_rating_threshold: float = 3.5
    ) -> np.ndarray:
        """
        Constructs a student volunteer's interest vector by aggregating TF-IDF vectors
        of activities they rated highly, weighted by the rating score.
        If user has no high ratings, returns zero vector (cold-start volunteer).
        """
        user_ratings = df_ratings[
            (df_ratings["user_id"] == user_id) &
            (df_ratings["rating"] >= min_rating_threshold)
        ]

        if user_ratings.empty or self.item_vectors is None:
            return np.zeros(len(self.feature_names))

        weighted_profile = np.zeros(len(self.feature_names))
        total_weight = 0.0

        for _, row in user_ratings.iterrows():
            item_id = row["item_id"]
            if item_id in self.item_ids:
                idx = self.item_ids.index(item_id)
                weight = float(row["rating"]) - 2.0  # Scale weight: 3.5 -> 1.5, 5.0 -> 3.0
                weighted_profile += self.item_vectors[idx] * weight
                total_weight += weight

        if total_weight > 0:
            norm = np.linalg.norm(weighted_profile)
            if norm > 0:
                weighted_profile = weighted_profile / norm

        return weighted_profile

    def get_top_keywords_for_vector(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Extracts top-k most important keywords and their weights from a vector.
        """
        if vector is None or np.all(vector == 0):
            return []
        top_indices = np.argsort(vector)[::-1][:top_k]
        return [(self.feature_names[i], float(vector[i])) for i in top_indices if vector[i] > 0]

    def explain_match(self, query_or_user_vec: np.ndarray, item_idx: int, top_k: int = 4) -> list[str]:
        """
        Identifies key matching terms between a query/profile vector and an item vector
        using element-wise TF-IDF product.
        """
        if self.item_vectors is None or np.all(query_or_user_vec == 0):
            return []
        item_vec = self.item_vectors[item_idx]
        overlap = query_or_user_vec * item_vec
        top_indices = np.argsort(overlap)[::-1][:top_k]
        matching_terms = [self.feature_names[i] for i in top_indices if overlap[i] > 0]
        return matching_terms
