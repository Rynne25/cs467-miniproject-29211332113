"""
Volunteer and Community Service Recommender - Recommendation Engine Module
Author: Le Van Hoai Rin (ID: 29211332113) - CS467

Includes:
1. Popularity Baseline Recommender
2. NLP Content-Based Recommender (User profile & Free-text query)
3. User-Based Collaborative Filtering (User-CF)
4. Weighted Hybrid Recommender with Cold-Start Fallback
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.nlp_pipeline import VolunteerNLPPipeline

class PopularityRecommender:
    """
    Ranks items based on average ratings and rating counts (IMDb-style weighted popularity score).
    Filters out items the target user has already rated.
    """
    def __init__(self, min_ratings_threshold: int = 2):
        self.min_ratings = min_ratings_threshold
        self.item_stats = None
        self.global_mean = 0.0

    def fit(self, df_ratings: pd.DataFrame, df_items: pd.DataFrame):
        self.global_mean = df_ratings["rating"].mean()
        stats = df_ratings.groupby("item_id")["rating"].agg(["count", "mean"]).reset_index()
        stats.columns = ["item_id", "rating_count", "avg_rating"]

        # Merge with all items (even those with 0 ratings)
        stats = pd.merge(df_items[["item_id", "activity_title", "cause_area"]], stats, on="item_id", how="left")
        stats["rating_count"] = stats["rating_count"].fillna(0).astype(int)
        stats["avg_rating"] = stats["avg_rating"].fillna(0.0)

        # Weighted score (Bayesian average formula):
        # W = (R * v + C * m) / (v + m)
        m = self.min_ratings
        C = self.global_mean
        stats["popularity_score"] = stats.apply(
            lambda r: (r["avg_rating"] * r["rating_count"] + C * m) / (r["rating_count"] + m)
            if r["rating_count"] > 0 else 0.0,
            axis=1
        )
        self.item_stats = stats.sort_values(by="popularity_score", ascending=False).reset_index(drop=True)

    def recommend(self, user_id: str, df_ratings: pd.DataFrame, top_n: int = 5) -> list[dict]:
        already_rated = set(df_ratings[df_ratings["user_id"] == user_id]["item_id"])
        recommendations = []

        for _, row in self.item_stats.iterrows():
            item_id = row["item_id"]
            if item_id in already_rated:
                continue
            if row["rating_count"] < self.min_ratings:
                continue

            recommendations.append({
                "item_id": item_id,
                "activity_title": row["activity_title"],
                "cause_area": row["cause_area"],
                "score": round(float(row["popularity_score"]), 3),
                "rating_count": int(row["rating_count"]),
                "avg_rating": round(float(row["avg_rating"]), 2),
                "method": "Popularity Baseline"
            })
            if len(recommendations) >= top_n:
                break

        return recommendations


class ContentBasedRecommender:
    """
    NLP Content-Based Recommender using TF-IDF and Cosine Similarity.
    Supports:
    - User Profile Recommendations (based on volunteer past ratings and feedback)
    - Free-Text Natural Language Query Search
    """
    def __init__(self, nlp_pipeline: VolunteerNLPPipeline):
        self.nlp = nlp_pipeline
        self.df_items = None

    def fit(self, df_items: pd.DataFrame):
        self.df_items = df_items.copy()
        self.nlp.fit_items(df_items)

    def recommend_for_user(
        self,
        user_id: str,
        df_ratings: pd.DataFrame,
        top_n: int = 5,
        filter_rated: bool = True
    ) -> list[dict]:
        """
        Recommends top-N volunteer programs matching the student's historical interest profile.
        """
        user_profile = self.nlp.build_user_interest_profile(user_id, df_ratings)
        already_rated = set(df_ratings[df_ratings["user_id"] == user_id]["item_id"]) if filter_rated else set()

        if np.all(user_profile == 0):
            return []

        # Cosine similarity between user profile and all item vectors
        similarities = cosine_similarity([user_profile], self.nlp.item_vectors)[0]
        ranked_indices = np.argsort(similarities)[::-1]

        recommendations = []
        for idx in ranked_indices:
            item_id = self.nlp.item_ids[idx]
            if item_id in already_rated:
                continue

            sim_score = float(similarities[idx])
            matching_keywords = self.nlp.explain_match(user_profile, idx, top_k=3)
            item_row = self.df_items[self.df_items["item_id"] == item_id].iloc[0]

            recommendations.append({
                "item_id": item_id,
                "activity_title": item_row["activity_title"],
                "cause_area": item_row["cause_area"],
                "score": round(sim_score, 4),
                "matching_keywords": matching_keywords,
                "method": "NLP Content-Based (Profile)"
            })
            if len(recommendations) >= top_n:
                break

        return recommendations

    def recommend_for_query(self, query: str, top_n: int = 5) -> list[dict]:
        """
        Recommends top-N activities matching an arbitrary natural language text query.
        """
        query_vec = self.nlp.vectorize_text(query)
        if np.all(query_vec == 0):
            return []

        similarities = cosine_similarity([query_vec], self.nlp.item_vectors)[0]
        ranked_indices = np.argsort(similarities)[::-1]

        recommendations = []
        for idx in ranked_indices:
            item_id = self.nlp.item_ids[idx]
            sim_score = float(similarities[idx])
            if sim_score <= 0:
                continue

            matching_keywords = self.nlp.explain_match(query_vec, idx, top_k=4)
            item_row = self.df_items[self.df_items["item_id"] == item_id].iloc[0]

            recommendations.append({
                "item_id": item_id,
                "activity_title": item_row["activity_title"],
                "cause_area": item_row["cause_area"],
                "score": round(sim_score, 4),
                "matching_keywords": matching_keywords,
                "method": "NLP Content-Based (Query)"
            })
            if len(recommendations) >= top_n:
                break

        return recommendations


class UserBasedCollaborativeFiltering:
    """
    User-Based Collaborative Filtering (User-CF) using Pearson / Centered-Cosine Similarity.
    Predicts volunteer ratings for unrated activities based on peer volunteer consensus.
    """
    def __init__(self, k_neighbors: int = 5):
        self.k = k_neighbors
        self.user_item_matrix = None
        self.user_means = {}
        self.user_similarity_matrix = None
        self.user_list = []
        self.item_list = []

    def fit(self, df_ratings: pd.DataFrame, df_items: pd.DataFrame):
        self.user_list = sorted(df_ratings["user_id"].unique().tolist())
        self.item_list = sorted(df_items["item_id"].unique().tolist())

        # Construct pivot matrix
        pivot = df_ratings.pivot(index="user_id", columns="item_id", values="rating")
        pivot = pivot.reindex(index=self.user_list, columns=self.item_list)
        self.user_item_matrix = pivot.values

        # Compute user means
        for u_idx, u_id in enumerate(self.user_list):
            ratings = self.user_item_matrix[u_idx]
            valid = ratings[~np.isnan(ratings)]
            self.user_means[u_id] = float(np.mean(valid)) if len(valid) > 0 else 3.5

        # Centered matrix for Pearson correlation
        centered = self.user_item_matrix.copy()
        for u_idx, u_id in enumerate(self.user_list):
            mask = ~np.isnan(centered[u_idx])
            centered[u_idx, mask] -= self.user_means[u_id]
        centered = np.nan_to_num(centered, nan=0.0)

        # Compute cosine similarity on centered ratings
        sim = cosine_similarity(centered)
        np.fill_diagonal(sim, 0.0)
        self.user_similarity_matrix = sim

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predicts rating for (user_id, item_id) using weighted k-nearest neighbors.
        """
        if user_id not in self.user_list or item_id not in self.item_list:
            return self.user_means.get(user_id, 3.5)

        u_idx = self.user_list.index(user_id)
        i_idx = self.item_list.index(item_id)

        # If already rated, return actual rating
        val = self.user_item_matrix[u_idx, i_idx]
        if not np.isnan(val):
            return float(val)

        # Find other users who rated item_id
        item_ratings = self.user_item_matrix[:, i_idx]
        rated_users = np.where(~np.isnan(item_ratings))[0]

        if len(rated_users) == 0:
            return self.user_means[user_id]

        similarities = self.user_similarity_matrix[u_idx, rated_users]
        # Top-k positive neighbors
        positive_idx = np.where(similarities > 0)[0]
        if len(positive_idx) == 0:
            return self.user_means[user_id]

        sorted_pos = positive_idx[np.argsort(similarities[positive_idx])[::-1][:self.k]]
        chosen_users = rated_users[sorted_pos]
        chosen_sims = similarities[sorted_pos]

        sim_sum = np.sum(np.abs(chosen_sims))
        if sim_sum == 0:
            return self.user_means[user_id]

        ratings_diff = self.user_item_matrix[chosen_users, i_idx] - [self.user_means[self.user_list[u]] for u in chosen_users]
        predicted = self.user_means[user_id] + np.sum(chosen_sims * ratings_diff) / sim_sum
        return float(np.clip(predicted, 1.0, 5.0))

    def recommend(self, user_id: str, df_ratings: pd.DataFrame, df_items: pd.DataFrame, top_n: int = 5) -> list[dict]:
        if user_id not in self.user_list:
            return []

        u_idx = self.user_list.index(user_id)
        already_rated = set(df_ratings[df_ratings["user_id"] == user_id]["item_id"])

        scores = []
        for i_idx, item_id in enumerate(self.item_list):
            if item_id in already_rated:
                continue

            # check if any neighbor rated this item
            item_ratings = self.user_item_matrix[:, i_idx]
            if np.all(np.isnan(item_ratings)):
                continue  # Cold start item for CF

            pred = self.predict_rating(user_id, item_id)
            item_row = df_items[df_items["item_id"] == item_id].iloc[0]
            scores.append({
                "item_id": item_id,
                "activity_title": item_row["activity_title"],
                "cause_area": item_row["cause_area"],
                "score": round(pred, 3),
                "method": "User-Based Collaborative Filtering"
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_n]


class HybridRecommender:
    """
    Weighted Hybrid Recommender combining NLP Content-Based and Collaborative Filtering:
    Final_Score = alpha * Normalized_Content_Score + (1 - alpha) * Normalized_CF_Score
    Features graceful fallback when CF is unavailable (Cold-Start user or item).
    """
    def __init__(self, content_recommender: ContentBasedRecommender, cf_recommender: UserBasedCollaborativeFiltering, alpha: float = 0.55):
        self.content_rec = content_recommender
        self.cf_rec = cf_recommender
        self.alpha = alpha

    def recommend(
        self,
        user_id: str,
        df_ratings: pd.DataFrame,
        df_items: pd.DataFrame,
        top_n: int = 5
    ) -> list[dict]:
        already_rated = set(df_ratings[df_ratings["user_id"] == user_id]["item_id"])

        # 1. Content scores
        user_profile = self.content_rec.nlp.build_user_interest_profile(user_id, df_ratings)
        is_user_cold_start = np.all(user_profile == 0)

        candidate_items = [it for it in df_items["item_id"].tolist() if it not in already_rated]
        if not candidate_items:
            return []

        # Raw scores
        raw_content = {}
        if not is_user_cold_start:
            sims = cosine_similarity([user_profile], self.content_rec.nlp.item_vectors)[0]
            for it in candidate_items:
                idx = self.content_rec.nlp.item_ids.index(it)
                raw_content[it] = float(sims[idx])

        raw_cf = {}
        if user_id in self.cf_rec.user_list:
            for it in candidate_items:
                raw_cf[it] = float(self.cf_rec.predict_rating(user_id, it))

        # Min-max normalization for fair scale blending
        norm_content = {}
        if raw_content:
            c_vals = list(raw_content.values())
            c_min, c_max = min(c_vals), max(c_vals)
            c_range = c_max - c_min if c_max > c_min else 1.0
            for it, val in raw_content.items():
                norm_content[it] = (val - c_min) / c_range

        norm_cf = {}
        if raw_cf:
            cf_vals = list(raw_cf.values())
            cf_min, cf_max = min(cf_vals), max(cf_vals)
            cf_range = cf_max - cf_min if cf_max > cf_min else 1.0
            for it, val in raw_cf.items():
                norm_cf[it] = (val - cf_min) / cf_range

        combined_results = []
        for item_id in candidate_items:
            has_content = item_id in norm_content
            has_cf = item_id in norm_cf

            if has_content and has_cf:
                c_score = norm_content[item_id]
                cf_score = norm_cf[item_id]
                hybrid_score = self.alpha * c_score + (1.0 - self.alpha) * cf_score
                note = "Hybrid (Content + CF)"
            elif has_content:
                hybrid_score = norm_content[item_id]
                note = "Fallback: NLP Content (Cold-start Item for CF)"
            elif has_cf:
                hybrid_score = norm_cf[item_id]
                note = "Fallback: User CF"
            else:
                hybrid_score = 0.0
                note = "Baseline"

            idx = self.content_rec.nlp.item_ids.index(item_id)
            matching_kw = self.content_rec.nlp.explain_match(user_profile, idx, top_k=3) if not is_user_cold_start else []
            item_row = df_items[df_items["item_id"] == item_id].iloc[0]

            combined_results.append({
                "item_id": item_id,
                "activity_title": item_row["activity_title"],
                "cause_area": item_row["cause_area"],
                "score": round(hybrid_score, 4),
                "matching_keywords": matching_kw,
                "method": note
            })

        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_n]
