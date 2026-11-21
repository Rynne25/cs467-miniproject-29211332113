"""
Volunteer and Community Service Recommender - Main CLI Entry Point
Author: Le Van Hoai Rin (ID: 29211332113) - CS467

Usage:
    python src/main.py                 # Runs the full automated demo & 3 test cases
    python src/main.py --interactive   # Launches interactive prompt
    python src/main.py --query "your interest text"
    python src/main.py --user USR_02
"""

import sys
import os
import argparse
import pandas as pd

# Ensure package imports work regardless of working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp_pipeline import VolunteerNLPPipeline
from src.recommender import (
    PopularityRecommender,
    ContentBasedRecommender,
    UserBasedCollaborativeFiltering,
    HybridRecommender
)
from src.evaluate import evaluate_models, plot_visualizations, run_test_cases


def print_banner():
    banner = r"""
================================================================================
   VOLUNTEER & COMMUNITY SERVICE RECOMMENDER (NLP + RECOMMENDATION SYSTEMS)
   Duy Tan University | CS467 Advanced Artificial Intelligence | 2026-2027
   Student : Le Van Hoai Rin      | Student ID : 29211332113
   Section : CS467C (No. 14)      | Project    : MP-C14
================================================================================
"""
    print(banner)


def display_recommendations(recs: list[dict], title: str):
    print(f"\n>>> {title}")
    print("-" * 78)
    if not recs:
        print("  No recommendations found matching the criteria.")
        return

    for rank, item in enumerate(recs, 1):
        print(f"[{rank}] [{item['item_id']}] {item['activity_title']}")
        print(f"     Cause Area : {item['cause_area']}")
        score_val = item.get('score', 0)
        method_name = item.get('method', 'Recommender')
        print(f"     Score      : {score_val} ({method_name})")
        if item.get("matching_keywords"):
            print(f"     Key Overlap: {', '.join(item['matching_keywords'])}")
        print()


def interactive_mode(cb_rec: ContentBasedRecommender, hybrid_rec: HybridRecommender, df_items: pd.DataFrame, df_ratings: pd.DataFrame):
    print("\n--- ENTERING INTERACTIVE MODE ---")
    print("Commands:")
    print("  1. Type 'q: <your search>' to search activities by free-text description.")
    print("  2. Type 'u: <USR_ID>' (e.g. u: USR_02) to get personalized hybrid recommendations.")
    print("  3. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("recsys-cli> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode. Goodbye!")
                break

            if user_input.startswith("q:"):
                query = user_input[2:].strip()
                print(f"\nSearching Top-5 volunteer activities for query: '{query}'...")
                recs = cb_rec.recommend_for_query(query, top_n=5)
                display_recommendations(recs, f"Top Matching Activities for '{query}'")

            elif user_input.startswith("u:"):
                uid = user_input[2:].strip().upper()
                if uid not in df_ratings["user_id"].values:
                    print(f"Warning: User '{uid}' not found in database. Using content-based cold start.")
                print(f"\nComputing Top-5 recommendations for Volunteer: {uid}...")
                recs = hybrid_rec.recommend(uid, df_ratings, df_items, top_n=5)
                display_recommendations(recs, f"Personalized Recommendations for {uid}")

            else:
                # Default treat as free text query
                print(f"\nSearching Top-5 volunteer activities for: '{user_input}'...")
                recs = cb_rec.recommend_for_query(user_input, top_n=5)
                display_recommendations(recs, f"Top Matching Activities for '{user_input}'")

        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break


def main():
    parser = argparse.ArgumentParser(description="Volunteer Recommender System")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI menu")
    parser.add_argument("--query", type=str, help="Search volunteer programs via natural language query")
    parser.add_argument("--user", type=str, help="Get personalized recommendations for a volunteer user ID")
    parser.add_argument("--eval", action="store_true", help="Run offline evaluation and generate charts")
    args = parser.parse_args()

    print_banner()

    # Load datasets
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    items_path = os.path.join(base_dir, "data", "items.csv")
    ratings_path = os.path.join(base_dir, "data", "ratings.csv")

    if not os.path.exists(items_path) or not os.path.exists(ratings_path):
        print(f"Error: Missing dataset files in data/ directory. Ensure items.csv and ratings.csv exist.")
        sys.exit(1)

    df_items = pd.read_csv(items_path)
    df_ratings = pd.read_csv(ratings_path)
    print(f"Loaded {len(df_items)} volunteer activities and {len(df_ratings)} user ratings.")

    # Initialize engines
    nlp = VolunteerNLPPipeline()
    nlp.fit_items(df_items)

    cb_rec = ContentBasedRecommender(nlp)
    cb_rec.fit(df_items)

    cf_rec = UserBasedCollaborativeFiltering()
    cf_rec.fit(df_ratings, df_items)

    hybrid_rec = HybridRecommender(cb_rec, cf_rec, alpha=0.55)

    if args.query:
        recs = cb_rec.recommend_for_query(args.query, top_n=5)
        display_recommendations(recs, f"Results for Query: '{args.query}'")
        return

    if args.user:
        recs = hybrid_rec.recommend(args.user, df_ratings, df_items, top_n=5)
        display_recommendations(recs, f"Personalized Recommendations for Volunteer '{args.user}'")
        return

    if args.eval:
        print("\n[Running Offline Evaluation and Chart Generation]")
        metrics_df, _, _ = evaluate_models(df_items, df_ratings)
        print("\n" + metrics_df.to_string(index=False))
        plot_visualizations(df_items, df_ratings, metrics_df, output_dir=os.path.join(base_dir, "results"))
        return

    if args.interactive:
        interactive_mode(cb_rec, hybrid_rec, df_items, df_ratings)
        return

    # Default run: Execute all 3 required test cases + brief evaluation summary
    print("\n[AUTOMATED VERIFICATION RUN: 3 MANDATORY TEST CASES]")
    results_dir = os.path.join(base_dir, "results")
    report = run_test_cases(df_items, df_ratings, output_dir=results_dir)
    print(report)

    print("\n[RUNNING OFFLINE EVALUATION BENCHMARK]")
    metrics_df, _, _ = evaluate_models(df_items, df_ratings)
    print("\n" + metrics_df.to_string(index=False))
    plot_visualizations(df_items, df_ratings, metrics_df, output_dir=results_dir)
    print("\nAll demonstration milestones and test cases executed successfully!")


if __name__ == "__main__":
    main()
