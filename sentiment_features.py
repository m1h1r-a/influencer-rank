import os
import json
import re
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Configuration ---
DATA_ROOT = "year_17"
GRAPHS_DIR = "graphs"
NUM_PROCESSES = cpu_count() - 1

# Initialize the sentiment analyzer once
SIA = SentimentIntensityAnalyzer()


def extract_post_features(file_path):
    """
    Worker function for multiprocessing. Extracts raw features from a single .info file.
    """
    try:
        influencer_name = os.path.basename(file_path).rsplit("-", 1)[0].lower()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- Text & Posting Features ---
        caption_text = ""
        caption_sentiment = 0.0
        if cap_edges := data.get("edge_media_to_caption", {}).get("edges", []):
            caption_text = cap_edges[0]["node"]["text"]
            caption_sentiment = SIA.polarity_scores(caption_text)["compound"]

        num_hashtags = len(re.findall(r"#(\w+)", caption_text))
        num_mentions = 0
        if tagged_edges := data.get("edge_media_to_tagged_user", {}).get("edges", []):
            num_mentions = len(tagged_edges)

        # --- Reaction Features ---
        comment_sentiments = []
        if comment_edges := data.get("edge_media_to_parent_comment", {}).get(
            "edges", []
        ):
            for edge in comment_edges:
                comment_text = edge.get("node", {}).get("text", "")
                if comment_text:
                    sentiment = SIA.polarity_scores(comment_text)["compound"]
                    comment_sentiments.append(sentiment)

        avg_comment_sentiment = (
            np.mean(comment_sentiments) if comment_sentiments else 0.0
        )

        return {
            "influencer": influencer_name,
            "caption_len": len(caption_text),
            "caption_sentiment": caption_sentiment,
            "num_hashtags": num_hashtags,
            "num_mentions": num_mentions,
            "avg_comment_sentiment": avg_comment_sentiment,
        }
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def engineer_rich_features():
    """
    Main function to orchestrate the feature engineering process for all months.
    """
    month_dirs = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    )

    for month_name in month_dirs:
        print(f"\n{'='*60}\nProcessing features for month: {month_name}\n{'='*60}")
        month_data_path = os.path.join(DATA_ROOT, month_name)
        graph_path = os.path.join(GRAPHS_DIR, f"{month_name.lower()}_graph.pt")

        # --- 1. Extract raw features in parallel ---
        file_paths = [
            os.path.join(month_data_path, f)
            for f in os.listdir(month_data_path)
            if f.endswith(".info")
        ]

        monthly_post_data = []
        with Pool(processes=NUM_PROCESSES) as pool:
            results_iterator = pool.imap_unordered(extract_post_features, file_paths)
            for result in tqdm(
                results_iterator, total=len(file_paths), desc="Extracting from posts"
            ):
                if result:
                    monthly_post_data.append(result)

        # --- 2. Aggregate features for each influencer ---
        influencer_posts = defaultdict(list)
        for post in monthly_post_data:
            influencer_posts[post["influencer"]].append(post)

        # Load graph to get the correct influencer order
        data_package = torch.load(graph_path, weights_only=False)
        graph = data_package["graph"]
        influencer_map = data_package["maps"]["influencer"]
        num_influencers = graph["influencer"].num_nodes

        # Define feature names for clarity
        feature_names = [
            "num_posts",
            "avg_caption_len",
            "avg_caption_sentiment",
            "avg_hashtags",
            "avg_mentions",
            "avg_comment_sentiment",
        ]
        num_features = len(feature_names)

        feature_matrix = torch.zeros(num_influencers, num_features)

        print("Aggregating features per influencer...")
        for name, idx in tqdm(influencer_map.items(), desc="Aggregating"):
            posts = influencer_posts.get(name, [])
            if not posts:
                continue  # Influencer might exist in graph but have no posts this month

            num_posts = len(posts)
            avg_caption_len = np.mean([p["caption_len"] for p in posts])
            avg_caption_sentiment = np.mean([p["caption_sentiment"] for p in posts])
            avg_hashtags = np.mean([p["num_hashtags"] for p in posts])
            avg_mentions = np.mean([p["num_mentions"] for p in posts])
            avg_comment_sentiment = np.mean([p["avg_comment_sentiment"] for p in posts])

            feature_vector = torch.tensor(
                [
                    num_posts,
                    avg_caption_len,
                    avg_caption_sentiment,
                    avg_hashtags,
                    avg_mentions,
                    avg_comment_sentiment,
                ]
            )
            feature_matrix[idx] = feature_vector

        # --- 3. Add features to the graph and save ---
        # Add rich features for influencers
        graph["influencer"].x = feature_matrix

        # Add simple one-hot features for other node types
        node_types = ["influencer", "hashtag", "user", "object"]
        feature_dim = (
            len(node_types) + num_features - 1
        )  # a bit of a hack to make all features same dimension

        for i, node_type in enumerate(node_types):
            # We want all feature vectors to have the same dimension for the GNN
            num_nodes = graph[node_type].num_nodes
            if num_nodes > 0:
                final_features = torch.zeros(num_nodes, feature_dim)
                if node_type == "influencer":
                    # First column is one-hot type, rest are rich features
                    final_features[:, 0] = 1.0
                    final_features[:, 1 : 1 + num_features] = feature_matrix
                else:
                    # Just the one-hot type encoding
                    final_features[:, i] = 1.0
                graph[node_type].x = final_features

        # Save the updated package
        torch.save(data_package, graph_path)
        print(f"✔️ Successfully added rich features to graph for {month_name}.")
        print(f"Feature vector size: {graph['influencer'].x.shape[1]}")


if __name__ == "__main__":
    engineer_rich_features()
