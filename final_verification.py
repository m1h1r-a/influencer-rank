import json
import os
import re

import torch
from tqdm import tqdm

# config
DATA_ROOT = "year_17"
GRAPHS_DIR = "graphs"
COMBINED_OBJECTS_CSV = "image_objects.csv"


def load_object_lookup():
    import ast

    import pandas as pd

    df = pd.read_csv(COMBINED_OBJECTS_CSV)
    df["detected_objects"] = df["detected_objects"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    return pd.Series(df.detected_objects.values, index=df.post_id).to_dict()


def run_forensic_validation():
    object_lookup = load_object_lookup()
    month_dirs = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    )

    all_months_ok = True
    for month_name in month_dirs:
        print(f"\n{'='*60}\n🔬 FORENSIC ANALYSIS FOR MONTH: {month_name}\n{'='*60}")
        graph_path = os.path.join(GRAPHS_DIR, f"{month_name.lower()}_graph.pt")
        month_data_path = os.path.join(DATA_ROOT, month_name)

        # load graph
        saved_data = torch.load(graph_path, weights_only=False)
        graph = saved_data["graph"]
        maps = saved_data["maps"]

        # get raw data
        raw_influencers, raw_hashtags, raw_users, raw_objects = (
            set(),
            set(),
            set(),
            set(),
        )
        raw_hashtag_edges, raw_mention_edges, raw_object_edges = 0, 0, 0

        for filename in tqdm(
            os.listdir(month_data_path), desc=f"Recounting {month_name}", leave=False
        ):
            if not filename.endswith(".info"):
                continue
            influencer_name = filename.rsplit("-", 1)[0].lower()
            post_id_str = os.path.splitext(filename)[0].rsplit("-", 1)[-1]
            raw_influencers.add(influencer_name)

            with open(os.path.join(month_data_path, filename), "r") as f:
                data = json.load(f)
                if cap_edges := data.get("edge_media_to_caption", {}).get("edges", []):
                    tags = {
                        h.lower()
                        for h in re.findall(r"#(\w+)", cap_edges[0]["node"]["text"])
                    }
                    raw_hashtags.update(tags)
                    raw_hashtag_edges += len(tags)
                if tagged_edges := data.get("edge_media_to_tagged_user", {}).get(
                    "edges", []
                ):
                    users = {
                        e["node"]["user"]["username"].lower()
                        for e in tagged_edges
                        if "user" in e.get("node", {})
                    }
                    raw_users.update(users)
                    raw_mention_edges += len(users)

            objects = object_lookup.get(int(post_id_str), [])
            raw_objects.update(objects)
            raw_object_edges += len(objects)

        # comparison
        month_ok = True
        print(f"{'METRIC':<25} | {'GRAPH':<12} | {'RAW DATA':<12} | {'STATUS':<10}")
        print(f"{'-'*25} | {'-'*12} | {'-'*12} | {'-'*10}")

        # Node count comparison
        for n_type in ["influencer", "hashtag", "user", "object"]:
            g_count = graph[n_type].num_nodes
            r_count = locals()[f"raw_{n_type}s"].__len__()
            status = "✔️ OK" if g_count == r_count else "❌ MISMATCH"
            if g_count != r_count:
                month_ok = False
            print(
                f"{n_type.capitalize()} Node Count {'':<8} | {g_count:<12,} | {r_count:<12,} | {status}"
            )

            # find difference
            if g_count != r_count:
                graph_nodes = set(maps[n_type].keys())
                recounted_nodes = locals()[f"raw_{n_type}s"]
                in_raw_not_graph = recounted_nodes - graph_nodes
                in_graph_not_raw = graph_nodes - recounted_nodes
                if in_raw_not_graph:
                    print(
                        f"  - In Raw, not Graph: {list(in_raw_not_graph)[:5]}"
                    )  # Show first 5
                if in_graph_not_raw:
                    print(f"  - In Graph, not Raw: {list(in_graph_not_raw)[:5]}")

        # Edge count comparison
        edge_map = {
            "hashtag": "posts_hashtag",
            "mention": "mentions",
            "object": "posted_object",
        }
        for e_type in ["hashtag", "mention", "object"]:
            edge_key = (
                "influencer",
                edge_map[e_type],
                e_type if e_type != "mention" else "user",
            )
            g_count = graph[edge_key].num_edges if edge_key in graph.edge_types else 0
            r_count = locals()[f"raw_{e_type}_edges"]
            status = "✔️ OK" if g_count == r_count else "❌ MISMATCH"
            if g_count != r_count:
                month_ok = False
            print(
                f"{e_type.capitalize()} Edge Count {'':<11} | {g_count:<12,} | {r_count:<12,} | {status}"
            )

        if not month_ok:
            all_months_ok = False

    print(f"\n{'='*60}\nFINAL CONCLUSION\n{'='*60}")
    if all_months_ok:
        print("Graph identical to raw source data")
    else:
        print(
            "‼️ One or more months failed validation. Please review the MISMATCH details above."
        )


if __name__ == "__main__":
    run_forensic_validation()
