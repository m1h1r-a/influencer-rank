import ast
import json
import os
import re

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

# config
DATA_ROOT = "year_17"
GRAPHS_DIR = "graphs"
COMBINED_OBJECTS_CSV = "image_objects.csv"


def load_object_data(csv_path):
    print(f"Loading object data from {csv_path}...")
    df = pd.read_csv(csv_path)

    def parse_object_list(objects_str):
        try:
            return ast.literal_eval(objects_str)
        except (ValueError, SyntaxError):
            return []

    df["detected_objects"] = df["detected_objects"].apply(parse_object_list)
    return pd.Series(df.detected_objects.values, index=df.post_id).to_dict()


def create_post_to_influencer_map(root_dir):
    print("Building post_id to influencer map...")
    post_map = {}
    month_dirs = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]
    for month in tqdm(month_dirs, desc="Scanning months"):
        month_path = os.path.join(root_dir, month)
        for filename in os.listdir(month_path):
            if filename.endswith(".info"):
                parts = os.path.splitext(filename)[0].rsplit("-", 1)
                if len(parts) == 2:
                    influencer, post_id = parts
                    post_map[post_id] = influencer.lower()
    print(f"Mapped {len(post_map)} unique posts to their influencers.")
    return post_map


def build_complete_graphs():
    object_lookup = load_object_data(COMBINED_OBJECTS_CSV)
    post_to_influencer_map = create_post_to_influencer_map(DATA_ROOT)

    month_dirs = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    )

    for month_name in month_dirs:
        graph_path = os.path.join(GRAPHS_DIR, f"{month_name.lower()}_graph.pt")
        month_data_path = os.path.join(DATA_ROOT, month_name)

        print(f"\n{'='*50}")
        print(f"Building complete graph for month: {month_name}")

        # get data from raw data
        influencers, hashtags, users, objects = set(), set(), set(), set()
        influencer_hashtag_edges, influencer_user_edges, influencer_object_edges = (
            [],
            [],
            [],
        )

        post_files = [f for f in os.listdir(month_data_path) if f.endswith(".info")]
        for filename in tqdm(post_files, desc=f"Processing {month_name}"):
            influencer_name = filename.rsplit("-", 1)[0].lower()
            post_id_str = os.path.splitext(filename)[0].rsplit("-", 1)[-1]
            influencers.add(influencer_name)

            with open(os.path.join(month_data_path, filename), "r") as f:
                data = json.load(f)
                # hashtags
                if cap_edges := data.get("edge_media_to_caption", {}).get("edges", []):
                    found_tags = {
                        h.lower()
                        for h in re.findall(r"#(\w+)", cap_edges[0]["node"]["text"])
                    }
                    hashtags.update(found_tags)
                    for tag in found_tags:
                        influencer_hashtag_edges.append((influencer_name, tag))
                # mentions
                if tagged_edges := data.get("edge_media_to_tagged_user", {}).get(
                    "edges", []
                ):
                    found_users = {
                        e["node"]["user"]["username"].lower()
                        for e in tagged_edges
                        if "user" in e.get("node", {})
                    }
                    users.update(found_users)
                    for user in found_users:
                        influencer_user_edges.append((influencer_name, user))

            # objects
            post_objects = object_lookup.get(int(post_id_str), [])
            objects.update(post_objects)
            for obj in post_objects:
                influencer_object_edges.append((influencer_name, obj))

        # make mappings
        influencer_map = {name: i for i, name in enumerate(influencers)}
        hashtag_map = {name: i for i, name in enumerate(hashtags)}
        user_map = {name: i for i, name in enumerate(users)}
        object_map = {name: i for i, name in enumerate(objects)}

        # build graph
        graph = HeteroData()
        graph["influencer"].num_nodes = len(influencers)
        graph["hashtag"].num_nodes = len(hashtags)
        graph["user"].num_nodes = len(users)
        graph["object"].num_nodes = len(objects)

        src, dst = zip(*influencer_hashtag_edges)
        graph["influencer", "posts_hashtag", "hashtag"].edge_index = torch.tensor(
            [[influencer_map[s] for s in src], [hashtag_map[d] for d in dst]],
            dtype=torch.long,
        )

        src, dst = zip(*influencer_user_edges)
        graph["influencer", "mentions", "user"].edge_index = torch.tensor(
            [[influencer_map[s] for s in src], [user_map[d] for d in dst]],
            dtype=torch.long,
        )

        if influencer_object_edges:
            src, dst = zip(*influencer_object_edges)
            graph["influencer", "posted_object", "object"].edge_index = torch.tensor(
                [[influencer_map[s] for s in src], [object_map[d] for d in dst]],
                dtype=torch.long,
            )

        # graph + mappings
        data_to_save = {
            "graph": graph,
            "maps": {
                "influencer": influencer_map,
                "hashtag": hashtag_map,
                "user": user_map,
                "object": object_map,
            },
        }
        torch.save(data_to_save, graph_path)
        print(f"Successfully built and saved complete graph for {month_name}!")


if __name__ == "__main__":
    build_complete_graphs()
