import os
import torch
from tqdm import tqdm

# config
GRAPHS_DIR = "graphs"


def add_placeholder_features():
    graph_files = sorted([f for f in os.listdir(GRAPHS_DIR) if f.endswith(".pt")])

    print(f"Found {len(graph_files)} graphs to process.")

    # order for one-hot encoding
    node_types = ["influencer", "hashtag", "user", "object"]

    for graph_file in tqdm(graph_files, desc="Adding features to graphs"):
        graph_path = os.path.join(GRAPHS_DIR, graph_file)

        # load data/graphs
        data = torch.load(graph_path, weights_only=False)
        graph = data["graph"]

        # add feature matrix
        for i, node_type in enumerate(node_types):
            num_nodes = graph[node_type].num_nodes

            if num_nodes > 0:
                one_hot_features = torch.zeros(num_nodes, len(node_types))
                one_hot_features[:, i] = 1

                graph[node_type].x = one_hot_features

        # save the updated data object (graph + maps + features) back to the file
        torch.save(data, graph_path)

    print("\n✔️ added placeholder features to all graph files.")


if __name__ == "__main__":
    add_placeholder_features()
