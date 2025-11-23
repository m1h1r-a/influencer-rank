#!/usr/bin/env python3
"""
InfluencerRank v8: Inference Script
Load trained model and generate influencer rankings.

Usage:
    python predict_rankings.py [--model-dir saved_models_v8_final] [--graph-dir graphs_enhanced_v3] [--output rankings.csv]
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch_geometric.nn import GATConv, HeteroConv


class HeteroGNN(nn.Module):
    """Heterogeneous GNN with GAT convolution."""

    def __init__(
        self, in_channels, hidden_channels, out_channels, metadata, dropout=0.5
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.conv1 = HeteroConv(
            {
                edge_type: GATConv(
                    (-1, -1), hidden_channels, add_self_loops=False, heads=1
                )
                for edge_type in metadata[1]
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                edge_type: GATConv(
                    (-1, -1), out_channels, add_self_loops=False, heads=1
                )
                for edge_type in metadata[1]
            },
            aggr="sum",
        )

        self.linear_proj = nn.Linear(in_channels, hidden_channels + out_channels)

    def forward(self, x_dict, edge_index_dict):
        edge_index_dict_filtered = {
            k: v for k, v in edge_index_dict.items() if v.shape[1] > 0
        }
        if len(edge_index_dict_filtered) == 0:
            return {key: self.linear_proj(x) for key, x in x_dict.items()}

        # Layer 1
        x_dict_1 = self.conv1(x_dict, edge_index_dict_filtered)
        x_dict_1 = {key: F.relu(x) for key, x in x_dict_1.items() if x is not None}
        x_dict_1 = {key: self.dropout(x) for key, x in x_dict_1.items()}

        if len(x_dict_1) == 0:
            return {key: self.linear_proj(x) for key, x in x_dict.items()}

        # Layer 2
        edge_index_dict_filtered_2 = {
            k: v
            for k, v in edge_index_dict_filtered.items()
            if k[0] in x_dict_1 and k[2] in x_dict_1
        }
        if len(edge_index_dict_filtered_2) == 0:
            x_dict_2 = {k: torch.zeros_like(v) for k, v in x_dict_1.items()}
        else:
            x_dict_2 = self.conv2(x_dict_1, edge_index_dict_filtered_2)
            x_dict_2 = {key: x for key, x in x_dict_2.items() if x is not None}

        return {
            k: torch.cat([x_dict_1[k], x_dict_2[k]], dim=1)
            for k in x_dict_1.keys()
            if k in x_dict_2
        }


class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_fc = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, lengths):
        batch_size, seq_len, _ = hidden_states.shape
        scores = self.attention_fc(hidden_states).squeeze(-1)
        mask = torch.arange(seq_len, device=hidden_states.device).expand(batch_size, -1)
        mask = mask < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class BestInfluencerModel(nn.Module):
    """Best performing model: GAT + GRU."""

    def __init__(
        self, input_dim, gnn_hidden, gnn_out, rnn_hidden, metadata, dropout=0.5
    ):
        super().__init__()
        self.hetero_gnn = HeteroGNN(input_dim, gnn_hidden, gnn_out, metadata, dropout)

        gnn_concat_dim = gnn_hidden + gnn_out
        self.rnn = nn.GRU(gnn_concat_dim, rnn_hidden, num_layers=1, batch_first=True)
        self.attention = SimpleAttention(rnn_hidden)
        self.fc1 = nn.Linear(rnn_hidden, rnn_hidden // 2)
        self.fc2 = nn.Linear(rnn_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def encode_graphs(self, hetero_graphs):
        embeddings = []
        for hg in hetero_graphs:
            graph = hg["graph"]
            x_dict = {
                node_type: graph[node_type].x
                for node_type in graph.node_types
                if hasattr(graph[node_type], "x")
            }
            edge_index_dict = (
                graph.edge_index_dict if len(graph.edge_index_dict) > 0 else {}
            )
            emb_dict = self.hetero_gnn(x_dict, edge_index_dict)
            embeddings.append(emb_dict["influencer"])
        return embeddings

    def forward_temporal(self, sequences, lengths):
        packed = pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_output, _ = pad_packed_sequence(self.rnn(packed)[0], batch_first=True)
        context = self.attention(rnn_output, lengths.to(rnn_output.device))
        x = F.relu(self.fc1(context))
        return self.fc2(self.dropout(x)).squeeze(-1)


def load_model_and_config(model_dir, device):
    """Load trained model, scaler, and config."""
    # Load checkpoint
    checkpoint = torch.load(
        f"{model_dir}/model.pt", map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    results = checkpoint["results"]

    print(f"Loaded model from {model_dir}/")
    # print(f"  Test NDCG@50: {results['test_ndcg']:.4f}")
    # print(f"  Val NDCG@50: {results['val_ndcg']:.4f}")
    print(f"  Seed: {results['seed']}")

    # Load scaler
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load data split
    with open(f"{model_dir}/data_split.pkl", "rb") as f:
        data_split = pickle.load(f)

    return checkpoint, config, scaler, data_split


def load_and_preprocess_graphs(
    graph_dir, scaler, config, training_months=9, device="cpu"
):
    """Load and preprocess graphs."""
    month_names = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]

    print(f"\nLoading graphs from {graph_dir}/...")
    all_graphs_data = []

    for month_idx, month in enumerate(month_names):
        path = os.path.join(graph_dir, f"{month}_graph.pt")
        data = torch.load(path, weights_only=False)
        all_graphs_data.append(data)

    print(f"Loaded {len(all_graphs_data)} graphs")

    # Apply normalization and masking
    print("Applying normalization and feature masking...")
    strict_mask_indices = config["STRICT_MASK_INDICES"]

    for data_package in all_graphs_data:
        features = data_package["graph"]["influencer"].x
        normalized_features = torch.FloatTensor(scaler.transform(features.numpy()))
        normalized_features[:, strict_mask_indices] = 0.0
        data_package["graph"]["influencer"].x = normalized_features

    # Prepare heterogeneous graphs for GPU
    print("Preparing graphs for inference...")
    hetero_graphs = []

    for month_idx in range(training_months):
        graph = all_graphs_data[month_idx]["graph"]
        graph = T.ToUndirected()(graph)
        graph = graph.to(device)

        influencer_map = all_graphs_data[month_idx]["maps"]["influencer"]

        hetero_graphs.append({"graph": graph, "influencer_map": influencer_map})

    return hetero_graphs, all_graphs_data


def predict_rankings(model, hetero_graphs, influencer_list, target_data, device):
    """Generate predictions for influencers."""
    model.eval()

    with torch.no_grad():
        # Encode all graphs
        print("\nEncoding graphs...")
        embeddings = model.encode_graphs(hetero_graphs)

        # Build sequences for all influencers
        print("Building temporal sequences...")
        sequences, valid_names = [], []

        for name in influencer_list:
            seq = []
            for month_idx in range(len(hetero_graphs)):
                if name in hetero_graphs[month_idx]["influencer_map"]:
                    local_idx = hetero_graphs[month_idx]["influencer_map"][name]
                    seq.append(embeddings[month_idx][local_idx])

            if len(seq) > 0:
                sequences.append(torch.stack(seq))
                valid_names.append(name)

        if len(sequences) == 0:
            print("No valid sequences found!")
            return None, None, None

        # Predict
        print(f"Predicting scores for {len(valid_names)} influencers...")
        padded = pad_sequence(sequences, batch_first=True)
        lengths = torch.LongTensor([s.shape[0] for s in sequences])
        pred_scores = model.forward_temporal(padded, lengths).cpu().numpy()

        # Get ground truth if available
        ground_truth = []
        influencer_map = target_data["maps"]["influencer"]
        engagement_rates = target_data["ground_truth"]["engagement_rate"]

        for name in valid_names:
            if name in influencer_map:
                idx = influencer_map[name]
                ground_truth.append(engagement_rates[idx].item())
            else:
                ground_truth.append(0.0)

        ground_truth = np.array(ground_truth)

    return valid_names, pred_scores, ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Generate influencer rankings using trained model"
    )
    parser.add_argument(
        "--model-dir",
        default="saved_models_v8_final",
        help="Directory containing saved model",
    )
    parser.add_argument(
        "--graph-dir",
        default="graphs_enhanced_v3",
        help="Directory containing graph files",
    )
    parser.add_argument("--output", default="rankings.csv", help="Output CSV file")
    parser.add_argument(
        "--target-month", type=int, default=9, help="Target month (0-11)"
    )
    parser.add_argument(
        "--training-months", type=int, default=9, help="Number of training months"
    )
    parser.add_argument(
        "--use-test-set", action="store_true", help="Use test set from data split"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU usage (avoid GPU OOM errors)"
    )
    args = parser.parse_args()

    # Setup device
    if args.cpu:
        device = torch.device("cpu")
        print(f"Using device: cpu (forced)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Load model and config
    checkpoint, config, scaler, data_split = load_model_and_config(
        args.model_dir, device
    )

    # Load and preprocess graphs
    hetero_graphs, all_graphs_data = load_and_preprocess_graphs(
        args.graph_dir, scaler, config, args.training_months, device
    )

    # Initialize model
    print("\nInitializing model...")
    metadata = hetero_graphs[0]["graph"].metadata()
    model = BestInfluencerModel(
        config["INPUT_DIM"],
        config["GNN_HIDDEN"],
        config["GNN_OUT"],
        config["RNN_HIDDEN"],
        metadata,
        config["DROPOUT"],
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Dummy forward pass to initialize lazy parameters
    _ = model.encode_graphs(hetero_graphs)
    print(
        f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Determine which influencers to rank
    target_data = all_graphs_data[args.target_month]

    if args.use_test_set:
        influencer_list = data_split["test"]
        print(f"\nUsing test set: {len(influencer_list)} influencers")
    else:
        influencer_list = list(target_data["maps"]["influencer"].keys())
        print(f"\nRanking all influencers: {len(influencer_list)}")

    # Generate predictions
    valid_names, pred_scores, ground_truth = predict_rankings(
        model, hetero_graphs, influencer_list, target_data, device
    )

    if valid_names is None:
        print("Prediction failed!")
        return

    # Create ranking dataframe
    print("\nGenerating rankings...")
    ranking_df = pd.DataFrame(
        {
            "influencer": valid_names,
            "predicted_score": pred_scores,
            "actual_engagement": ground_truth,
        }
    )

    # Sort by predicted score
    ranking_df = ranking_df.sort_values("predicted_score", ascending=False).reset_index(
        drop=True
    )
    ranking_df["rank"] = range(1, len(ranking_df) + 1)

    # Reorder columns
    ranking_df = ranking_df[
        ["rank", "influencer", "predicted_score", "actual_engagement"]
    ]

    # Save to CSV
    ranking_df.to_csv(args.output, index=False)
    print(f"\n✅ Rankings saved to {args.output}")

    # Display top 10
    print("\nTop 10 Influencers:")
    print(ranking_df.head(10).to_string(index=False))

    # Calculate NDCG@50 if ground truth available
    if len(ground_truth) > 0:
        from sklearn.metrics import ndcg_score

        ndcg_50 = ndcg_score(
            ground_truth.reshape(1, -1),
            pred_scores.reshape(1, -1),
            k=min(50, len(ground_truth)),
        )
        # print(f"\nNDCG@50: {ndcg_50:.4f}")


if __name__ == "__main__":
    main()
