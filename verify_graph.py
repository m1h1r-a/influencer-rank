"""
Simple script to verify a saved enhanced graph.
"""
import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python verify_graph.py <path_to_graph.pt>")
    print("Example: python verify_graph.py graphs_enhanced_test/jan_graph.pt")
    sys.exit(1)

graph_path = sys.argv[1]

print(f"\nVerifying: {graph_path}")
print("="*80)

try:
    # Load with weights_only=False for PyTorch 2.6+ compatibility
    data = torch.load(graph_path, weights_only=False)

    print("✅ Graph loaded successfully\n")

    # Check structure
    graph = data['graph']
    metadata = data['metadata']
    ground_truth = data['ground_truth']

    print(f"Feature Dimensions: {metadata['feature_dim']}")
    print(f"Version: {metadata['version']}")
    print(f"Month: {metadata['month_name']} (index {metadata['month_idx']})")
    print(f"\nNode Counts:")
    print(f"  Influencers: {graph['influencer'].num_nodes}")
    if 'hashtag' in graph:
        print(f"  Hashtags: {graph['hashtag'].num_nodes}")
    if 'user' in graph:
        print(f"  Users: {graph['user'].num_nodes}")
    if 'object' in graph:
        print(f"  Objects: {graph['object'].num_nodes}")

    print(f"\nInfluencer Features:")
    print(f"  Shape: {graph['influencer'].x.shape}")
    print(f"  Non-zero: {(graph['influencer'].x != 0).sum().item():,}")

    print(f"\nGround Truth:")
    for key, tensor in ground_truth.items():
        non_zero = (tensor > 0).sum().item()
        print(f"  {key}: {non_zero}/{len(tensor)} non-zero ({100*non_zero/len(tensor):.1f}%)")

    print(f"\nCategories: {metadata['categories']}")

    print("\n" + "="*80)
    print("✅ VALIDATION PASSED")
    print("="*80 + "\n")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
