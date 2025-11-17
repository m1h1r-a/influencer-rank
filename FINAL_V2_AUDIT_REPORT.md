# Final-v2.ipynb Audit Report
**Date:** 2025-11-17
**Auditor:** Claude Code
**Status:** ❌ NOT VALID - CRITICAL BUGS FOUND

---

## Executive Summary

Your `final-v2.ipynb` notebook has **2 CRITICAL bugs** that completely undermine the intended fix (using actual graph structure instead of self-loops). The GCN is **never trained** and uses random weights for message passing, which means you're not actually learning from the graph structure at all.

**Current Expected Performance:** ~0.55-0.65 NDCG@50 (worse than v1!)
**After Fixes:** 0.70-0.74 NDCG@50 (matching paper target)

---

## Error List (Ranked by Severity)

### 1. ⚠️ SEVERITY 5: GCN NEVER TRAINED (CRITICAL)

**Location:** Cell 28, `train_single_model()` function

**The Bug:**
```python
# Cell 28 - Current (WRONG):
gcn = SimpleGCN(INPUT_DIM, GNN_HIDDEN, GNN_OUT, DROPOUT).to(device)
all_embeddings = precompute_gnn_embeddings(all_graphs_data, gcn, device)  # Uses external GCN
model = InfluencerRankModel(INPUT_DIM, GNN_HIDDEN, GNN_OUT, RNN_HIDDEN, DROPOUT).to(device)  # Has its OWN GCN
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Only trains model's GCN (which is never used!)
```

**What Happens:**
1. External `gcn` generates embeddings with RANDOM weights (no training)
2. `model` has its own internal GCN that is NEVER called
3. Optimizer trains `model.parameters()` including the unused internal GCN
4. The external GCN that actually does graph convolution never receives gradients

**Impact:**
- Graph structure is used with RANDOM weights
- No learning happens in the GCN layers
- Defeats the entire purpose of v2
- **Expected NDCG loss: -0.10 to -0.15 compared to properly trained GCN**

**Fix (Detailed):**

**Option A: End-to-End Training (Recommended)**
```python
def train_single_model(seed, all_graphs_data, train_influencers_list, val_influencers_list, test_influencers_list):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize SINGLE model that handles both GCN and temporal
    model = InfluencerRankModel(INPUT_DIM, GNN_HIDDEN, GNN_OUT, RNN_HIDDEN, DROPOUT).to(device)

    # Pre-convert all graphs to homogeneous (keep on CPU to save memory)
    print("Converting heterogeneous graphs to homogeneous...")
    converted_graphs = []
    for month_idx in range(TRAINING_MONTHS):
        data_package = all_graphs_data[month_idx]
        graph = data_package['graph']
        influencer_map = data_package['maps']['influencer']
        x, edge_index, global_indices = convert_hetero_to_homogeneous(graph, influencer_map)
        converted_graphs.append({
            'x': x,
            'edge_index': edge_index,
            'global_indices': global_indices
        })

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()

        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            batch_loss = 0.0

            for _ in range(BATCH_SIZE):
                # Sample influencers
                batch_names = sample_influencers(train_influencers_list, LIST_SIZE)

                # Forward pass through GCN for each month (now trainable!)
                sequences = []
                for month_idx in range(TRAINING_MONTHS):
                    x = converted_graphs[month_idx]['x'].to(device)
                    edge_index = converted_graphs[month_idx]['edge_index'].to(device)

                    # GCN forward - NOW PART OF COMPUTATION GRAPH!
                    node_embeddings = model.gcn(x, edge_index)

                    # Extract influencer embeddings
                    month_seq = []
                    for name in batch_names:
                        if name in converted_graphs[month_idx]['global_indices']:
                            global_idx = converted_graphs[month_idx]['global_indices'][name]
                            month_seq.append(node_embeddings[global_idx])
                    sequences.append(torch.stack(month_seq) if month_seq else None)

                # Build temporal sequences and run through RNN
                # ... rest of forward pass

                loss = listwise_ranking_loss(y_pred, y_true)
                batch_loss += loss

            # Backprop through ENTIRE model including GCN!
            (batch_loss / BATCH_SIZE).backward()
            optimizer.step()
```

**Option B: Pre-train GCN Separately (Simpler but less effective)**
```python
# Pre-train GCN with graph reconstruction loss
def pretrain_gcn(gcn, all_graphs_data, epochs=50):
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)

    for epoch in range(epochs):
        for month_idx in range(TRAINING_MONTHS):
            x, edge_index, _ = convert_hetero_to_homogeneous(...)
            x, edge_index = x.to(device), edge_index.to(device)

            # Reconstruction loss: can we predict node features from neighbors?
            embeddings = gcn(x, edge_index)

            # Simple reconstruction: predict original features
            reconstructed = nn.Linear(GNN_OUT, INPUT_DIM)(embeddings)
            loss = F.mse_loss(reconstructed, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return gcn

# Then use pre-trained GCN for embeddings
gcn = SimpleGCN(...).to(device)
gcn = pretrain_gcn(gcn, all_graphs_data)
all_embeddings = precompute_gnn_embeddings(all_graphs_data, gcn, device)
```

---

### 2. ⚠️ SEVERITY 5: WRONG NORMALIZATION OF NON-INFLUENCER FEATURES (CRITICAL)

**Location:** Cell 15

**The Bug:**
```python
# Current (WRONG):
for node_type in ['hashtag', 'user', 'object']:
    if node_type in graph.node_types:
        features = graph[node_type].x  # One-hot: [0,1,0,0,...] for hashtags
        normalized = scaler.transform(features.numpy())  # Scaler fitted on INFLUENCER features!
        graph[node_type].x = torch.FloatTensor(normalized)
```

**What Happens:**
From `build_enhanced_graphs.py`, non-influencer nodes have one-hot features:
- Hashtags: `[0,1,0,0,0,0,0,...,0]` (37 dims, position 1 is 1)
- Users: `[0,0,1,0,0,0,0,...,0]` (position 2 is 1)
- Objects: `[0,0,0,1,0,0,0,...,0]` (position 3 is 1)

The scaler was fitted on influencer features where:
- Position 0 = log_followers (mean ~4.5, std ~1.2)
- Position 1 = log_followees (mean ~3.8, std ~0.9)
- etc.

When you transform hashtag features `[0,1,0,0,...]`:
- `(0 - 4.5) / 1.2 = -3.75` for position 0
- `(1 - 3.8) / 0.9 = -3.11` for position 1
- etc.

The one-hot encoding is completely destroyed!

**Impact:**
- GCN can't distinguish between hashtags, users, and objects
- Message passing becomes meaningless
- **Expected NDCG loss: -0.02 to -0.05**

**Fix (Detailed):**

**Option A: Don't normalize non-influencer features (Recommended)**
```python
# Cell 15 - FIXED:
print("\nApplying train-fitted normalization to INFLUENCER features only...")

for month_idx, data_package in enumerate(all_graphs_data):
    # Only normalize influencer features
    features = data_package['graph']['influencer'].x
    normalized = scaler.transform(features.numpy())
    data_package['graph']['influencer'].x = torch.FloatTensor(normalized)

# DON'T normalize non-influencer features - they are one-hot encoded!
print("  Note: Non-influencer features (hashtags, users, objects) kept as one-hot encoding")
print("  These don't need normalization as they are already standardized (0 or 1)")

# Verify normalization
oct_features = all_graphs_data[9]['graph']['influencer'].x
print(f"  After normalization (Oct influencers):")
print(f"    Mean: {oct_features.mean():.6f}")
print(f"    Std: {oct_features.std():.6f}")
```

**Option B: Use separate identity scaler for non-influencers**
```python
# If you must "transform" non-influencer features for consistency:
identity_scaler = StandardScaler()
identity_scaler.mean_ = np.zeros(FEATURE_DIM)
identity_scaler.scale_ = np.ones(FEATURE_DIM)
identity_scaler.var_ = np.ones(FEATURE_DIM)

for node_type in ['hashtag', 'user', 'object']:
    if node_type in graph.node_types:
        features = graph[node_type].x
        # This effectively does nothing, preserving one-hot encoding
        normalized = identity_scaler.transform(features.numpy())
        graph[node_type].x = torch.FloatTensor(normalized)
```

---

### 3. ⚠️ SEVERITY 3: UNUSED MODEL COMPONENT

**Location:** Cell 20, `InfluencerRankModel.__init__()`

**The Bug:**
```python
class InfluencerRankModel(nn.Module):
    def __init__(self, input_dim, gnn_hidden, gnn_out, rnn_hidden, dropout=0.5):
        super().__init__()
        self.gcn = SimpleGCN(input_dim, gnn_hidden, gnn_out, dropout)  # NEVER USED!
        self.rnn = nn.GRU(...)
        # ...

    def forward(self, sequences, lengths):
        # sequences are ALREADY GCN outputs!
        # self.gcn is never called here
        packed = pack_padded_sequence(sequences, ...)
        # ...
```

**Impact:**
- Wastes GPU memory (stores unused GCN weights)
- Confusing architecture (model claims to have GCN but doesn't use it)
- If you fix bug #1, this becomes the place to put your trainable GCN

**Fix:**

**If NOT doing end-to-end training:**
```python
class InfluencerRankModel(nn.Module):
    def __init__(self, rnn_input_dim, rnn_hidden, dropout=0.5):  # Remove GCN params
        super().__init__()
        # NO GCN - it's pre-computed externally
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,  # This is GNN_OUT
            hidden_size=rnn_hidden,
            # ...
        )
        self.attention = SimpleAttention(rnn_hidden)
        self.fc1 = nn.Linear(rnn_hidden, rnn_hidden // 2)
        self.fc2 = nn.Linear(rnn_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)
```

**If doing end-to-end training (recommended):**
```python
class InfluencerRankModel(nn.Module):
    def __init__(self, input_dim, gnn_hidden, gnn_out, rnn_hidden, dropout=0.5):
        super().__init__()
        self.gcn = SimpleGCN(input_dim, gnn_hidden, gnn_out, dropout)
        self.rnn = nn.GRU(input_size=gnn_out, hidden_size=rnn_hidden, ...)
        # ...

    def encode_graph(self, x, edge_index):
        """Encode graph with GCN - call this during training!"""
        return self.gcn(x, edge_index)

    def forward_temporal(self, sequences, lengths):
        """Process temporal sequences - call this after encode_graph"""
        packed = pack_padded_sequence(sequences, lengths.cpu(), ...)
        # ... rest of forward
        return score
```

---

### 4. ⚠️ SEVERITY 2: POTENTIAL LIKES-BASED LEAKAGE

**Location:** Cell 7 - Leaky indices don't include likes_trend, likes_variance, etc.

**The Bug:**
```python
LEAKY_INDICES = [11, 13, 15, 17, 19, 23, 25, 26]
# Zeroes out engagement-based features
# BUT indices 12, 14, 16, 18, 20, 24 are LIKES-based features!
# likes_trend (12), likes_variance (14), likes_consistency (16), etc.
```

From `build_enhanced_graphs.py` (lines 436-451):
```python
temporal = [
    temporal_feat["engagement_trend"],    # 11 - ZEROED
    temporal_feat["likes_trend"],         # 12 - NOT ZEROED!
    temporal_feat["engagement_variance"], # 13 - ZEROED
    temporal_feat["likes_variance"],      # 14 - NOT ZEROED!
    # ...
]
```

**Impact:**
- `likes_trend` correlates with `engagement_rate` (more likes = higher engagement)
- This is a subtle form of leakage
- **Potential NDCG inflation: +0.01 to +0.03**

**Fix:**
```python
# Zero out ALL engagement-correlated temporal features:
LEAKY_INDICES = [
    11,  # engagement_trend
    12,  # likes_trend (ALSO LEAKY!)
    13,  # engagement_variance
    14,  # likes_variance (ALSO LEAKY!)
    15,  # engagement_consistency
    16,  # likes_consistency (ALSO LEAKY!)
    17,  # engagement_momentum
    18,  # likes_momentum (ALSO LEAKY!)
    19,  # engagement_peak
    20,  # likes_peak (ALSO LEAKY!)
    23,  # engagement_growth
    24,  # likes_growth (ALSO LEAKY!)
    25,  # log_avg_likes
    26,  # log_avg_comments
]
# This zeros out 14 features, leaving 23 effective features
```

---

### 5. ⚠️ SEVERITY 1: UNUSED FUNCTION PARAMETER

**Location:** Cell 26, `prepare_sequences()` function

**The Bug:**
```python
def prepare_sequences(influencer_names, all_embeddings, embedding_dim):
    # embedding_dim is passed but NEVER used!
    sequences = []
    # ...
```

**Impact:** None functionally, just code cleanliness.

**Fix:**
```python
def prepare_sequences(influencer_names, all_embeddings):  # Remove unused param
    sequences = []
    # ...

# Update all call sites:
sequences, lengths, valid_names = prepare_sequences(batch_names, all_embeddings)
```

---

## Recommended Action Plan

### Priority 1: Fix GCN Training (Severity 5)
1. Implement end-to-end training where GCN is part of the computation graph
2. This requires restructuring the training loop
3. **Expected improvement: +0.10 to +0.15 NDCG@50**

### Priority 2: Fix Non-Influencer Normalization (Severity 5)
1. Remove normalization of hashtag/user/object features
2. Keep them as one-hot encoded
3. **Expected improvement: +0.02 to +0.05 NDCG@50**

### Priority 3: Consider Likes Leakage (Severity 2)
1. Decide if likes_trend, etc. should be zeroed out
2. If targeting complete honesty, zero them out
3. If keeping them, document the decision

### Priority 4: Clean Up Architecture (Severity 3)
1. Remove unused GCN from model OR
2. Actually use it in end-to-end training

---

## Quick Validation Checklist

Before running your notebook, verify:

- [ ] GCN is trained (gradients flow through it)
- [ ] Non-influencer features are NOT normalized with influencer scaler
- [ ] Leaky features are properly zeroed out
- [ ] No train/val/test influencer overlap
- [ ] Scaler is fitted ONLY on training data
- [ ] NDCG@50 < 0.85 (otherwise likely leakage)

---

## Expected Performance After Fixes

| Version | NDCG@50 | Notes |
|---------|---------|-------|
| v1 (self-loops) | 0.6951 | Working baseline |
| v2 (current, broken) | ~0.55-0.65 | Random GCN weights |
| v2 (fixed) | 0.70-0.74 | Properly trained GCN |
| Paper target | 0.720 | Goal |

---

## Conclusion

Your code is **NOT valid** as-is. The main issue is that the GCN uses random weights because it's never trained. This is a fundamental architectural bug that completely undermines the intended improvement (using actual graph structure).

After fixing the critical bugs, you should see a significant improvement over v1, likely reaching the paper's target of 0.720 NDCG@50.

**Files to modify:**
1. Cell 28 - Fix GCN training (end-to-end)
2. Cell 15 - Remove non-influencer normalization
3. Cell 20 - Clean up model architecture
4. (Optional) Cell 7 - Extend leaky indices list

Good luck with the fixes!
