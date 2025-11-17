# InfluencerRank

Temporal graph neural network for predicting influencer engagement rates. Uses GCN + GRU architecture to model influencer behavior over time.

### Results

- **NDCG@50: 0.70** (ensemble of 5 models)
- Predicts October engagement using January-September data
- No data leakage (engagement-derived features removed)

### File Overview

- `parse_profiles.py`: Parses influencer profiles from `influencers.txt` to create `profiles_lookup.pkl`
- `build_enhanced_graphs.py`: Builds monthly heterogeneous graphs with 37-dim features
- `verify_graph.py`: Verifies graph structure and feature integrity
- `final-v1.ipynb`: Model using self-loop GCN (baseline)
- `final-v2.ipynb`: Model using actual graph structure (influencer→hashtag/user/object)

### Usage

1. Prepare your data:
   - `year_17/` - monthly post data (Jan-Dec subdirectories)
   - `influencers.txt` - profile information
   - `image_objects.csv` - detected objects per post

2. Build graphs:
   ```bash
   python parse_profiles.py
   python build_enhanced_graphs.py
   python verify_graph.py graphs_enhanced_v2/oct_graph.pt
   ```

3. Train model:
   - Run `final-v2.ipynb` in Jupyter/Kaggle
   - Outputs: NDCG@50 score and ranking predictions
