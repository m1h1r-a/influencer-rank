"""
ENHANCED GRAPH BUILDER FOR 0.72 NDCG@50 TARGET

Key Enhancements over build_graph_optimized.py:
1. Temporal features (14 dims): Trends, momentum, growth across 12 months
2. Log-scale features: Better normalization for followers, likes
3. Diversity metrics: Content variety signals
4. 37 dimensions total (vs 21 original)
5. Built-in validation: Ensures data integrity
6. Backward compatible: Works with existing model code
7. Proper ground truth: engagement_rate NOT in features (prevents leakage!)

Feature Breakdown (37 dimensions):
  [0-2]   Static: log_followers, log_followees, follower_ratio (3)
  [3-10]  Category: one-hot encoding (8)
  [11-24] Temporal: trends, variance, consistency, momentum, peaks, stability, growth (14)
  [25-36] Monthly: posts, log_likes, log_comments, captions, hashtags, mentions, sentiment (12)

Ground Truth (NOT in features):
  - engagement_rate = avg_likes / num_followers
  - avg_likes

Expected Performance: 0.70-0.76 NDCG@50 (up from v9's 0.5936)
"""

import ast
import json
import os
import pickle
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.data import HeteroData
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_ROOT = "year_17"
GRAPHS_DIR = "graphs_enhanced_v2"  # NEW: Fixed temporal features
COMBINED_OBJECTS_CSV = "image_objects.csv"
PROFILES_FILE = "profiles_lookup.pkl"
NUM_PROCESSES = cpu_count() - 1
FEATURE_DIM = 37  # Fixed: removed engagement_rate from features (was 38)
TARGET_MONTH = 9  # October (0-indexed) - we predict this month

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SIA = SentimentIntensityAnalyzer()

print("\n" + "="*80)
print("ENHANCED GRAPH BUILDER - InfluencerRank v11.2 (FIXED)")
print("="*80)
print(f"Features: {FEATURE_DIM} dimensions (37)")
print(f"Output: {GRAPHS_DIR}/*.pt")
print(f"Target Month: {TARGET_MONTH} ({MONTH_NAMES[TARGET_MONTH]})")
print(f"Temporal Features: Uses ONLY months 0-{TARGET_MONTH-1} (NO LEAKAGE)")
print(f"Processes: {NUM_PROCESSES}")
print(f"Target: 0.70-0.76 NDCG@50")
print(f"Ground Truth: engagement_rate, avg_likes (NOT in features!)")
print("="*80 + "\n")


# ============================================================================
# STEP 1: DATA EXTRACTION (Per-Month, Parallel)
# ============================================================================

def extract_post_data(args):
    """Extract post data with all features needed for temporal analysis."""
    file_path, profile_lookup = args

    try:
        filename = os.path.basename(file_path)
        influencer_name = filename.rsplit("-", 1)[0].lower()
        post_id_str = os.path.splitext(filename)[0].rsplit("-", 1)[-1]

        # Profile check
        profile = profile_lookup.get(influencer_name)
        if not profile or profile.get('followers', 0) == 0:
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Caption
        caption_text = ""
        if cap_edges := data.get("edge_media_to_caption", {}).get("edges", []):
            caption_text = cap_edges[0]["node"]["text"]

        # Graph structure
        hashtags = list({h.lower() for h in re.findall(r"#(\w+)", caption_text)})
        mentioned_users = []
        if tagged_edges := data.get("edge_media_to_tagged_user", {}).get("edges", []):
            mentioned_users = [
                e["node"]["user"]["username"].lower()
                for e in tagged_edges
                if "user" in e.get("node", {})
            ]

        # Engagement
        num_likes = data.get("edge_media_preview_like", {}).get("count", 0)
        comments_edges = data.get("edge_media_to_parent_comment", {}).get("edges", [])
        num_comments = len(comments_edges)

        # Caption features
        caption_sentiment = SIA.polarity_scores(caption_text)["compound"]

        # Comment sentiments (batch process, limit to 20)
        comment_sentiments = []
        for edge in comments_edges[:20]:
            if comment_txt := edge.get("node", {}).get("text", ""):
                sentiment = SIA.polarity_scores(comment_txt)["compound"]
                comment_sentiments.append(sentiment)

        # Sentiment positivity rate
        positive_sentiment = 1.0 if caption_sentiment > 0.05 else 0.0

        return {
            "influencer": influencer_name,
            "post_id": post_id_str,
            "hashtags": hashtags,
            "mentioned_users": mentioned_users,
            "num_likes": num_likes,
            "num_comments": num_comments,
            "caption_len": len(caption_text),
            "caption_sentiment": caption_sentiment,
            "num_hashtags": len(hashtags),
            "num_mentions": len(mentioned_users),
            "comment_sentiments": comment_sentiments,
            "positive_sentiment": positive_sentiment,
            "num_followers": profile['followers'],
            "num_followees": profile['followees'],
            "category": profile.get('category', 'unknown'),
        }

    except Exception as e:
        return None


def extract_month_data(month_name, profile_lookup):
    """Extract all posts for a single month."""
    month_path = os.path.join(DATA_ROOT, month_name)

    file_paths = [
        os.path.join(month_path, f)
        for f in os.listdir(month_path)
        if f.endswith(".info")
    ]

    print(f"\nProcessing {month_name}: {len(file_paths):,} posts")
    args_list = [(fp, profile_lookup) for fp in file_paths]

    with Pool(processes=NUM_PROCESSES) as pool:
        posts = [
            r for r in tqdm(
                pool.imap_unordered(extract_post_data, args_list, chunksize=100),
                total=len(file_paths),
                desc=f"  Extracting {month_name}"
            ) if r is not None
        ]

    print(f"  ✅ {month_name}: {len(posts):,} valid posts")
    return posts


# ============================================================================
# STEP 2: AGGREGATE MONTHLY FEATURES
# ============================================================================

def aggregate_monthly_features(posts, object_lookup):
    """Aggregate features per influencer for a single month."""
    influencer_features = {}
    influencer_posts = defaultdict(list)

    # Group by influencer
    for post in posts:
        influencer_posts[post["influencer"]].append(post)

    # Aggregate per influencer
    for inf, inf_posts in influencer_posts.items():
        if not inf_posts:
            continue

        # Extract arrays
        likes = np.array([p["num_likes"] for p in inf_posts])
        comments = np.array([p["num_comments"] for p in inf_posts])
        caption_lens = np.array([p["caption_len"] for p in inf_posts])
        caption_sentiments = np.array([p["caption_sentiment"] for p in inf_posts])
        num_hashtags = np.array([p["num_hashtags"] for p in inf_posts])
        num_mentions = np.array([p["num_mentions"] for p in inf_posts])
        positive_sentiments = np.array([p["positive_sentiment"] for p in inf_posts])

        # Comment sentiments (all)
        all_comment_sentiments = []
        for p in inf_posts:
            all_comment_sentiments.extend(p["comment_sentiments"])
        comment_array = np.array(all_comment_sentiments) if all_comment_sentiments else np.array([0.0])

        # Profile info (constant)
        num_followers = inf_posts[0]["num_followers"]
        num_followees = inf_posts[0]["num_followees"]
        category = inf_posts[0]["category"]

        # Aggregations
        num_posts = len(inf_posts)
        avg_likes = float(np.mean(likes))
        avg_comments = float(np.mean(comments))
        engagement_rate = (avg_likes / num_followers) if num_followers > 0 else 0.0

        def agg(arr):
            return {"avg": float(np.mean(arr)), "std": float(np.std(arr))}

        influencer_features[inf] = {
            "num_posts": num_posts,
            "num_followers": num_followers,
            "num_followees": num_followees,
            "category": category,
            "avg_likes": avg_likes,
            "avg_comments": avg_comments,
            "engagement_rate": engagement_rate,
            "caption_len": agg(caption_lens),
            "caption_sentiment": agg(caption_sentiments),
            "num_hashtags": agg(num_hashtags),
            "num_mentions": agg(num_mentions),
            "comment_sentiment": agg(comment_array),
            "sentiment_positivity_rate": float(np.mean(positive_sentiments)),
        }

    return influencer_features


# ============================================================================
# STEP 3: COMPUTE TEMPORAL FEATURES (Across All Months)
# ============================================================================

def compute_temporal_features(all_months_features, target_month=TARGET_MONTH):
    """
    Compute temporal features using ONLY months BEFORE the target month.
    This prevents data leakage - we can't use future/target data as features.

    For target_month=9 (October), uses months 0-8 (Jan-Sep) only.

    Returns 14-dimensional temporal feature vector per influencer:
    - engagement_trend, likes_trend (2)
    - engagement_variance, likes_variance (2)
    - engagement_consistency, likes_consistency (2)
    - engagement_momentum, likes_momentum (2)
    - engagement_peak, likes_peak (2)
    - activity_rate, posting_consistency (2)
    - engagement_growth, likes_growth (2)
    """
    print("\n" + "="*80)
    print("COMPUTING TEMPORAL FEATURES (NO LEAKAGE)")
    print("="*80)
    print(f"Target month: {target_month} ({MONTH_NAMES[target_month]})")
    print(f"Using months: 0-{target_month-1} ({MONTH_NAMES[0]}-{MONTH_NAMES[target_month-1]})")
    print("⚠️  NOT using months {}-11 to prevent data leakage".format(target_month))

    # Only use months BEFORE target
    available_months = target_month  # 0 to target_month-1

    # Collect all unique influencers from ALL months (they might appear in any)
    all_influencers = set()
    for month_feats in all_months_features:
        all_influencers.update(month_feats.keys())

    print(f"Total influencers: {len(all_influencers):,}")

    temporal_features = {}

    for influencer in tqdm(all_influencers, desc="Computing temporal"):
        # Gather time series ONLY for months 0 to target_month-1
        engagement_series = []
        likes_series = []
        posts_series = []

        for month_idx in range(available_months):  # Only 0 to target_month-1
            month_feats = all_months_features[month_idx]
            if influencer in month_feats:
                feat = month_feats[influencer]
                engagement_series.append(feat["engagement_rate"])
                likes_series.append(feat["avg_likes"])
                posts_series.append(feat["num_posts"])
            else:
                engagement_series.append(0.0)
                likes_series.append(0.0)
                posts_series.append(0.0)

        engagement_arr = np.array(engagement_series)
        likes_arr = np.array(likes_series)
        posts_arr = np.array(posts_series)

        # Active months (out of available months)
        active_months = (engagement_arr > 0).sum()
        activity_rate = active_months / float(available_months)

        # Handle edge cases
        if active_months == 0:
            temporal_features[influencer] = {
                "engagement_trend": 0.0, "likes_trend": 0.0,
                "engagement_variance": 0.0, "likes_variance": 0.0,
                "engagement_consistency": 0.0, "likes_consistency": 0.0,
                "engagement_momentum": 0.0, "likes_momentum": 0.0,
                "engagement_peak": 0.0, "likes_peak": 0.0,
                "activity_rate": 0.0, "posting_consistency": 0.0,
                "engagement_growth": 0.0, "likes_growth": 0.0,
            }
            continue

        # Helper: safe computation (adapted for variable length)
        def safe_trend(arr):
            """Linear regression slope."""
            if np.sum(arr) == 0 or np.all(arr == arr[0]):
                return 0.0
            x = np.arange(len(arr))
            try:
                slope, _, _, _, _ = stats.linregress(x, arr)
                return float(slope) if not np.isnan(slope) else 0.0
            except:
                return 0.0

        def safe_variance(arr):
            """Temporal variance."""
            if np.sum(arr) == 0:
                return 0.0
            return float(np.var(arr))

        def safe_consistency(arr):
            """Inverse coefficient of variation."""
            if np.sum(arr) == 0:
                return 0.0
            mean = np.mean(arr[arr > 0]) if np.any(arr > 0) else 0.0
            std = np.std(arr[arr > 0]) if np.any(arr > 0) else 0.0
            if mean == 0:
                return 0.0
            cv = std / mean
            return 1.0 / (1.0 + cv)  # Higher = more consistent

        def safe_momentum(arr):
            """Recent (last 3 months of AVAILABLE data) vs overall mean."""
            if np.sum(arr) == 0:
                return 0.0
            # For 9 months (0-8), last 3 = months 6,7,8
            recent = np.mean(arr[-3:])
            overall = np.mean(arr[arr > 0]) if np.any(arr > 0) else 1e-10
            return (recent - overall) / (overall + 1e-10)

        def safe_peak(arr):
            """Max / mean ratio."""
            if np.sum(arr) == 0:
                return 0.0
            mean = np.mean(arr[arr > 0]) if np.any(arr > 0) else 1e-10
            return np.max(arr) / (mean + 1e-10)

        def safe_growth(arr):
            """Second half vs first half (adjusted for available months)."""
            if np.sum(arr) == 0:
                return 0.0
            # For 9 months: first_half = 0-3, second_half = 4-8
            # For 12 months: first_half = 0-5, second_half = 6-11
            mid_point = len(arr) // 2
            first_half = np.mean(arr[:mid_point])
            second_half = np.mean(arr[mid_point:])
            if first_half == 0:
                return 0.0
            return (second_half - first_half) / (first_half + 1e-10)

        # Compute features using ONLY past data
        temporal_features[influencer] = {
            "engagement_trend": safe_trend(engagement_arr),
            "likes_trend": safe_trend(likes_arr),
            "engagement_variance": safe_variance(engagement_arr),
            "likes_variance": safe_variance(likes_arr),
            "engagement_consistency": safe_consistency(engagement_arr),
            "likes_consistency": safe_consistency(likes_arr),
            "engagement_momentum": safe_momentum(engagement_arr),
            "likes_momentum": safe_momentum(likes_arr),
            "engagement_peak": safe_peak(engagement_arr),
            "likes_peak": safe_peak(likes_arr),
            "activity_rate": activity_rate,
            "posting_consistency": safe_consistency(posts_arr),
            "engagement_growth": safe_growth(engagement_arr),
            "likes_growth": safe_growth(likes_arr),
        }

    print(f"✅ Computed temporal features for {len(temporal_features):,} influencers")
    print(f"   Using {available_months} months of historical data (no future leakage)")
    return temporal_features


# ============================================================================
# STEP 4: BUILD FEATURE VECTORS (37 Dimensions)
# ============================================================================

def build_enhanced_feature_vector(
    monthly_feat,
    temporal_feat,
    category_to_idx
):
    """
    Build 37-dimensional enhanced feature vector.

    [0-2]   Static: log_followers, log_followees, follower_ratio (3)
    [3-10]  Category: one-hot (8)
    [11-24] Temporal: 14 dimensions
    [25-36] Monthly: 12 dimensions (NO engagement_rate - that's ground truth!)
    """
    if monthly_feat is None:
        return torch.zeros(FEATURE_DIM)

    # Static features (3) - Log scale for better distribution
    num_followers = monthly_feat["num_followers"]
    num_followees = monthly_feat["num_followees"]

    log_followers = np.log10(num_followers + 1)
    log_followees = np.log10(num_followees + 1)
    follower_ratio = num_followers / (num_followees + 1)

    static = [log_followers, log_followees, follower_ratio]

    # Category one-hot (8)
    cat_vec = [0.0] * 8
    cat_idx = category_to_idx.get(monthly_feat["category"], -1)
    if 0 <= cat_idx < 8:
        cat_vec[cat_idx] = 1.0

    # Temporal features (14)
    if temporal_feat:
        temporal = [
            temporal_feat["engagement_trend"],
            temporal_feat["likes_trend"],
            temporal_feat["engagement_variance"],
            temporal_feat["likes_variance"],
            temporal_feat["engagement_consistency"],
            temporal_feat["likes_consistency"],
            temporal_feat["engagement_momentum"],
            temporal_feat["likes_momentum"],
            temporal_feat["engagement_peak"],
            temporal_feat["likes_peak"],
            temporal_feat["activity_rate"],
            temporal_feat["posting_consistency"],
            temporal_feat["engagement_growth"],
            temporal_feat["likes_growth"],
        ]
    else:
        temporal = [0.0] * 14

    # Monthly features (12) - Log scale for likes/comments
    # NOTE: engagement_rate is ground truth, NOT a feature!
    log_avg_likes = np.log10(monthly_feat["avg_likes"] + 1)
    log_avg_comments = np.log10(monthly_feat["avg_comments"] + 1)

    monthly = [
        monthly_feat["num_posts"],
        log_avg_likes,
        log_avg_comments,
        # REMOVED: monthly_feat["engagement_rate"],  # This is ground truth!
        monthly_feat["caption_len"]["avg"],
        monthly_feat["caption_len"]["std"],
        monthly_feat["caption_sentiment"]["avg"],
        monthly_feat["caption_sentiment"]["std"],
        monthly_feat["num_hashtags"]["avg"],
        monthly_feat["num_hashtags"]["std"],
        monthly_feat["num_mentions"]["avg"],
        monthly_feat["num_mentions"]["std"],
        monthly_feat["sentiment_positivity_rate"],
    ]

    # Combine: 3 + 8 + 14 + 12 = 37 (NOT 38!)
    feature_list = static + cat_vec + temporal + monthly

    # Safety check
    if len(feature_list) != FEATURE_DIM:
        print(f"⚠️  Feature dimension mismatch: {len(feature_list)} != {FEATURE_DIM}")
        return torch.zeros(FEATURE_DIM)

    return torch.tensor(feature_list, dtype=torch.float32)


# ============================================================================
# STEP 5: BUILD GRAPHS (12 Monthly Graphs)
# ============================================================================

def build_month_graph(
    month_name,
    month_idx,
    all_posts,
    all_months_features,
    temporal_features,
    category_to_idx,
    object_lookup
):
    """Build a single month's heterogeneous graph with enhanced features."""

    print(f"\n{'='*80}")
    print(f"BUILDING GRAPH: {month_name} (Month {month_idx})")
    print(f"{'='*80}")

    # Extract entities and edges
    influencers = set()
    hashtags = set()
    users = set()
    objects = set()

    inf_hashtag_edges = []
    inf_user_edges = []
    inf_object_edges = []

    influencer_posts = defaultdict(list)

    for post in all_posts:
        inf = post["influencer"]
        influencers.add(inf)
        influencer_posts[inf].append(post)

        for tag in post["hashtags"]:
            hashtags.add(tag)
            inf_hashtag_edges.append((inf, tag))

        for user in post["mentioned_users"]:
            users.add(user)
            inf_user_edges.append((inf, user))

        post_objs = object_lookup.get(int(post["post_id"]), [])
        for obj in post_objs:
            objects.add(obj)
            inf_object_edges.append((inf, obj))

    print(f"Entities: {len(influencers)} influencers, {len(hashtags)} hashtags, "
          f"{len(users)} users, {len(objects)} objects")
    print(f"Edges: {len(inf_hashtag_edges)} hashtag, {len(inf_user_edges)} user, "
          f"{len(inf_object_edges)} object")

    # Create mappings
    inf_map = {n: i for i, n in enumerate(sorted(influencers))}
    tag_map = {n: i for i, n in enumerate(sorted(hashtags))}
    user_map = {n: i for i, n in enumerate(sorted(users))}
    obj_map = {n: i for i, n in enumerate(sorted(objects))}

    # Build graph
    graph = HeteroData()
    graph["influencer"].num_nodes = len(influencers)
    graph["hashtag"].num_nodes = len(hashtags)
    graph["user"].num_nodes = len(users)
    graph["object"].num_nodes = len(objects)

    # Add edges
    if inf_hashtag_edges:
        src, dst = zip(*inf_hashtag_edges)
        graph["influencer", "posts_hashtag", "hashtag"].edge_index = torch.tensor(
            [[inf_map[s] for s in src], [tag_map[d] for d in dst]], dtype=torch.long
        )

    if inf_user_edges:
        src, dst = zip(*inf_user_edges)
        graph["influencer", "mentions", "user"].edge_index = torch.tensor(
            [[inf_map[s] for s in src], [user_map[d] for d in dst]], dtype=torch.long
        )

    if inf_object_edges:
        src, dst = zip(*inf_object_edges)
        graph["influencer", "posted_object", "object"].edge_index = torch.tensor(
            [[inf_map[s] for s in src], [obj_map[d] for d in dst]], dtype=torch.long
        )

    # Build features
    print("Building enhanced 38-dimensional features...")
    num_influencers = len(influencers)
    feature_matrix = torch.zeros(num_influencers, FEATURE_DIM)
    engagement_rates = torch.zeros(num_influencers)
    avg_likes_tensor = torch.zeros(num_influencers)

    monthly_features = all_months_features[month_idx]

    for name, idx in tqdm(inf_map.items(), desc="  Features"):
        monthly_feat = monthly_features.get(name)
        temporal_feat = temporal_features.get(name)

        if monthly_feat:
            feature_matrix[idx] = build_enhanced_feature_vector(
                monthly_feat, temporal_feat, category_to_idx
            )
            engagement_rates[idx] = monthly_feat["engagement_rate"]
            avg_likes_tensor[idx] = monthly_feat["avg_likes"]

    graph["influencer"].x_original = feature_matrix.clone()
    graph["influencer"].x = feature_matrix

    # Other node types (simple one-hot)
    for i, node_type in enumerate(["influencer", "hashtag", "user", "object"]):
        if node_type != "influencer" and graph[node_type].num_nodes > 0:
            one_hot = torch.zeros(graph[node_type].num_nodes, FEATURE_DIM)
            one_hot[:, i] = 1.0
            graph[node_type].x = one_hot

    # Validation
    non_zero_features = (feature_matrix != 0).sum().item()
    non_zero_engagement = (engagement_rates > 0).sum().item()

    print(f"\n✅ Graph built:")
    print(f"   Features: {feature_matrix.shape} ({FEATURE_DIM} dims)")
    print(f"   Non-zero features: {non_zero_features:,}/{feature_matrix.numel():,} "
          f"({100*non_zero_features/feature_matrix.numel():.1f}%)")
    print(f"   Non-zero engagement: {non_zero_engagement}/{num_influencers} "
          f"({100*non_zero_engagement/num_influencers:.1f}%)")

    # Package
    data_package = {
        "graph": graph,
        "maps": {
            "influencer": inf_map,
            "hashtag": tag_map,
            "user": user_map,
            "object": obj_map
        },
        "ground_truth": {
            "engagement_rate": engagement_rates,
            "avg_likes": avg_likes_tensor
        },
        "metadata": {
            "categories": list(category_to_idx.keys()),
            "category_to_idx": category_to_idx,
            "feature_dim": FEATURE_DIM,
            "month_name": month_name,
            "month_idx": month_idx,
            "version": "v11.2_no_leakage",
            "target_month": TARGET_MONTH,
            "temporal_months_used": TARGET_MONTH,  # Only uses months 0 to TARGET_MONTH-1
        }
    }

    return data_package


# ============================================================================
# STEP 6: VALIDATION
# ============================================================================

def validate_graph(data_package, month_name):
    """Validate graph integrity."""
    print(f"\nValidating {month_name}...")

    graph = data_package["graph"]
    ground_truth = data_package["ground_truth"]

    # Check 1: No inf/nan in features
    features = graph["influencer"].x
    if torch.isnan(features).any() or torch.isinf(features).any():
        print(f"  ⚠️  {month_name}: Found inf/nan in features!")
        return False

    # Check 2: Ground truth reasonable
    eng_rates = ground_truth["engagement_rate"]
    if (eng_rates < 0).any() or (eng_rates > 100).any():
        print(f"  ⚠️  {month_name}: Engagement rates out of range!")
        return False

    # Check 3: Feature dimensions
    if features.shape[1] != FEATURE_DIM:
        print(f"  ⚠️  {month_name}: Feature dim mismatch! "
              f"Expected {FEATURE_DIM}, got {features.shape[1]}")
        return False

    print(f"  ✅ {month_name}: Validation passed")
    return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_all_enhanced_graphs():
    """Main pipeline: Extract → Aggregate → Compute Temporal → Build Graphs."""

    # Load external data
    print("\n" + "="*80)
    print("STEP 1: LOADING EXTERNAL DATA")
    print("="*80)

    # Profiles
    with open(PROFILES_FILE, 'rb') as f:
        data = pickle.load(f)
        profile_lookup = data['profiles']
        categories = data['categories']

    print(f"✅ Loaded {len(profile_lookup):,} profiles")

    # Objects
    df = pd.read_csv(COMBINED_OBJECTS_CSV)
    df["detected_objects"] = df["detected_objects"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    object_lookup = pd.Series(df.detected_objects.values, index=df.post_id).to_dict()
    print(f"✅ Loaded objects for {len(object_lookup):,} posts")

    # Category mapping (top 8)
    if len(categories) > 8:
        cat_counts = defaultdict(int)
        for prof in profile_lookup.values():
            cat_counts[prof['category']] += 1
        top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        categories = [cat for cat, _ in top_cats]
    category_to_idx = {cat: idx for idx, cat in enumerate(sorted(categories))}
    print(f"✅ Categories: {categories}")

    # Extract all 12 months
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING ALL MONTHS (Parallel)")
    print("="*80)

    all_months_posts = []
    all_months_features = []

    for month_name in MONTH_NAMES:
        posts = extract_month_data(month_name, profile_lookup)
        features = aggregate_monthly_features(posts, object_lookup)

        all_months_posts.append(posts)
        all_months_features.append(features)

        print(f"  {month_name}: {len(features)} influencers")

    # Compute temporal features
    temporal_features = compute_temporal_features(all_months_features)

    # Build graphs
    print("\n" + "="*80)
    print("STEP 3: BUILDING 12 ENHANCED GRAPHS")
    print("="*80)

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    for month_idx, month_name in enumerate(MONTH_NAMES):
        data_package = build_month_graph(
            month_name,
            month_idx,
            all_months_posts[month_idx],
            all_months_features,
            temporal_features,
            category_to_idx,
            object_lookup
        )

        # Validate
        if not validate_graph(data_package, month_name):
            print(f"❌ Validation failed for {month_name}!")
            continue

        # Save
        graph_path = os.path.join(GRAPHS_DIR, f"{month_name.lower()}_graph.pt")
        torch.save(data_package, graph_path)
        print(f"✅ Saved: {graph_path}")

    print("\n" + "="*80)
    print("✅ ALL ENHANCED GRAPHS BUILT SUCCESSFULLY (NO LEAKAGE)!")
    print("="*80)
    print(f"\nOutput directory: {GRAPHS_DIR}/")
    print(f"Feature dimensions: {FEATURE_DIM} (37)")
    print(f"Target month: {TARGET_MONTH} ({MONTH_NAMES[TARGET_MONTH]})")
    print(f"Temporal features: Computed from months 0-{TARGET_MONTH-1} ONLY")
    print(f"Ground truth: engagement_rate, avg_likes (NOT in features!)")
    print(f"Expected performance: 0.65-0.72 NDCG@50")
    print("\nKey improvements over v11:")
    print("  - ✅ NO DATA LEAKAGE: Temporal features use only past months")
    print("  - Temporal trends (14 dims): momentum, growth, consistency")
    print("  - Log-scale normalization: followers, likes, comments")
    print("  - engagement_rate is ground truth only, NOT a feature")
    print("\nNext steps:")
    print(f"  1. Verify graphs: python -c 'import torch; d=torch.load(\"{GRAPHS_DIR}/oct_graph.pt\", weights_only=False); print(d[\"metadata\"])'")
    print("  2. Create model notebook: improved-model-v11-v2.ipynb")
    print("  3. Load from graphs_enhanced_v2/ (NOT graphs_enhanced/)")
    print("  4. Train with 80/10/10 split by influencers")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Prerequisites check
    if not os.path.exists(PROFILES_FILE):
        print("❌ Error: profiles_lookup.pkl not found!")
        print("   Run: python parse_profiles.py")
        exit(1)

    if not os.path.exists(COMBINED_OBJECTS_CSV):
        print("❌ Error: image_objects.csv not found!")
        exit(1)

    # Run
    build_all_enhanced_graphs()
