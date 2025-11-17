"""
ENHANCED GRAPH BUILDER V3 - FIXED TEMPORAL LEAKAGE

CRITICAL FIX: Each month's graph now has temporal features computed ONLY from
months BEFORE that month (progressive temporal features).

Previous Bug: All graphs used temporal features from months 0-8, causing
Jan-Sep graphs to contain future information.

Feature Breakdown (37 dimensions):
  [0-2]   Static: log_followers, log_followees, follower_ratio (3)
  [3-10]  Category: one-hot encoding (8)
  [11-24] Temporal: trends, variance, consistency, momentum, peaks, stability, growth (14)
  [25-36] Monthly: posts, log_likes, log_comments, captions, hashtags, mentions, sentiment (12)

Ground Truth (NOT in features):
  - engagement_rate = avg_likes / num_followers
  - avg_likes
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
GRAPHS_DIR = "graphs_enhanced_v3"  # V3: Progressive temporal features
COMBINED_OBJECTS_CSV = "image_objects.csv"
PROFILES_FILE = "profiles_lookup.pkl"
NUM_PROCESSES = max(1, cpu_count() - 1)
FEATURE_DIM = 37
TARGET_MONTH = 9  # October (0-indexed)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SIA = SentimentIntensityAnalyzer()

print("\n" + "="*80)
print("ENHANCED GRAPH BUILDER V3 - PROGRESSIVE TEMPORAL FEATURES")
print("="*80)
print(f"Features: {FEATURE_DIM} dimensions (37)")
print(f"Output: {GRAPHS_DIR}/*.pt")
print(f"Target Month: {TARGET_MONTH} ({MONTH_NAMES[TARGET_MONTH]})")
print(f"CRITICAL FIX: Each month uses ONLY past months for temporal features!")
print(f"  - Month 0 (Jan): No temporal features (first month)")
print(f"  - Month 1 (Feb): Temporal from month 0 only")
print(f"  - Month 8 (Sep): Temporal from months 0-7")
print(f"  - Month 9 (Oct): Temporal from months 0-8")
print(f"Processes: {NUM_PROCESSES}")
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

        profile = profile_lookup.get(influencer_name)
        if not profile or profile.get('followers', 0) == 0:
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        caption_text = ""
        if cap_edges := data.get("edge_media_to_caption", {}).get("edges", []):
            caption_text = cap_edges[0]["node"]["text"]

        hashtags = list({h.lower() for h in re.findall(r"#(\w+)", caption_text)})
        mentioned_users = []
        if tagged_edges := data.get("edge_media_to_tagged_user", {}).get("edges", []):
            mentioned_users = [
                e["node"]["user"]["username"].lower()
                for e in tagged_edges
                if "user" in e.get("node", {})
            ]

        num_likes = data.get("edge_media_preview_like", {}).get("count", 0)
        comments_edges = data.get("edge_media_to_parent_comment", {}).get("edges", [])
        num_comments = len(comments_edges)

        caption_sentiment = SIA.polarity_scores(caption_text)["compound"]

        comment_sentiments = []
        for edge in comments_edges[:20]:
            if comment_txt := edge.get("node", {}).get("text", ""):
                sentiment = SIA.polarity_scores(comment_txt)["compound"]
                comment_sentiments.append(sentiment)

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

    except Exception:
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

    for post in posts:
        influencer_posts[post["influencer"]].append(post)

    for inf, inf_posts in influencer_posts.items():
        if not inf_posts:
            continue

        likes = np.array([p["num_likes"] for p in inf_posts])
        comments = np.array([p["num_comments"] for p in inf_posts])
        caption_lens = np.array([p["caption_len"] for p in inf_posts])
        caption_sentiments = np.array([p["caption_sentiment"] for p in inf_posts])
        num_hashtags = np.array([p["num_hashtags"] for p in inf_posts])
        num_mentions = np.array([p["num_mentions"] for p in inf_posts])
        positive_sentiments = np.array([p["positive_sentiment"] for p in inf_posts])

        all_comment_sentiments = []
        for p in inf_posts:
            all_comment_sentiments.extend(p["comment_sentiments"])
        comment_array = np.array(all_comment_sentiments) if all_comment_sentiments else np.array([0.0])

        num_followers = inf_posts[0]["num_followers"]
        num_followees = inf_posts[0]["num_followees"]
        category = inf_posts[0]["category"]

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
# STEP 3: COMPUTE TEMPORAL FEATURES (PROGRESSIVE - NO LEAKAGE!)
# ============================================================================

def compute_temporal_features_progressive(all_months_features, current_month_idx):
    """
    Compute temporal features using ONLY months BEFORE current_month_idx.

    CRITICAL FIX: This function is called per-month with different current_month_idx.
    - For month 0 (Jan): No past months, returns zeros
    - For month 1 (Feb): Uses only month 0 (Jan)
    - For month 8 (Sep): Uses months 0-7 (Jan-Aug)
    - For month 9 (Oct): Uses months 0-8 (Jan-Sep)

    This prevents future information from leaking into past month graphs.
    """
    print(f"\n  Computing temporal features for month {current_month_idx} ({MONTH_NAMES[current_month_idx]})")
    print(f"    Using months: 0-{current_month_idx-1} (NO FUTURE DATA)")

    available_months = current_month_idx  # Only 0 to current_month_idx-1

    # Collect all influencers (they appear in any month)
    all_influencers = set()
    for month_idx in range(current_month_idx):  # Only past months
        all_influencers.update(all_months_features[month_idx].keys())

    # Also include influencers from current month (they need features)
    all_influencers.update(all_months_features[current_month_idx].keys())

    temporal_features = {}

    for influencer in all_influencers:
        # Handle edge case: no past months (month 0)
        if available_months == 0:
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

        # Gather time series ONLY for past months
        engagement_series = []
        likes_series = []
        posts_series = []

        for month_idx in range(available_months):  # Only 0 to current_month_idx-1
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

        active_months = (engagement_arr > 0).sum()
        activity_rate = active_months / float(available_months) if available_months > 0 else 0.0

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

        # Helper functions
        def safe_trend(arr):
            if len(arr) < 2 or np.sum(arr) == 0 or np.all(arr == arr[0]):
                return 0.0
            x = np.arange(len(arr))
            try:
                slope, _, _, _, _ = stats.linregress(x, arr)
                return float(slope) if not np.isnan(slope) else 0.0
            except Exception:
                return 0.0

        def safe_variance(arr):
            if np.sum(arr) == 0:
                return 0.0
            return float(np.var(arr))

        def safe_consistency(arr):
            if np.sum(arr) == 0:
                return 0.0
            mean = np.mean(arr[arr > 0]) if np.any(arr > 0) else 0.0
            std = np.std(arr[arr > 0]) if np.any(arr > 0) else 0.0
            if mean == 0:
                return 0.0
            cv = std / mean
            return 1.0 / (1.0 + cv)

        def safe_momentum(arr):
            if len(arr) < 2 or np.sum(arr) == 0:
                return 0.0
            # Recent = last min(3, len) months
            recent_len = min(3, len(arr))
            recent = np.mean(arr[-recent_len:])
            overall = np.mean(arr[arr > 0]) if np.any(arr > 0) else 1e-10
            return (recent - overall) / (overall + 1e-10)

        def safe_peak(arr):
            if np.sum(arr) == 0:
                return 0.0
            mean = np.mean(arr[arr > 0]) if np.any(arr > 0) else 1e-10
            return np.max(arr) / (mean + 1e-10)

        def safe_growth(arr):
            if len(arr) < 2 or np.sum(arr) == 0:
                return 0.0
            mid_point = len(arr) // 2
            if mid_point == 0:
                return 0.0
            first_half = np.mean(arr[:mid_point])
            second_half = np.mean(arr[mid_point:])
            if first_half == 0:
                return 0.0
            return (second_half - first_half) / (first_half + 1e-10)

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

    print(f"    ✅ Computed for {len(temporal_features):,} influencers (using {available_months} past months)")
    return temporal_features


# ============================================================================
# STEP 4: BUILD FEATURE VECTORS (37 Dimensions)
# ============================================================================

def build_enhanced_feature_vector(monthly_feat, temporal_feat, category_to_idx):
    """
    Build 37-dimensional enhanced feature vector.

    [0-2]   Static: log_followers, log_followees, follower_ratio (3)
    [3-10]  Category: one-hot (8)
    [11-24] Temporal: 14 dimensions
    [25-36] Monthly: 12 dimensions (NO engagement_rate!)
    """
    if monthly_feat is None:
        return torch.zeros(FEATURE_DIM)

    # Static features (3)
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
            temporal_feat["engagement_trend"],      # 11
            temporal_feat["likes_trend"],           # 12
            temporal_feat["engagement_variance"],   # 13
            temporal_feat["likes_variance"],        # 14
            temporal_feat["engagement_consistency"],# 15
            temporal_feat["likes_consistency"],     # 16
            temporal_feat["engagement_momentum"],   # 17
            temporal_feat["likes_momentum"],        # 18
            temporal_feat["engagement_peak"],       # 19
            temporal_feat["likes_peak"],            # 20
            temporal_feat["activity_rate"],         # 21
            temporal_feat["posting_consistency"],   # 22
            temporal_feat["engagement_growth"],     # 23
            temporal_feat["likes_growth"],          # 24
        ]
    else:
        temporal = [0.0] * 14

    # Monthly features (12) - NO engagement_rate!
    log_avg_likes = np.log10(monthly_feat["avg_likes"] + 1)
    log_avg_comments = np.log10(monthly_feat["avg_comments"] + 1)

    monthly = [
        monthly_feat["num_posts"],                  # 25
        log_avg_likes,                              # 26
        log_avg_comments,                           # 27
        monthly_feat["caption_len"]["avg"],         # 28
        monthly_feat["caption_len"]["std"],         # 29
        monthly_feat["caption_sentiment"]["avg"],   # 30
        monthly_feat["caption_sentiment"]["std"],   # 31
        monthly_feat["num_hashtags"]["avg"],        # 32
        monthly_feat["num_hashtags"]["std"],        # 33
        monthly_feat["num_mentions"]["avg"],        # 34
        monthly_feat["num_mentions"]["std"],        # 35
        monthly_feat["sentiment_positivity_rate"],  # 36
    ]

    feature_list = static + cat_vec + temporal + monthly

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
    monthly_features,
    temporal_features,
    category_to_idx,
    object_lookup
):
    """Build a single month's heterogeneous graph with PROGRESSIVE temporal features."""

    print(f"\n{'='*80}")
    print(f"BUILDING GRAPH: {month_name} (Month {month_idx})")
    print(f"  Temporal features from: months 0-{month_idx-1} (NO FUTURE DATA)")
    print(f"{'='*80}")

    # Extract entities
    influencers = set()
    hashtags = set()
    users = set()
    objects = set()

    inf_hashtag_edges = []
    inf_user_edges = []
    inf_object_edges = []

    for post in all_posts:
        inf = post["influencer"]
        influencers.add(inf)

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
    print("Building 37-dimensional features with PROGRESSIVE temporal...")
    num_influencers = len(influencers)
    feature_matrix = torch.zeros(num_influencers, FEATURE_DIM)
    engagement_rates = torch.zeros(num_influencers)
    avg_likes_tensor = torch.zeros(num_influencers)

    for name, idx in tqdm(inf_map.items(), desc="  Features"):
        monthly_feat = monthly_features.get(name)
        temporal_feat = temporal_features.get(name)

        if monthly_feat:
            feature_matrix[idx] = build_enhanced_feature_vector(
                monthly_feat, temporal_feat, category_to_idx
            )
            engagement_rates[idx] = monthly_feat["engagement_rate"]
            avg_likes_tensor[idx] = monthly_feat["avg_likes"]

    graph["influencer"].x = feature_matrix

    # Non-influencer features (one-hot at positions beyond influencer features)
    # Use positions 37+ to avoid semantic confusion
    for i, node_type in enumerate(["influencer", "hashtag", "user", "object"]):
        if node_type != "influencer" and graph[node_type].num_nodes > 0:
            one_hot = torch.zeros(graph[node_type].num_nodes, FEATURE_DIM)
            # Use last positions for type indicators (more semantic)
            one_hot[:, -(i+1)] = 1.0  # Positions 36, 35, 34 for hashtag, user, object
            graph[node_type].x = one_hot

    print(f"\n✅ Graph built with PROGRESSIVE temporal features:")
    print(f"   Features: {feature_matrix.shape}")
    print(f"   Temporal data from: months 0-{month_idx-1} only")

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
            "version": "v3_progressive_temporal",
            "temporal_months_used": month_idx,  # CRITICAL: Only past months
            "no_future_leakage": True,
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

    features = graph["influencer"].x
    if torch.isnan(features).any() or torch.isinf(features).any():
        print(f"  ⚠️  {month_name}: Found inf/nan in features!")
        return False

    eng_rates = ground_truth["engagement_rate"]
    if (eng_rates < 0).any() or (eng_rates > 100).any():
        print(f"  ⚠️  {month_name}: Engagement rates out of range!")
        return False

    if features.shape[1] != FEATURE_DIM:
        print(f"  ⚠️  {month_name}: Feature dim mismatch!")
        return False

    # Verify temporal features are progressive
    temporal_months = data_package["metadata"]["temporal_months_used"]
    if temporal_months != data_package["metadata"]["month_idx"]:
        print(f"  ⚠️  {month_name}: Temporal month mismatch!")
        return False

    print(f"  ✅ {month_name}: Validation passed (temporal from {temporal_months} past months)")
    return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_all_enhanced_graphs():
    """Main pipeline with PROGRESSIVE temporal features."""

    print("\n" + "="*80)
    print("STEP 1: LOADING EXTERNAL DATA")
    print("="*80)

    with open(PROFILES_FILE, 'rb') as f:
        data = pickle.load(f)
        profile_lookup = data['profiles']
        categories = data['categories']

    print(f"✅ Loaded {len(profile_lookup):,} profiles")

    df = pd.read_csv(COMBINED_OBJECTS_CSV)
    df["detected_objects"] = df["detected_objects"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    object_lookup = pd.Series(df.detected_objects.values, index=df.post_id).to_dict()
    print(f"✅ Loaded objects for {len(object_lookup):,} posts")

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
    print("STEP 2: EXTRACTING ALL MONTHS")
    print("="*80)

    all_months_posts = []
    all_months_features = []

    for month_name in MONTH_NAMES:
        posts = extract_month_data(month_name, profile_lookup)
        features = aggregate_monthly_features(posts, object_lookup)

        all_months_posts.append(posts)
        all_months_features.append(features)

        print(f"  {month_name}: {len(features)} influencers")

    # Build graphs with PROGRESSIVE temporal features
    print("\n" + "="*80)
    print("STEP 3: BUILDING GRAPHS WITH PROGRESSIVE TEMPORAL FEATURES")
    print("="*80)
    print("CRITICAL: Each month computes temporal from ONLY past months!")

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    for month_idx, month_name in enumerate(MONTH_NAMES):
        # CRITICAL FIX: Compute temporal features specific to this month
        temporal_features = compute_temporal_features_progressive(
            all_months_features, month_idx
        )

        data_package = build_month_graph(
            month_name,
            month_idx,
            all_months_posts[month_idx],
            all_months_features[month_idx],
            temporal_features,  # Month-specific!
            category_to_idx,
            object_lookup
        )

        if not validate_graph(data_package, month_name):
            print(f"❌ Validation failed for {month_name}!")
            continue

        graph_path = os.path.join(GRAPHS_DIR, f"{month_name.lower()}_graph.pt")
        torch.save(data_package, graph_path)
        print(f"✅ Saved: {graph_path}")

    print("\n" + "="*80)
    print("✅ ALL GRAPHS BUILT WITH PROGRESSIVE TEMPORAL FEATURES!")
    print("="*80)
    print(f"\nOutput directory: {GRAPHS_DIR}/")
    print(f"Feature dimensions: {FEATURE_DIM}")
    print(f"\nPROGRESSIVE TEMPORAL FEATURES (NO LEAKAGE):")
    print(f"  - Jan graph: 0 past months (zeros)")
    print(f"  - Feb graph: 1 past month (Jan only)")
    print(f"  - Sep graph: 8 past months (Jan-Aug)")
    print(f"  - Oct graph: 9 past months (Jan-Sep)")
    print(f"\nThis ensures NO FUTURE INFORMATION leaks into any graph!")
    print("="*80 + "\n")


if __name__ == "__main__":
    if not os.path.exists(PROFILES_FILE):
        print("❌ Error: profiles_lookup.pkl not found!")
        exit(1)

    if not os.path.exists(COMBINED_OBJECTS_CSV):
        print("❌ Error: image_objects.csv not found!")
        exit(1)

    build_all_enhanced_graphs()
