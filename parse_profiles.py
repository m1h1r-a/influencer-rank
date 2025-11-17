"""
Profile Data Parser for influencers.txt

Parses tab/space-separated profile data and creates clean lookup dictionary.
Handles missing values and validates data.
"""

import pickle
import re
from collections import defaultdict

PROFILE_FILE = "influencers.txt"
OUTPUT_FILE = "profiles_lookup.pkl"


def parse_profiles(filename):
    """
    Parse influencers.txt into clean lookup dictionary.
    
    Format expected:
    Username	Category	#Followers	#Followees	#Posts
    =======================================================================
    makeupbynvs	beauty	1432	1089	363
    ...
    
    Returns:
    {
        'makeupbynvs': {
            'followers': 1432,
            'followees': 1089,
            'total_posts': 363,
            'category': 'beauty'
        },
        ...
    }
    """
    print(f"\n{'='*70}")
    print(f"PARSING PROFILE DATA: {filename}")
    print(f"{'='*70}\n")
    
    profiles = {}
    categories_found = set()
    
    # Statistics
    total_lines = 0
    skipped_lines = 0
    missing_posts = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip first 2 lines (header + separator)
    data_lines = lines[2:]
    
    print(f"Total lines to process: {len(data_lines)}")
    print(f"Processing...\n")
    
    for line_num, line in enumerate(data_lines, start=3):  # Start at 3 (after skipping 2)
        line = line.strip()
        if not line:
            continue
        
        total_lines += 1
        
        # Split by whitespace (handles both tabs and spaces)
        parts = re.split(r'\s+', line)
        
        # Expected: [username, category, followers, followees, posts]
        if len(parts) < 4:
            print(f"⚠️  Line {line_num}: Not enough fields, skipping: {line[:50]}")
            skipped_lines += 1
            continue
        
        try:
            username = parts[0].lower()  # Normalize to lowercase
            category = parts[1].lower() if len(parts) > 1 else 'unknown'
            followers = int(parts[2]) if len(parts) > 2 else 0
            followees = int(parts[3]) if len(parts) > 3 else 0
            
            # Handle missing #Posts (simplest: use 0)
            if len(parts) > 4 and parts[4].strip():
                total_posts = int(parts[4])
            else:
                total_posts = 0
                missing_posts += 1
            
            profiles[username] = {
                'followers': followers,
                'followees': followees,
                'total_posts': total_posts,
                'category': category
            }
            
            categories_found.add(category)
            
        except (ValueError, IndexError) as e:
            print(f"⚠️  Line {line_num}: Error parsing, skipping: {e}")
            print(f"    Line content: {line[:50]}")
            skipped_lines += 1
            continue
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"PARSING COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successfully parsed: {len(profiles):,} influencers")
    print(f"⚠️  Skipped lines: {skipped_lines}")
    print(f"⚠️  Missing #Posts: {missing_posts} (set to 0)")
    print(f"\n📊 Categories found ({len(categories_found)}):")
    for cat in sorted(categories_found):
        count = sum(1 for p in profiles.values() if p['category'] == cat)
        print(f"   - {cat:20s}: {count:5,d} influencers")
    
    # Statistics on followers
    follower_counts = [p['followers'] for p in profiles.values()]
    print(f"\n📈 Follower Statistics:")
    print(f"   Min:      {min(follower_counts):,}")
    print(f"   Max:      {max(follower_counts):,}")
    print(f"   Median:   {sorted(follower_counts)[len(follower_counts)//2]:,}")
    print(f"   Mean:     {sum(follower_counts)//len(follower_counts):,}")
    
    # Check for zero followers (potential issues)
    zero_followers = sum(1 for p in profiles.values() if p['followers'] == 0)
    if zero_followers > 0:
        print(f"\n⚠️  Warning: {zero_followers} influencers have 0 followers")
        print(f"   These will have engagement_rate = 0")
    
    return profiles, categories_found


def save_profiles(profiles, categories, output_file):
    """Save parsed profiles to pickle file for fast loading."""
    data = {
        'profiles': profiles,
        'categories': sorted(list(categories))
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✅ Saved profile lookup to: {output_file}")
    print(f"   Size: {len(profiles):,} profiles")
    print(f"   Categories: {len(categories)}")


def load_profiles(filename):
    """Load pre-parsed profiles from pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['profiles'], data['categories']


if __name__ == "__main__":
    # Parse profiles
    profiles, categories = parse_profiles(PROFILE_FILE)
    
    # Save to pickle
    save_profiles(profiles, categories, OUTPUT_FILE)
    
    print(f"\n{'='*70}")
    print(f"PROFILE PARSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nYou can now run: python build_graph_with_profiles.py")
    print(f"{'='*70}\n")
