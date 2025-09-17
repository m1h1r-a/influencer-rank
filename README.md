# Influencer Rank

This project processes social media data to build monthly interaction graphs, capturing relationships between influencers, users, hashtags, and objects.

### File Overview

- `combine_csv.py`: Merges object detection CSVs.
- `build_graph.py`: Builds monthly graphs from raw data.
- `add_features.py`: Adds features to the graph nodes.
- `final_verification.py`: Verifies the integrity of the graphs.

### Usage

1.  Add your data to the `year_17` and `object_csvs` directories.
2.  Run the scripts in order:
    - `python combine_csv.py`
    - `python build_graph.py`
    - You can verify the output with `python final_verification.py`.
3. `python add_features.py`
