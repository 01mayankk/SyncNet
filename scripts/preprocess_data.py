"""
SyncNet Data Preprocessing & Behavioral Feature Pipeline
=========================================================
Processes raw user profiles, posts, and edge lists into clean Parquet data tables
and computes standard behavioral feature representations.

Data Leakage Safeguards:
- Isolates Train / Validation / Test splits with fixed seed (42) prior to feature scaling.
- Ensures no future-information or label leakage occurs during feature calculation.

Outputs:
- data/processed/clean_users.parquet (.csv)
- data/processed/clean_posts.parquet (.csv)
- data/processed/clean_edges.parquet (.csv)
- data/features/behavioral_features.parquet (.csv)
"""

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DATA_FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
RANDOM_SEED = 42

def clean_text(text: str) -> str:
    """Removes URLs, mentions, hashtags symbols, and extra whitespace for NLP ingestion."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_syncnet_data():
    os.makedirs(DATA_PROC_DIR, exist_ok=True)
    os.makedirs(DATA_FEAT_DIR, exist_ok=True)
    
    users_file = os.path.join(DATA_RAW_DIR, "users.jsonl")
    posts_file = os.path.join(DATA_RAW_DIR, "posts.jsonl")
    edges_file = os.path.join(DATA_RAW_DIR, "edges.jsonl")
    
    if not os.path.exists(users_file):
        raise FileNotFoundError(f"Raw data file {users_file} not found. Run ingest_dataset.py first.")
        
    print("--- 1. Loading Raw Data ---")
    users_data = [json.loads(line) for line in open(users_file, "r", encoding="utf-8")]
    posts_data = [json.loads(line) for line in open(posts_file, "r", encoding="utf-8")]
    edges_data = [json.loads(line) for line in open(edges_file, "r", encoding="utf-8")]
    
    df_users = pd.DataFrame(users_data)
    df_posts = pd.DataFrame(posts_data)
    df_edges = pd.DataFrame(edges_data)
    
    print(f"Loaded Raw Users : {df_users.shape}")
    print(f"Loaded Raw Posts : {df_posts.shape}")
    print(f"Loaded Raw Edges : {df_edges.shape}")
    
    print("\n--- 2. Cleaning Text & Preprocessing Posts ---")
    df_posts["clean_text"] = df_posts["text"].apply(clean_text)
    
    # User-level post metric aggregations
    post_stats = df_posts.groupby("user_id").agg(
        num_posts_collected=("post_id", "count"),
        avg_retweet_count=("retweet_count", "mean"),
        avg_reply_count=("reply_count", "mean"),
        avg_like_count=("like_count", "mean"),
        url_density=("has_urls", "mean"),
        mention_density=("has_mentions", "mean"),
        hashtag_density=("has_hashtags", "mean"),
    ).reset_index()
    
    print("\n--- 3. Behavioral Feature Mining ---")
    ref_date = pd.to_datetime("2024-06-01")
    df_users["created_at_dt"] = pd.to_datetime(df_users["created_at"])
    df_users["account_age_days"] = (ref_date - df_users["created_at_dt"]).dt.days.clip(lower=1)
    
    df_users["follower_following_ratio"] = df_users["followers_count"] / (df_users["following_count"] + 1.0)
    df_users["tweet_frequency"] = df_users["tweet_count"] / df_users["account_age_days"]
    
    # Calculate digit ratio in username
    df_users["screen_name_digit_ratio"] = df_users["screen_name"].apply(
        lambda s: sum(c.isdigit() for c in str(s)) / max(len(str(s)), 1)
    )
    
    df_users["default_profile_image_int"] = df_users["default_profile_image"].astype(int)
    df_users["verified_int"] = df_users["verified"].astype(int)
    
    # Merge user metadata with aggregated post statistics
    df_merged = pd.merge(df_users, post_stats, on="user_id", how="left").fillna(0)
    
    print("\n--- 4. Data Leakage Isolation: Train/Val/Test Split ---")
    # Split: 80% Train, 10% Validation, 10% Test
    train_ids, test_ids = train_test_split(
        df_merged["user_id"], test_size=0.20, random_state=RANDOM_SEED, stratify=df_merged["label"]
    )
    val_ids, test_ids = train_test_split(
        test_ids, test_size=0.50, random_state=RANDOM_SEED, stratify=df_merged.set_index("user_id").loc[test_ids, "label"]
    )
    
    split_map = {}
    for uid in train_ids:
        split_map[uid] = "train"
    for uid in val_ids:
        split_map[uid] = "val"
    for uid in test_ids:
        split_map[uid] = "test"
        
    df_merged["split"] = df_merged["user_id"].map(split_map)
    print(f"Split Distribution: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    print("\n--- 5. Saving Processed Tables & Behavioral Feature Matrix ---")
    # Export clean processed files
    df_merged.to_parquet(os.path.join(DATA_PROC_DIR, "clean_users.parquet"), index=False)
    df_merged.to_csv(os.path.join(DATA_PROC_DIR, "clean_users.csv"), index=False)
    
    df_posts.to_parquet(os.path.join(DATA_PROC_DIR, "clean_posts.parquet"), index=False)
    df_posts.to_csv(os.path.join(DATA_PROC_DIR, "clean_posts.csv"), index=False)
    
    df_edges.to_parquet(os.path.join(DATA_PROC_DIR, "clean_edges.parquet"), index=False)
    df_edges.to_csv(os.path.join(DATA_PROC_DIR, "clean_edges.csv"), index=False)
    
    # Behavioral Feature Matrix
    feature_cols = [
        "user_id",
        "split",
        "label",
        "account_age_days",
        "followers_count",
        "following_count",
        "follower_following_ratio",
        "tweet_count",
        "tweet_frequency",
        "screen_name_digit_ratio",
        "default_profile_image_int",
        "verified_int",
        "avg_retweet_count",
        "url_density",
        "mention_density",
        "hashtag_density",
    ]
    
    df_features = df_merged[feature_cols].copy()
    
    df_features.to_parquet(os.path.join(DATA_FEAT_DIR, "behavioral_features.parquet"), index=False)
    df_features.to_csv(os.path.join(DATA_FEAT_DIR, "behavioral_features.csv"), index=False)
    
    print(f"[x] PREPROCESSING COMPLETE!")
    print(f"    - Clean Users  : {os.path.join(DATA_PROC_DIR, 'clean_users.parquet')}")
    print(f"    - Clean Posts  : {os.path.join(DATA_PROC_DIR, 'clean_posts.parquet')}")
    print(f"    - Clean Edges  : {os.path.join(DATA_PROC_DIR, 'clean_edges.parquet')}")
    print(f"    - Feature Matrix: {os.path.join(DATA_FEAT_DIR, 'behavioral_features.parquet')}")
    print(f"    - Feature Shape : {df_features.shape}")
    
    return df_merged, df_features

if __name__ == "__main__":
    preprocess_syncnet_data()
