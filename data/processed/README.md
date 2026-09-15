# Clean Processed Datasets — SyncNet

This directory contains cleaned, tokenized, and structured datasets produced by `scripts/preprocess_data.py`.

## Files & Schemas
- `clean_users.parquet` (`clean_users.csv`): Clean user profiles containing metadata, calculated account age, follower-following ratio, tweet frequency, post engagement averages, bot labels, and data split tags (`train`, `val`, `test`).
- `clean_posts.parquet` (`clean_posts.csv`): Clean post items with original text, normalized text (`clean_text`), engagement counts, and URL/hashtag/mention boolean indicators.
- `clean_edges.parquet` (`clean_edges.csv`): User-user interaction graph edges containing `source`, `target`, `relation` (retweet, mention, reply), and timestamps.

## Data Leakage Safeguards
Data splits (`train`, `val`, `test`) are assigned with a fixed seed (`RANDOM_SEED = 42`) prior to scaling or graph dataset construction.
