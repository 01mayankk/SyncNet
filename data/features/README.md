# Feature Matrices & Embeddings — SyncNet

This directory stores extracted feature representations for behavioral baselines, Transformer content embeddings, and PyTorch Geometric node feature matrices.

## Files & Schemas
- `behavioral_features.parquet` (`behavioral_features.csv`):
  - `user_id`: Unique account identifier string (`usr_XXXX`).
  - `split`: Partition assignment (`train`, `val`, `test`).
  - `label`: Ground-truth label (`0`: Human, `1`: Bot).
  - `account_age_days`: Account age computed relative to reference snapshot.
  - `followers_count`: Number of followers.
  - `following_count`: Number of accounts followed.
  - `follower_following_ratio`: $\text{followers} / (\text{following} + 1)$.
  - `tweet_count`: Total posted tweets.
  - `tweet_frequency`: $\text{tweets} / \text{account\_age\_days}$.
  - `screen_name_digit_ratio`: Proportion of digits in screen name string.
  - `default_profile_image_int`: Binary flag for default profile image.
  - `verified_int`: Binary flag for verified status.
  - `avg_retweet_count`: Average retweets per user post.
  - `url_density`: Proportion of user posts containing URLs.
  - `mention_density`: Proportion of user posts mentioning other users.
  - `hashtag_density`: Proportion of user posts containing hashtags.
