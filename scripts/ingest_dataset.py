"""
SyncNet Data Ingestion Script
=============================
Generates/loads a reproducible social media bot-network detection dataset based on
the canonical TwiBot-22 / Cresci benchmark schemas.

Dataset Schema:
- users.jsonl: User profile metadata and ground-truth bot labels (0: human, 1: bot).
- posts.jsonl: Content posts, text, engagement metrics, and timestamps.
- edges.jsonl: User-user interaction graph links (retweet, mention, reply).

Output Directory: data/raw/
"""

import os
import json
import random
from datetime import datetime, timedelta

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RANDOM_SEED = 42

BOT_DESCRIPTIONS = [
    "Crypto enthusiast | Daily news bot | Auto-retweet #Crypto #NFT",
    "Follow for instant back follow! 100% follow back. Free giveaways daily!",
    "Automated breaking news aggregator for trending topics worldwide.",
    "Promoting the best deals and discounts 24/7! Click link in bio.",
    "Official alert bot for network updates and server notifications.",
    "AIRDROP ALERT! Claim your free tokens now by following and retweeting!",
    "Daily quote generator and automated inspiration feed.",
    "E-commerce deal finder. Cheap prices every hour!",
]

HUMAN_DESCRIPTIONS = [
    "Software engineer, coffee lover, open source contributor. Views are my own.",
    "Data scientist exploring GNNs & NLP. Outdoor enthusiast and photographer.",
    "Journalist covering tech, internet culture, and privacy.",
    "Researcher in machine learning at AI lab. Reading books and writing code.",
    "UX designer. Passionate about accessible web interfaces and clean design.",
    "Graduate student in computer science. Learning PyTorch & graph algorithms.",
    "Product manager, runner, father. Tweets about technology and design.",
    "Digital artist & illustrator. Creating art and sharing daily sketches.",
]

BOT_POST_TEMPLATES = [
    "URGENT: Claim your free airdrop now at http://bit.ly/crypto-reward #Airdrop #Crypto",
    "Massive discount on latest gadgets! Check out the sale here: http://deals.shop/deal123",
    "Win $1000 in our daily giveaway! RT + Follow to enter now! #Giveaway",
    "Breaking News: New market trends emerging today! Learn more at http://news-bot.info",
    "Follow back guaranteed! Expanding my follower count today #FollowBack #GainTrain",
]

HUMAN_POST_TEMPLATES = [
    "Just finished reading a fascinating paper on Graph Neural Networks and GraphSAGE!",
    "Had a great cup of coffee this morning. Excited to deploy our latest PyTorch model.",
    "Does anyone have recommendations for good FastAPI async ORM libraries?",
    "Attended an awesome meetup on NLP text embeddings and modern Transformers today.",
    "Spent the weekend hiking in the mountains. Fresh air and zero notifications!",
]

def generate_synthetic_benchmark_dataset(num_users=1000, bot_ratio=0.3, num_posts_per_user=5, seed=RANDOM_SEED):
    """
    Generates a realistic, reproducible benchmark social network dataset for SyncNet research.
    """
    random.seed(seed)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    
    users_file = os.path.join(DATA_RAW_DIR, "users.jsonl")
    posts_file = os.path.join(DATA_RAW_DIR, "posts.jsonl")
    edges_file = os.path.join(DATA_RAW_DIR, "edges.jsonl")
    
    num_bots = int(num_users * bot_ratio)
    num_humans = num_users - num_bots
    
    users = []
    posts = []
    edges = []
    
    base_date = datetime(2024, 1, 1)
    
    user_ids = [f"usr_{i:04d}" for i in range(1, num_users + 1)]
    
    # 1. Generate User Profiles
    for i, u_id in enumerate(user_ids):
        is_bot = 1 if i < num_bots else 0
        
        if is_bot:
            # Bot characteristics: higher followings, higher digit density in username, lower account age
            created_days_ago = random.randint(10, 180)
            followers_count = random.randint(10, 500)
            following_count = random.randint(500, 4500) # Unbalanced ratio
            tweet_count = random.randint(200, 10000)   # High post volume relative to age
            listed_count = random.randint(0, 2)
            verified = False
            default_profile_image = random.choice([True, False, True]) # Higher default pic rate
            screen_name = f"bot_user_{random.randint(100000, 999999)}"
            name = f"Promo Bot {i}"
            description = random.choice(BOT_DESCRIPTIONS)
        else:
            # Human characteristics: balanced follower/following, older accounts, lower digit density
            created_days_ago = random.randint(300, 3600)
            followers_count = random.randint(100, 5000)
            following_count = random.randint(100, 1500)
            tweet_count = random.randint(50, 3000)
            listed_count = random.randint(1, 50)
            verified = random.choice([False, False, False, True])
            default_profile_image = False
            screen_name = f"user_{i}_{random.choice(['dev', 'tech', 'data', 'coder'])}"
            name = f"User Person {i}"
            description = random.choice(HUMAN_DESCRIPTIONS)
            
        created_at = (base_date - timedelta(days=created_days_ago)).isoformat()
        
        user_doc = {
            "user_id": u_id,
            "screen_name": screen_name,
            "name": name,
            "description": description,
            "created_at": created_at,
            "followers_count": followers_count,
            "following_count": following_count,
            "tweet_count": tweet_count,
            "listed_count": listed_count,
            "verified": verified,
            "default_profile_image": default_profile_image,
            "label": is_bot, # 1 = bot, 0 = human
        }
        users.append(user_doc)
        
    # 2. Generate Posts
    post_counter = 1
    for u in users:
        u_id = u["user_id"]
        is_bot = u["label"]
        num_p = random.randint(2, num_posts_per_user * 2) if is_bot else random.randint(1, num_posts_per_user)
        
        for p_idx in range(num_p):
            p_id = f"post_{post_counter:06d}"
            post_counter += 1
            
            if is_bot:
                text = random.choice(BOT_POST_TEMPLATES)
                retweet_count = random.randint(10, 500) # Coordinated bot boosting
                reply_count = random.randint(0, 5)
                like_count = random.randint(5, 50)
                has_urls = True
            else:
                text = random.choice(HUMAN_POST_TEMPLATES)
                retweet_count = random.randint(0, 20)
                reply_count = random.randint(0, 15)
                like_count = random.randint(2, 100)
                has_urls = random.choice([True, False, False])
                
            post_date = (base_date + timedelta(days=random.randint(1, 100), hours=random.randint(0, 23))).isoformat()
            
            post_doc = {
                "post_id": p_id,
                "user_id": u_id,
                "text": text,
                "created_at": post_date,
                "retweet_count": retweet_count,
                "reply_count": reply_count,
                "like_count": like_count,
                "has_urls": has_urls,
                "has_mentions": "@" in text,
                "has_hashtags": "#" in text,
            }
            posts.append(post_doc)

    # 3. Generate Interaction Edges (Retweets, Mentions, Replies)
    # Coordinated bots interact heavily within their own cluster (cluster coordination pattern)
    bot_ids = [u["user_id"] for u in users if u["label"] == 1]
    human_ids = [u["user_id"] for u in users if u["label"] == 0]
    
    edge_set = set()
    
    # Dense bot-bot interaction edges (retweets & mentions among bots)
    for src in bot_ids:
        # Each bot retweets / mentions 5-10 other bots to simulate coordinated amplification
        targets = random.sample(bot_ids, random.randint(5, 12))
        for tgt in targets:
            if src != tgt:
                rel = random.choice(["retweet", "retweet", "mention"])
                edge_set.add((src, tgt, rel))
                
    # Human-human and human-bot organic interactions
    for src in human_ids:
        targets = random.sample(user_ids, random.randint(1, 4))
        for tgt in targets:
            if src != tgt:
                rel = random.choice(["reply", "mention", "retweet"])
                edge_set.add((src, tgt, rel))
                
    for src, tgt, rel in edge_set:
        edge_doc = {
            "source": src,
            "target": tgt,
            "relation": rel,
            "timestamp": base_date.isoformat(),
        }
        edges.append(edge_doc)
        
    # Write JSONL outputs
    with open(users_file, "w", encoding="utf-8") as f:
        for u in users:
            f.write(json.dumps(u) + "\n")
            
    with open(posts_file, "w", encoding="utf-8") as f:
        for p in posts:
            f.write(json.dumps(p) + "\n")
            
    with open(edges_file, "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e) + "\n")
            
    print(f"[x] INGESTION COMPLETE: Data saved to {DATA_RAW_DIR}")
    print(f"    - Users : {len(users)} (Bots: {num_bots}, Humans: {num_humans})")
    print(f"    - Posts : {len(posts)}")
    print(f"    - Edges : {len(edges)}")
    return users_file, posts_file, edges_file

if __name__ == "__main__":
    generate_synthetic_benchmark_dataset()
