#!/usr/bin/env python3
"""
Reddit Dataset Builder - Incremental Scraper

This script incrementally builds a Reddit training dataset over time,
respecting rate limits. Run periodically (e.g., via cron or manually)
to grow the dataset.

Usage:
    python3 reddit_dataset_builder.py [--subreddits sub1,sub2] [--max-per-run 50]
"""

import requests
import json
import time
import os
import hashlib
import argparse
from datetime import datetime
from pathlib import Path

# Default subreddits with natural/colorful language
DEFAULT_SUBREDDITS = [
    'tifu', 'offmychest', 'AmItheAsshole', 'confession', 'TrueOffMyChest',
    'relationship_advice', 'pettyrevenge', 'MaliciousCompliance', 'entitledparents',
    'ChoosingBeggars', 'talesfromtechsupport', 'talesfromretail', 'IDontWorkHereLady',
    'ProRevenge', 'antiwork', 'gaming', 'pcmasterrace', 'rant', 'unpopularopinion',
    'askreddit', 'CasualConversation', 'self'
]

# Human-readable descriptions for subreddits
SUBREDDIT_DESCRIPTIONS = {
    'tifu': 'today I fucked up',
    'offmychest': 'getting something off my chest',
    'AmItheAsshole': 'am I the asshole',
    'confession': 'confession',
    'TrueOffMyChest': 'getting something off my chest',
    'relationship_advice': 'relationship advice',
    'pettyrevenge': 'petty revenge',
    'MaliciousCompliance': 'malicious compliance',
    'entitledparents': 'entitled parents',
    'ChoosingBeggars': 'choosing beggars',
    'talesfromtechsupport': 'tech support stories',
    'talesfromretail': 'retail worker stories',
    'IDontWorkHereLady': 'mistaken for an employee',
    'ProRevenge': 'pro revenge',
    'antiwork': 'work frustrations',
    'gaming': 'gaming',
    'pcmasterrace': 'PC gaming',
    'rant': 'rant',
    'unpopularopinion': 'unpopular opinion',
    'askreddit': 'interesting questions',
    'CasualConversation': 'casual conversation',
    'self': 'personal story',
}

# File paths
DATASET_PATH = Path('/data/finetune_datasets/reddit_incremental.json')
SEEN_IDS_PATH = Path('/data/finetune_datasets/.reddit_seen_ids.json')
INDEX_PATH = Path('/data/finetune_datasets/datasets_index.json')

# Rate limiting
REQUEST_DELAY = 2.0  # Seconds between requests


def load_seen_ids():
    """Load set of already-scraped post IDs."""
    if SEEN_IDS_PATH.exists():
        with open(SEEN_IDS_PATH, 'r') as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen_ids):
    """Save seen post IDs."""
    with open(SEEN_IDS_PATH, 'w') as f:
        json.dump(list(seen_ids), f)


def load_dataset():
    """Load existing dataset."""
    if DATASET_PATH.exists():
        with open(DATASET_PATH, 'r') as f:
            return json.load(f)
    return []


def save_dataset(data):
    """Save dataset."""
    with open(DATASET_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def update_index(num_examples):
    """Update the datasets index."""
    if INDEX_PATH.exists():
        with open(INDEX_PATH, 'r') as f:
            index = json.load(f)
    else:
        index = []
    
    # Find or create entry
    entry = None
    for d in index:
        if d['id'] == 'reddit_incremental':
            entry = d
            break
    
    if entry is None:
        entry = {
            'id': 'reddit_incremental',
            'name': f'Reddit Incremental ({num_examples})',
            'description': 'Growing Reddit dataset built incrementally over time. Contains posts and top comments.',
            'format': 'alpaca',
            'status': 'ready',
            'file_path': str(DATASET_PATH),
            'created_at': datetime.utcnow().isoformat() + 'Z',
        }
        index.append(entry)
    
    entry['num_examples'] = num_examples
    entry['name'] = f'Reddit Incremental ({num_examples})'
    entry['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    entry['total_tokens'] = num_examples * 300  # Rough estimate
    
    with open(INDEX_PATH, 'w') as f:
        json.dump(index, f, indent=2)


def get_headers():
    """Get request headers with rotating user agent."""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
    }


def fetch_subreddit_posts(subreddit, sort='top', time_filter='month', limit=25):
    """Fetch posts from a subreddit."""
    url = f'https://www.reddit.com/r/{subreddit}/{sort}.json'
    params = {'t': time_filter, 'limit': limit}
    
    try:
        resp = requests.get(url, headers=get_headers(), params=params, timeout=30)
        if resp.status_code == 429:
            print(f"  Rate limited! Waiting 60s...")
            time.sleep(60)
            return []
        resp.raise_for_status()
        return resp.json().get('data', {}).get('children', [])
    except Exception as e:
        print(f"  Error fetching r/{subreddit}: {e}")
        return []


def fetch_post_comments(permalink, limit=5):
    """Fetch top comments from a post."""
    url = f'https://www.reddit.com{permalink}.json'
    params = {'limit': limit, 'sort': 'top'}
    
    try:
        resp = requests.get(url, headers=get_headers(), params=params, timeout=30)
        if resp.status_code == 429:
            return []
        resp.raise_for_status()
        data = resp.json()
        
        comments = []
        if len(data) > 1:
            for item in data[1].get('data', {}).get('children', [])[:limit]:
                if item.get('kind') == 't1':
                    cd = item.get('data', {})
                    body = cd.get('body', '')
                    score = cd.get('score', 0)
                    if score >= 20 and len(body) >= 40 and body not in ['[deleted]', '[removed]']:
                        comments.append({
                            'body': body[:1200],
                            'score': score
                        })
        return comments
    except:
        return []


def scrape_subreddit(subreddit, seen_ids, max_new=10):
    """Scrape a subreddit for new examples."""
    examples = []
    new_ids = set()
    
    posts = fetch_subreddit_posts(subreddit, limit=50)
    time.sleep(REQUEST_DELAY)
    
    for post in posts:
        if len(examples) >= max_new:
            break
            
        pd = post.get('data', {})
        post_id = pd.get('id', '')
        
        if post_id in seen_ids:
            continue
        
        title = pd.get('title', '')
        body = pd.get('selftext', '')
        permalink = pd.get('permalink', '')
        score = pd.get('score', 0)
        
        if score < 50:
            continue
        
        new_ids.add(post_id)
        
        # Get human-readable description for this subreddit
        sub_desc = SUBREDDIT_DESCRIPTIONS.get(subreddit, subreddit)
        
        # Add post as example (story/experience)
        if body and len(body) >= 80 and body not in ['[deleted]', '[removed]']:
            examples.append({
                'instruction': f'Write a {sub_desc} story: {title[:150]}',
                'input': '',
                'output': body[:1800]
            })
        
        # Fetch and add ONE top comment to avoid duplicates
        if permalink:
            comments = fetch_post_comments(permalink, limit=1)
            if comments:
                # Use the post body as context for the response
                examples.append({
                    'instruction': f'Respond to this {sub_desc} post: {title[:120]}',
                    'input': body[:500] if body else '',
                    'output': comments[0]['body']
                })
            time.sleep(REQUEST_DELAY)
    
    return examples, new_ids


def main():
    parser = argparse.ArgumentParser(description='Incrementally build Reddit training dataset')
    parser.add_argument('--subreddits', type=str, help='Comma-separated subreddit list')
    parser.add_argument('--max-per-run', type=int, default=50, help='Max new examples per run')
    args = parser.parse_args()
    
    subreddits = args.subreddits.split(',') if args.subreddits else DEFAULT_SUBREDDITS
    
    print(f"Reddit Dataset Builder")
    print(f"=" * 50)
    print(f"Subreddits: {len(subreddits)}")
    print(f"Max new examples: {args.max_per_run}")
    print()
    
    # Load existing state
    seen_ids = load_seen_ids()
    dataset = load_dataset()
    initial_count = len(dataset)
    
    print(f"Existing examples: {initial_count}")
    print(f"Already seen posts: {len(seen_ids)}")
    print()
    
    total_new = 0
    
    for subreddit in subreddits:
        if total_new >= args.max_per_run:
            break
        
        remaining = args.max_per_run - total_new
        print(f"r/{subreddit}...", end=' ', flush=True)
        
        examples, new_ids = scrape_subreddit(subreddit, seen_ids, max_new=min(10, remaining))
        
        if examples:
            dataset.extend(examples)
            seen_ids.update(new_ids)
            total_new += len(examples)
            print(f"+{len(examples)}")
        else:
            print("0")
        
        time.sleep(REQUEST_DELAY)
    
    print()
    print(f"=" * 50)
    print(f"New examples added: {total_new}")
    print(f"Total dataset size: {len(dataset)}")
    
    # Save everything
    save_dataset(dataset)
    save_seen_ids(seen_ids)
    update_index(len(dataset))
    
    print(f"Saved to {DATASET_PATH}")


if __name__ == '__main__':
    main()
