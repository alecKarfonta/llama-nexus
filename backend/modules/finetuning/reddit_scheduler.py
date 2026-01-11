"""
Reddit Crawler Scheduler - Background async task for periodic Reddit scraping.

Environment Variables:
    REDDIT_CRAWLER_ENABLED: Enable auto-start on boot (default: false)
    REDDIT_CRAWLER_INTERVAL_HOURS: Hours between runs (default: 6)
    REDDIT_CRAWLER_MAX_PER_RUN: Max examples per run (default: 50)
    REDDIT_CRAWLER_SUBREDDITS: Comma-separated list of subreddits
"""

import asyncio
import json
import os
import logging
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Default subreddits
DEFAULT_SUBREDDITS = [
    'tifu', 'offmychest', 'AmItheAsshole', 'confession', 'TrueOffMyChest',
    'relationship_advice', 'pettyrevenge', 'MaliciousCompliance', 'entitledparents',
    'ChoosingBeggars', 'talesfromtechsupport', 'talesfromretail', 'IDontWorkHereLady',
    'ProRevenge', 'antiwork', 'gaming', 'pcmasterrace', 'rant', 'unpopularopinion',
    'askreddit', 'CasualConversation', 'self'
]

# Human-readable descriptions for subreddits (used in training prompts)
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
CONFIG_PATH = Path('/data/finetune_datasets/.reddit_crawler_config.json')


@dataclass
class CrawlerConfig:
    """Configuration for the Reddit crawler."""
    enabled: bool = False
    interval_hours: float = 6.0
    max_per_run: int = 50
    subreddits: List[str] = field(default_factory=lambda: DEFAULT_SUBREDDITS.copy())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrawlerConfig":
        return cls(
            enabled=data.get('enabled', False),
            interval_hours=data.get('interval_hours', 6.0),
            max_per_run=data.get('max_per_run', 50),
            subreddits=data.get('subreddits', DEFAULT_SUBREDDITS.copy())
        )
    
    @classmethod
    def from_env(cls) -> "CrawlerConfig":
        """Load configuration from environment variables."""
        subreddits_str = os.getenv('REDDIT_CRAWLER_SUBREDDITS', '')
        subreddits = subreddits_str.split(',') if subreddits_str else DEFAULT_SUBREDDITS.copy()
        subreddits = [s.strip() for s in subreddits if s.strip()]
        
        return cls(
            enabled=os.getenv('REDDIT_CRAWLER_ENABLED', 'false').lower() == 'true',
            interval_hours=float(os.getenv('REDDIT_CRAWLER_INTERVAL_HOURS', '6')),
            max_per_run=int(os.getenv('REDDIT_CRAWLER_MAX_PER_RUN', '50')),
            subreddits=subreddits
        )


@dataclass
class CrawlerStatus:
    """Status of the crawler."""
    running: bool = False
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    last_run_examples: int = 0
    total_examples: int = 0
    error: Optional[str] = None


class RedditCrawlerScheduler:
    """
    Async scheduler for periodic Reddit scraping.
    
    Usage:
        scheduler = RedditCrawlerScheduler()
        await scheduler.start()  # Starts background task
        await scheduler.stop()   # Stops background task
    """
    
    REQUEST_DELAY = 2.0  # Seconds between requests
    
    def __init__(self):
        self.config = self._load_config()
        self.status = CrawlerStatus()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._load_status()
    
    def _load_config(self) -> CrawlerConfig:
        """Load config from file, falling back to env vars."""
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'r') as f:
                    return CrawlerConfig.from_dict(json.load(f))
            except Exception:
                pass
        return CrawlerConfig.from_env()
    
    def _save_config(self):
        """Save current config to file."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def _load_status(self):
        """Load dataset stats."""
        if DATASET_PATH.exists():
            try:
                with open(DATASET_PATH, 'r') as f:
                    data = json.load(f)
                    self.status.total_examples = len(data) if isinstance(data, list) else 0
            except Exception:
                self.status.total_examples = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'running': self.status.running,
            'last_run': self.status.last_run,
            'next_run': self.status.next_run,
            'last_run_examples': self.status.last_run_examples,
            'total_examples': self.status.total_examples,
            'error': self.status.error,
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.to_dict()
    
    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration and save."""
        if 'enabled' in updates:
            self.config.enabled = bool(updates['enabled'])
        if 'interval_hours' in updates:
            self.config.interval_hours = max(0.5, min(168, float(updates['interval_hours'])))
        if 'max_per_run' in updates:
            self.config.max_per_run = max(5, min(500, int(updates['max_per_run'])))
        if 'subreddits' in updates:
            subs = updates['subreddits']
            if isinstance(subs, str):
                subs = [s.strip() for s in subs.split(',') if s.strip()]
            self.config.subreddits = subs
        
        self._save_config()
        return self.config.to_dict()
    
    async def start(self):
        """Start the background scheduler."""
        if self._task is not None and not self._task.done():
            logger.info("Reddit crawler scheduler already running")
            return
        
        self._stop_event.clear()
        self._task = asyncio.create_task(self._scheduler_loop())
        self.status.running = True
        logger.info("Reddit crawler scheduler started")
    
    async def stop(self):
        """Stop the background scheduler."""
        if self._task is None or self._task.done():
            return
        
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        
        self.status.running = False
        self.status.next_run = None
        logger.info("Reddit crawler scheduler stopped")
    
    async def run_now(self) -> Dict[str, Any]:
        """Trigger an immediate run."""
        logger.info("Running Reddit crawler now...")
        try:
            examples_added = await self._run_scrape()
            self.status.last_run = datetime.utcnow().isoformat() + 'Z'
            self.status.last_run_examples = examples_added
            self.status.error = None
            self._load_status()  # Refresh total count
            return {
                'success': True,
                'examples_added': examples_added,
                'total_examples': self.status.total_examples
            }
        except Exception as e:
            self.status.error = str(e)
            logger.error(f"Reddit crawler error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                # Calculate next run time
                next_run = datetime.utcnow() + timedelta(hours=self.config.interval_hours)
                self.status.next_run = next_run.isoformat() + 'Z'
                
                # Wait for interval or stop event
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.interval_hours * 3600
                    )
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    pass  # Time to run
                
                # Run the scrape
                if not self._stop_event.is_set():
                    await self.run_now()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                self.status.error = str(e)
                await asyncio.sleep(60)  # Wait before retry
    
    async def _run_scrape(self) -> int:
        """Run the actual scraping (in thread pool to avoid blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._scrape_sync)
    
    def _scrape_sync(self) -> int:
        """Synchronous scraping logic."""
        # Load existing data
        seen_ids = self._load_seen_ids()
        dataset = self._load_dataset()
        initial_count = len(dataset)
        
        total_new = 0
        
        for subreddit in self.config.subreddits:
            if total_new >= self.config.max_per_run:
                break
            
            remaining = self.config.max_per_run - total_new
            examples, new_ids = self._scrape_subreddit(subreddit, seen_ids, max_new=min(10, remaining))
            
            if examples:
                dataset.extend(examples)
                seen_ids.update(new_ids)
                total_new += len(examples)
            
            time.sleep(self.REQUEST_DELAY)
        
        # Save everything
        self._save_dataset(dataset)
        self._save_seen_ids(seen_ids)
        self._update_index(len(dataset))
        
        logger.info(f"Reddit crawler: added {total_new} examples, total: {len(dataset)}")
        return total_new
    
    def _load_seen_ids(self) -> set:
        if SEEN_IDS_PATH.exists():
            try:
                with open(SEEN_IDS_PATH, 'r') as f:
                    return set(json.load(f))
            except Exception:
                pass
        return set()
    
    def _save_seen_ids(self, seen_ids: set):
        SEEN_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SEEN_IDS_PATH, 'w') as f:
            json.dump(list(seen_ids), f)
    
    def _load_dataset(self) -> list:
        if DATASET_PATH.exists():
            try:
                with open(DATASET_PATH, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []
    
    def _save_dataset(self, data: list):
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _update_index(self, num_examples: int):
        if INDEX_PATH.exists():
            try:
                with open(INDEX_PATH, 'r') as f:
                    index = json.load(f)
            except Exception:
                index = []
        else:
            index = []
        
        entry = None
        for d in index:
            if d['id'] == 'reddit_incremental':
                entry = d
                break
        
        if entry is None:
            entry = {
                'id': 'reddit_incremental',
                'name': f'Reddit Incremental ({num_examples})',
                'description': 'Growing Reddit dataset built incrementally over time.',
                'format': 'alpaca',
                'status': 'ready',
                'file_path': str(DATASET_PATH),
                'created_at': datetime.utcnow().isoformat() + 'Z',
            }
            index.append(entry)
        
        entry['num_examples'] = num_examples
        entry['name'] = f'Reddit Incremental ({num_examples})'
        entry['updated_at'] = datetime.utcnow().isoformat() + 'Z'
        entry['total_tokens'] = num_examples * 300
        
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _get_headers(self) -> dict:
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
    
    def _fetch_posts(self, subreddit: str, limit: int = 50) -> list:
        url = f'https://www.reddit.com/r/{subreddit}/top.json'
        params = {'t': 'month', 'limit': limit}
        try:
            resp = requests.get(url, headers=self._get_headers(), params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(60)
                return []
            resp.raise_for_status()
            return resp.json().get('data', {}).get('children', [])
        except Exception as e:
            logger.warning(f"Error fetching r/{subreddit}: {e}")
            return []
    
    def _fetch_comments(self, permalink: str, limit: int = 5) -> list:
        url = f'https://www.reddit.com{permalink}.json'
        params = {'limit': limit, 'sort': 'top'}
        try:
            resp = requests.get(url, headers=self._get_headers(), params=params, timeout=30)
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
                            comments.append({'body': body[:1200], 'score': score})
            return comments
        except Exception:
            return []
    
    def _scrape_subreddit(self, subreddit: str, seen_ids: set, max_new: int = 10) -> tuple:
        examples = []
        new_ids = set()
        
        posts = self._fetch_posts(subreddit, limit=50)
        time.sleep(self.REQUEST_DELAY)
        
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
                comments = self._fetch_comments(permalink, limit=1)
                if comments:
                    # Use the post body as context for the response
                    examples.append({
                        'instruction': f'Respond to this {sub_desc} post: {title[:120]}',
                        'input': body[:500] if body else '',
                        'output': comments[0]['body']
                    })
                time.sleep(self.REQUEST_DELAY)
        
        return examples, new_ids


# Singleton instance
_scheduler: Optional[RedditCrawlerScheduler] = None


def get_reddit_scheduler() -> RedditCrawlerScheduler:
    """Get or create the singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = RedditCrawlerScheduler()
    return _scheduler
