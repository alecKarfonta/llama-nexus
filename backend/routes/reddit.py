"""
Reddit Crawler API Routes

Provides endpoints for controlling the Reddit crawler scheduler
and viewing collected data samples and statistics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import logging
from pathlib import Path

try:
    from modules.finetuning.reddit_scheduler import get_reddit_scheduler, DATASET_PATH, SEEN_IDS_PATH
except ImportError:
    from finetuning.reddit_scheduler import get_reddit_scheduler, DATASET_PATH, SEEN_IDS_PATH

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/reddit", tags=["reddit"])


class ConfigUpdate(BaseModel):
    """Request model for updating crawler configuration."""
    enabled: Optional[bool] = None
    interval_hours: Optional[float] = None
    max_per_run: Optional[int] = None
    subreddits: Optional[List[str]] = None


@router.get("/status")
def get_crawler_status():
    """Get current crawler status."""
    scheduler = get_reddit_scheduler()
    return {
        "status": scheduler.get_status(),
        "config": scheduler.get_config()
    }


@router.get("/config")
def get_crawler_config():
    """Get current crawler configuration."""
    scheduler = get_reddit_scheduler()
    return scheduler.get_config()


@router.put("/config")
def update_crawler_config(update: ConfigUpdate):
    """Update crawler configuration."""
    scheduler = get_reddit_scheduler()
    updates = update.dict(exclude_none=True)
    new_config = scheduler.update_config(updates)
    return {"config": new_config, "message": "Configuration updated"}


@router.post("/start")
async def start_crawler():
    """Start the crawler scheduler."""
    scheduler = get_reddit_scheduler()
    await scheduler.start()
    return {"message": "Crawler started", "status": scheduler.get_status()}


@router.post("/stop")
async def stop_crawler():
    """Stop the crawler scheduler."""
    scheduler = get_reddit_scheduler()
    await scheduler.stop()
    return {"message": "Crawler stopped", "status": scheduler.get_status()}


@router.post("/run-now")
async def run_crawler_now():
    """Trigger an immediate crawl run."""
    scheduler = get_reddit_scheduler()
    result = await scheduler.run_now()
    return {
        "result": result,
        "status": scheduler.get_status()
    }


@router.get("/samples")
def get_dataset_samples(limit: int = 20, offset: int = 0):
    """
    Get sample examples from the Reddit dataset.
    
    Args:
        limit: Number of samples to return (max 100)
        offset: Starting offset for pagination
    """
    limit = min(limit, 100)
    
    if not DATASET_PATH.exists():
        return {"samples": [], "total": 0, "offset": offset, "limit": limit}
    
    try:
        with open(DATASET_PATH, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return {"samples": [], "total": 0, "offset": offset, "limit": limit}
        
        total = len(data)
        samples = data[offset:offset + limit]
        
        return {
            "samples": samples,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total
        }
    except Exception as e:
        logger.error(f"Error reading dataset samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
def get_dataset_stats():
    """
    Get detailed statistics about the Reddit dataset.
    
    Returns:
        - Total examples
        - Examples by subreddit
        - Average output length
        - Instruction type breakdown
    """
    if not DATASET_PATH.exists():
        return {
            "total_examples": 0,
            "subreddit_counts": {},
            "avg_output_length": 0,
            "instruction_types": {},
            "seen_posts": 0
        }
    
    try:
        with open(DATASET_PATH, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = []
        
        # Calculate stats
        subreddit_counts: Dict[str, int] = {}
        instruction_types: Dict[str, int] = {}
        total_output_length = 0
        
        for item in data:
            instruction = item.get('instruction', '')
            output = item.get('output', '')
            
            # Extract subreddit from instruction
            if 'r/' in instruction:
                try:
                    sub = instruction.split('r/')[1].split(':')[0].split(' ')[0]
                    subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
                except:
                    pass
            
            # Categorize instruction type
            if instruction.startswith('Write a story'):
                instruction_types['post'] = instruction_types.get('post', 0) + 1
            elif instruction.startswith('Respond to'):
                instruction_types['comment'] = instruction_types.get('comment', 0) + 1
            else:
                instruction_types['other'] = instruction_types.get('other', 0) + 1
            
            total_output_length += len(output)
        
        # Load seen posts count
        seen_posts = 0
        if SEEN_IDS_PATH.exists():
            try:
                with open(SEEN_IDS_PATH, 'r') as f:
                    seen_posts = len(json.load(f))
            except:
                pass
        
        avg_output_length = total_output_length / len(data) if data else 0
        
        # Sort subreddit counts
        sorted_subs = dict(sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "total_examples": len(data),
            "subreddit_counts": sorted_subs,
            "avg_output_length": round(avg_output_length, 1),
            "instruction_types": instruction_types,
            "seen_posts": seen_posts,
            "estimated_tokens": len(data) * 300  # Rough estimate
        }
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/dataset")
def clear_dataset():
    """
    Clear the Reddit dataset and reset tracking.
    Use with caution - this deletes all collected data!
    """
    try:
        if DATASET_PATH.exists():
            DATASET_PATH.unlink()
        if SEEN_IDS_PATH.exists():
            SEEN_IDS_PATH.unlink()
        
        # Recreate empty dataset
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_PATH, 'w') as f:
            json.dump([], f)
        with open(SEEN_IDS_PATH, 'w') as f:
            json.dump([], f)
        
        return {"message": "Dataset cleared", "total_examples": 0}
    except Exception as e:
        logger.error(f"Error clearing dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))
