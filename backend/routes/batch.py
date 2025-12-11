"""Batch processing routes."""
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import Response
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])


def get_batch_processor(request: Request):
    """Get the batch processor from app state."""
    processor = getattr(request.app.state, 'batch_processor', None)
    if processor is None:
        raise HTTPException(status_code=503, detail="Batch processor not available")
    return processor


@router.get("/stats")
async def get_batch_stats(request: Request):
    """Get batch processing statistics."""
    processor = get_batch_processor(request)
    return processor.get_stats()


@router.get("/jobs")
async def list_batch_jobs(
    request: Request,
    status: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """List batch jobs."""
    processor = get_batch_processor(request)
    return processor.list_jobs(status=status, limit=limit, offset=offset)


@router.post("/jobs")
async def create_batch_job(request: Request):
    """Create a new batch job."""
    processor = get_batch_processor(request)
    data = await request.json()
    
    # Parse input data
    items = []
    if 'items' in data:
        # Direct items array
        items = data['items']
    elif 'content' in data and 'file_type' in data:
        # File content to parse
        items = processor.parse_input_file(data['content'], data['file_type'])
    else:
        raise HTTPException(status_code=400, detail="Must provide 'items' or 'content' with 'file_type'")
    
    if not items:
        raise HTTPException(status_code=400, detail="No items to process")
    
    config = data.get('config', {})
    name = data.get('name', f"Batch Job {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    job = processor.create_batch_job(name=name, items=items, config=config)
    
    return {
        "status": "created",
        "job_id": job.id,
        "total_items": job.total_items,
    }


@router.post("/jobs/{job_id}/run")
async def run_batch_job(request: Request, job_id: str, background_tasks: BackgroundTasks):
    """Start running a batch job."""
    processor = get_batch_processor(request)
    
    job = processor.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] not in ['pending', 'failed']:
        raise HTTPException(status_code=400, detail=f"Job cannot be run (status: {job['status']})")
    
    data = await request.json() if request.headers.get('content-type') == 'application/json' else {}
    api_key = data.get('api_key')
    
    # Run in background
    async def run_in_background():
        try:
            await processor.run_batch_job(job_id, api_key=api_key)
        except Exception as e:
            logger.error(f"Background batch job failed: {e}")
    
    loop = asyncio.get_event_loop()
    loop.create_task(run_in_background())
    
    return {"status": "started", "job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_batch_job(request: Request, job_id: str):
    """Get batch job details."""
    processor = get_batch_processor(request)
    job = processor.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/items")
async def get_batch_job_items(
    request: Request,
    job_id: str,
    status: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """Get items for a batch job."""
    processor = get_batch_processor(request)
    return processor.get_job_items(job_id, status=status, limit=limit, offset=offset)


@router.post("/jobs/{job_id}/cancel")
async def cancel_batch_job(request: Request, job_id: str):
    """Cancel a running batch job."""
    processor = get_batch_processor(request)
    
    job = processor.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] != 'running':
        raise HTTPException(status_code=400, detail="Job is not running")
    
    processor.cancel_job(job_id)
    return {"status": "cancelling", "job_id": job_id}


@router.get("/jobs/{job_id}/export")
async def export_batch_job(request: Request, job_id: str, format: str = "json"):
    """Export batch job results."""
    processor = get_batch_processor(request)
    
    try:
        content = processor.export_results(job_id, format=format)
        
        if format == 'csv':
            return Response(
                content=content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=batch_{job_id}.csv"}
            )
        else:
            return Response(
                content=content,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=batch_{job_id}.json"}
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/jobs/{job_id}")
async def delete_batch_job(request: Request, job_id: str):
    """Delete a batch job."""
    processor = get_batch_processor(request)
    deleted = processor.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}
