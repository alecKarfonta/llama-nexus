"""LlamaCPP commit management routes."""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import re
import subprocess
import httpx

from enhanced_logger import enhanced_logger as logger

router = APIRouter(prefix="/api/v1", tags=["llamacpp"])


def get_current_llamacpp_commit() -> str:
    """Get the current llama.cpp commit from Dockerfile"""
    try:
        dockerfile_path = Path("/home/alec/git/llama-nexus/Dockerfile")
        if not dockerfile_path.exists():
            return "unknown"
        
        content = dockerfile_path.read_text()
        match = re.search(r'git checkout ([^\s&\\]+)', content)
        if match:
            return match.group(1)
        return "unknown"
    except Exception:
        return "unknown"


@router.get("/llamacpp/commits")
async def get_llamacpp_commits():
    """Get available llama.cpp commits/releases"""
    try:
        async with httpx.AsyncClient() as client:
            # Get latest releases
            releases_response = await client.get(
                "https://api.github.com/repos/ggml-org/llama.cpp/releases",
                params={"per_page": 20}
            )
            releases_response.raise_for_status()
            releases = releases_response.json()
            
            # Get current commit from Dockerfile
            current_commit = get_current_llamacpp_commit()
            
            # Format releases
            formatted_releases = []
            for release in releases:
                formatted_releases.append({
                    "tag": release["tag_name"],
                    "name": release["name"] or release["tag_name"],
                    "published_at": release["published_at"],
                    "body": release["body"][:200] + "..." if len(release["body"]) > 200 else release["body"],
                    "is_current": release["tag_name"] == current_commit
                })
            
            # Also get some recent commits from master branch
            commits_response = await client.get(
                "https://api.github.com/repos/ggml-org/llama.cpp/commits",
                params={"per_page": 10}
            )
            commits_response.raise_for_status()
            commits = commits_response.json()
            
            formatted_commits = []
            for commit in commits:
                formatted_commits.append({
                    "tag": commit["sha"][:8],
                    "name": f"{commit['sha'][:8]} - {commit['commit']['message'].split(chr(10))[0][:50]}",
                    "published_at": commit["commit"]["committer"]["date"],
                    "body": commit["commit"]["message"],
                    "is_current": commit["sha"][:8] == current_commit or commit["sha"] == current_commit
                })
            
            return {
                "current_commit": current_commit,
                "releases": formatted_releases,
                "recent_commits": formatted_commits
            }
    except Exception as e:
        logger.error(f"Failed to fetch llama.cpp commits: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch commits: {str(e)}")


@router.get("/llamacpp/commits/{commit_id}/validate")
async def validate_llamacpp_commit(commit_id: str):
    """Validate that a commit exists in the llama.cpp repository"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.github.com/repos/ggml-org/llama.cpp/commits/{commit_id}",
                timeout=10.0
            )
            
            if response.status_code == 404:
                return {"valid": False, "error": "Commit not found in repository"}
            elif response.status_code != 200:
                return {"valid": False, "error": f"GitHub API error: {response.status_code}"}
            
            commit_data = response.json()
            
            return {
                "valid": True,
                "commit": {
                    "sha": commit_data["sha"],
                    "short_sha": commit_data["sha"][:8],
                    "message": commit_data["commit"]["message"].split('\n')[0][:100],
                    "author": commit_data["commit"]["author"]["name"],
                    "date": commit_data["commit"]["author"]["date"],
                    "url": commit_data["html_url"]
                }
            }
    except httpx.TimeoutException:
        return {"valid": False, "error": "Timeout while validating commit"}
    except Exception as e:
        logger.error(f"Failed to validate commit {commit_id}: {e}")
        return {"valid": False, "error": f"Validation error: {str(e)}"}


@router.post("/llamacpp/commits/{commit_id}/apply")
async def apply_llamacpp_commit(commit_id: str):
    """Update Dockerfile to use a specific llama.cpp commit"""
    try:
        validation = await validate_llamacpp_commit(commit_id)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid commit: {validation['error']}")
        
        dockerfile_path = Path("/home/alec/git/llama-nexus/Dockerfile")
        
        if not dockerfile_path.exists():
            raise HTTPException(status_code=404, detail="Dockerfile not found")
        
        content = dockerfile_path.read_text()
        updated_content = re.sub(
            r'git checkout [^\s&\\]+',
            f'git checkout {commit_id}',
            content
        )
        
        if updated_content == content:
            raise HTTPException(status_code=400, detail="Could not find git checkout line in Dockerfile")
        
        dockerfile_path.write_text(updated_content)
        logger.info(f"Updated Dockerfile to use llama.cpp commit: {commit_id}")
        
        return {
            "message": f"Dockerfile updated to use commit {commit_id}",
            "commit": commit_id,
            "commit_info": validation.get("commit"),
            "requires_rebuild": True
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update Dockerfile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update Dockerfile: {str(e)}")


@router.post("/llamacpp/rebuild")
async def rebuild_llamacpp():
    """Rebuild the llama.cpp containers with the current Dockerfile"""
    try:
        compose_file = "/home/alec/git/llama-nexus/docker-compose.yml"
        
        if not Path(compose_file).exists():
            raise HTTPException(status_code=500, detail=f"Docker compose file not found: {compose_file}")
        
        result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "up", "-d", "--build"],
            cwd="/home/alec/git/llama-nexus",
            capture_output=True,
            text=True,
            timeout=1800
        )
        
        if result.returncode != 0:
            logger.error(f"Docker rebuild failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Rebuild failed: {result.stderr}")
        
        logger.info("LlamaCPP containers rebuilt successfully")
        
        return {
            "message": "Containers rebuilt successfully",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Rebuild timed out after 30 minutes")
    except Exception as e:
        logger.error(f"Failed to rebuild containers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild: {str(e)}")
