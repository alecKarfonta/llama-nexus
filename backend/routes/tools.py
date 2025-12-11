"""Tools routes for research agents."""
import os
import logging
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tools", tags=["tools"])


class WebSearchRequest(BaseModel):
    query: str
    num_results: int = 5


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source: Optional[str] = None


class FetchUrlRequest(BaseModel):
    url: str
    extract_type: str = "text"  # text, markdown, summary


@router.post("/web-search")
async def web_search(request: WebSearchRequest):
    """
    Search the web using configured search API.
    Supports: Tavily, Serper, Brave, or SearXNG.
    """
    query = request.query
    num_results = min(request.num_results, 10)
    
    # Try different search providers based on environment
    tavily_key = os.getenv("TAVILY_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")
    brave_key = os.getenv("BRAVE_API_KEY")
    searxng_url = os.getenv("SEARXNG_URL")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Try Tavily
        if tavily_key:
            try:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": num_results,
                        "search_depth": "advanced"
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "query": query,
                        "results": [
                            {
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "snippet": r.get("content", ""),
                                "source": r.get("url", "").split("/")[2] if r.get("url") else None
                            }
                            for r in data.get("results", [])
                        ],
                        "source": "tavily"
                    }
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}")
        
        # Try Serper (Google Search API)
        if serper_key:
            try:
                response = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_key},
                    json={"q": query, "num": num_results}
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "query": query,
                        "results": [
                            {
                                "title": r.get("title", ""),
                                "url": r.get("link", ""),
                                "snippet": r.get("snippet", ""),
                                "source": r.get("link", "").split("/")[2] if r.get("link") else None
                            }
                            for r in data.get("organic", [])
                        ],
                        "source": "serper"
                    }
            except Exception as e:
                logger.warning(f"Serper search failed: {e}")
        
        # Try Brave Search
        if brave_key:
            try:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": brave_key},
                    params={"q": query, "count": num_results}
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "query": query,
                        "results": [
                            {
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "snippet": r.get("description", ""),
                                "source": r.get("url", "").split("/")[2] if r.get("url") else None
                            }
                            for r in data.get("web", {}).get("results", [])
                        ],
                        "source": "brave"
                    }
            except Exception as e:
                logger.warning(f"Brave search failed: {e}")
        
        # Try SearXNG (self-hosted)
        if searxng_url:
            try:
                response = await client.get(
                    f"{searxng_url}/search",
                    params={"q": query, "format": "json", "engines": "google,bing,duckduckgo"}
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "query": query,
                        "results": [
                            {
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "snippet": r.get("content", ""),
                                "source": r.get("engine", "searxng")
                            }
                            for r in data.get("results", [])[:num_results]
                        ],
                        "source": "searxng"
                    }
            except Exception as e:
                logger.warning(f"SearXNG search failed: {e}")
    
    # No search provider configured
    return {
        "query": query,
        "results": [],
        "source": "none",
        "message": "No search API configured. Set TAVILY_API_KEY, SERPER_API_KEY, BRAVE_API_KEY, or SEARXNG_URL."
    }


@router.post("/fetch-url")
async def fetch_url(request: FetchUrlRequest):
    """
    Fetch and extract content from a URL.
    """
    url = request.url
    extract_type = request.extract_type
    
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://github.com/llama-nexus)"
                }
            )
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "")
            
            if "text/html" in content_type:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                
                # Get title
                title = soup.title.string if soup.title else url
                
                # Extract text
                if extract_type == "markdown":
                    # Convert to basic markdown
                    content = []
                    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
                        if tag.name.startswith("h"):
                            level = int(tag.name[1])
                            content.append(f"{'#' * level} {tag.get_text().strip()}\n")
                        elif tag.name == "p":
                            text = tag.get_text().strip()
                            if text:
                                content.append(f"{text}\n")
                        elif tag.name == "li":
                            content.append(f"- {tag.get_text().strip()}\n")
                        elif tag.name in ["pre", "code"]:
                            content.append(f"```\n{tag.get_text()}\n```\n")
                    text_content = "\n".join(content)
                elif extract_type == "summary":
                    # Get first few paragraphs
                    paragraphs = soup.find_all("p")
                    text_content = " ".join(
                        p.get_text().strip() 
                        for p in paragraphs[:5] 
                        if len(p.get_text().strip()) > 50
                    )
                else:
                    # Raw text
                    text_content = soup.get_text(separator="\n", strip=True)
                
                # Limit content length
                max_chars = 10000 if extract_type == "text" else 5000
                if len(text_content) > max_chars:
                    text_content = text_content[:max_chars] + "...[truncated]"
                
                return {
                    "url": url,
                    "title": title,
                    "content": text_content,
                    "extract_type": extract_type,
                    "content_type": content_type
                }
            else:
                # Non-HTML content
                return {
                    "url": url,
                    "title": url,
                    "content": f"[Non-HTML content: {content_type}]",
                    "extract_type": extract_type,
                    "content_type": content_type
                }
                
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Failed to fetch URL: {e.response.status_code}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching URL: {str(e)}"
            )
