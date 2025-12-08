"""
Document Discovery System

Automated document gathering from web searches with review queue.
Features:
- Web search integration (DuckDuckGo, Serper)
- Content extraction from URLs
- Relevance scoring
- Review and approval workflow
"""

import logging
import asyncio
import aiohttp
import aiosqlite
import os
import json
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class DiscoveryStatus(str, Enum):
    """Status of discovered document"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class DiscoveredDocument:
    """A document discovered from search"""
    id: str
    title: str
    url: str
    snippet: str
    content: str = ""
    status: DiscoveryStatus = DiscoveryStatus.PENDING
    # Scoring
    relevance_score: float = 0.0
    quality_score: float = 0.0
    # Metadata
    source: str = ""  # Search provider
    search_query: str = ""
    domain_suggestion: Optional[str] = None
    # Content info
    word_count: int = 0
    content_hash: str = ""
    extracted_at: Optional[str] = None
    # Timestamps
    discovered_at: str = ""
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'status': self.status.value,
            'relevance_score': self.relevance_score,
            'quality_score': self.quality_score,
            'source': self.source,
            'search_query': self.search_query,
            'domain_suggestion': self.domain_suggestion,
            'word_count': self.word_count,
            'discovered_at': self.discovered_at,
            'reviewed_at': self.reviewed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscoveredDocument':
        data['status'] = DiscoveryStatus(data['status'])
        return cls(**data)


class DocumentDiscovery:
    """
    Document discovery and review system.
    
    Features:
    - Multiple search provider support
    - Automatic content extraction
    - Duplicate detection
    - Relevance scoring
    - Review queue management
    """
    
    # Search providers
    DUCKDUCKGO_URL = "https://api.duckduckgo.com/"
    SERPER_URL = "https://google.serper.dev/search"
    
    def __init__(
        self,
        db_path: str = "data/rag/discovery.db",
        serper_api_key: Optional[str] = None,
        llm_endpoint: Optional[str] = None
    ):
        self.db_path = db_path
        self.serper_api_key = serper_api_key
        self.llm_endpoint = llm_endpoint or "http://localhost:8600/v1/chat/completions"
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialized = False
    
    async def initialize(self):
        """Initialize database"""
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS discovered_documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    snippet TEXT DEFAULT '',
                    content TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    relevance_score REAL DEFAULT 0,
                    quality_score REAL DEFAULT 0,
                    source TEXT DEFAULT '',
                    search_query TEXT DEFAULT '',
                    domain_suggestion TEXT,
                    word_count INTEGER DEFAULT 0,
                    content_hash TEXT DEFAULT '',
                    extracted_at TEXT,
                    discovered_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewed_by TEXT
                )
            """)
            
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_discovered_status ON discovered_documents(status)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_discovered_url ON discovered_documents(url)"
            )
            await db.commit()
        
        self._initialized = True
        logger.info("DocumentDiscovery initialized")
    
    async def search_web(
        self,
        query: str,
        max_results: int = 10,
        provider: str = "duckduckgo"
    ) -> List[DiscoveredDocument]:
        """
        Search the web for documents matching query.
        Returns list of discovered documents.
        """
        await self.initialize()
        
        if provider == "serper" and self.serper_api_key:
            results = await self._search_serper(query, max_results)
        else:
            results = await self._search_duckduckgo(query, max_results)
        
        # Save to database
        saved = []
        for doc in results:
            try:
                await self._save_discovered(doc)
                saved.append(doc)
            except Exception as e:
                logger.warning(f"Failed to save discovered document: {e}")
        
        return saved
    
    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int
    ) -> List[DiscoveredDocument]:
        """Search using DuckDuckGo Instant Answer API"""
        # Note: DuckDuckGo doesn't have a direct web search API
        # This uses the instant answer API which is limited
        # For production, use Serper or similar
        
        params = {
            'q': query,
            'format': 'json',
            'no_redirect': '1',
            'no_html': '1'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.DUCKDUCKGO_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"DuckDuckGo search failed: {response.status}")
                        return []
                    
                    data = await response.json()
                    results = []
                    now = datetime.utcnow().isoformat()
                    
                    # Process related topics
                    topics = data.get('RelatedTopics', [])
                    for topic in topics[:max_results]:
                        if isinstance(topic, dict) and 'FirstURL' in topic:
                            results.append(DiscoveredDocument(
                                id=str(uuid.uuid4()),
                                title=topic.get('Text', '')[:200],
                                url=topic.get('FirstURL', ''),
                                snippet=topic.get('Text', ''),
                                source='duckduckgo',
                                search_query=query,
                                discovered_at=now
                            ))
                    
                    return results
                    
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    async def _search_serper(
        self,
        query: str,
        max_results: int
    ) -> List[DiscoveredDocument]:
        """Search using Serper (Google) API"""
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': max_results
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.SERPER_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Serper search failed: {response.status}")
                        return []
                    
                    data = await response.json()
                    results = []
                    now = datetime.utcnow().isoformat()
                    
                    for item in data.get('organic', [])[:max_results]:
                        results.append(DiscoveredDocument(
                            id=str(uuid.uuid4()),
                            title=item.get('title', ''),
                            url=item.get('link', ''),
                            snippet=item.get('snippet', ''),
                            source='serper',
                            search_query=query,
                            discovered_at=now
                        ))
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []
    
    async def extract_content(self, doc_id: str) -> Optional[DiscoveredDocument]:
        """Extract content from URL"""
        await self.initialize()
        
        doc = await self.get_discovered(doc_id)
        if not doc:
            return None
        
        try:
            content = await self._fetch_and_extract(doc.url)
            
            if content:
                doc.content = content
                doc.word_count = len(content.split())
                doc.content_hash = hashlib.md5(content.encode()).hexdigest()
                doc.extracted_at = datetime.utcnow().isoformat()
                doc.status = DiscoveryStatus.PENDING
                
                # Score relevance
                doc.relevance_score = await self._score_relevance(
                    doc.search_query, content
                )
                doc.quality_score = self._score_quality(content)
                
                await self._update_discovered(doc)
            
            return doc
            
        except Exception as e:
            logger.error(f"Content extraction failed for {doc.url}: {e}")
            doc.status = DiscoveryStatus.ERROR
            await self._update_discovered(doc)
            return doc
    
    async def _fetch_and_extract(self, url: str) -> Optional[str]:
        """Fetch URL and extract main content"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; LlamaNexus/1.0)'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return None
                    
                    html = await response.text()
                    return self._extract_text_from_html(html)
                    
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract main text content from HTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove scripts, styles, etc.
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            
            # Get text
            text = soup.get_text(separator='\n')
            
            # Clean up
            lines = [line.strip() for line in text.splitlines()]
            text = '\n'.join(line for line in lines if line)
            
            return text
            
        except ImportError:
            # Fallback without BeautifulSoup
            # Simple regex-based extraction
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    
    async def _score_relevance(self, query: str, content: str) -> float:
        """Score content relevance to query using LLM"""
        if not self.llm_endpoint:
            # Simple keyword-based scoring
            query_terms = query.lower().split()
            content_lower = content.lower()
            matches = sum(1 for term in query_terms if term in content_lower)
            return min(matches / len(query_terms), 1.0)
        
        prompt = f"""Rate the relevance of this content to the search query on a scale of 0-1.
        
Query: {query}

Content (first 1000 chars):
{content[:1000]}

Return only a number between 0 and 1:"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.llm_endpoint,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 10
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return 0.5
                    
                    result = await response.json()
                    text = result['choices'][0]['message']['content']
                    
                    # Extract number
                    match = re.search(r'[\d.]+', text)
                    if match:
                        return min(float(match.group()), 1.0)
                    return 0.5
                    
        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            return 0.5
    
    def _score_quality(self, content: str) -> float:
        """Score content quality based on heuristics"""
        score = 0.5
        
        word_count = len(content.split())
        
        # Length scoring
        if word_count > 500:
            score += 0.1
        if word_count > 1000:
            score += 0.1
        if word_count > 2000:
            score += 0.1
        
        # Penalty for very short content
        if word_count < 100:
            score -= 0.2
        
        # Check for structure (headers, paragraphs)
        if '\n\n' in content:
            score += 0.1
        
        # Penalty for too many special characters (likely not text)
        special_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', content)) / max(len(content), 1)
        if special_ratio > 0.3:
            score -= 0.2
        
        return max(0, min(score, 1.0))
    
    # CRUD Operations
    
    async def _save_discovered(self, doc: DiscoveredDocument):
        """Save discovered document to database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR IGNORE INTO discovered_documents
                (id, title, url, snippet, content, status, relevance_score,
                 quality_score, source, search_query, domain_suggestion,
                 word_count, content_hash, extracted_at, discovered_at,
                 reviewed_at, reviewed_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, doc.title, doc.url, doc.snippet, doc.content,
                doc.status.value, doc.relevance_score, doc.quality_score,
                doc.source, doc.search_query, doc.domain_suggestion,
                doc.word_count, doc.content_hash, doc.extracted_at,
                doc.discovered_at, doc.reviewed_at, doc.reviewed_by
            ))
            await db.commit()
    
    async def _update_discovered(self, doc: DiscoveredDocument):
        """Update discovered document"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE discovered_documents SET
                    title = ?, snippet = ?, content = ?, status = ?,
                    relevance_score = ?, quality_score = ?, domain_suggestion = ?,
                    word_count = ?, content_hash = ?, extracted_at = ?,
                    reviewed_at = ?, reviewed_by = ?
                WHERE id = ?
            """, (
                doc.title, doc.snippet, doc.content, doc.status.value,
                doc.relevance_score, doc.quality_score, doc.domain_suggestion,
                doc.word_count, doc.content_hash, doc.extracted_at,
                doc.reviewed_at, doc.reviewed_by, doc.id
            ))
            await db.commit()
    
    async def get_discovered(self, doc_id: str) -> Optional[DiscoveredDocument]:
        """Get discovered document by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM discovered_documents WHERE id = ?",
                (doc_id,)
            )
            row = await cursor.fetchone()
            if row:
                return DiscoveredDocument.from_dict(dict(row))
        return None
    
    async def get_review_queue(
        self,
        status: Optional[DiscoveryStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[DiscoveredDocument], int]:
        """Get documents in review queue"""
        await self.initialize()
        
        conditions = []
        params = []
        
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        else:
            conditions.append("status = ?")
            params.append(DiscoveryStatus.PENDING.value)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM discovered_documents {where_clause}",
                params
            )
            total = (await cursor.fetchone())[0]
            
            cursor = await db.execute(
                f"""SELECT * FROM discovered_documents {where_clause}
                    ORDER BY relevance_score DESC, discovered_at DESC
                    LIMIT ? OFFSET ?""",
                params + [limit, offset]
            )
            rows = await cursor.fetchall()
            docs = [DiscoveredDocument.from_dict(dict(row)) for row in rows]
        
        return docs, total
    
    async def approve_document(
        self,
        doc_id: str,
        domain_id: Optional[str] = None,
        reviewed_by: Optional[str] = None
    ) -> Optional[DiscoveredDocument]:
        """Approve a discovered document"""
        doc = await self.get_discovered(doc_id)
        if not doc:
            return None
        
        doc.status = DiscoveryStatus.APPROVED
        doc.reviewed_at = datetime.utcnow().isoformat()
        doc.reviewed_by = reviewed_by
        if domain_id:
            doc.domain_suggestion = domain_id
        
        await self._update_discovered(doc)
        return doc
    
    async def reject_document(
        self,
        doc_id: str,
        reviewed_by: Optional[str] = None
    ) -> Optional[DiscoveredDocument]:
        """Reject a discovered document"""
        doc = await self.get_discovered(doc_id)
        if not doc:
            return None
        
        doc.status = DiscoveryStatus.REJECTED
        doc.reviewed_at = datetime.utcnow().isoformat()
        doc.reviewed_by = reviewed_by
        
        await self._update_discovered(doc)
        return doc
    
    async def bulk_approve(
        self,
        doc_ids: List[str],
        domain_id: Optional[str] = None,
        reviewed_by: Optional[str] = None
    ) -> int:
        """Approve multiple documents"""
        count = 0
        for doc_id in doc_ids:
            result = await self.approve_document(doc_id, domain_id, reviewed_by)
            if result:
                count += 1
        return count
    
    async def check_duplicate(self, url: str) -> bool:
        """Check if URL already exists"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id FROM discovered_documents WHERE url = ?",
                (url,)
            )
            return await cursor.fetchone() is not None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            stats = {}
            
            cursor = await db.execute("SELECT COUNT(*) FROM discovered_documents")
            stats['total'] = (await cursor.fetchone())[0]
            
            cursor = await db.execute(
                "SELECT status, COUNT(*) FROM discovered_documents GROUP BY status"
            )
            stats['by_status'] = dict(await cursor.fetchall())
            
            cursor = await db.execute(
                "SELECT source, COUNT(*) FROM discovered_documents GROUP BY source"
            )
            stats['by_source'] = dict(await cursor.fetchall())
            
            cursor = await db.execute(
                "SELECT AVG(relevance_score), AVG(quality_score) FROM discovered_documents"
            )
            row = await cursor.fetchone()
            stats['avg_relevance'] = row[0] or 0
            stats['avg_quality'] = row[1] or 0
        
        return stats
