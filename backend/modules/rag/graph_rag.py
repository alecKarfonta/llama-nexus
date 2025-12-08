"""
GraphRAG Implementation

Knowledge graph construction and retrieval for RAG.
Features:
- Entity extraction using LLM
- Relationship extraction
- Graph storage and querying
- Graph-enhanced retrieval
"""

import json
import logging
import asyncio
import aiosqlite
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import uuid
import re

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Standard entity types"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    PROCESS = "process"
    CUSTOM = "custom"


class RelationshipType(str, Enum):
    """Standard relationship types"""
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PROPERTY = "has_property"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    CREATED_BY = "created_by"
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CONTRADICTS = "contradicts"
    CUSTOM = "custom"


@dataclass
class Entity:
    """A knowledge graph entity"""
    id: str
    name: str
    entity_type: EntityType
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    # Source tracking
    source_documents: List[str] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)
    # Metadata
    confidence: float = 1.0
    created_at: str = ""
    updated_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['entity_type'] = self.entity_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        data['entity_type'] = EntityType(data['entity_type'])
        if isinstance(data.get('aliases'), str):
            data['aliases'] = json.loads(data['aliases'])
        if isinstance(data.get('properties'), str):
            data['properties'] = json.loads(data['properties'])
        if isinstance(data.get('source_documents'), str):
            data['source_documents'] = json.loads(data['source_documents'])
        if isinstance(data.get('source_chunks'), str):
            data['source_chunks'] = json.loads(data['source_chunks'])
        return cls(**data)


@dataclass
class Relationship:
    """A knowledge graph relationship"""
    id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    # Bidirectional
    bidirectional: bool = False
    # Source tracking
    source_documents: List[str] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)
    # Metadata
    confidence: float = 1.0
    weight: float = 1.0
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['relationship_type'] = self.relationship_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        data['relationship_type'] = RelationshipType(data['relationship_type'])
        if isinstance(data.get('properties'), str):
            data['properties'] = json.loads(data['properties'])
        if isinstance(data.get('source_documents'), str):
            data['source_documents'] = json.loads(data['source_documents'])
        if isinstance(data.get('source_chunks'), str):
            data['source_chunks'] = json.loads(data['source_chunks'])
        return cls(**data)


@dataclass
class GraphNode:
    """Node in visualization graph"""
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge in visualization graph"""
    id: str
    source: str
    target: str
    label: str
    weight: float = 1.0


class GraphRAG:
    """
    GraphRAG implementation for knowledge-enhanced retrieval.
    
    Features:
    - Entity and relationship extraction using LLM
    - SQLite-based graph storage
    - Graph traversal and querying
    - Community detection
    - Graph-enhanced retrieval
    """
    
    # Prompt templates for extraction
    ENTITY_EXTRACTION_PROMPT = """Extract entities from the following text. Return a JSON array of entities.

Each entity should have:
- name: The entity name
- type: One of [person, organization, location, date, event, product, technology, concept, process]
- description: Brief description (optional)
- aliases: Alternative names (optional array)

Text:
{text}

Return only valid JSON array. Example:
[
  {"name": "OpenAI", "type": "organization", "description": "AI research company"},
  {"name": "GPT-4", "type": "technology", "description": "Large language model"}
]

JSON:"""

    RELATIONSHIP_EXTRACTION_PROMPT = """Extract relationships between entities from the following text.

Known entities: {entities}

Return a JSON array of relationships. Each relationship should have:
- source: Source entity name
- target: Target entity name  
- type: One of [is_a, part_of, has_property, located_in, works_for, created_by, depends_on, related_to, causes, precedes, follows, contradicts]
- description: Brief description (optional)

Text:
{text}

Return only valid JSON array. Example:
[
  {"source": "OpenAI", "target": "GPT-4", "type": "created_by", "description": "OpenAI developed GPT-4"}
]

JSON:"""

    def __init__(self, db_path: str = "data/rag/graph.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialized = False
        
        # In-memory graph for fast traversal
        self._entity_cache: Dict[str, Entity] = {}
        self._adjacency: Dict[str, Set[str]] = {}  # entity_id -> connected entity_ids
    
    async def initialize(self):
        """Initialize database schema"""
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            # Entities table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    aliases TEXT DEFAULT '[]',
                    properties TEXT DEFAULT '{}',
                    source_documents TEXT DEFAULT '[]',
                    source_chunks TEXT DEFAULT '[]',
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Relationships table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    properties TEXT DEFAULT '{}',
                    bidirectional INTEGER DEFAULT 0,
                    source_documents TEXT DEFAULT '[]',
                    source_chunks TEXT DEFAULT '[]',
                    confidence REAL DEFAULT 1.0,
                    weight REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                )
            """)
            
            # Indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relationship_type)")
            
            await db.commit()
        
        # Load into memory
        await self._load_graph()
        
        self._initialized = True
        logger.info(f"GraphRAG initialized with database: {self.db_path}")
    
    async def _load_graph(self):
        """Load graph into memory for fast traversal"""
        self._entity_cache.clear()
        self._adjacency.clear()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Load entities
            cursor = await db.execute("SELECT * FROM entities")
            rows = await cursor.fetchall()
            for row in rows:
                entity = Entity.from_dict(dict(row))
                self._entity_cache[entity.id] = entity
                self._adjacency[entity.id] = set()
            
            # Load relationships into adjacency
            cursor = await db.execute("SELECT source_id, target_id, bidirectional FROM relationships")
            rows = await cursor.fetchall()
            for row in rows:
                source_id, target_id, bidirectional = row
                if source_id in self._adjacency:
                    self._adjacency[source_id].add(target_id)
                if bidirectional and target_id in self._adjacency:
                    self._adjacency[target_id].add(source_id)
        
        logger.info(f"Loaded {len(self._entity_cache)} entities into memory")
    
    # Entity Operations
    
    async def create_entity(self, entity: Entity) -> Entity:
        """Create a new entity"""
        now = datetime.utcnow().isoformat()
        entity.created_at = entity.created_at or now
        entity.updated_at = now
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO entities (id, name, entity_type, description, aliases,
                    properties, source_documents, source_chunks, confidence,
                    created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.id, entity.name, entity.entity_type.value, entity.description,
                json.dumps(entity.aliases), json.dumps(entity.properties),
                json.dumps(entity.source_documents), json.dumps(entity.source_chunks),
                entity.confidence, entity.created_at, entity.updated_at
            ))
            await db.commit()
        
        # Update cache
        self._entity_cache[entity.id] = entity
        self._adjacency[entity.id] = set()
        
        logger.info(f"Created entity: {entity.name} ({entity.id})")
        return entity
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        if entity_id in self._entity_cache:
            return self._entity_cache[entity_id]
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM entities WHERE id = ?",
                (entity_id,)
            )
            row = await cursor.fetchone()
            if row:
                entity = Entity.from_dict(dict(row))
                self._entity_cache[entity.id] = entity
                return entity
        return None
    
    async def find_entity_by_name(self, name: str) -> Optional[Entity]:
        """Find entity by name or alias"""
        # Check cache first
        for entity in self._entity_cache.values():
            if entity.name.lower() == name.lower():
                return entity
            if name.lower() in [a.lower() for a in entity.aliases]:
                return entity
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM entities WHERE LOWER(name) = LOWER(?)",
                (name,)
            )
            row = await cursor.fetchone()
            if row:
                return Entity.from_dict(dict(row))
        return None
    
    async def list_entities(
        self,
        entity_type: Optional[EntityType] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Entity], int]:
        """List entities with filtering"""
        conditions = []
        params = []
        
        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type.value)
        if search:
            conditions.append("(name LIKE ? OR description LIKE ? OR aliases LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM entities {where_clause}",
                params
            )
            total = (await cursor.fetchone())[0]
            
            cursor = await db.execute(
                f"SELECT * FROM entities {where_clause} ORDER BY name LIMIT ? OFFSET ?",
                params + [limit, offset]
            )
            rows = await cursor.fetchall()
            entities = [Entity.from_dict(dict(row)) for row in rows]
        
        return entities, total
    
    async def update_entity(self, entity: Entity) -> Entity:
        """Update entity"""
        entity.updated_at = datetime.utcnow().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE entities SET
                    name = ?, entity_type = ?, description = ?, aliases = ?,
                    properties = ?, source_documents = ?, source_chunks = ?,
                    confidence = ?, updated_at = ?
                WHERE id = ?
            """, (
                entity.name, entity.entity_type.value, entity.description,
                json.dumps(entity.aliases), json.dumps(entity.properties),
                json.dumps(entity.source_documents), json.dumps(entity.source_chunks),
                entity.confidence, entity.updated_at, entity.id
            ))
            await db.commit()
        
        self._entity_cache[entity.id] = entity
        return entity
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete entity and its relationships"""
        async with aiosqlite.connect(self.db_path) as db:
            # Delete relationships
            await db.execute(
                "DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id)
            )
            # Delete entity
            await db.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            await db.commit()
        
        # Update cache
        if entity_id in self._entity_cache:
            del self._entity_cache[entity_id]
        if entity_id in self._adjacency:
            del self._adjacency[entity_id]
        
        # Remove from other adjacency lists
        for adj_set in self._adjacency.values():
            adj_set.discard(entity_id)
        
        logger.info(f"Deleted entity: {entity_id}")
        return True
    
    async def merge_entities(
        self,
        entity_ids: List[str],
        merged_name: str
    ) -> Optional[Entity]:
        """Merge multiple entities into one"""
        if len(entity_ids) < 2:
            return None
        
        entities = []
        for eid in entity_ids:
            entity = await self.get_entity(eid)
            if entity:
                entities.append(entity)
        
        if len(entities) < 2:
            return None
        
        # Create merged entity
        primary = entities[0]
        merged = Entity(
            id=str(uuid.uuid4()),
            name=merged_name,
            entity_type=primary.entity_type,
            description=primary.description,
            aliases=list(set(
                [e.name for e in entities] +
                [alias for e in entities for alias in e.aliases]
            )),
            properties={k: v for e in entities for k, v in e.properties.items()},
            source_documents=list(set(
                doc for e in entities for doc in e.source_documents
            )),
            source_chunks=list(set(
                chunk for e in entities for chunk in e.source_chunks
            )),
            confidence=min(e.confidence for e in entities)
        )
        
        await self.create_entity(merged)
        
        # Update relationships to point to merged entity
        async with aiosqlite.connect(self.db_path) as db:
            for entity in entities:
                await db.execute(
                    "UPDATE relationships SET source_id = ? WHERE source_id = ?",
                    (merged.id, entity.id)
                )
                await db.execute(
                    "UPDATE relationships SET target_id = ? WHERE target_id = ?",
                    (merged.id, entity.id)
                )
            await db.commit()
        
        # Delete old entities
        for entity in entities:
            await self.delete_entity(entity.id)
        
        # Reload graph
        await self._load_graph()
        
        return merged
    
    # Relationship Operations
    
    async def create_relationship(self, relationship: Relationship) -> Relationship:
        """Create a new relationship"""
        relationship.created_at = relationship.created_at or datetime.utcnow().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO relationships (id, source_id, target_id, relationship_type,
                    description, properties, bidirectional, source_documents,
                    source_chunks, confidence, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship.id, relationship.source_id, relationship.target_id,
                relationship.relationship_type.value, relationship.description,
                json.dumps(relationship.properties), int(relationship.bidirectional),
                json.dumps(relationship.source_documents),
                json.dumps(relationship.source_chunks), relationship.confidence,
                relationship.weight, relationship.created_at
            ))
            await db.commit()
        
        # Update adjacency
        if relationship.source_id in self._adjacency:
            self._adjacency[relationship.source_id].add(relationship.target_id)
        if relationship.bidirectional and relationship.target_id in self._adjacency:
            self._adjacency[relationship.target_id].add(relationship.source_id)
        
        logger.info(f"Created relationship: {relationship.source_id} -> {relationship.target_id}")
        return relationship
    
    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Relationship]:
        """Get relationships for an entity"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if direction == "outgoing":
                cursor = await db.execute(
                    "SELECT * FROM relationships WHERE source_id = ?",
                    (entity_id,)
                )
            elif direction == "incoming":
                cursor = await db.execute(
                    "SELECT * FROM relationships WHERE target_id = ?",
                    (entity_id,)
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?",
                    (entity_id, entity_id)
                )
            
            rows = await cursor.fetchall()
            return [Relationship.from_dict(dict(row)) for row in rows]
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT source_id, target_id FROM relationships WHERE id = ?",
                (relationship_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                source_id, target_id = row
                await db.execute(
                    "DELETE FROM relationships WHERE id = ?",
                    (relationship_id,)
                )
                await db.commit()
                
                # Update adjacency (reload to be safe)
                await self._load_graph()
                return True
        return False
    
    # Graph Operations
    
    async def get_subgraph(
        self,
        entity_ids: List[str],
        depth: int = 1
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Get subgraph around specified entities"""
        visited = set(entity_ids)
        frontier = set(entity_ids)
        
        # BFS to find connected entities
        for _ in range(depth):
            new_frontier = set()
            for entity_id in frontier:
                if entity_id in self._adjacency:
                    for neighbor in self._adjacency[entity_id]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_frontier.add(neighbor)
            frontier = new_frontier
        
        # Build nodes and edges
        nodes = []
        edges = []
        
        for entity_id in visited:
            entity = await self.get_entity(entity_id)
            if entity:
                nodes.append(GraphNode(
                    id=entity.id,
                    label=entity.name,
                    type=entity.entity_type.value,
                    properties={
                        'description': entity.description,
                        'confidence': entity.confidence
                    }
                ))
        
        # Get edges between visited nodes
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            placeholders = ','.join(['?' for _ in visited])
            cursor = await db.execute(
                f"""SELECT * FROM relationships 
                    WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})""",
                list(visited) + list(visited)
            )
            rows = await cursor.fetchall()
            
            for row in rows:
                rel = Relationship.from_dict(dict(row))
                edges.append(GraphEdge(
                    id=rel.id,
                    source=rel.source_id,
                    target=rel.target_id,
                    label=rel.relationship_type.value,
                    weight=rel.weight
                ))
        
        return nodes, edges
    
    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[List[str]]:
        """Find paths between two entities"""
        if source_id not in self._adjacency or target_id not in self._adjacency:
            return []
        
        paths = []
        
        def dfs(current: str, target: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return
            
            if current == target:
                paths.append(path.copy())
                return
            
            for neighbor in self._adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(source_id, target_id, [source_id], {source_id})
        return paths
    
    async def get_communities(self, min_size: int = 3) -> List[List[str]]:
        """Detect communities using connected components"""
        visited = set()
        communities = []
        
        for entity_id in self._adjacency:
            if entity_id in visited:
                continue
            
            # BFS to find connected component
            community = []
            queue = [entity_id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                community.append(current)
                
                for neighbor in self._adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if len(community) >= min_size:
                communities.append(community)
        
        return communities
    
    # Extraction Methods
    
    async def extract_entities_from_text(
        self,
        text: str,
        llm_endpoint: str = "http://localhost:8600/v1/chat/completions",
        api_key: Optional[str] = None
    ) -> List[Entity]:
        """Extract entities from text using LLM"""
        import aiohttp
        
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])  # Limit text
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    llm_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        logger.error(f"LLM request failed: {response.status}")
                        return []
                    
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # Parse JSON from response
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if not json_match:
                        return []
                    
                    entities_data = json.loads(json_match.group())
                    
                    entities = []
                    for data in entities_data:
                        entity = Entity(
                            id=str(uuid.uuid4()),
                            name=data.get('name', ''),
                            entity_type=EntityType(data.get('type', 'concept')),
                            description=data.get('description', ''),
                            aliases=data.get('aliases', [])
                        )
                        if entity.name:
                            entities.append(entity)
                    
                    return entities
                    
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def extract_relationships_from_text(
        self,
        text: str,
        entities: List[Entity],
        llm_endpoint: str = "http://localhost:8600/v1/chat/completions",
        api_key: Optional[str] = None
    ) -> List[Relationship]:
        """Extract relationships from text using LLM"""
        import aiohttp
        
        entity_names = [e.name for e in entities]
        prompt = self.RELATIONSHIP_EXTRACTION_PROMPT.format(
            text=text[:4000],
            entities=", ".join(entity_names[:50])  # Limit entities
        )
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    llm_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        logger.error(f"LLM request failed: {response.status}")
                        return []
                    
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # Parse JSON
                    json_match = re.search(r'\[.*\]', content, re.DOTALL)
                    if not json_match:
                        return []
                    
                    rels_data = json.loads(json_match.group())
                    
                    # Map entity names to IDs
                    name_to_entity = {e.name.lower(): e for e in entities}
                    
                    relationships = []
                    for data in rels_data:
                        source_name = data.get('source', '').lower()
                        target_name = data.get('target', '').lower()
                        
                        source = name_to_entity.get(source_name)
                        target = name_to_entity.get(target_name)
                        
                        if source and target:
                            try:
                                rel_type = RelationshipType(data.get('type', 'related_to'))
                            except ValueError:
                                rel_type = RelationshipType.RELATED_TO
                            
                            rel = Relationship(
                                id=str(uuid.uuid4()),
                                source_id=source.id,
                                target_id=target.id,
                                relationship_type=rel_type,
                                description=data.get('description', '')
                            )
                            relationships.append(rel)
                    
                    return relationships
                    
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    # Statistics
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            stats = {}
            
            cursor = await db.execute("SELECT COUNT(*) FROM entities")
            stats['entity_count'] = (await cursor.fetchone())[0]
            
            cursor = await db.execute("SELECT COUNT(*) FROM relationships")
            stats['relationship_count'] = (await cursor.fetchone())[0]
            
            cursor = await db.execute(
                "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type"
            )
            stats['entities_by_type'] = dict(await cursor.fetchall())
            
            cursor = await db.execute(
                "SELECT relationship_type, COUNT(*) FROM relationships GROUP BY relationship_type"
            )
            stats['relationships_by_type'] = dict(await cursor.fetchall())
            
            # Community count
            communities = await self.get_communities()
            stats['community_count'] = len(communities)
        
        return stats
    
    async def get_visualization_data(
        self,
        limit: int = 500
    ) -> Dict[str, Any]:
        """Get full graph data for visualization"""
        nodes = []
        edges = []
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get entities
            cursor = await db.execute(
                "SELECT * FROM entities LIMIT ?",
                (limit,)
            )
            rows = await cursor.fetchall()
            entity_ids = set()
            
            for row in rows:
                entity = Entity.from_dict(dict(row))
                entity_ids.add(entity.id)
                nodes.append({
                    'id': entity.id,
                    'label': entity.name,
                    'type': entity.entity_type.value,
                    'description': entity.description,
                    'confidence': entity.confidence
                })
            
            # Get relationships between these entities
            if entity_ids:
                placeholders = ','.join(['?' for _ in entity_ids])
                cursor = await db.execute(
                    f"""SELECT * FROM relationships 
                        WHERE source_id IN ({placeholders}) 
                        AND target_id IN ({placeholders})""",
                    list(entity_ids) + list(entity_ids)
                )
                rows = await cursor.fetchall()
                
                for row in rows:
                    rel = Relationship.from_dict(dict(row))
                    edges.append({
                        'id': rel.id,
                        'source': rel.source_id,
                        'target': rel.target_id,
                        'label': rel.relationship_type.value,
                        'weight': rel.weight,
                        'description': rel.description
                    })
        
        return {'nodes': nodes, 'edges': edges}
