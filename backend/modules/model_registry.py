"""
Model Registry Module
Provides local caching and registry for model metadata, usage tracking, and recommendations.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import hashlib
from local_paths import backend_data_path

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Local model registry with metadata caching.
    Stores model information, usage statistics, user ratings, and notes.
    """
    
    def __init__(self, db_path: str = None):
        """Initialize the model registry."""
        if db_path is None:
            db_path = os.getenv('MODEL_REGISTRY_DB_PATH') or str(backend_data_path('model_registry.db'))
        
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
        logger.info(f"Model registry initialized with database at: {self.db_path}")
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Model metadata cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_cache (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    downloads INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    tags TEXT,  -- JSON array
                    model_type TEXT,
                    license TEXT,
                    last_modified TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT  -- JSON object for additional metadata
                )
            ''')
            
            # Quantization variants table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_variants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    quantization TEXT NOT NULL,
                    size_bytes INTEGER,
                    vram_required_mb INTEGER,
                    quality_score REAL,
                    speed_score REAL,
                    created_at TEXT,
                    FOREIGN KEY (model_id) REFERENCES model_cache(id),
                    UNIQUE (model_id, filename)
                )
            ''')
            
            # Model usage statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    variant TEXT,
                    load_count INTEGER DEFAULT 0,
                    inference_count INTEGER DEFAULT 0,
                    total_tokens_generated INTEGER DEFAULT 0,
                    total_inference_time_ms INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT,
                    FOREIGN KEY (model_id) REFERENCES model_cache(id),
                    UNIQUE (model_id, variant)
                )
            ''')
            
            # User ratings and notes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    variant TEXT,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    notes TEXT,
                    tags TEXT,  -- JSON array for user tags
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (model_id) REFERENCES model_cache(id),
                    UNIQUE (model_id, variant)
                )
            ''')
            
            # Hardware compatibility recommendations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hardware_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    min_vram_gb REAL,
                    recommended_vram_gb REAL,
                    min_ram_gb REAL,
                    recommended_context_size INTEGER,
                    gpu_layers_recommendation INTEGER,
                    notes TEXT,
                    FOREIGN KEY (model_id) REFERENCES model_cache(id),
                    UNIQUE (model_id, variant)
                )
            ''')
            
            # Create indices for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_cache_repo ON model_cache(repo_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_variants_model ON model_variants(model_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_usage_model ON model_usage(model_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_usage_last_used ON model_usage(last_used)')
            
            conn.commit()
    
    def _generate_model_id(self, repo_id: str) -> str:
        """Generate a unique model ID from the repository ID."""
        return hashlib.sha256(repo_id.encode()).hexdigest()[:16]
    
    # Model Cache Operations
    
    def cache_model(
        self,
        repo_id: str,
        name: str,
        description: str = None,
        author: str = None,
        downloads: int = 0,
        likes: int = 0,
        tags: List[str] = None,
        model_type: str = None,
        license: str = None,
        last_modified: str = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Cache model metadata from HuggingFace."""
        model_id = self._generate_model_id(repo_id)
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_cache 
                (id, repo_id, name, description, author, downloads, likes, tags, 
                 model_type, license, last_modified, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    author = excluded.author,
                    downloads = excluded.downloads,
                    likes = excluded.likes,
                    tags = excluded.tags,
                    model_type = excluded.model_type,
                    license = excluded.license,
                    last_modified = excluded.last_modified,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
            ''', (
                model_id, repo_id, name, description, author, downloads, likes,
                json.dumps(tags or []), model_type, license, last_modified,
                now, now, json.dumps(metadata or {})
            ))
            conn.commit()
        
        logger.info(f"Cached model: {repo_id} (id: {model_id})")
        return model_id
    
    def get_cached_model(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Get cached model metadata."""
        model_id = self._generate_model_id(repo_id)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM model_cache WHERE id = ?', (model_id,))
            row = cursor.fetchone()
            
            if row:
                data = dict(row)
                data['tags'] = json.loads(data.get('tags') or '[]')
                data['metadata'] = json.loads(data.get('metadata') or '{}')
                return data
        
        return None
    
    def list_cached_models(
        self,
        limit: int = 50,
        offset: int = 0,
        search: str = None,
        model_type: str = None,
    ) -> Dict[str, Any]:
        """List all cached models with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM model_cache WHERE 1=1'
            params = []
            
            if search:
                query += ' AND (name LIKE ? OR description LIKE ? OR repo_id LIKE ?)'
                search_pattern = f'%{search}%'
                params.extend([search_pattern, search_pattern, search_pattern])
            
            if model_type:
                query += ' AND model_type = ?'
                params.append(model_type)
            
            # Get total count
            count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # Get paginated results
            query += ' ORDER BY downloads DESC, likes DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            cursor.execute(query, params)
            
            models = []
            for row in cursor.fetchall():
                data = dict(row)
                data['tags'] = json.loads(data.get('tags') or '[]')
                data['metadata'] = json.loads(data.get('metadata') or '{}')
                models.append(data)
            
            return {
                'models': models,
                'total': total,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total,
            }
    
    # Variant Operations
    
    def add_variant(
        self,
        repo_id: str,
        filename: str,
        quantization: str,
        size_bytes: int = None,
        vram_required_mb: int = None,
        quality_score: float = None,
        speed_score: float = None,
    ) -> None:
        """Add a quantization variant for a model."""
        model_id = self._generate_model_id(repo_id)
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_variants 
                (model_id, filename, quantization, size_bytes, vram_required_mb, 
                 quality_score, speed_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, filename) DO UPDATE SET
                    quantization = excluded.quantization,
                    size_bytes = excluded.size_bytes,
                    vram_required_mb = excluded.vram_required_mb,
                    quality_score = excluded.quality_score,
                    speed_score = excluded.speed_score
            ''', (model_id, filename, quantization, size_bytes, vram_required_mb,
                  quality_score, speed_score, now))
            conn.commit()
    
    def get_variants(self, repo_id: str) -> List[Dict[str, Any]]:
        """Get all quantization variants for a model."""
        model_id = self._generate_model_id(repo_id)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM model_variants 
                WHERE model_id = ? 
                ORDER BY size_bytes DESC
            ''', (model_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # Usage Statistics
    
    def record_model_load(self, repo_id: str, variant: str = None) -> None:
        """Record that a model was loaded."""
        model_id = self._generate_model_id(repo_id)
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_usage (model_id, variant, load_count, last_used, created_at)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(model_id, variant) DO UPDATE SET
                    load_count = load_count + 1,
                    last_used = excluded.last_used
            ''', (model_id, variant or '', now, now))
            conn.commit()
    
    def record_inference(
        self,
        repo_id: str,
        variant: str = None,
        tokens_generated: int = 0,
        inference_time_ms: int = 0,
    ) -> None:
        """Record inference statistics."""
        model_id = self._generate_model_id(repo_id)
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_usage 
                (model_id, variant, inference_count, total_tokens_generated, 
                 total_inference_time_ms, last_used, created_at)
                VALUES (?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(model_id, variant) DO UPDATE SET
                    inference_count = inference_count + 1,
                    total_tokens_generated = total_tokens_generated + excluded.total_tokens_generated,
                    total_inference_time_ms = total_inference_time_ms + excluded.total_inference_time_ms,
                    last_used = excluded.last_used
            ''', (model_id, variant or '', tokens_generated, inference_time_ms, now, now))
            conn.commit()
    
    def get_usage_stats(self, repo_id: str = None) -> List[Dict[str, Any]]:
        """Get usage statistics for models."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if repo_id:
                model_id = self._generate_model_id(repo_id)
                cursor.execute('''
                    SELECT u.*, c.name, c.repo_id 
                    FROM model_usage u
                    LEFT JOIN model_cache c ON u.model_id = c.id
                    WHERE u.model_id = ?
                    ORDER BY u.last_used DESC
                ''', (model_id,))
            else:
                cursor.execute('''
                    SELECT u.*, c.name, c.repo_id 
                    FROM model_usage u
                    LEFT JOIN model_cache c ON u.model_id = c.id
                    ORDER BY u.last_used DESC
                    LIMIT 100
                ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_most_used_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most frequently used models."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    c.id, c.repo_id, c.name, c.model_type,
                    SUM(u.load_count) as total_loads,
                    SUM(u.inference_count) as total_inferences,
                    MAX(u.last_used) as last_used
                FROM model_cache c
                LEFT JOIN model_usage u ON c.id = u.model_id
                GROUP BY c.id
                ORDER BY total_loads DESC, total_inferences DESC
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # Rating and Notes
    
    def set_rating(
        self,
        repo_id: str,
        rating: int,
        variant: str = None,
        notes: str = None,
        tags: List[str] = None,
    ) -> None:
        """Set user rating and notes for a model."""
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        model_id = self._generate_model_id(repo_id)
        now = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_ratings 
                (model_id, variant, rating, notes, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, variant) DO UPDATE SET
                    rating = excluded.rating,
                    notes = COALESCE(excluded.notes, notes),
                    tags = COALESCE(excluded.tags, tags),
                    updated_at = excluded.updated_at
            ''', (model_id, variant or '', rating, notes, json.dumps(tags or []), now, now))
            conn.commit()
    
    def get_rating(self, repo_id: str, variant: str = None) -> Optional[Dict[str, Any]]:
        """Get user rating for a model."""
        model_id = self._generate_model_id(repo_id)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM model_ratings 
                WHERE model_id = ? AND variant = ?
            ''', (model_id, variant or ''))
            row = cursor.fetchone()
            
            if row:
                data = dict(row)
                data['tags'] = json.loads(data.get('tags') or '[]')
                return data
        
        return None
    
    # Hardware Recommendations
    
    def set_hardware_recommendation(
        self,
        repo_id: str,
        variant: str,
        min_vram_gb: float = None,
        recommended_vram_gb: float = None,
        min_ram_gb: float = None,
        recommended_context_size: int = None,
        gpu_layers_recommendation: int = None,
        notes: str = None,
    ) -> None:
        """Set hardware recommendations for a model variant."""
        model_id = self._generate_model_id(repo_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO hardware_recommendations 
                (model_id, variant, min_vram_gb, recommended_vram_gb, min_ram_gb,
                 recommended_context_size, gpu_layers_recommendation, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, variant) DO UPDATE SET
                    min_vram_gb = COALESCE(excluded.min_vram_gb, min_vram_gb),
                    recommended_vram_gb = COALESCE(excluded.recommended_vram_gb, recommended_vram_gb),
                    min_ram_gb = COALESCE(excluded.min_ram_gb, min_ram_gb),
                    recommended_context_size = COALESCE(excluded.recommended_context_size, recommended_context_size),
                    gpu_layers_recommendation = COALESCE(excluded.gpu_layers_recommendation, gpu_layers_recommendation),
                    notes = COALESCE(excluded.notes, notes)
            ''', (model_id, variant, min_vram_gb, recommended_vram_gb, min_ram_gb,
                  recommended_context_size, gpu_layers_recommendation, notes))
            conn.commit()
    
    def get_recommendations_for_hardware(
        self,
        available_vram_gb: float,
        available_ram_gb: float = None,
    ) -> List[Dict[str, Any]]:
        """Get model recommendations based on available hardware."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
                SELECT 
                    c.id, c.repo_id, c.name, c.model_type, c.description,
                    v.filename, v.quantization, v.size_bytes, v.quality_score, v.speed_score,
                    h.min_vram_gb, h.recommended_vram_gb, h.min_ram_gb,
                    h.recommended_context_size, h.gpu_layers_recommendation,
                    r.rating, r.notes as user_notes
                FROM model_cache c
                INNER JOIN model_variants v ON c.id = v.model_id
                LEFT JOIN hardware_recommendations h ON c.id = h.model_id AND v.quantization = h.variant
                LEFT JOIN model_ratings r ON c.id = r.model_id AND (r.variant = v.quantization OR r.variant = '')
                WHERE (h.min_vram_gb IS NULL OR h.min_vram_gb <= ?)
            '''
            params = [available_vram_gb]
            
            if available_ram_gb:
                query += ' AND (h.min_ram_gb IS NULL OR h.min_ram_gb <= ?)'
                params.append(available_ram_gb)
            
            query += ' ORDER BY v.quality_score DESC, v.size_bytes ASC LIMIT 20'
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_model_cache(self, repo_id: str) -> bool:
        """Delete a model from the cache."""
        model_id = self._generate_model_id(repo_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM model_variants WHERE model_id = ?', (model_id,))
            cursor.execute('DELETE FROM model_usage WHERE model_id = ?', (model_id,))
            cursor.execute('DELETE FROM model_ratings WHERE model_id = ?', (model_id,))
            cursor.execute('DELETE FROM hardware_recommendations WHERE model_id = ?', (model_id,))
            cursor.execute('DELETE FROM model_cache WHERE id = ?', (model_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM model_cache')
            cached_models = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM model_variants')
            total_variants = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(load_count) FROM model_usage')
            total_loads = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(inference_count) FROM model_usage')
            total_inferences = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT COUNT(*) FROM model_ratings WHERE rating IS NOT NULL')
            rated_models = cursor.fetchone()[0]
            
            return {
                'cached_models': cached_models,
                'total_variants': total_variants,
                'total_loads': total_loads,
                'total_inferences': total_inferences,
                'rated_models': rated_models,
            }

    def seed_from_catalog(self, catalog_path: str = None, force: bool = False) -> int:
        """[2026-10-15] | INFO | MODEL_REGISTRY.SEED_FROM_CATALOG | Populate the
        registry with curated models on first run so new installs have
        recommendations out of the box. No-op when the cache already has models
        (unless force=True).
        """
        if not force:
            stats = self.get_registry_stats()
            if stats.get('cached_models', 0) > 0:
                return 0

        if catalog_path is None:
            catalog_path = str(Path(__file__).resolve().parents[1] / "data" / "model_catalog_seed.json")

        try:
            with open(catalog_path, "r") as fh:
                catalog = json.load(fh)
        except FileNotFoundError:
            logger.warning(f"Model catalog seed not found: {catalog_path}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"Model catalog seed invalid JSON: {e}")
            return 0

        seeded = 0
        for model in catalog.get("models", []):
            repo_id = model.get("repo_id")
            if not repo_id:
                continue
            model_id = self.cache_model(
                repo_id=repo_id,
                name=model.get("name", repo_id),
                description=model.get("description"),
                author=model.get("author"),
                downloads=model.get("downloads", 0),
                likes=model.get("likes", 0),
                tags=model.get("tags", []),
                model_type=model.get("model_type"),
                license=model.get("license"),
                last_modified=catalog.get("updated"),
                metadata={"curated": True},
            )
            for variant in model.get("variants", []):
                size_gb = variant.get("size_gb")
                self.add_variant(
                    repo_id=repo_id,
                    filename=variant.get("filename", ""),
                    quantization=variant.get("quantization", ""),
                    size_bytes=int(size_gb * (1024 ** 3)) if size_gb else None,
                    vram_required_mb=int(variant.get("min_vram_gb", 0) * 1024) or None,
                    quality_score=variant.get("quality_score"),
                    speed_score=variant.get("speed_score"),
                )
                hw = variant.get("recommended_vram_gb")
                if hw:
                    self.set_hardware_recommendation(
                        repo_id=repo_id,
                        variant=variant.get("quantization", ""),
                        min_vram_gb=variant.get("min_vram_gb"),
                        recommended_vram_gb=variant.get("recommended_vram_gb"),
                        min_ram_gb=variant.get("min_ram_gb"),
                        recommended_context_size=variant.get("recommended_context_size"),
                        gpu_layers_recommendation=None,
                        notes="Curated seed recommendation",
                    )
            seeded += 1

        logger.info(f"Seeded model registry with {seeded} curated models from {catalog_path}")
        return seeded


# Global instance
model_registry = ModelRegistry()
