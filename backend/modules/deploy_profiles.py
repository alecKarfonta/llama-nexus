"""
Deploy Profiles: SQLite-backed storage for named deployment configurations.
Allows users to save, load, and manage model deployment presets.
"""
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DeployProfile:
    """Represents a saved deployment configuration."""
    
    def __init__(
        self,
        id: str,
        name: str,
        config: Dict[str, Any],
        created_at: str,
        updated_at: str,
        is_active: bool = False,
        tags: List[str] = None,
        description: str = "",
    ):
        self.id = id
        self.name = name
        self.config = config
        self.created_at = created_at
        self.updated_at = updated_at
        self.is_active = is_active
        self.tags = tags or []
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "tags": self.tags,
            "description": self.description,
        }
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DeployProfile":
        return cls(
            id=row["id"],
            name=row["name"],
            config=json.loads(row["config"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=bool(row["is_active"]),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            description=row["description"] or "",
        )


class DeployProfileStore:
    """SQLite-backed storage for deploy profiles."""
    
    def __init__(self, db_path: str = "data/deploy_profiles.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deploy_profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                config TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 0,
                tags TEXT DEFAULT '[]',
                description TEXT DEFAULT ''
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Deploy profiles database initialized at {self.db_path}")
    
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all saved profiles."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM deploy_profiles ORDER BY updated_at DESC"
            ).fetchall()
            return [DeployProfile.from_row(r).to_dict() for r in rows]
        finally:
            conn.close()
    
    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific profile by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM deploy_profiles WHERE id = ?", (profile_id,)
            ).fetchone()
            return DeployProfile.from_row(row).to_dict() if row else None
        finally:
            conn.close()
    
    def save_profile(
        self,
        name: str,
        config: Dict[str, Any],
        tags: List[str] = None,
        description: str = "",
        profile_id: str = None,
    ) -> Dict[str, Any]:
        """Save a new profile or update an existing one."""
        conn = self._get_conn()
        now = datetime.now().isoformat()
        
        try:
            if profile_id:
                # Update existing
                conn.execute(
                    """UPDATE deploy_profiles 
                       SET name = ?, config = ?, updated_at = ?, tags = ?, description = ?
                       WHERE id = ?""",
                    (name, json.dumps(config), now, json.dumps(tags or []), description, profile_id),
                )
                conn.commit()
                return self.get_profile(profile_id)
            else:
                # Create new
                new_id = str(uuid.uuid4())[:8]
                conn.execute(
                    """INSERT INTO deploy_profiles (id, name, config, created_at, updated_at, tags, description)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (new_id, name, json.dumps(config), now, now, json.dumps(tags or []), description),
                )
                conn.commit()
                return self.get_profile(new_id)
        except sqlite3.IntegrityError:
            raise ValueError(f"A profile named '{name}' already exists")
        finally:
            conn.close()
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "DELETE FROM deploy_profiles WHERE id = ?", (profile_id,)
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()
    
    def set_active(self, profile_id: str) -> bool:
        """Mark a profile as active (clears all others)."""
        conn = self._get_conn()
        try:
            conn.execute("UPDATE deploy_profiles SET is_active = 0")
            conn.execute(
                "UPDATE deploy_profiles SET is_active = 1 WHERE id = ?",
                (profile_id,),
            )
            conn.commit()
            return True
        finally:
            conn.close()
    
    def clear_active(self):
        """Clear all active flags."""
        conn = self._get_conn()
        try:
            conn.execute("UPDATE deploy_profiles SET is_active = 0")
            conn.commit()
        finally:
            conn.close()
    
    def get_active_profile(self) -> Optional[Dict[str, Any]]:
        """Get the currently active profile, if any."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM deploy_profiles WHERE is_active = 1"
            ).fetchone()
            return DeployProfile.from_row(row).to_dict() if row else None
        finally:
            conn.close()
