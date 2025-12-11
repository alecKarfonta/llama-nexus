"""
Workflow Storage - SQLite persistence for workflows and executions
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager

from .models import (
    Workflow, 
    WorkflowCreate, 
    WorkflowUpdate,
    WorkflowExecution,
    NodeExecution,
    ExecutionStatus,
)

logger = logging.getLogger(__name__)


class WorkflowStorage:
    """SQLite storage for workflows and executions"""
    
    def __init__(self, db_path: str = "data/workflows.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with context manager"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Workflows table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    nodes_json TEXT NOT NULL,
                    connections_json TEXT NOT NULL,
                    variables_json TEXT,
                    settings_json TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1
                )
            """)
            
            # Workflow versions table (for history)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    change_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE,
                    UNIQUE(workflow_id, version)
                )
            """)
            
            # Workflow executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    workflow_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    trigger_data_json TEXT,
                    node_executions_json TEXT,
                    final_output_json TEXT,
                    error TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                )
            """)
            
            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflows_active 
                ON workflows(is_active)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_workflow 
                ON workflow_executions(workflow_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_status 
                ON workflow_executions(status)
            """)
            
            logger.info("Workflow database initialized")
    
    # ----- Workflow CRUD -----
    
    def create_workflow(self, workflow: WorkflowCreate) -> Workflow:
        """Create a new workflow"""
        now = datetime.utcnow()
        wf = Workflow(
            name=workflow.name,
            description=workflow.description,
            nodes=workflow.nodes,
            connections=workflow.connections,
            variables=workflow.variables,
            settings=workflow.settings or Workflow.__fields__['settings'].default_factory(),
            createdAt=now,
            updatedAt=now,
        )
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workflows 
                (id, name, description, nodes_json, connections_json, variables_json, settings_json, created_at, updated_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                wf.id,
                wf.name,
                wf.description,
                json.dumps([n.dict() for n in wf.nodes]),
                json.dumps([c.dict() for c in wf.connections]),
                json.dumps(wf.variables),
                json.dumps(wf.settings.dict()),
                wf.createdAt.isoformat(),
                wf.updatedAt.isoformat(),
                wf.version,
            ))
            
            # Save initial version
            self._save_version(cursor, wf, "Initial creation")
        
        logger.info(f"Created workflow: {wf.id} - {wf.name}")
        return wf
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_workflow(row)
    
    def list_workflows(
        self, 
        limit: int = 50, 
        offset: int = 0,
        active_only: bool = True,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        """List workflows with pagination"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query
            where_clauses = []
            params = []
            
            if active_only:
                where_clauses.append("is_active = 1")
            
            if search:
                where_clauses.append("(name LIKE ? OR description LIKE ?)")
                params.extend([f"%{search}%", f"%{search}%"])
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM workflows WHERE {where_sql}", params)
            total = cursor.fetchone()[0]
            
            # Get workflows
            cursor.execute(f"""
                SELECT * FROM workflows 
                WHERE {where_sql}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            workflows = [self._row_to_workflow(row) for row in cursor.fetchall()]
            
            return {
                "workflows": workflows,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(workflows) < total,
            }
    
    def update_workflow(self, workflow_id: str, update: WorkflowUpdate) -> Optional[Workflow]:
        """Update an existing workflow"""
        existing = self.get_workflow(workflow_id)
        if not existing:
            return None
        
        now = datetime.utcnow()
        new_version = existing.version + 1
        
        # Apply updates
        updated_data = existing.dict()
        update_dict = update.dict(exclude_unset=True)
        
        for key, value in update_dict.items():
            if value is not None:
                updated_data[key] = value
        
        updated_data['updatedAt'] = now
        updated_data['version'] = new_version
        
        wf = Workflow(**updated_data)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE workflows SET
                    name = ?,
                    description = ?,
                    nodes_json = ?,
                    connections_json = ?,
                    variables_json = ?,
                    settings_json = ?,
                    is_active = ?,
                    updated_at = ?,
                    version = ?
                WHERE id = ?
            """, (
                wf.name,
                wf.description,
                json.dumps([n.dict() for n in wf.nodes]),
                json.dumps([c.dict() for c in wf.connections]),
                json.dumps(wf.variables),
                json.dumps(wf.settings.dict()),
                wf.isActive,
                wf.updatedAt.isoformat(),
                wf.version,
                workflow_id,
            ))
            
            # Save version
            self._save_version(cursor, wf, f"Update v{new_version}")
        
        logger.info(f"Updated workflow: {workflow_id} to version {new_version}")
        return wf
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow (soft delete by setting inactive)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE workflows SET is_active = 0, updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), workflow_id))
            
            if cursor.rowcount > 0:
                logger.info(f"Deleted workflow: {workflow_id}")
                return True
            return False
    
    def hard_delete_workflow(self, workflow_id: str) -> bool:
        """Permanently delete a workflow"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))
            
            if cursor.rowcount > 0:
                logger.info(f"Hard deleted workflow: {workflow_id}")
                return True
            return False
    
    # ----- Workflow Versions -----
    
    def _save_version(self, cursor, workflow: Workflow, summary: str):
        """Save a workflow version snapshot"""
        cursor.execute("""
            INSERT INTO workflow_versions (workflow_id, version, snapshot_json, change_summary)
            VALUES (?, ?, ?, ?)
        """, (
            workflow.id,
            workflow.version,
            json.dumps(workflow.dict(), default=str),
            summary,
        ))
    
    def get_workflow_versions(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get version history for a workflow"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version, change_summary, created_at
                FROM workflow_versions
                WHERE workflow_id = ?
                ORDER BY version DESC
            """, (workflow_id,))
            
            return [
                {
                    "version": row["version"],
                    "summary": row["change_summary"],
                    "createdAt": row["created_at"],
                }
                for row in cursor.fetchall()
            ]
    
    def restore_workflow_version(self, workflow_id: str, version: int) -> Optional[Workflow]:
        """Restore a workflow to a specific version"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT snapshot_json FROM workflow_versions
                WHERE workflow_id = ? AND version = ?
            """, (workflow_id, version))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            snapshot = json.loads(row["snapshot_json"])
            
            # Create update from snapshot
            update = WorkflowUpdate(
                name=snapshot.get("name"),
                description=snapshot.get("description"),
                nodes=snapshot.get("nodes"),
                connections=snapshot.get("connections"),
                variables=snapshot.get("variables"),
                settings=snapshot.get("settings"),
            )
            
            return self.update_workflow(workflow_id, update)
    
    # ----- Workflow Executions -----
    
    def create_execution(self, workflow_id: str, trigger_data: Dict[str, Any] = None) -> WorkflowExecution:
        """Create a new execution record"""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        execution = WorkflowExecution(
            workflowId=workflow_id,
            workflowVersion=workflow.version,
            triggerData=trigger_data or {},
        )
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workflow_executions
                (id, workflow_id, workflow_version, status, trigger_data_json, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                execution.id,
                execution.workflowId,
                execution.workflowVersion,
                execution.status.value,
                json.dumps(execution.triggerData),
                execution.startedAt.isoformat(),
            ))
        
        logger.info(f"Created execution: {execution.id} for workflow {workflow_id}")
        return execution
    
    def update_execution(self, execution: WorkflowExecution) -> WorkflowExecution:
        """Update an execution record"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE workflow_executions SET
                    status = ?,
                    node_executions_json = ?,
                    final_output_json = ?,
                    error = ?,
                    completed_at = ?
                WHERE id = ?
            """, (
                execution.status.value,
                json.dumps({k: v.dict() for k, v in execution.nodeExecutions.items()}, default=str),
                json.dumps(execution.finalOutput) if execution.finalOutput else None,
                execution.error,
                execution.completedAt.isoformat() if execution.completedAt else None,
                execution.id,
            ))
        
        return execution
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get an execution by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workflow_executions WHERE id = ?", (execution_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_execution(row)
    
    def list_executions(
        self, 
        workflow_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List executions with filtering"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            where_clauses = []
            params = []
            
            if workflow_id:
                where_clauses.append("workflow_id = ?")
                params.append(workflow_id)
            
            if status:
                where_clauses.append("status = ?")
                params.append(status.value)
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Get total
            cursor.execute(f"SELECT COUNT(*) FROM workflow_executions WHERE {where_sql}", params)
            total = cursor.fetchone()[0]
            
            # Get executions
            cursor.execute(f"""
                SELECT * FROM workflow_executions
                WHERE {where_sql}
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            executions = [self._row_to_execution(row) for row in cursor.fetchall()]
            
            return {
                "executions": executions,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(executions) < total,
            }
    
    # ----- Helper Methods -----
    
    def _row_to_workflow(self, row: sqlite3.Row) -> Workflow:
        """Convert a database row to a Workflow object"""
        from .models import WorkflowNode, WorkflowConnection, WorkflowSettings, Position, WorkflowNodeData, PortDefinition
        
        nodes_data = json.loads(row["nodes_json"])
        connections_data = json.loads(row["connections_json"])
        variables = json.loads(row["variables_json"]) if row["variables_json"] else {}
        settings_data = json.loads(row["settings_json"]) if row["settings_json"] else {}
        
        # Parse nodes
        nodes = []
        for n in nodes_data:
            node_data = n.get("data", {})
            inputs = [PortDefinition(**p) for p in node_data.get("inputs", [])]
            outputs = [PortDefinition(**p) for p in node_data.get("outputs", [])]
            
            nodes.append(WorkflowNode(
                id=n["id"],
                type=n["type"],
                position=Position(**n["position"]),
                data=WorkflowNodeData(
                    label=node_data.get("label", ""),
                    nodeType=node_data.get("nodeType", n["type"]),
                    config=node_data.get("config", {}),
                    inputs=inputs,
                    outputs=outputs,
                ),
            ))
        
        # Parse connections
        connections = [WorkflowConnection(**c) for c in connections_data]
        
        return Workflow(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            nodes=nodes,
            connections=connections,
            variables=variables,
            settings=WorkflowSettings(**settings_data) if settings_data else WorkflowSettings(),
            createdAt=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
            updatedAt=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.utcnow(),
            version=row["version"],
            isActive=bool(row["is_active"]),
        )
    
    def _row_to_execution(self, row: sqlite3.Row) -> WorkflowExecution:
        """Convert a database row to a WorkflowExecution object"""
        node_executions_data = json.loads(row["node_executions_json"]) if row["node_executions_json"] else {}
        
        node_executions = {}
        for node_id, ne_data in node_executions_data.items():
            node_executions[node_id] = NodeExecution(**ne_data)
        
        return WorkflowExecution(
            id=row["id"],
            workflowId=row["workflow_id"],
            workflowVersion=row["workflow_version"],
            status=ExecutionStatus(row["status"]),
            startedAt=datetime.fromisoformat(row["started_at"]) if row["started_at"] else datetime.utcnow(),
            completedAt=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            triggerData=json.loads(row["trigger_data_json"]) if row["trigger_data_json"] else {},
            nodeExecutions=node_executions,
            finalOutput=json.loads(row["final_output_json"]) if row["final_output_json"] else None,
            error=row["error"],
        )
