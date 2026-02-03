#!/bin/bash
# Database backup script for llama-nexus
# Uses docker exec to safely backup SQLite databases from the container
# Keeps only the last 3 backups

BACKUP_DIR="/home/alec/git/llama-nexus/data/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAX_BACKUPS=3
CONTAINER="llamacpp-backend"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "$(date): Starting database backup..."

# Function to backup a database using docker exec
backup_db() {
    local container_path="$1"
    local db_name="$2"
    
    # Check if file exists in container
    if docker exec "$CONTAINER" test -f "$container_path"; then
        # Copy from container to host
        docker cp "$CONTAINER:$container_path" "$BACKUP_DIR/${db_name}_${TIMESTAMP}.db"
        if [ $? -eq 0 ]; then
            echo "$(date): Backed up ${db_name}.db"
        else
            echo "$(date): Failed to backup ${db_name}.db"
        fi
    else
        echo "$(date): Skipping ${db_name}.db (not found)"
    fi
}

# Backup main RAG databases
backup_db "/data/rag/documents.db" "documents"
backup_db "/data/rag/graph.db" "graph"
backup_db "/data/rag/discovery.db" "discovery"
backup_db "/data/rag_domains.db" "rag_domains"
backup_db "/data/document_manager.db" "document_manager"

# Backup other important databases
backup_db "/data/model_registry.db" "model_registry"
backup_db "/data/prompt_library.db" "prompt_library"
backup_db "/data/benchmarks.db" "benchmarks"
backup_db "/data/token_usage.db" "token_usage"
backup_db "/data/quantization.db" "quantization"
backup_db "/data/batch_jobs.db" "batch_jobs"

# Cleanup old backups - keep only the last MAX_BACKUPS for each type
for db_name in documents graph discovery rag_domains document_manager model_registry prompt_library benchmarks token_usage quantization batch_jobs; do
    backup_count=$(ls -1 "$BACKUP_DIR/${db_name}_"*.db 2>/dev/null | wc -l)
    if [ "$backup_count" -gt "$MAX_BACKUPS" ]; then
        # Remove oldest backups, keep most recent MAX_BACKUPS
        ls -1t "$BACKUP_DIR/${db_name}_"*.db | tail -n +$((MAX_BACKUPS + 1)) | xargs -r rm
        echo "$(date): Cleaned up old ${db_name} backups, kept last ${MAX_BACKUPS}"
    fi
done

echo "$(date): Backup completed successfully"
