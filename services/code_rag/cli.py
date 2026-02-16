"""
Command-line interface for Code RAG system.
"""

import typer
import asyncio
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import track
import time

from .parsers.python_parser import PythonParser
from .search.search_engine import CodeSearchEngine, SearchContext
from .models.entities import EntityType

app = typer.Typer(help="Code RAG - Intelligent Code Search CLI")
console = Console()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return"),
    threshold: float = typer.Option(0.0, "--threshold", "-t", help="Minimum similarity threshold"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Filter by language")
):
    """Search for code entities."""
    # Initialize search engine (in a real app, this would be persistent)
    search_engine = CodeSearchEngine()
    
    if not search_engine._indexed:
        console.print("[red]No code indexed yet. Use 'index' command first.[/red]")
        return
    
    # Build context
    context = SearchContext(language=language) if language else None
    
    # Perform search
    with console.status("[bold green]Searching..."):
        response = search_engine.search(query, context, top_k, threshold)
    
    # Display results
    console.print(f"\n[bold]Search Results for: '{query}'[/bold]")
    console.print(f"Found {response.total_results} results in {response.search_time_ms:.1f}ms")
    console.print(f"Query intent: {response.query_intent.value}")
    
    if response.results:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Score", style="green", width=8)
        table.add_column("Type", style="blue", width=10)
        table.add_column("Name", style="yellow", width=20)
        table.add_column("File", style="cyan", width=30)
        table.add_column("Explanation", style="white")
        
        for result in response.results:
            entity = result.entity
            table.add_row(
                f"{result.score:.3f}",
                entity.entity_type.value,
                entity.name,
                Path(entity.file_path).name,
                result.explanation[:50] + "..." if len(result.explanation) > 50 else result.explanation
            )
        
        console.print(table)
    else:
        console.print("[yellow]No results found.[/yellow]")
    
    # Show suggestions
    if response.suggestions:
        console.print(f"\n[bold]Suggestions:[/bold]")
        for suggestion in response.suggestions:
            console.print(f"  • {suggestion}")


@app.command()
def index(
    path: str = typer.Argument(..., help="Path to code file or directory"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Search recursively"),
    language: Optional[str] = typer.Option("python", "--language", "-l", help="Programming language")
):
    """Index code files for searching."""
    path_obj = Path(path)
    
    if not path_obj.exists():
        console.print(f"[red]Path does not exist: {path}[/red]")
        return
    
    # Find code files
    files = []
    if path_obj.is_file():
        files = [str(path_obj)]
    else:
        if language == "python":
            pattern = "**/*.py" if recursive else "*.py"
            files = [str(f) for f in path_obj.glob(pattern)]
    
    if not files:
        console.print("[yellow]No code files found.[/yellow]")
        return
    
    console.print(f"Found {len(files)} files to index")
    
    # Initialize parser and search engine
    if language == "python":
        parser = PythonParser()
    else:
        console.print(f"[red]Unsupported language: {language}[/red]")
        return
    
    search_engine = CodeSearchEngine()
    
    # Process files
    all_entities = []
    total_files = len(files)
    successful_files = 0
    
    for file_path in track(files, description="Indexing files..."):
        try:
            result = parser.parse_file(file_path)
            if result.success:
                all_entities.extend(result.entities)
                successful_files += 1
            else:
                console.print(f"[red]Failed to parse {file_path}: {result.error}[/red]")
        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {e}[/red]")
    
    # Add to search engine
    if all_entities:
        with console.status("[bold green]Building search index..."):
            search_engine.add_entities(all_entities)
    
    # Show statistics
    console.print(f"\n[bold green]Indexing complete![/bold green]")
    console.print(f"Files processed: {successful_files}/{total_files}")
    console.print(f"Total entities: {len(all_entities)}")
    
    # Show entity breakdown
    entity_counts = {}
    for entity in all_entities:
        entity_type = entity.entity_type.value
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    if entity_counts:
        table = Table(title="Entity Breakdown")
        table.add_column("Type", style="blue")
        table.add_column("Count", style="green")
        
        for entity_type, count in sorted(entity_counts.items()):
            table.add_row(entity_type.title(), str(count))
        
        console.print(table)


@app.command()
def analyze(
    file_path: str = typer.Argument(..., help="Path to code file to analyze"),
    show_relationships: bool = typer.Option(False, "--relationships", "-r", help="Show relationships")
):
    """Analyze a single code file."""
    path_obj = Path(file_path)
    
    if not path_obj.exists():
        console.print(f"[red]File does not exist: {file_path}[/red]")
        return
    
    # Initialize parser
    parser = PythonParser()
    
    if not parser.can_parse(file_path):
        console.print(f"[red]Unsupported file type: {file_path}[/red]")
        return
    
    # Parse file
    with console.status("[bold green]Analyzing file..."):
        result = parser.parse_file(file_path)
    
    if not result.success:
        console.print(f"[red]Failed to parse file: {result.error}[/red]")
        return
    
    # Display results
    console.print(f"\n[bold]Analysis of: {file_path}[/bold]")
    console.print(f"Language: {result.language}")
    console.print(f"Parse time: {result.parse_time_ms:.1f}ms")
    console.print(f"Entities found: {len(result.entities)}")
    console.print(f"Relationships found: {len(result.relationships)}")
    
    # Show entities
    if result.entities:
        table = Table(title="Entities", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="blue", width=12)
        table.add_column("Name", style="yellow", width=25)
        table.add_column("Lines", style="green", width=10)
        table.add_column("Details", style="white")
        
        for entity in result.entities:
            details = ""
            if hasattr(entity, 'parameters') and entity.parameters:
                details = f"{len(entity.parameters)} params"
            elif hasattr(entity, 'methods') and entity.methods:
                details = f"{len(entity.methods)} methods"
            elif hasattr(entity, 'scope'):
                details = entity.scope
            
            table.add_row(
                entity.entity_type.value,
                entity.name,
                f"{entity.line_start}-{entity.line_end}",
                details
            )
        
        console.print(table)
    
    # Show relationships
    if show_relationships and result.relationships:
        table = Table(title="Relationships", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="blue", width=15)
        table.add_column("Source", style="yellow", width=20)
        table.add_column("Target", style="green", width=20)
        table.add_column("Context", style="white")
        
        for rel in result.relationships:
            table.add_row(
                rel.relationship_type.value,
                rel.source_entity_id[:20] + "..." if len(rel.source_entity_id) > 20 else rel.source_entity_id,
                rel.target_entity_id[:20] + "..." if len(rel.target_entity_id) > 20 else rel.target_entity_id,
                rel.context[:40] + "..." if len(rel.context) > 40 else rel.context
            )
        
        console.print(table)


@app.command()
def demo():
    """Run a demo with sample code."""
    console.print("[bold blue]Code RAG Demo[/bold blue]")
    console.print("This will create sample Python code and demonstrate search capabilities.\n")
    
    # Create sample code
    sample_code = '''
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password."""
    if not username or not password:
        return False
    
    # Hash the password
    hashed_password = hash_password(password)
    
    # Check against database
    user = get_user_from_database(username)
    if user and user.password_hash == hashed_password:
        create_session(user.id)
        return True
    
    return False

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

class UserManager:
    """Manages user operations."""
    
    def __init__(self, database_connection):
        self.db = database_connection
    
    def create_user(self, username: str, email: str, password: str):
        """Create a new user account."""
        hashed_password = hash_password(password)
        user_data = {
            "username": username,
            "email": email,
            "password_hash": hashed_password
        }
        return self.db.insert("users", user_data)
    
    def delete_user(self, user_id: int):
        """Delete a user account."""
        return self.db.delete("users", {"id": user_id})

def log_message(level: str, message: str):
    """Log a message with specified level."""
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] {level.upper()}: {message}")
'''
    
    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_code)
        temp_file = f.name
    
    try:
        # Parse the sample code
        parser = PythonParser()
        result = parser.parse_file(temp_file)
        
        if result.success:
            # Initialize search engine
            search_engine = CodeSearchEngine()
            search_engine.add_entities(result.entities)
            
            console.print(f"✅ Parsed sample code: {len(result.entities)} entities found\n")
            
            # Demo searches
            demo_queries = [
                "authentication",
                "password hashing",
                "user management",
                "logging",
                "UserManager",
                "create_user"
            ]
            
            for query in demo_queries:
                console.print(f"[bold]Searching for: '{query}'[/bold]")
                response = search_engine.search(query, top_k=3)
                
                if response.results:
                    for i, result in enumerate(response.results[:2], 1):
                        entity = result.entity
                        console.print(f"  {i}. {entity.entity_type.value}: {entity.name} (score: {result.score:.3f})")
                else:
                    console.print("  No results found")
                console.print()
        
        else:
            console.print(f"[red]Failed to parse sample code: {result.error}[/red]")
    
    finally:
        # Clean up
        import os
        os.unlink(temp_file)


if __name__ == "__main__":
    app() 