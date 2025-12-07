"""
Tests for new features added in Phase 4-5:
- Prompt Library
- Model Registry
- Benchmark Runner
- Batch Processor
- Conversation Stats
"""

import pytest
import json
import os
import tempfile
import asyncio
from datetime import datetime

# Test Prompt Library
class TestPromptLibrary:
    """Tests for the Prompt Library module."""
    
    @pytest.fixture
    def prompt_library(self):
        """Create a prompt library instance with temp database."""
        from modules.prompt_library import PromptLibrary
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        lib = PromptLibrary(db_path=db_path)
        yield lib
        os.unlink(db_path)
    
    def test_create_prompt(self, prompt_library):
        """Test creating a prompt."""
        prompt = prompt_library.create_prompt(
            name="Test Prompt",
            content="Hello {{name}}, how are you?",
            description="A test prompt",
            category="general",
            tags=["test", "greeting"],
        )
        
        assert prompt['id'] is not None
        assert prompt['name'] == "Test Prompt"
        assert prompt['content'] == "Hello {{name}}, how are you?"
        assert prompt['variables'] == ["name"]
        assert "test" in prompt['tags']
    
    def test_get_prompt(self, prompt_library):
        """Test getting a prompt by ID."""
        created = prompt_library.create_prompt(
            name="Get Test",
            content="Test content",
        )
        
        fetched = prompt_library.get_prompt(created['id'])
        assert fetched is not None
        assert fetched['name'] == "Get Test"
    
    def test_update_prompt(self, prompt_library):
        """Test updating a prompt."""
        created = prompt_library.create_prompt(
            name="Update Test",
            content="Original content",
        )
        
        updated = prompt_library.update_prompt(
            created['id'],
            name="Updated Name",
            content="Updated content {{var}}",
        )
        
        assert updated['name'] == "Updated Name"
        assert updated['content'] == "Updated content {{var}}"
        assert updated['variables'] == ["var"]
    
    def test_delete_prompt(self, prompt_library):
        """Test deleting a prompt."""
        created = prompt_library.create_prompt(
            name="Delete Test",
            content="To be deleted",
        )
        
        deleted = prompt_library.delete_prompt(created['id'])
        assert deleted is True
        
        fetched = prompt_library.get_prompt(created['id'])
        assert fetched is None
    
    def test_list_prompts(self, prompt_library):
        """Test listing prompts."""
        prompt_library.create_prompt(name="Prompt 1", content="Content 1", category="coding")
        prompt_library.create_prompt(name="Prompt 2", content="Content 2", category="writing")
        prompt_library.create_prompt(name="Prompt 3", content="Content 3", category="coding")
        
        # List all
        result = prompt_library.list_prompts()
        assert result['total'] == 3
        
        # Filter by category
        result = prompt_library.list_prompts(category="coding")
        assert result['total'] == 2
    
    def test_render_prompt(self, prompt_library):
        """Test rendering a prompt with variables."""
        created = prompt_library.create_prompt(
            name="Render Test",
            content="Hello {{name}}, welcome to {{place}}!",
        )
        
        rendered = prompt_library.render_prompt(
            created['id'],
            variables={"name": "Alice", "place": "Wonderland"}
        )
        
        assert rendered == "Hello Alice, welcome to Wonderland!"
    
    def test_prompt_versioning(self, prompt_library):
        """Test prompt version history."""
        created = prompt_library.create_prompt(
            name="Version Test",
            content="Version 1",
        )
        
        # Update to create new version
        prompt_library.update_prompt(created['id'], content="Version 2")
        prompt_library.update_prompt(created['id'], content="Version 3")
        
        versions = prompt_library.get_versions(created['id'])
        assert len(versions) == 3
        assert versions[0]['version'] == 3  # Most recent first
    
    def test_categories(self, prompt_library):
        """Test prompt categories."""
        categories = prompt_library.list_categories()
        assert len(categories) >= 6  # Default categories
        
        # Create custom category
        new_cat = prompt_library.create_category(
            name="Custom Category",
            description="A custom category",
            color="#FF0000",
        )
        assert new_cat['name'] == "Custom Category"
    
    def test_export_import(self, prompt_library):
        """Test exporting and importing prompts."""
        prompt_library.create_prompt(name="Export 1", content="Content 1")
        prompt_library.create_prompt(name="Export 2", content="Content 2")
        
        # Export
        exported = prompt_library.export_prompts()
        data = json.loads(exported)
        assert len(data['prompts']) == 2
        
        # Import to new library
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            new_db_path = f.name
        
        try:
            from modules.prompt_library import PromptLibrary
            new_lib = PromptLibrary(db_path=new_db_path)
            result = new_lib.import_prompts(exported)
            assert result['imported'] == 2
        finally:
            os.unlink(new_db_path)


# Test Model Registry
class TestModelRegistry:
    """Tests for the Model Registry module."""
    
    @pytest.fixture
    def model_registry(self):
        """Create a model registry instance with temp database."""
        from modules.model_registry import ModelRegistry
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        registry = ModelRegistry(db_path=db_path)
        yield registry
        os.unlink(db_path)
    
    def test_cache_model(self, model_registry):
        """Test caching model metadata."""
        model_id = model_registry.cache_model(
            repo_id="test/model-7b",
            name="Test Model 7B",
            description="A test model",
            author="Test Author",
            downloads=1000,
            likes=100,
            tags=["test", "7b"],
            model_type="text-generation",
        )
        
        assert model_id is not None
    
    def test_get_cached_model(self, model_registry):
        """Test getting a cached model."""
        model_registry.cache_model(
            repo_id="test/get-model",
            name="Get Test Model",
        )
        
        model = model_registry.get_cached_model("test/get-model")
        assert model is not None
        assert model['name'] == "Get Test Model"
    
    def test_add_variant(self, model_registry):
        """Test adding model variants."""
        model_registry.cache_model(repo_id="test/variant-model", name="Variant Test")
        
        model_registry.add_variant(
            repo_id="test/variant-model",
            filename="model-Q4_K_M.gguf",
            quantization="Q4_K_M",
            size_bytes=5000000000,
            vram_required_mb=6000,
        )
        
        variants = model_registry.get_variants("test/variant-model")
        assert len(variants) == 1
        assert variants[0]['quantization'] == "Q4_K_M"
    
    def test_usage_tracking(self, model_registry):
        """Test usage statistics tracking."""
        model_registry.cache_model(repo_id="test/usage-model", name="Usage Test")
        
        model_registry.record_model_load("test/usage-model", "Q4_K_M")
        model_registry.record_model_load("test/usage-model", "Q4_K_M")
        model_registry.record_inference(
            "test/usage-model",
            variant="Q4_K_M",
            tokens_generated=100,
            inference_time_ms=1000,
        )
        
        stats = model_registry.get_usage_stats("test/usage-model")
        assert len(stats) > 0
        assert stats[0]['load_count'] == 2
    
    def test_rating(self, model_registry):
        """Test model rating."""
        model_registry.cache_model(repo_id="test/rating-model", name="Rating Test")
        
        model_registry.set_rating(
            repo_id="test/rating-model",
            rating=5,
            notes="Great model!",
            tags=["fast", "accurate"],
        )
        
        rating = model_registry.get_rating("test/rating-model")
        assert rating['rating'] == 5
        assert rating['notes'] == "Great model!"


# Test Benchmark Runner
class TestBenchmarkRunner:
    """Tests for the Benchmark Runner module."""
    
    @pytest.fixture
    def benchmark_runner(self):
        """Create a benchmark runner instance with temp database."""
        from modules.benchmark import BenchmarkRunner
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        runner = BenchmarkRunner(db_path=db_path)
        yield runner
        os.unlink(db_path)
    
    def test_list_presets(self, benchmark_runner):
        """Test listing benchmark presets."""
        presets = benchmark_runner.list_presets()
        assert 'quick' in presets
        assert 'standard' in presets
        assert 'long_context' in presets
        assert 'max_speed' in presets
    
    def test_get_preset(self, benchmark_runner):
        """Test getting a specific preset."""
        preset = benchmark_runner.get_preset('quick')
        assert preset.prompt_tokens == 128
        assert preset.max_output_tokens == 64
        assert preset.num_runs == 3
    
    def test_stats(self, benchmark_runner):
        """Test getting benchmark stats."""
        stats = benchmark_runner.get_stats()
        assert 'total_benchmarks' in stats
        assert 'completed' in stats
        assert 'failed' in stats


# Test Batch Processor
class TestBatchProcessor:
    """Tests for the Batch Processor module."""
    
    @pytest.fixture
    def batch_processor(self):
        """Create a batch processor instance with temp database."""
        from modules.batch_processor import BatchProcessor
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        processor = BatchProcessor(db_path=db_path)
        yield processor
        os.unlink(db_path)
    
    def test_parse_json_input(self, batch_processor):
        """Test parsing JSON input."""
        content = json.dumps([
            {"input": "Hello"},
            {"input": "World", "metadata": {"key": "value"}},
        ])
        
        items = batch_processor.parse_input_file(content, 'json')
        assert len(items) == 2
        assert items[0]['input'] == "Hello"
        assert items[1]['metadata']['key'] == "value"
    
    def test_parse_csv_input(self, batch_processor):
        """Test parsing CSV input."""
        content = "input,category\nHello,greeting\nWorld,test"
        
        items = batch_processor.parse_input_file(content, 'csv')
        assert len(items) == 2
        assert items[0]['input'] == "Hello"
        assert items[0]['metadata']['category'] == "greeting"
    
    def test_parse_txt_input(self, batch_processor):
        """Test parsing TXT input."""
        content = "Line 1\nLine 2\nLine 3"
        
        items = batch_processor.parse_input_file(content, 'txt')
        assert len(items) == 3
        assert items[0]['input'] == "Line 1"
    
    def test_create_job(self, batch_processor):
        """Test creating a batch job."""
        items = [
            {"input": "Question 1"},
            {"input": "Question 2"},
        ]
        
        job = batch_processor.create_batch_job(
            name="Test Job",
            items=items,
            config={"max_tokens": 100},
        )
        
        assert job.id is not None
        assert job.name == "Test Job"
        assert job.total_items == 2
        assert job.status == "pending"
    
    def test_get_job(self, batch_processor):
        """Test getting a job by ID."""
        items = [{"input": "Test"}]
        created = batch_processor.create_batch_job(name="Get Test", items=items)
        
        fetched = batch_processor.get_job(created.id)
        assert fetched is not None
        assert fetched['name'] == "Get Test"
    
    def test_list_jobs(self, batch_processor):
        """Test listing jobs."""
        batch_processor.create_batch_job(name="Job 1", items=[{"input": "Test 1"}])
        batch_processor.create_batch_job(name="Job 2", items=[{"input": "Test 2"}])
        
        result = batch_processor.list_jobs()
        assert result['total'] == 2
    
    def test_get_job_items(self, batch_processor):
        """Test getting job items."""
        items = [
            {"input": "Item 1"},
            {"input": "Item 2"},
            {"input": "Item 3"},
        ]
        job = batch_processor.create_batch_job(name="Items Test", items=items)
        
        result = batch_processor.get_job_items(job.id)
        assert result['total'] == 3
        assert len(result['items']) == 3
    
    def test_delete_job(self, batch_processor):
        """Test deleting a job."""
        items = [{"input": "Delete me"}]
        job = batch_processor.create_batch_job(name="Delete Test", items=items)
        
        deleted = batch_processor.delete_job(job.id)
        assert deleted is True
        
        fetched = batch_processor.get_job(job.id)
        assert fetched is None
    
    def test_export_results(self, batch_processor):
        """Test exporting job results."""
        items = [{"input": "Export test"}]
        job = batch_processor.create_batch_job(name="Export Test", items=items)
        
        # Export as JSON
        json_export = batch_processor.export_results(job.id, format='json')
        data = json.loads(json_export)
        assert 'job' in data
        assert 'items' in data
        
        # Export as CSV
        csv_export = batch_processor.export_results(job.id, format='csv')
        assert 'input_text' in csv_export


# Test Conversation Store Statistics
class TestConversationStats:
    """Tests for the Conversation Store statistics."""
    
    @pytest.fixture
    def conversation_store(self):
        """Create a conversation store instance with temp directory."""
        from modules.conversation_store import ConversationStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConversationStore(storage_dir=tmpdir)
            yield store
    
    def test_get_statistics_empty(self, conversation_store):
        """Test getting statistics with no conversations."""
        stats = conversation_store.get_statistics()
        assert stats['total_conversations'] == 0
        assert stats['total_messages'] == 0
        assert stats['models_used'] == []
    
    def test_get_statistics_with_data(self, conversation_store):
        """Test getting statistics with conversations."""
        # Create some conversations
        conv1 = conversation_store.create_conversation(
            title="Test 1",
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        
        conv2 = conversation_store.create_conversation(
            title="Test 2",
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Question"},
            ]
        )
        
        stats = conversation_store.get_statistics()
        assert stats['total_conversations'] == 2
        assert stats['active_conversations'] == 2
        assert stats['total_messages'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
