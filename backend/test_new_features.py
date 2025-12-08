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


# Test VRAM Estimator
class TestVRAMEstimator:
    """Tests for the VRAM Estimator module."""
    
    def test_quantization_detection_q4km(self):
        """Test detecting Q4_K_M quantization from filename."""
        from modules.vram_estimator import detect_quantization
        
        assert detect_quantization("model-Q4_K_M.gguf") == "Q4_K_M"
        assert detect_quantization("llama-3-8b-q4_k_m.gguf") == "Q4_K_M"
        assert detect_quantization("MODEL-Q4-K-M.GGUF") == "Q4_K_M"
    
    def test_quantization_detection_other(self):
        """Test detecting various quantization levels."""
        from modules.vram_estimator import detect_quantization
        
        assert detect_quantization("model-Q8_0.gguf") == "Q8_0"
        assert detect_quantization("model-Q5_K_S.gguf") == "Q5_K_S"
        assert detect_quantization("model-F16.gguf") == "F16"
        assert detect_quantization("model-IQ4_XS.gguf") == "IQ4_XS"
    
    def test_quantization_detection_default(self):
        """Test default quantization when not detected."""
        from modules.vram_estimator import detect_quantization
        
        # Unknown patterns should default to Q4_K_M
        assert detect_quantization("model.gguf") == "Q4_K_M"
        assert detect_quantization("unknown-model") == "Q4_K_M"
    
    def test_model_architecture_detection_llama(self):
        """Test detecting Llama model architecture."""
        from modules.vram_estimator import detect_model_architecture
        
        arch = detect_model_architecture("llama-3.1-8b-instruct")
        assert arch is not None
        assert arch['params_b'] == 8
        assert arch['layers'] == 32
    
    def test_model_architecture_detection_qwen(self):
        """Test detecting Qwen model architecture."""
        from modules.vram_estimator import detect_model_architecture
        
        arch = detect_model_architecture("Qwen2.5-72B-Instruct")
        assert arch is not None
        assert arch['params_b'] == 72
        assert arch['layers'] == 80
    
    def test_model_architecture_from_params(self):
        """Test estimating architecture from parameter count."""
        from modules.vram_estimator import estimate_architecture_from_params
        
        # Small model
        arch = estimate_architecture_from_params(3)
        assert arch['layers'] == 24
        
        # Medium model
        arch = estimate_architecture_from_params(7)
        assert arch['layers'] == 32
        
        # Large model
        arch = estimate_architecture_from_params(70)
        assert arch['layers'] == 80
    
    def test_model_weights_vram_calculation(self):
        """Test model weights VRAM calculation."""
        from modules.vram_estimator import calculate_model_weights_vram
        
        # 7B model with Q4_K_M (4.5 bits per weight)
        vram = calculate_model_weights_vram(
            params_b=7,
            quantization="Q4_K_M",
            gpu_layers=32,
            total_layers=32
        )
        
        # Should be around 3.5-4GB
        assert 3000 < vram < 5000
    
    def test_model_weights_partial_offload(self):
        """Test VRAM calculation with partial GPU offload."""
        from modules.vram_estimator import calculate_model_weights_vram
        
        # Full GPU
        full_vram = calculate_model_weights_vram(
            params_b=7,
            quantization="Q4_K_M",
            gpu_layers=32,
            total_layers=32
        )
        
        # Half GPU
        half_vram = calculate_model_weights_vram(
            params_b=7,
            quantization="Q4_K_M",
            gpu_layers=16,
            total_layers=32
        )
        
        # Half should be ~50% of full
        assert 0.45 < (half_vram / full_vram) < 0.55
    
    def test_kv_cache_vram_calculation(self):
        """Test KV cache VRAM calculation."""
        from modules.vram_estimator import calculate_kv_cache_vram
        
        kv_cache = calculate_kv_cache_vram(
            context_size=4096,
            hidden_size=4096,
            num_layers=32,
            kv_heads=8,
            batch_size=1,
            kv_cache_type='f16'
        )
        
        # Should be some positive value
        assert kv_cache > 0
    
    def test_kv_cache_scales_with_context(self):
        """Test that KV cache scales with context size."""
        from modules.vram_estimator import calculate_kv_cache_vram
        
        small_ctx = calculate_kv_cache_vram(
            context_size=2048,
            hidden_size=4096,
            num_layers=32,
            kv_heads=8
        )
        
        large_ctx = calculate_kv_cache_vram(
            context_size=8192,
            hidden_size=4096,
            num_layers=32,
            kv_heads=8
        )
        
        # 4x context should be 4x KV cache
        assert 3.5 < (large_ctx / small_ctx) < 4.5
    
    def test_estimate_vram_basic(self):
        """Test basic VRAM estimation."""
        from modules.vram_estimator import estimate_vram
        
        estimate = estimate_vram(
            model_name="llama-3.1-8b-instruct-Q4_K_M.gguf",
            context_size=4096,
            available_vram_gb=24
        )
        
        assert estimate.total_gb > 0
        assert estimate.model_weights_mb > 0
        assert estimate.kv_cache_mb > 0
        assert estimate.fits_in_vram is True
        assert estimate.utilization_percent < 100
    
    def test_estimate_vram_warns_on_high_usage(self):
        """Test that high VRAM usage generates warnings."""
        from modules.vram_estimator import estimate_vram
        
        # Large model on small GPU
        estimate = estimate_vram(
            params_b=70,
            quantization="Q4_K_M",
            context_size=32768,
            available_vram_gb=8
        )
        
        assert estimate.fits_in_vram is False
        assert len(estimate.warnings) > 0
    
    def test_estimate_vram_with_explicit_params(self):
        """Test VRAM estimation with explicit parameters."""
        from modules.vram_estimator import estimate_vram
        
        estimate = estimate_vram(
            params_b=7,
            quantization="Q8_0",
            context_size=8192,
            batch_size=2,
            available_vram_gb=24
        )
        
        assert estimate.total_gb > 0
        assert estimate.utilization_percent > 0
    
    def test_get_quantization_options(self):
        """Test getting quantization options."""
        from modules.vram_estimator import get_quantization_options
        
        options = get_quantization_options()
        
        assert 'Q4_K_M' in options
        assert 'F16' in options
        assert 'Q8_0' in options
        assert options['F16'] == 16.0
        assert options['Q4_K_M'] == 4.5
    
    def test_get_model_architectures(self):
        """Test getting model architectures."""
        from modules.vram_estimator import get_model_architectures
        
        archs = get_model_architectures()
        
        assert 'llama-7b' in archs
        assert 'mistral-7b' in archs
        assert archs['llama-7b']['params_b'] == 7


# Test Token Tracker
class TestTokenTrackerUnit:
    """Unit tests for the Token Tracker module."""
    
    @pytest.fixture
    def token_tracker(self):
        """Create a token tracker instance with temp database."""
        from modules.token_tracker import TokenTracker
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        tracker = TokenTracker(db_path=db_path)
        yield tracker
        os.unlink(db_path)
    
    def test_record_token_usage(self, token_tracker):
        """Test recording token usage."""
        result = token_tracker.record_token_usage(
            model_id="test-model",
            prompt_tokens=100,
            completion_tokens=50
        )
        
        assert result is True
    
    def test_record_token_usage_with_metadata(self, token_tracker):
        """Test recording token usage with full metadata."""
        result = token_tracker.record_token_usage(
            model_id="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            model_name="Test Model",
            request_id="req-123",
            user_id="user-456",
            endpoint="/v1/chat/completions",
            metadata={"custom": "data"}
        )
        
        assert result is True
    
    def test_get_token_usage_empty(self, token_tracker):
        """Test getting token usage with no data."""
        usage = token_tracker.get_token_usage(time_range='24h')
        
        assert isinstance(usage, list)
        assert len(usage) == 0
    
    def test_get_token_usage_with_data(self, token_tracker):
        """Test getting token usage after recording."""
        # Record some usage
        token_tracker.record_token_usage(
            model_id="model-1",
            prompt_tokens=100,
            completion_tokens=50,
            model_name="Model One"
        )
        token_tracker.record_token_usage(
            model_id="model-1",
            prompt_tokens=200,
            completion_tokens=100,
            model_name="Model One"
        )
        
        usage = token_tracker.get_token_usage(time_range='24h')
        
        assert len(usage) == 1
        assert usage[0]['modelId'] == 'model-1'
        assert usage[0]['promptTokens'] == 300
        assert usage[0]['completionTokens'] == 150
        assert usage[0]['requests'] == 2
    
    def test_get_token_usage_multiple_models(self, token_tracker):
        """Test token usage with multiple models."""
        token_tracker.record_token_usage(
            model_id="model-a",
            prompt_tokens=100,
            completion_tokens=50
        )
        token_tracker.record_token_usage(
            model_id="model-b",
            prompt_tokens=200,
            completion_tokens=100
        )
        
        usage = token_tracker.get_token_usage(time_range='24h')
        
        assert len(usage) == 2
        model_ids = [u['modelId'] for u in usage]
        assert 'model-a' in model_ids
        assert 'model-b' in model_ids
    
    def test_get_total_token_usage_empty(self, token_tracker):
        """Test getting total token usage with no data."""
        total = token_tracker.get_total_token_usage(time_range='all')
        
        assert total['promptTokens'] == 0
        assert total['completionTokens'] == 0
        assert total['totalTokens'] == 0
        assert total['requests'] == 0
    
    def test_get_total_token_usage_with_data(self, token_tracker):
        """Test getting total token usage."""
        token_tracker.record_token_usage("m1", 100, 50)
        token_tracker.record_token_usage("m2", 200, 100)
        
        total = token_tracker.get_total_token_usage(time_range='all')
        
        assert total['promptTokens'] == 300
        assert total['completionTokens'] == 150
        assert total['totalTokens'] == 450
        assert total['requests'] == 2
        assert total['models'] == 2
    
    def test_get_token_usage_over_time(self, token_tracker):
        """Test getting token usage over time."""
        # Record some data
        for i in range(5):
            token_tracker.record_token_usage(
                model_id="test",
                prompt_tokens=100 * (i + 1),
                completion_tokens=50 * (i + 1)
            )
        
        timeline = token_tracker.get_token_usage_over_time(time_range='24h')
        
        assert isinstance(timeline, list)
        # All should be in same time interval since recorded quickly
        if len(timeline) > 0:
            assert 'timeInterval' in timeline[0]
            assert 'promptTokens' in timeline[0]
            assert 'completionTokens' in timeline[0]
    
    def test_time_range_filtering(self, token_tracker):
        """Test different time range filters."""
        token_tracker.record_token_usage("test", 100, 50)
        
        # All time ranges should work
        for time_range in ['1h', '24h', '7d', '30d']:
            usage = token_tracker.get_token_usage(time_range=time_range)
            assert isinstance(usage, list)


# Test Event Bus
class TestEventBus:
    """Tests for the Event Bus module."""
    
    @pytest.fixture
    def event_bus(self):
        """Create an event bus instance."""
        from modules.event_bus import EventBus
        bus = EventBus()
        yield bus
    
    def test_event_bus_creation(self, event_bus):
        """Test creating an event bus."""
        assert event_bus is not None
        assert event_bus.is_connected is False  # Not connected without Redis
    
    def test_subscribe_local(self, event_bus):
        """Test subscribing to local events."""
        messages = []
        
        def handler(msg):
            messages.append(msg)
        
        event_bus.subscribe_local("test-channel", handler)
        
        assert "test-channel" in event_bus.subscribers
        assert len(event_bus.subscribers["test-channel"]) == 1
    
    def test_unsubscribe_local(self, event_bus):
        """Test unsubscribing from local events."""
        def handler(msg):
            pass
        
        event_bus.subscribe_local("test-channel", handler)
        event_bus.unsubscribe_local("test-channel", handler)
        
        assert len(event_bus.subscribers["test-channel"]) == 0
    
    @pytest.mark.asyncio
    async def test_publish_local(self, event_bus):
        """Test publishing events locally."""
        messages = []
        
        def handler(msg):
            messages.append(msg)
        
        event_bus.subscribe_local("test-channel", handler)
        
        await event_bus.publish("test-channel", "test_event", {"key": "value"})
        
        assert len(messages) == 1
        assert messages[0]['type'] == 'test_event'
        assert messages[0]['data']['key'] == 'value'
        assert 'timestamp' in messages[0]
    
    @pytest.mark.asyncio
    async def test_publish_async_handler(self, event_bus):
        """Test publishing with async handler."""
        messages = []
        
        async def async_handler(msg):
            messages.append(msg)
        
        event_bus.subscribe_local("test-channel", async_handler)
        
        await event_bus.publish("test-channel", "async_event", {"async": True})
        
        assert len(messages) == 1
    
    @pytest.mark.asyncio
    async def test_emit_status_change(self, event_bus):
        """Test emitting status change events."""
        messages = []
        
        def handler(msg):
            messages.append(msg)
        
        event_bus.subscribe_local(event_bus.CHANNEL_STATUS, handler)
        
        await event_bus.emit_status_change("running", {"uptime": 100})
        
        assert len(messages) == 1
        assert messages[0]['type'] == 'status_change'
        assert messages[0]['data']['status'] == 'running'
    
    @pytest.mark.asyncio
    async def test_emit_model_event(self, event_bus):
        """Test emitting model events."""
        messages = []
        
        def handler(msg):
            messages.append(msg)
        
        event_bus.subscribe_local(event_bus.CHANNEL_MODEL, handler)
        
        await event_bus.emit_model_event("model_loaded", "llama-7b", {"vram_used": 5000})
        
        assert len(messages) == 1
        assert messages[0]['data']['model'] == 'llama-7b'
    
    @pytest.mark.asyncio
    async def test_emit_download_progress(self, event_bus):
        """Test emitting download progress events."""
        messages = []
        
        def handler(msg):
            messages.append(msg)
        
        event_bus.subscribe_local(event_bus.CHANNEL_DOWNLOAD, handler)
        
        await event_bus.emit_download_progress("dl-123", 0.75, {"speed": "10MB/s"})
        
        assert len(messages) == 1
        assert messages[0]['data']['download_id'] == 'dl-123'
        assert messages[0]['data']['progress'] == 0.75
    
    @pytest.mark.asyncio
    async def test_emit_metrics(self, event_bus):
        """Test emitting metrics events."""
        messages = []
        
        def handler(msg):
            messages.append(msg)
        
        event_bus.subscribe_local(event_bus.CHANNEL_METRICS, handler)
        
        await event_bus.emit_metrics({"cpu": 50, "memory": 4096})
        
        assert len(messages) == 1
        assert messages[0]['data']['cpu'] == 50
    
    @pytest.mark.asyncio
    async def test_emit_log(self, event_bus):
        """Test emitting log events."""
        messages = []
        
        def handler(msg):
            messages.append(msg)
        
        event_bus.subscribe_local(event_bus.CHANNEL_LOGS, handler)
        
        await event_bus.emit_log("INFO", "Test message", "test_source")
        
        assert len(messages) == 1
        assert messages[0]['data']['level'] == 'INFO'
        assert messages[0]['data']['message'] == 'Test message'


# Test Fixed Chunker
class TestFixedChunker:
    """Tests for the Fixed Chunker module."""
    
    @pytest.fixture
    def chunker(self):
        """Create a fixed chunker instance."""
        from modules.rag.chunkers.fixed_chunker import FixedChunker
        from modules.rag.chunkers.base import ChunkingConfig
        
        config = ChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=20,
            preserve_sentences=False
        )
        return FixedChunker(config=config)
    
    @pytest.fixture
    def chunker_with_sentences(self):
        """Create a chunker that preserves sentences."""
        from modules.rag.chunkers.fixed_chunker import FixedChunker
        from modules.rag.chunkers.base import ChunkingConfig
        
        config = ChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=20,
            preserve_sentences=True
        )
        return FixedChunker(config=config)
    
    def test_empty_text(self, chunker):
        """Test chunking empty text."""
        chunks = chunker.chunk("")
        assert chunks == []
    
    def test_whitespace_only(self, chunker):
        """Test chunking whitespace-only text."""
        chunks = chunker.chunk("   \n\t\n   ")
        assert chunks == []
    
    def test_short_text(self, chunker):
        """Test chunking text shorter than chunk size."""
        text = "This is a short text."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert "short text" in chunks[0].content
    
    def test_long_text_creates_multiple_chunks(self, chunker):
        """Test that long text creates multiple chunks."""
        text = "Lorem ipsum dolor sit amet. " * 50
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1
    
    def test_chunk_indices_sequential(self, chunker):
        """Test that chunk indices are sequential."""
        text = "This is a test sentence. " * 100
        chunks = chunker.chunk(text)
        
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
    
    def test_chunk_positions_valid(self, chunker):
        """Test that chunk positions are valid."""
        text = "Test text for chunking. " * 50
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert chunk.start_char < len(text)
    
    def test_chunk_metadata_includes_strategy(self, chunker):
        """Test that chunk metadata includes strategy."""
        text = "Some test text for the chunker to process."
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert chunks[0].metadata['chunking_strategy'] == 'fixed'
    
    def test_chunk_metadata_includes_total(self, chunker):
        """Test that chunks include total_chunks metadata."""
        text = "Test text. " * 100
        chunks = chunker.chunk(text)
        
        total = len(chunks)
        for chunk in chunks:
            assert chunk.metadata['total_chunks'] == total
    
    def test_custom_metadata_preserved(self, chunker):
        """Test that custom metadata is preserved."""
        text = "Test text. " * 100
        metadata = {"source": "test", "author": "pytest"}
        chunks = chunker.chunk(text, metadata=metadata)
        
        for chunk in chunks:
            assert chunk.metadata.get('source') == 'test'
            assert chunk.metadata.get('author') == 'pytest'
    
    def test_overlap_working(self, chunker):
        """Test that chunk overlap is working."""
        text = "Word" + " word" * 200  # Create repeating text
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            # This is a basic test - actual overlap depends on config
            first_end = chunks[0].end_char
            second_start = chunks[1].start_char
            assert second_start < first_end  # Overlap exists
    
    def test_sentence_preservation(self, chunker_with_sentences):
        """Test that sentence boundaries are respected when enabled."""
        text = "First sentence here. Second sentence here. Third sentence here. " * 20
        chunks = chunker_with_sentences.chunk(text)
        
        # Most chunks should end with sentence-ending punctuation
        sentence_ending_count = sum(
            1 for c in chunks 
            if c.content.rstrip().endswith(('.', '!', '?'))
        )
        
        # At least half should end with sentence boundaries
        # (last chunk might not)
        assert sentence_ending_count >= len(chunks) // 2
    
    def test_chunk_length_property(self, chunker):
        """Test that chunk length property works."""
        text = "Test " * 100
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            assert chunk.length == len(chunk.content)


# Additional Conversation Store Tests
class TestConversationStoreEdgeCases:
    """Additional edge case tests for the Conversation Store."""
    
    @pytest.fixture
    def conversation_store(self):
        """Create a conversation store instance with temp directory."""
        from modules.conversation_store import ConversationStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConversationStore(storage_dir=tmpdir)
            yield store
    
    def test_create_conversation_auto_title(self, conversation_store):
        """Test auto-generating title from first user message."""
        conv = conversation_store.create_conversation(
            messages=[
                {"role": "user", "content": "Hello, how are you today?"}
            ]
        )
        
        assert conv.title.startswith("Hello, how are you")
    
    def test_create_conversation_truncates_long_title(self, conversation_store):
        """Test that auto-generated titles are truncated."""
        long_message = "A" * 100
        conv = conversation_store.create_conversation(
            messages=[
                {"role": "user", "content": long_message}
            ]
        )
        
        assert len(conv.title) <= 53  # 50 + "..."
    
    def test_get_nonexistent_conversation(self, conversation_store):
        """Test getting a conversation that doesn't exist."""
        result = conversation_store.get_conversation("nonexistent-id")
        assert result is None
    
    def test_update_nonexistent_conversation(self, conversation_store):
        """Test updating a conversation that doesn't exist."""
        result = conversation_store.update_conversation(
            "nonexistent-id",
            title="New Title"
        )
        assert result is None
    
    def test_delete_nonexistent_conversation(self, conversation_store):
        """Test deleting a conversation that doesn't exist."""
        result = conversation_store.delete_conversation("nonexistent-id")
        assert result is False
    
    def test_add_message_to_nonexistent(self, conversation_store):
        """Test adding message to nonexistent conversation."""
        result = conversation_store.add_message(
            "nonexistent-id",
            {"role": "user", "content": "Hello"}
        )
        assert result is None
    
    def test_export_json_format(self, conversation_store):
        """Test exporting conversation as JSON."""
        conv = conversation_store.create_conversation(
            title="Export Test",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )
        
        exported = conversation_store.export_conversation(conv.id, format='json')
        
        assert exported is not None
        data = json.loads(exported)
        assert data['title'] == "Export Test"
        assert len(data['messages']) == 2
    
    def test_export_markdown_format(self, conversation_store):
        """Test exporting conversation as Markdown."""
        conv = conversation_store.create_conversation(
            title="Markdown Test",
            messages=[
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"}
            ]
        )
        
        exported = conversation_store.export_conversation(conv.id, format='markdown')
        
        assert exported is not None
        assert "# Markdown Test" in exported
        assert "### User" in exported
        assert "### Assistant" in exported
    
    def test_export_unsupported_format(self, conversation_store):
        """Test exporting with unsupported format."""
        conv = conversation_store.create_conversation(title="Test")
        
        result = conversation_store.export_conversation(conv.id, format='xml')
        assert result is None
    
    def test_list_with_search(self, conversation_store):
        """Test listing conversations with search."""
        conversation_store.create_conversation(
            title="Python Tutorial",
            messages=[{"role": "user", "content": "How to use Python?"}]
        )
        conversation_store.create_conversation(
            title="JavaScript Guide",
            messages=[{"role": "user", "content": "What is JavaScript?"}]
        )
        
        # Search by title
        result = conversation_store.list_conversations(search="Python")
        assert result['total'] == 1
        
        # Search by content
        result = conversation_store.list_conversations(search="JavaScript")
        assert result['total'] == 1
    
    def test_list_with_tags(self, conversation_store):
        """Test listing conversations filtered by tags."""
        conversation_store.create_conversation(
            title="Tagged 1",
            tags=["python", "tutorial"]
        )
        conversation_store.create_conversation(
            title="Tagged 2",
            tags=["javascript"]
        )
        
        result = conversation_store.list_conversations(tags=["python"])
        assert result['total'] == 1
        assert result['conversations'][0]['title'] == "Tagged 1"
    
    def test_archived_filtering(self, conversation_store):
        """Test filtering archived conversations."""
        conv = conversation_store.create_conversation(title="Active")
        archived = conversation_store.create_conversation(title="Archived")
        conversation_store.update_conversation(archived.id, is_archived=True)
        
        # Default excludes archived
        result = conversation_store.list_conversations()
        assert result['total'] == 1
        
        # Include archived
        result = conversation_store.list_conversations(include_archived=True)
        assert result['total'] == 2
    
    def test_conversation_with_tool_calls(self, conversation_store):
        """Test conversation with tool calls."""
        conv = conversation_store.create_conversation(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "get_weather"}}
                    ]
                },
                {
                    "role": "tool",
                    "content": "Sunny, 72F",
                    "tool_call_id": "call_1"
                }
            ]
        )
        
        loaded = conversation_store.get_conversation(conv.id)
        assert loaded is not None
        assert loaded.messages[0].tool_calls is not None
        assert loaded.messages[1].tool_call_id == "call_1"
    
    def test_conversation_with_reasoning(self, conversation_store):
        """Test conversation with reasoning content."""
        conv = conversation_store.create_conversation(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning_content": "I need to add 2 and 2 together."
                }
            ]
        )
        
        loaded = conversation_store.get_conversation(conv.id)
        assert loaded.messages[1].reasoning_content is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
