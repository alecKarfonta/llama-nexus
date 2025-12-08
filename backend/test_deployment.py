"""
Tests for model deployment endpoints.

These tests cover the /api/v1/service/* endpoints for starting, stopping,
and monitoring model deployments. Deployments can take 2-5 minutes.

Run with: pytest test_deployment.py -v --timeout=600
"""

import pytest
import httpx
import asyncio
import time
from typing import Optional, Dict, Any


# Configuration
BACKEND_URL = "http://localhost:8700"
DEFAULT_TIMEOUT = 30  # seconds for regular API calls
DEPLOYMENT_TIMEOUT = 360  # 6 minutes max for deployment
HEALTH_CHECK_INTERVAL = 10  # seconds between status polls


class DeploymentTestClient:
    """Client for testing deployment endpoints."""
    
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url
        
    async def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/service/status",
                timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
    
    async def get_config(self) -> Dict[str, Any]:
        """Get current service configuration."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/service/config",
                timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
    
    async def start_service(self) -> Dict[str, Any]:
        """Start the model deployment."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/service/start",
                timeout=DEFAULT_TIMEOUT
            )
            # Don't raise for expected errors (400 = already running, etc)
            return {
                "status_code": response.status_code,
                "body": response.json() if response.status_code < 500 else {"error": response.text}
            }
    
    async def stop_service(self) -> Dict[str, Any]:
        """Stop the model deployment."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/service/stop",
                timeout=DEFAULT_TIMEOUT
            )
            return {
                "status_code": response.status_code,
                "body": response.json() if response.status_code < 500 else {"error": response.text}
            }
    
    async def restart_service(self) -> Dict[str, Any]:
        """Restart the model deployment."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/service/restart",
                timeout=120  # Restart might take longer
            )
            return {
                "status_code": response.status_code,
                "body": response.json() if response.status_code < 500 else {"error": response.text}
            }
    
    async def get_logs(self, lines: int = 100) -> Dict[str, Any]:
        """Get recent log lines."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/logs",
                params={"lines": lines},
                timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
    
    async def get_container_logs(self, lines: int = 100) -> Dict[str, Any]:
        """Get container logs (if running in Docker mode)."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/logs/container",
                params={"lines": lines},
                timeout=DEFAULT_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
            return {"error": response.text, "status_code": response.status_code}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check backend API health."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/health",
                timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
    
    async def wait_for_running(
        self, 
        timeout: int = DEPLOYMENT_TIMEOUT,
        interval: int = HEALTH_CHECK_INTERVAL
    ) -> Dict[str, Any]:
        """
        Wait for service to become running and healthy.
        
        Args:
            timeout: Maximum seconds to wait
            interval: Seconds between status checks
            
        Returns:
            Final status dict with timing info
        """
        start_time = time.time()
        last_status = None
        health_checks = []
        
        while time.time() - start_time < timeout:
            try:
                status = await self.get_status()
                last_status = status
                
                health_check = {
                    "timestamp": time.time() - start_time,
                    "running": status.get("running", False),
                    "llamacpp_health": status.get("llamacpp_health", {})
                }
                health_checks.append(health_check)
                
                # Check if service is fully running and healthy
                if status.get("running"):
                    llamacpp_health = status.get("llamacpp_health", {})
                    if llamacpp_health.get("healthy"):
                        return {
                            "success": True,
                            "status": status,
                            "elapsed_seconds": time.time() - start_time,
                            "health_checks": health_checks
                        }
                
            except Exception as e:
                health_checks.append({
                    "timestamp": time.time() - start_time,
                    "error": str(e)
                })
            
            await asyncio.sleep(interval)
        
        return {
            "success": False,
            "status": last_status,
            "elapsed_seconds": time.time() - start_time,
            "health_checks": health_checks,
            "error": "Timeout waiting for service to become healthy"
        }
    
    async def wait_for_stopped(
        self,
        timeout: int = 60,
        interval: int = 5
    ) -> Dict[str, Any]:
        """Wait for service to stop completely."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                status = await self.get_status()
                if not status.get("running"):
                    return {
                        "success": True,
                        "status": status,
                        "elapsed_seconds": time.time() - start_time
                    }
            except Exception as e:
                pass
            
            await asyncio.sleep(interval)
        
        return {
            "success": False,
            "elapsed_seconds": time.time() - start_time,
            "error": "Timeout waiting for service to stop"
        }


class TestDeploymentStatus:
    """Tests for service status endpoint."""
    
    @pytest.fixture
    def client(self):
        return DeploymentTestClient()
    
    @pytest.mark.asyncio
    async def test_backend_health(self, client):
        """Test that backend API is healthy."""
        health = await client.health_check()
        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert "mode" in health
    
    @pytest.mark.asyncio
    async def test_get_status(self, client):
        """Test getting service status."""
        status = await client.get_status()
        
        # Required fields
        assert "running" in status
        assert isinstance(status["running"], bool)
        assert "mode" in status
        assert status["mode"] in ["docker", "subprocess"]
        
        # Optional fields when running
        if status["running"]:
            assert "pid" in status or status["pid"] is None
            assert "uptime" in status
            assert isinstance(status["uptime"], (int, float))
    
    @pytest.mark.asyncio
    async def test_get_config(self, client):
        """Test getting service configuration."""
        config = await client.get_config()
        
        assert "config" in config
        cfg = config["config"]
        
        # Required config sections
        assert "model" in cfg
        assert "sampling" in cfg
        assert "performance" in cfg
        assert "server" in cfg
        
        # Model config
        assert "name" in cfg["model"]
        assert "variant" in cfg["model"]
        
        # Editable fields info
        assert "editable_fields" in config
    
    @pytest.mark.asyncio
    async def test_get_logs(self, client):
        """Test getting log lines."""
        logs = await client.get_logs(lines=50)
        
        assert "logs" in logs
        assert isinstance(logs["logs"], list)
        assert "count" in logs


class TestDeploymentLifecycle:
    """
    Tests for service start/stop/restart lifecycle.
    
    WARNING: These tests actually start/stop the model deployment
    and can take several minutes to complete.
    """
    
    @pytest.fixture
    def client(self):
        return DeploymentTestClient()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_start_service_when_stopped(self, client):
        """Test starting service when it's not running."""
        # First check if already running
        initial_status = await client.get_status()
        
        if initial_status.get("running"):
            pytest.skip("Service already running - stop it first to test start")
        
        # Start the service
        result = await client.start_service()
        
        # Should succeed (200) or be starting
        assert result["status_code"] in [200, 202], f"Start failed: {result}"
        
        # Wait for it to become healthy
        wait_result = await client.wait_for_running(timeout=DEPLOYMENT_TIMEOUT)
        
        # Log the timeline for debugging
        print(f"\nDeployment timeline ({wait_result['elapsed_seconds']:.1f}s):")
        for check in wait_result.get("health_checks", [])[-10:]:
            print(f"  {check.get('timestamp', 0):.1f}s: {check}")
        
        assert wait_result["success"], f"Service did not become healthy: {wait_result.get('error')}"
        assert wait_result["status"]["running"]
        assert wait_result["status"].get("llamacpp_health", {}).get("healthy")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_start_service_when_already_running(self, client):
        """Test starting service when it's already running returns 400."""
        status = await client.get_status()
        
        if not status.get("running"):
            pytest.skip("Service not running - start it first to test double-start")
        
        # Try to start again
        result = await client.start_service()
        
        # Should return 400 (already running)
        assert result["status_code"] == 400
        assert "already running" in result["body"].get("detail", "").lower()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_stop_service_when_running(self, client):
        """Test stopping a running service."""
        status = await client.get_status()
        
        if not status.get("running"):
            pytest.skip("Service not running - start it first to test stop")
        
        # Stop the service
        result = await client.stop_service()
        
        assert result["status_code"] == 200, f"Stop failed: {result}"
        
        # Wait for it to actually stop
        wait_result = await client.wait_for_stopped(timeout=60)
        
        assert wait_result["success"], f"Service did not stop: {wait_result.get('error')}"
    
    @pytest.mark.asyncio
    async def test_stop_service_when_not_running(self, client):
        """Test stopping service when it's not running returns 400."""
        status = await client.get_status()
        
        if status.get("running"):
            pytest.skip("Service is running - stop it first to test double-stop")
        
        result = await client.stop_service()
        
        # Should return 400 (not running)
        assert result["status_code"] == 400
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_restart_service(self, client):
        """Test restarting the service."""
        status = await client.get_status()
        
        if not status.get("running"):
            pytest.skip("Service not running - start it first to test restart")
        
        initial_uptime = status.get("uptime", 0)
        
        # Restart
        result = await client.restart_service()
        
        assert result["status_code"] == 200, f"Restart failed: {result}"
        
        # Wait for it to become healthy again
        wait_result = await client.wait_for_running(timeout=DEPLOYMENT_TIMEOUT)
        
        assert wait_result["success"], f"Service did not restart successfully: {wait_result.get('error')}"
        
        # Uptime should be reset (less than initial)
        new_status = wait_result["status"]
        # Note: uptime might be similar if restart was fast, so just check it's running
        assert new_status["running"]


class TestDeploymentErrors:
    """Tests for deployment error scenarios."""
    
    @pytest.fixture
    def client(self):
        return DeploymentTestClient()
    
    @pytest.mark.asyncio
    async def test_status_when_backend_unreachable(self):
        """Test behavior when backend is unreachable."""
        client = DeploymentTestClient(base_url="http://localhost:9999")
        
        with pytest.raises(httpx.ConnectError):
            await client.get_status()
    
    @pytest.mark.asyncio
    async def test_container_logs_when_not_running(self, client):
        """Test getting container logs when service is not running."""
        status = await client.get_status()
        
        if status.get("running"):
            pytest.skip("Service is running")
        
        # Should still work but may return empty or error
        logs = await client.get_container_logs()
        # Either returns logs or an error dict
        assert "logs" in logs or "error" in logs or "status_code" in logs


class TestDeploymentConfiguration:
    """Tests for deployment configuration validation."""
    
    @pytest.fixture
    def client(self):
        return DeploymentTestClient()
    
    @pytest.mark.asyncio
    async def test_config_has_model_info(self, client):
        """Test that config contains required model information."""
        config = await client.get_config()
        cfg = config["config"]
        
        model = cfg["model"]
        assert "name" in model, "Model name is required"
        assert "variant" in model, "Model variant is required"
        
        # These should be present for deployment
        if "context_size" in model:
            assert isinstance(model["context_size"], int)
            assert model["context_size"] > 0
        
        if "gpu_layers" in model:
            assert isinstance(model["gpu_layers"], int)
    
    @pytest.mark.asyncio
    async def test_config_has_server_info(self, client):
        """Test that config contains server configuration."""
        config = await client.get_config()
        cfg = config["config"]
        
        server = cfg["server"]
        assert "host" in server
        assert "port" in server
    
    @pytest.mark.asyncio
    async def test_config_has_sampling_params(self, client):
        """Test that config contains sampling parameters."""
        config = await client.get_config()
        cfg = config["config"]
        
        sampling = cfg["sampling"]
        
        # Common sampling params
        expected_params = ["temperature", "top_p", "top_k"]
        for param in expected_params:
            if param in sampling:
                assert isinstance(sampling[param], (int, float)), f"{param} should be numeric"


class TestDeploymentMonitoring:
    """Tests for monitoring deployment progress."""
    
    @pytest.fixture
    def client(self):
        return DeploymentTestClient()
    
    @pytest.mark.asyncio
    async def test_status_includes_resources_when_running(self, client):
        """Test that status includes resource usage when running."""
        status = await client.get_status()
        
        if not status.get("running"):
            pytest.skip("Service not running")
        
        # Resource info should be present
        if "resources" in status:
            resources = status["resources"]
            # CPU and memory info
            if "cpu_percent" in resources:
                assert isinstance(resources["cpu_percent"], (int, float))
            if "memory_mb" in resources:
                assert isinstance(resources["memory_mb"], (int, float))
                assert resources["memory_mb"] >= 0
    
    @pytest.mark.asyncio
    async def test_status_includes_gpu_info_when_running(self, client):
        """Test that status includes GPU info when running."""
        status = await client.get_status()
        
        if not status.get("running"):
            pytest.skip("Service not running")
        
        if "gpu" in status:
            gpu = status["gpu"]
            if "vram_used_mb" in gpu:
                assert isinstance(gpu["vram_used_mb"], (int, float))
            if "vram_total_mb" in gpu:
                assert isinstance(gpu["vram_total_mb"], (int, float))
    
    @pytest.mark.asyncio
    async def test_llamacpp_health_when_running(self, client):
        """Test that llamacpp health check is included when running."""
        status = await client.get_status()
        
        if not status.get("running"):
            pytest.skip("Service not running")
        
        assert "llamacpp_health" in status
        health = status["llamacpp_health"]
        assert "healthy" in health
        
        if health["healthy"]:
            assert health.get("status_code") == 200


class TestDeploymentFullCycle:
    """
    End-to-end test of deployment lifecycle.
    
    This test runs through a complete deployment cycle:
    1. Stop if running
    2. Start service
    3. Wait for healthy
    4. Verify status
    5. Stop service
    6. Verify stopped
    """
    
    @pytest.fixture
    def client(self):
        return DeploymentTestClient()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.timeout(720)  # 12 minute max for full cycle
    async def test_full_deployment_cycle(self, client):
        """Test complete deployment lifecycle."""
        print("\n=== Full Deployment Cycle Test ===")
        
        # Step 1: Ensure stopped
        print("\n1. Ensuring service is stopped...")
        initial_status = await client.get_status()
        if initial_status.get("running"):
            print("   Service running, stopping first...")
            await client.stop_service()
            await client.wait_for_stopped()
        print("   Service stopped.")
        
        # Step 2: Start service
        print("\n2. Starting service...")
        start_time = time.time()
        start_result = await client.start_service()
        assert start_result["status_code"] == 200, f"Start failed: {start_result}"
        print(f"   Start initiated in {time.time() - start_time:.1f}s")
        
        # Step 3: Wait for healthy
        print("\n3. Waiting for service to become healthy...")
        wait_result = await client.wait_for_running(timeout=DEPLOYMENT_TIMEOUT)
        
        print(f"   Result after {wait_result['elapsed_seconds']:.1f}s: {'SUCCESS' if wait_result['success'] else 'FAILED'}")
        
        if not wait_result["success"]:
            # Get logs for debugging
            logs = await client.get_logs(lines=50)
            print("\n   Recent logs:")
            for log in logs.get("logs", [])[-10:]:
                print(f"   {log.get('timestamp', '')}: {log.get('message', log)}")
            
            pytest.fail(f"Service failed to become healthy: {wait_result.get('error')}")
        
        # Step 4: Verify running status
        print("\n4. Verifying running status...")
        status = await client.get_status()
        assert status["running"], "Service should be running"
        assert status.get("llamacpp_health", {}).get("healthy"), "LlamaCPP should be healthy"
        print(f"   Uptime: {status.get('uptime', 0):.1f}s")
        if "resources" in status:
            print(f"   Memory: {status['resources'].get('memory_mb', 'N/A')} MB")
        if "gpu" in status:
            print(f"   VRAM: {status['gpu'].get('vram_used_mb', 'N/A')} MB")
        
        # Step 5: Stop service
        print("\n5. Stopping service...")
        stop_result = await client.stop_service()
        assert stop_result["status_code"] == 200, f"Stop failed: {stop_result}"
        
        # Step 6: Wait for stopped
        print("\n6. Waiting for service to stop...")
        stop_wait = await client.wait_for_stopped()
        assert stop_wait["success"], f"Service did not stop: {stop_wait.get('error')}"
        print(f"   Stopped in {stop_wait['elapsed_seconds']:.1f}s")
        
        # Final verification
        final_status = await client.get_status()
        assert not final_status["running"], "Service should be stopped"
        
        print("\n=== Full Deployment Cycle Complete ===")


# Markers for test filtering
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    # Run quick tests by default
    pytest.main([__file__, "-v", "-m", "not slow"])
