"""
Shared base imports and utilities for manager classes.
"""
import os
import asyncio
import subprocess
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

from enhanced_logger import enhanced_logger as logger

# Try to import docker, fallback to subprocess if not available
try:
    import docker
    DOCKER_AVAILABLE = True
    # Try different approaches to initialize Docker client
    docker_client = None
    
    # Method 1: Default from environment
    try:
        docker_client = docker.from_env()
        docker_client.ping()  # Verify connection
    except Exception as e1:
        logger.debug(f"Docker from_env() failed: {e1}")
        # Method 2: Try specific socket
        try:
            docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')
            docker_client.ping()
        except Exception as e2:
            logger.debug(f"Docker direct socket failed: {e2}")
            # Method 3: Keep client None, we'll use CLI fallback
            docker_client = None
            
    if docker_client is None:
        logger.warning("All Docker connection methods failed, fallback_mode='subprocess'")
        
except ImportError:
    DOCKER_AVAILABLE = False
    docker_client = None
    logger.warning("Docker SDK not available, fallback_mode='subprocess'")
except Exception as e:
    DOCKER_AVAILABLE = False
    docker_client = None
    logger.warning(f"Docker client initialization failed with error='{e}', fallback_mode='subprocess'")
